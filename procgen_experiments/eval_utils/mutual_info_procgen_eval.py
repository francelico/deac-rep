import gc
from typing import Callable, Any, List, NamedTuple
import sys
import time
import wandb
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from rich.pretty import pprint
from tensorboardX import SummaryWriter

import utils
from utils import job_util, model_util, mi_util, log_util, repmetric_util

# jax.config.update('jax_platform_name', 'cpu')

@dataclass
class Args:
  run_dir: str = "runs"
  "the directory where the experiment results are stored"
  wandb_dir: str = "wandb"
  "the directory where the wandb results are stored"
  exp_name: str = "change_me"
  "the name of the experiment to load"
  env_id: str = "BigfishEasy-v0"
  "the id of the environment of the experiment to load"
  seed: int = 1
  "random seed"
  total_timesteps: int = 65536
  "the total number of timesteps to run during evaluation"
  num_envs: int = 64
  "the number of parallel game environments"
  downsample_to_n: int = 4096
  "the number of samples to use to compute mi Should be less than total_timesteps/2"
  upload_to_wandb_run: bool = False
  "if True, we will upload the experiment results to the corresponding wandb run"
  wandb_project_name: str = "cleanRL"
  "the wandb's project name"
  wandb_entity: str = None
  "the entity (team) of wandb's project"
  wandb_group: str = None
  "the wandb group to add this run to"
  overwrite_completed: bool = False
  "if True, we will overwrite the results of a past evaluation"
  evaluate_mi_zo_train_only: bool = False
  "if True, we will only evaluate I(Z;O), on the training set only"

class Storage(NamedTuple):
  obs: list
  dones: list
  actions: list
  logprobs: list
  logits: list
  values: list
  aux_values: list
  aux_advantages: list
  reps_a: list
  reps_c: list
  nreps_a: list
  nreps_c: list
  rewards: list
  truncations: list
  terminations: list
  level_ids: list

# @jax.jit
def flatten(arr):
  return arr.reshape(arr.shape[0] * arr.shape[1], -1)

# @jax.jit
def prepare_data(storage: List[Storage]) -> Storage:
  return jax.tree_map(lambda *xs: jnp.stack(xs), *storage)

def step_envs_once(envs, action):
  cpu_action = np.array(action)
  next_obs, next_reward, next_done, next_info = envs.step(cpu_action)

  # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
  # so we use our own truncated flag
  next_truncated = next_info["elapsed_step"] >= envs.spec.config.max_episode_steps
  next_ts = next_info["elapsed_step"]

  # Procgen only - info:terminated and info:reward are missing in the procgen envstate
  next_terminated = next_done * (1 - next_truncated)
  next_info["terminated"] = next_terminated.copy()
  next_info["truncated"] = next_truncated.copy()
  next_info["reward"] = next_reward.copy()

  return next_obs, next_reward, next_done, next_terminated, next_truncated, next_ts, next_info

def evaluate_mi(
    checkpoint_path: str,
    make_env: Callable,
    make_agent_fn: Callable,
    eval_fns: List[Callable],
    run_args: Any,
    total_timesteps: int,
    seed=1,
    num_levels: int = 0,
    start_level: int = 0,
    num_envs=1,
):
  out = {}
  eval_mi_start = time.time()
  assert total_timesteps % num_envs == 0, "Ensure --total_timesteps divisible by --num_envs"
  envs = make_env(run_args.env_id, seed, num_envs=num_envs, num_levels=num_levels, start_level=start_level)()
  out['walltime:make_env_time'] = time.time() - eval_mi_start
  key = jax.random.PRNGKey(seed)

  make_agent_start = time.time()
  agent_state, model_modules, key = make_agent_fn(args=run_args, envs=envs, key=key, print_model=False)
  agent_state, _, _ = \
    job_util.restore_agent_state(
      checkpoint_path,
      (agent_state, None, None),
    )
  params = agent_state.get_params()
  out['walltime:make_agent_time'] = time.time() - make_agent_start

  print("Loaded agent from checkpoint")

  @jax.jit
  def get_model_outputs(
      params,
      obs: np.ndarray,
      key: jax.random.PRNGKey,
  ):
    obs = jnp.array(obs)
    reps_a = model_modules['actor_base'].apply(params.actor_params.base_params, obs)
    logits = model_modules['policy_head'].apply(params.actor_params.policy_head_params, reps_a)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

    if 'critic_base' in model_modules:
      reps_c = model_modules['critic_base'].apply(params.critic_params.base_params, obs)
      value_pred = model_modules['value_head'].apply(params.critic_params.value_head_params, reps_c)
    else:
      reps_c = model_modules['actor_base'].apply(params.actor_params.base_params, obs)
      value_pred = model_modules['value_head'].apply(params.actor_params.value_head_params, reps_c)
    if 'auxiliary_head' in model_modules:
      aux_value_pred = model_modules['auxiliary_head'].apply(params.actor_params.auxiliary_head_params, reps_a)
    else:
      aux_value_pred = jnp.zeros_like(value_pred)
    if 'auxiliary_advantage_head' in model_modules:
      aux_adv_pred = model_modules['auxiliary_advantage_head'].apply(params.actor_params.auxiliary_advantage_head_params, reps_a, action)
    else:
      aux_adv_pred = jnp.zeros_like(value_pred)

    return obs, action, logprob, logits, value_pred.squeeze(), aux_value_pred.squeeze(), aux_adv_pred.squeeze(), reps_a, reps_c, key

  next_obs = envs.reset()
  next_done = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
  next_terminated = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
  next_truncated = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
  storage = []

  rollout_start = time.time()
  for _ in range(0, total_timesteps // num_envs):
    cached_next_obs = next_obs
    cached_next_done = next_done
    cached_next_terminated = next_terminated
    cached_next_truncated = next_truncated

    cached_next_obs, action, logprob, logits, value_pred, aux_value_pred, aux_adv_pred, reps_a, reps_c, key = \
      jax.lax.stop_gradient(get_model_outputs(params, cached_next_obs, key))
    next_obs, next_reward, next_done, next_terminated, next_truncated, next_ts, next_info = \
      step_envs_once(envs, action)
    nreps_a = jax.lax.stop_gradient(model_modules['actor_base'].apply(params.actor_params.base_params, np.array(next_obs)))
    if 'critic_base' in model_modules:
      nreps_c = jax.lax.stop_gradient(model_modules['critic_base'].apply(params.critic_params.base_params, np.array(next_obs)))
    else:
      nreps_c = nreps_a

    storage.append(
      Storage(
        obs=cached_next_obs,
        dones=cached_next_done,
        actions=action,
        logprobs=logprob,
        logits=logits,
        values=value_pred,
        aux_values=aux_value_pred,
        aux_advantages=aux_adv_pred,
        reps_a=reps_a,
        reps_c=reps_c,
        nreps_a=nreps_a,
        nreps_c=nreps_c,
        rewards=next_reward,
        truncations=cached_next_truncated,
        terminations=cached_next_terminated,
        level_ids=next_info['prev_level_seed'],
      )
    )

  print(f"Rollout collected in {time.time() - rollout_start}s")

  storage = prepare_data(storage)
  envs.close()
  out['walltime:eval_mi_rollout_time'] = time.time() - rollout_start
  print(f"Rollout phase completed in {out['walltime:eval_mi_rollout_time']}s")
  ref = time.time()
  for i, fn in enumerate(eval_fns):
    stat, key = fn(storage, key)
    print(f"MI eval on levelset completed for fn n{i} in {time.time() - ref}s")
    ref = time.time()
    out.update(stat)
  out['walltime:eval_mi_time'] = time.time() - eval_mi_start
  print(f"MI eval on levelset completed in {out['walltime:eval_mi_time']}s")
  return out

def compute_mi_zo(storage, key, n_samples):
  def downsample(storage, key, n_samples):
    key, subkey = jax.random.split(key)
    reps_a, reps_c, obs = jax.tree_map(lambda x: jax.random.permutation(subkey, flatten(x))[0:n_samples],
      [storage.reps_a, storage.reps_c, storage.obs])
    return reps_a, reps_c, obs

  start = time.time()
  reps_a, reps_c, obs = downsample(storage, key, n_samples)
  mi_reps_a = mi_util.compute_mi_cc(obs, reps_a, n_neighbors=3)
  mi_reps_c = mi_util.compute_mi_cc(obs, reps_c, n_neighbors=3)

  key, subkey = jax.random.split(key)
  randz_proj = model_util.random_projection(obs, subkey, reps_a.shape[-1])
  mi_rand_proj = mi_util.compute_mi_cc(obs, randz_proj, n_neighbors=3)

  return {
    "reps_actor:mi_zO": mi_reps_a,
    "reps_critic:mi_zO": mi_reps_c,
    "random_projection:mi_zO": mi_rand_proj,
    "walltime:mi_zo_time": time.time() - start,
  }, key

def compute_mi_levelset(storage, key, n_samples):

  # @functools.partial(jax.jit, static_argnames=('n_samples',))
  def downsample(storage, key, n_samples):
    key, subkey = jax.random.split(key)
    reps_a, reps_c, obs, level_ids = jax.tree_map(lambda x: jax.random.permutation(subkey, flatten(x))[0:n_samples],
      [storage.reps_a, storage.reps_c, storage.obs, storage.level_ids])
    return reps_a, reps_c, obs, level_ids

  start = time.time()
  reps_a, reps_c, obs, level_ids = downsample(storage, key, n_samples)
  mi_reps_a = mi_util.compute_mi_cd(reps_a, np.ravel(level_ids), n_neighbors=3)
  mi_reps_c = mi_util.compute_mi_cd(reps_c, np.ravel(level_ids), n_neighbors=3)
  mi_obs = mi_util.compute_mi_cd(obs, np.ravel(level_ids), n_neighbors=3)

  key, subkey = jax.random.split(key)
  randz_proj = model_util.random_projection(obs, subkey, reps_a.shape[-1])
  mi_rand_proj = mi_util.compute_mi_cd(randz_proj, np.ravel(level_ids), n_neighbors=3)

  return {
    "reps_actor:mi_zL" : mi_reps_a,
    "reps_critic:mi_zL": mi_reps_c,
    "random_projection:mi_zL": mi_rand_proj,
    "obs:mi_zL": mi_obs,
    "walltime:mi_levelset_time": time.time() - start,
  }, key

def compute_mi_vf(storage, key, n_samples, gamma):

  def compute_true_returns(storage, gamma):
    nsteps, nenvs = storage.rewards.shape[:2]
    returns = np.zeros((nsteps, nenvs))
    returns[-1] = storage.rewards[-1]
    completed_episodes = np.zeros((nsteps, nenvs), dtype=jnp.bool_)
    episodes_last_ts_trunc = np.zeros((nsteps, nenvs), dtype=jnp.bool_)
    completed_episodes[-1] = storage.dones[-1]
    episodes_last_ts_trunc[-1] = storage.truncations[-1]
    for i in range(nsteps - 2, -1, -1):
      returns[i] = storage.rewards[i] + gamma * returns[i + 1] * (1 - storage.dones[i])
      completed_episodes[i] = np.maximum(completed_episodes[i + 1], storage.dones[i])
      episodes_last_ts_trunc[i] = np.maximum(episodes_last_ts_trunc[i + 1], storage.truncations[i]) * (
            1 - storage.terminations[i])

    return returns, completed_episodes + (1 - episodes_last_ts_trunc)

  start = time.time()
  nsteps, nenvs = storage.rewards.shape[:2]
  returns, pos_mask = compute_true_returns(storage, gamma)
  p = pos_mask * jnp.ones((nsteps, nenvs), dtype=jnp.float32)
  p = p / jnp.sum(p)
  key, subkey = jax.random.split(key)
  idx = jax.random.choice(subkey, jnp.arange(nsteps*nenvs), shape=(n_samples,), p=p.ravel(), replace=False)
  returns = np.array(returns.reshape(-1,1)[idx])
  reps_a = np.array(flatten(storage.reps_a)[idx])
  reps_c = np.array(flatten(storage.reps_c)[idx])
  obs = np.array(flatten(storage.obs)[idx])
  value_err = np.abs(np.array(flatten(storage.values)[idx]) - returns).mean()
  aux_value_err = np.abs(np.array(flatten(storage.aux_values)[idx]) - returns).mean()
  mi_zV_actor = mi_util.compute_mi_cc(reps_a, returns, n_neighbors=3)
  mi_zV_critic = mi_util.compute_mi_cc(reps_c, returns, n_neighbors=3)
  mi_zV_obs = mi_util.compute_mi_cc(obs, returns, n_neighbors=3)

  key, subkey = jax.random.split(key)
  randz_proj = model_util.random_projection(obs, subkey, reps_a.shape[-1])
  mi_zV_randz_p = mi_util.compute_mi_cc(randz_proj, returns, n_neighbors=3)

  return {
    "reps_actor:mi_zV": mi_zV_actor,
    "reps_critic:mi_zV": mi_zV_critic,
    "obs:mi_zV": mi_zV_obs,
    "random_projection:mi_zV": mi_zV_randz_p,
    "value_error": value_err,
    "aux_value_error": aux_value_err,
    "walltime:mi_vf_time": time.time() - start,
  }, key

def compute_mi_markov(storage, key, n_samples):
  # I(Z,Z',A) = I((Z,Z'),A) + I(Z,Z')

  def subsample_storage(storage, key, n_samples):
    """Subsample n_samples (reps, nreps, actions) from the storage,
    while enforcing the following conditions:
    - no overlap between nreps and reps i.e. cannot sample both
      (storage.reps[idx], storage.reps[idx+1], storage.actions[idx]) and
      (storage.reps[idx+1], storage.reps[idx+2])
    - no (rep, nreps) sampled across episode termination, i.e.:
      cannot sample (storage.reps[idx], storage.rep[idx+1], storage.actions[idx])
      where storage.dones[idx] = 1
    - no repetition of samples
    """
    nsteps, nenvs = storage.actions.shape[:2]
    even_idx = jnp.arange(0, nsteps - 1, 2)
    pos_mask = jnp.zeros((nsteps, nenvs), dtype=jnp.bool_)
    pos_mask = pos_mask.at[even_idx].set(True, indices_are_sorted=True)
    pos_mask = jnp.logical_and(pos_mask, ~storage.dones.reshape(nsteps, nenvs))
    p = pos_mask * jnp.ones((nsteps, nenvs), dtype=jnp.float32)
    p = p / jnp.sum(p)

    idx = jax.random.choice(key, jnp.arange(nsteps*nenvs), shape=(n_samples,), p=p.ravel(), replace=False)
    actions = np.array(jnp.ravel(flatten(storage.actions[:-1]))[idx])
    reps_a = np.array(flatten(storage.reps_a)[idx])
    nreps_a = np.array(flatten(storage.reps_a)[idx + 1])
    reps_c = np.array(flatten(storage.reps_c)[idx])
    nreps_c = np.array(flatten(storage.reps_c)[idx + 1])
    obs = np.array(flatten(storage.obs)[idx])
    nobs = np.array(flatten(storage.obs)[idx + 1])
    return actions, reps_a, nreps_a, reps_c, nreps_c, obs, nobs

  time_start = time.time()
  key, subkey = jax.random.split(key)
  actions, reps_a, nreps_a, reps_c, nreps_c, obs, nobs = subsample_storage(storage, subkey, n_samples)

  def compute_muts(st, nst, act, n_neighbors=3):
    mi_zz = mi_util.compute_mi_cc(st, nst, n_neighbors)
    mi_zz_a = mi_util.compute_mi_cd(np.hstack((st, nst)), act, n_neighbors)
    mi_za = mi_util.compute_mi_cd(st, act, n_neighbors)
    return mi_zz, mi_zz_a, mi_za

  mi_zz_actor, mi_zz_a_actor, mi_za_actor = compute_muts(reps_a, nreps_a, actions, n_neighbors=3)
  mi_zz_critic, mi_zz_a_critic, mi_za_critic = compute_muts(reps_c, nreps_c, actions, n_neighbors=3)
  mi_zz_obs, mi_zz_a_obs, mi_za_obs = compute_muts(obs, nobs, actions, n_neighbors=3)

  key, subkey = jax.random.split(key)
  randz_proj = model_util.random_projection(obs, subkey, reps_a.shape[-1])
  nrandz_proj = model_util.random_projection(nobs, subkey, reps_a.shape[-1])
  mi_zz_randz_p, mi_zz_a_randz_p, mi_za_randz_p = compute_muts(randz_proj, nrandz_proj, actions, n_neighbors=3)

  total_c_actor = mi_zz_actor + mi_zz_a_actor
  total_c_critic = mi_zz_critic + mi_zz_a_critic
  total_c_obs = mi_zz_obs + mi_zz_a_obs
  total_c_randz_p = mi_zz_randz_p + mi_zz_a_randz_p
  return {
    "reps_actor:mi_zz": mi_zz_actor,
    "reps_actor:mi_zz-a": mi_zz_a_actor,
    "reps_actor:mi_za": mi_za_actor,
    "reps_actor:mi_zza": total_c_actor,
    "reps_critic:mi_zz": mi_zz_critic,
    "reps_critic:mi_zz-a": mi_zz_a_critic,
    "reps_critic:mi_za": mi_za_critic,
    "reps_critic:mi_zza": total_c_critic,
    "obs:mi_zz": mi_zz_obs,
    "obs:mi_zz-a": mi_zz_a_obs,
    "obs:mi_za": mi_za_obs,
    "obs:mi_zza": total_c_obs,
    "random_projection:mi_zz": mi_zz_randz_p,
    "random_projection:mi_zz-a": mi_zz_a_randz_p,
    "random_projection:mi_zza": total_c_randz_p,
    "random_projection:mi_za": mi_za_randz_p,
    "walltime:mi_markov_time": time.time() - time_start,
  }, key

def compute_rep_stats(storage, key, model_modules, num_steps, num_rollouts):
  # Consider:
  # implement entk[grad], norm[grad] from https://github.com/google/dopamine/blob/4552f69af4763053d87ee4ce6d3da59ca3232f3c/dopamine/labs/moes/agents/full_rainbow_moe_agent.py#L253
  # implement stiffness = (grad(s1).T @ grad(s2)) / (||grad(s1)|| ||grad(s2)||)
  # But need to pick a loss function to compute the gradient of.

  # for simplicity here we downsample to num_rollout slices of len num_steps.
  key, subkey = jax.random.split(key)
  start_steps = jax.random.randint(subkey, (num_rollouts,), 0, len(storage.reps_a) - num_steps)
  storage = jax.tree_map(lambda x: jnp.stack([x[start:start + num_steps, i] for i, start in enumerate(start_steps)], 1), storage)
  gc.collect()

  start = time.time()
  stats = {}
  # Compute representation stats
  key, subkey = jax.random.split(key)

  # Critic
  stats.update(repmetric_util.compute_nn_latent_stats(subkey, flatten(storage.reps_c), flatten(storage.nreps_c), flatten(storage.dones), label='reps_critic'))
  stats.update(repmetric_util.compute_nn_latent_out_stats(flatten(storage.reps_c), flatten(storage.values),
                                                          metric_fn_out=utils.absolute_diff,
                                                          label='reps_critic:value:l1'))

  # Actor
  stats.update(repmetric_util.compute_nn_latent_stats(subkey, flatten(storage.reps_a), flatten(storage.nreps_a), flatten(storage.dones), label='reps_actor'))
  stats.update(repmetric_util.compute_nn_latent_out_stats(flatten(storage.reps_a), flatten(storage.logits),
                                                          metric_fn_out=utils.total_variation,
                                                          label='reps_actor:logits:tv'))
  stats.update(repmetric_util.compute_nn_latent_out_stats(flatten(storage.reps_a), flatten(storage.logits),
                                                          metric_fn_out=utils.jsd_categorical_categorical,
                                                          label='reps_actor:logits:jsd'))
  if 'auxiliary_advantage_head' in model_modules:
    stats.update(repmetric_util.compute_nn_latent_out_stats(flatten(storage.reps_a), flatten(storage.aux_advantages),
                                                            metric_fn_out=utils.absolute_diff,
                                                            label='reps_actor:aux_adv:l1'))
  if 'auxiliary_head' in model_modules:
    stats.update(repmetric_util.compute_nn_latent_out_stats(flatten(storage.reps_a), flatten(storage.aux_values),
                                                            metric_fn_out=utils.absolute_diff,
                                                            label='reps_actor:aux_value:l1'))
  stats = {k: v.item() for k, v in stats.items()}
  stats['walltime:rep_stats_time'] = time.time() - start
  return stats, key

if __name__ == "__main__":
  import tyro
  args = tyro.cli(Args)

  run_name, _ = job_util.generate_run_name(args, args.run_dir, resume_only=True)
  checkpoint_path = f"{args.run_dir}/{run_name}/{args.exp_name}.ckpt"
  walltimes = {}

  wandb_import_start = time.time()
  runscript = log_util.import_script_from_wandb(run_name, args, check_local_dir_for_file=True)
  walltimes['walltime:wandb_script_import'] = time.time() - wandb_import_start
  print(f"Imported runscript from wandb: {runscript.__name__}. \n"
        f"It took {walltimes['walltime:wandb_script_import']}s")

  preload_start = time.time()
  run_args = runscript.Args()
  runstate = job_util.RunState(checkpoint_path, save_fn=partial(job_util.save_agent_state, unreplicate=False))
  _, run_args, runstate.metadata = \
    job_util.restore_agent_state(
      checkpoint_path,
      (None, run_args, runstate.metadata),
      lax_entries = {str(2): "update"}
    )
  dummy_env = runscript.make_env(run_args.env_id, 0, num_envs=1)()
  agent_state, model_modules, key = runscript.make_agent(args=run_args, envs=dummy_env, key=jax.random.PRNGKey(0),
                                                         print_model=False)
  walltimes['walltime:preload_time'] = time.time() - preload_start


  if runstate.metadata['post_eval_completed']:
    if not args.overwrite_completed:
      print(f"Post eval already completed for {run_name}, skipping")
      sys.exit()
    else:
      print(f"Overwriting existing post eval results for {run_name}")

  assert run_args.env_id == args.env_id, "The env_id in the checkpoint does not match the env_id in the args"
  assert run_args.exp_name == args.exp_name, "The exp_name in the checkpoint does not match the exp_name in the args"

  config = vars(args)
  slurm_metadata = job_util.gather_slurm_metadata()
  if slurm_metadata:
    print("SLURM METADATA:")
    pprint(slurm_metadata)
    config.update(slurm=slurm_metadata)

  wandb_init_start = time.time()
  if args.upload_to_wandb_run:
    wandb.init(
      project=args.wandb_project_name,
      entity=args.wandb_entity,
      group=args.wandb_group,
      dir=args.wandb_dir,
      sync_tensorboard=True,
      config=config,
      name=run_name,
      id=run_name,
      resume="allow",
      monitor_gym=True,
      # save_code=True,
    )
  walltimes['walltime:wandb_init'] = time.time() - wandb_init_start
  writer_init_start = time.time()
  writer = SummaryWriter(f"{args.run_dir}/{run_name}", flush_secs=50, purge_step=run_args.total_timesteps + 2)
  walltimes['walltime:writer_init'] = time.time() - writer_init_start

  if args.evaluate_mi_zo_train_only:
    eval_fns_train = [
      partial(compute_mi_zo, n_samples=args.downsample_to_n),
    ]

    print(f"Starting I(Z,O) eval on train set")
    stats_train = evaluate_mi(
      checkpoint_path,
      runscript.make_env,
      runscript.make_agent,
      eval_fns_train,
      run_args,
      seed=args.seed,
      total_timesteps=args.total_timesteps,
      num_envs=args.num_envs,
      num_levels=run_args.num_train_levels
    )

    print("Saving results to wandb/tensorboard")
    for k, v in stats_train.items():
      if k.startswith('walltime'):
        writer.add_scalar(f"stats/{k}", v, run_args.total_timesteps + 3)
      else:
        writer.add_scalar(f"rep_eval_train/{k}", v, run_args.total_timesteps + 3)

      print("Representation eval stats [TRAIN SET]")
      pprint(stats_train)
      print("MI eval completed")
      time.sleep(60)
      dummy_env.close()
      sys.exit()

  eval_fns_train = [
    partial(compute_mi_levelset, n_samples=args.downsample_to_n),
    partial(compute_mi_markov, n_samples=args.downsample_to_n),
    partial(compute_mi_vf, n_samples=args.downsample_to_n, gamma=run_args.gamma),
    partial(compute_rep_stats, model_modules=model_modules, num_steps=run_args.num_steps,
              num_rollouts=args.downsample_to_n // run_args.num_steps),
  ]
  eval_fns_test = [
    partial(compute_mi_markov, n_samples=args.downsample_to_n),
    partial(compute_mi_vf, n_samples=args.downsample_to_n, gamma=run_args.gamma),
    partial(compute_rep_stats, model_modules=model_modules, num_steps=run_args.num_steps,
              num_rollouts=args.downsample_to_n // run_args.num_steps),
  ]

  print(f"Starting MI eval on test set")
  stats_test = evaluate_mi(
    checkpoint_path,
    runscript.make_env,
    runscript.make_agent,
    eval_fns_test,
    run_args,
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    num_envs=args.num_envs,
    start_level=run_args.num_train_levels
  )

  print(f"Starting MI eval on train set")
  stats_train = evaluate_mi(
    checkpoint_path,
    runscript.make_env,
    runscript.make_agent,
    eval_fns_train,
    run_args,
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    num_envs=args.num_envs,
    num_levels=run_args.num_train_levels
  )

  print("Saving results to wandb/tensorboard")
  for k, v in stats_train.items():
    if k.startswith('walltime'):
      writer.add_scalar(f"stats/{k}", v, run_args.total_timesteps + 2)
    else:
      writer.add_scalar(f"rep_eval_train/{k}", v, run_args.total_timesteps + 2)
  for k, v in stats_test.items():
    if k.startswith('walltime'):
      writer.add_scalar(f"stats/{k}", v, run_args.total_timesteps + 2)
    else:
      writer.add_scalar(f"rep_eval_test/{k}", v, run_args.total_timesteps + 2)
  for k, v in walltimes.items():
    writer.add_scalar(f"stats/{k}", v, run_args.total_timesteps + 2)

  print("Representation eval stats [TRAIN SET]")
  pprint(stats_train)
  print("Representation eval stats [TEST SET]")
  pprint(stats_test)

  agent_state, _, _ = \
    job_util.restore_agent_state(
      checkpoint_path,
      (agent_state, None, None),
    )
  runstate.after_post_eval(agent_state, run_args)
  print("MI eval completed")
  time.sleep(60)
  dummy_env.close()
