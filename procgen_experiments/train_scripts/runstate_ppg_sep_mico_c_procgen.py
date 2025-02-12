import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"
# Can be tried if you are running into issues with your specific GPU
# os.environ["TF_USE_NVLINK_FOR_PARALLEL_COMPILATION"] = "0" #NOTE: only necessary when driver only supports cuda 12.0
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import gc
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Sequence
import functools
from functools import partial
import envpool
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal, lecun_normal
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

import utils
from utils import job_util, constants, model_util, gym_util, kl_categorical_categorical, bisim_util, repmetric_util, \
  binary_cross_entropy_with_logits, data_augmentation

#TASKS
# [x] DCPG shared implemented
# [x] PPG shared implemented
# - PPO sep implemented?
# RUNS:
# [x] PPG shared,
# [x] DCPG shared
# - PPG sep:
  # - PPG sep + MICo sep rollouts
    # [x] with --detach_value_grads_policy_phase
    # [x] without it
  # - [x] PPG sep + dyna
  # - PPG sep + dyna + MICo
# - PPO sep:
  # - PPO sep
  # - PPO sep + dyna
  # - PPO sep + MICo
  # - PPO sep + dyna + MICo

@dataclass
class Args:
  exp_name: str = os.path.basename(__file__).rstrip(".py")
  "the name of this experiment"
  seed: int = 1
  "seed of the experiment"
  track: bool = False
  "if toggled, this experiment will be tracked with Weights and Biases"
  wandb_project_name: str = "cleanRL"
  "the wandb's project name"
  wandb_entity: str = None
  "the entity (team) of wandb's project"
  wandb_group: str = None
  "the wandb group to add this run to"
  capture_video: bool = False
  "whether to capture videos of the agent performances (check out `videos` folder)"
  save_model: bool = False
  "whether to save model into the `runs/{run_name}` folder"
  preemptible: bool = False
  "Whether to save the model on job preemption or timeout. SLURM only."
  start_from_checkpoint: bool = False
  "Begin training from checkpoint (if it exists). Set to True to allow resuming training on preemptible clusters."
  checkpoint_frequency: int = 0
  "Save a checkpoint every n seconds. Default 0 means signal-based checkpointing only."
  log_frequency: int = 32
  "the logging frequency of the model performance (in terms of `updates`)"
  eval_every_n_log_cycles: int = 2
  "eval the model on the test environments every n log cycles"
  num_eval_episodes: int = 1000
  "the number of episodes to evaluate the model on at the end of training (both on training and testing levels)"
  mi_eval_downsample_to_n: int = 4096
  "the number of samples to use to compute mi Should be less than mi_eval_total_timesteps/2"
  mi_eval_total_timesteps: int = 65536
  "the total number of timesteps to run during mi evaluation"

  # Algorithm specific arguments
  env_id: str = "BigfishEasy-v0"
  "the id of the environment"
  num_train_levels: int = 200
  "number of Procgen levels to use for training"
  total_timesteps: int = 25000000
  "total timesteps of the experiments"
  norm_and_clip_rewards: bool = True
  "whether to normalize and clip the rewards"
  learning_rate: float = 5e-4
  "the learning rate of the optimizer"
  adam_eps: float = 1e-8
  "the epsilon parameter of the Adam optimizer"
  local_num_envs: int = 32
  "the number of parallel game environments"
  num_actor_threads: int = 2
  "the number of actor threads to use"
  num_steps: int = 256
  "the number of steps to run in each environment per policy rollout"
  anneal_lr: bool = False
  "Toggle learning rate annealing for policy and value networks"
  gamma: float = 0.999
  "the discount factor gamma"
  gae_lambda: float = 0.95
  "the lambda for the general advantage estimation"
  num_minibatches: int = 8
  "the number of mini-batches in the policy phase"
  num_rollouts_in_aux_mb: int = 4
  """the number of rollouts in each minibatch during the auxiliary phase"""
  gradient_accumulation_steps: int = 1
  "the number of gradient accumulation steps before performing an optimization step"
  num_policy_phases: int = 32
  "the number of policy phases to run before running an auxiliary phase"
  num_policy_phase_batches_for_auxiliary_phase: int = -1
  "Run the auxiliary phase on data from the last N policy phase batches. " \
  "A value of -1 will automatically set this parameter to num_policy_phases"
  policy_phase_epochs: int = 1
  "the K epochs to run during the policy phase"
  auxiliary_phase_epochs: int = 6
  "the K epochs to run during the auxiliary phase"
  norm_adv: str = "batch"
  "Type of advantage normalization. Options: ['batch', 'minibatch', 'none']"
  clip_coef: float = 0.2
  "the surrogate clipping coefficient"
  ent_coef: float = 0.01
  "coefficient of the entropy"
  bc_coef: float = 1.0
  "coefficient for the behavior cloning loss during the auxiliary phase"
  max_grad_norm: float = 0.5
  "the maximum norm for the gradient clipping"
  channels: List[int] = field(default_factory=lambda: [16, 32, 32])
  "the channels of the CNN"
  hiddens: List[int] = field(default_factory=lambda: [256])
  "the hiddens size of the MLP"
  kernel_init_method: str = "ppg_cleanrl_procgen"
  "which reference implementation to follow for weight initialization"

  #Aux losses

  # drAC arguments
  drA_loss_coef: float = 0.0
  "coefficient for the Actor drAC loss"
  drC_loss_coef: float = 0.0
  "coefficient for the Critic drAC loss"
  data_augmentation_fn: str = "random_crop"
  "the data augmentation function to use"

  # aux value loss arg
  aux_vf_coef_aux_phase: float = 1.0
  "coefficient for the value prediction loss during the auxiliary phase"

  # Advantage loss arg
  adv_coef_aux_phase: float = 0.0
  "coefficient for the advantage loss applied to the actor representation during auxiliary phase"

  # Markov loss arg
  markov_coef_a: float = 0.0
  "coefficient for the Markov forward-inverse dynamics loss during the auxiliary phase for the Actor"
  markov_coef_c: float = 0.0
  "coefficient for the Markov forward-inverse dynamics loss during the auxiliary phase for the Critic"

  # MICo arguments
  detach_value_grads_policy_phase: bool = False
  "whether to detach gradients after the value head during the policy phase"
  mico_coef_actor_pol_phase: float = 0.0
  "coefficient for the MICO loss applied to the actor representation during policy phase"
  mico_coef_critic_pol_phase: float = 0.0
  "coefficient for the MICO loss applied to the critic representation during policy phase"
  mico_coef_actor_aux_phase: float = 0.0
  "coefficient for the MICO loss applied to the actor representation during auxiliary phase"
  mico_coef_critic_aux_phase: float = 0.0
  "coefficient for the MICo loss applied to the critic representation during auxiliary phase"
  target_smoothing_coef: float = 0.0
  "the coefficient for soft target network updates"
  enable_mico_pol_phase: bool = False
  "whether to enable/compute mico loss and components during policy phase"
  remove_same_rollout_pairs: bool = False
  "Whether to mask any pairs coming from the same rollouts when computing MICo"

  actor_device_ids: List[int] = field(default_factory=lambda: [0])
  "the device ids that actor workers will use"
  learner_device_ids: List[int] = field(default_factory=lambda: [0])
  "the device ids that learner workers will use"
  distributed: bool = False
  "whether to use `jax.distirbuted`"
  no_cuda: bool = False
  "whether to use CPU instead of GPU"
  concurrency: bool = False
  "whether to run the actor and learner concurrently"

  # runtime arguments to be filled in
  local_batch_size: int = 0
  local_minibatch_size: int = 0
  num_updates: int = 0
  world_size: int = 0
  local_rank: int = 0
  num_envs: int = 0
  batch_size: int = 0
  minibatch_size: int = 0
  aux_batch_rollouts: int = 0
  aux_batch_size: int = 0
  aux_minibatch_size: int = 0
  num_aux_minibatches: int = 0
  num_sharded_rollouts_aux_phase: int = 0
  global_learner_devices: Optional[List[str]] = None
  actor_devices: Optional[List[str]] = None
  learner_devices: Optional[List[str]] = None
  baseline_score: Optional[float] = None # the baseline score for the environment for reporting normalized scores

def make_env(env_id, seed, num_envs, num_levels=0, start_level=0):
  def thunk():
    envs = envpool.make(
      env_id,
      env_type="gym",
      num_envs=num_envs,
      seed=seed,
      num_levels=num_levels,
      start_level=start_level,
    )
    envs.num_envs = num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    return envs

  return thunk

def make_agent(args, envs, key, print_model=False):
  (actor_params, critic_params), modules, key = \
    make_ppg_model(args, envs, key, print_model=print_model)

  linear_schedule_fn = partial(linear_schedule, update_epochs=args.policy_phase_epochs)

  actor_state = TrainState.create(
    apply_fn=None,
    params=actor_params,
    tx=make_opt(args, linear_schedule_fn)
  )
  critic_state = TrainState.create(
    apply_fn=None,
    params=critic_params,
    tx=make_opt(args, linear_schedule_fn)
  )

  agent_state = AgentTrainStateWithTarget(actor_state, critic_state, actor_target_params=actor_params, critic_target_params=critic_params)

  return agent_state, modules, key

def make_ppg_model(args, envs, key, print_model=False):
  key, actor_base_key, policy_head_key, auxiliary_head_key, auxiliary_advantage_head_key, dyna_head_actor_key, \
  critic_base_key, value_head_key, dyna_head_critic_key = jax.random.split(key, 9)

  kernel_init_dict = model_util.set_layer_init_fn(args)

  actor_base = ResNetBase(args.channels, args.hiddens, kernel_init_dict)
  policy_head = PolicyHead(envs.single_action_space.n, kernel_init_dict["policy_head_dense"])
  dyna_head_actor = DynaHead(args.hiddens[-1], envs.single_action_space.n, kernel_init_dict["dyna_head_actor_dense"])
  auxiliary_head = ValueHead(kernel_init_dict["auxiliary_head_dense"])
  auxiliary_advantage_head = AdvantageHead(envs.single_action_space.n, kernel_init_dict["auxiliary_advantage_head_dense"])
  critic_base = ResNetBase(args.channels, args.hiddens, kernel_init_dict)
  value_head = ValueHead(kernel_init_dict["value_head_dense"])
  dyna_head_critic = DynaHead(args.hiddens[-1], envs.single_action_space.n, kernel_init_dict["dyna_head_critic_dense"])
  actor_base_params = actor_base.init(actor_base_key, np.array([envs.single_observation_space.sample()]))
  critic_base_params = critic_base.init(critic_base_key, np.array([envs.single_observation_space.sample()]))
  hidden_sample_actor = actor_base.apply(actor_base_params, np.array([envs.single_observation_space.sample()]))
  hidden_sample_critic = critic_base.apply(critic_base_params, np.array([envs.single_observation_space.sample()]))

  actor_params = ActorParams(
    actor_base_params,
    policy_head.init(policy_head_key, hidden_sample_actor),
    auxiliary_head.init(auxiliary_head_key, hidden_sample_actor),
    auxiliary_advantage_head.init(auxiliary_advantage_head_key, hidden_sample_actor,
                                  np.array([envs.single_action_space.sample()])),
    dyna_head_actor.init(dyna_head_actor_key, hidden_sample_actor, np.array([envs.single_action_space.sample()]),
                         hidden_sample_actor)
  )
  critic_params = CriticParams(
    critic_base_params,
    value_head.init(value_head_key, hidden_sample_critic),
    dyna_head_critic.init(dyna_head_critic_key, hidden_sample_critic, np.array([envs.single_action_space.sample()]),
                          hidden_sample_critic)
  )

  if print_model:
    print(actor_base.tabulate(actor_base_key, np.array([envs.single_observation_space.sample()])))
    print(policy_head.tabulate(policy_head_key, hidden_sample_actor))
    print(auxiliary_head.tabulate(auxiliary_head_key, hidden_sample_actor))
    print(auxiliary_advantage_head.tabulate(auxiliary_advantage_head_key, hidden_sample_actor, np.array([envs.single_action_space.sample()])))
    print(dyna_head_actor.tabulate(auxiliary_head_key, hidden_sample_actor,
                                   np.array([envs.single_action_space.sample()]), hidden_sample_actor))

    print(critic_base.tabulate(critic_base_key, np.array([envs.single_observation_space.sample()])))
    print(value_head.tabulate(value_head_key, hidden_sample_critic))
    print(dyna_head_critic.tabulate(dyna_head_critic_key, hidden_sample_critic,
                                    np.array([envs.single_action_space.sample()]), hidden_sample_critic))

  modules = {
    "actor_base": actor_base,
    "policy_head": policy_head,
    "auxiliary_head": auxiliary_head,
    "auxiliary_advantage_head": auxiliary_advantage_head,
    "dyna_head_actor": dyna_head_actor,
    "critic_base": critic_base,
    "value_head": value_head,
    "dyna_head_critic": dyna_head_critic,
  }

  return (actor_params, critic_params), modules, key

def make_opt(args, lr_scheduler_fn):
  return optax.MultiSteps(
    optax.chain(
      optax.clip_by_global_norm(args.max_grad_norm),
      optax.inject_hyperparams(optax.adam)(
        learning_rate=lr_scheduler_fn if args.anneal_lr else args.learning_rate, eps=args.adam_eps
      ),
    ),
    every_k_schedule=args.gradient_accumulation_steps,
  )

class ResidualBlock(nn.Module):
  channels: int
  kernel_init_fn: nn.initializers.Initializer

  @nn.compact
  def __call__(self, x):
    inputs = x
    x = nn.relu(x)
    x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn)(x)
    x = nn.relu(x)
    x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn)(x)
    return x + inputs

class ConvSequence(nn.Module):
  channels: int
  kernel_init_fn_conv: nn.initializers.Initializer = lecun_normal()
  kernel_init_fn_resblock: nn.initializers.Initializer = lecun_normal()

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn_conv)(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
    x = ResidualBlock(self.channels, self.kernel_init_fn_resblock)(x)
    x = ResidualBlock(self.channels, self.kernel_init_fn_resblock)(x)
    return x

class ResNetBase(nn.Module):
  channels: Sequence[int] = (16, 32, 32)
  hiddens: Sequence[int] = (256,)
  kernel_init_dict: dict = field(default_factory=lambda: {
    "convsequence_conv": lecun_normal(),
    "convsequence_resblock": lecun_normal(),
    "resnet_dense": orthogonal(2**0.5)
  })

  @nn.compact
  def __call__(self, x):
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = x / (255.0)
    for channels in self.channels:
      x = ConvSequence(channels,
                       kernel_init_fn_conv=self.kernel_init_dict["convsequence_conv"],
                       kernel_init_fn_resblock=self.kernel_init_dict["convsequence_resblock"])(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))
    for hidden in self.hiddens:
      x = nn.Dense(hidden, kernel_init=self.kernel_init_dict["resnet_dense"], bias_init=constant(0.0))(x)
      x = nn.relu(x)
    return x

class ValueHead(nn.Module):
  kernel_init_fn: nn.initializers.Initializer = orthogonal(1.0)

  @nn.compact
  def __call__(self, x):
    return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)

class AdvantageHead(nn.Module):
  action_dim: int
  kernel_init_fn: nn.initializers.Initializer = orthogonal(1.0)

  @nn.compact
  def __call__(self, x, actions):
    onehot_actions = nn.one_hot(actions, self.action_dim)
    gae_inputs = jnp.concatenate([x, onehot_actions], axis=1)
    return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(gae_inputs)

class PolicyHead(nn.Module):
  action_dim: int
  kernel_init_fn: nn.initializers.Initializer = orthogonal(0.01)

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.action_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)

class DynaHead(nn.Module):
  hidden_dim: int
  action_dim: int
  kernel_init_fn: nn.initializers.Initializer = orthogonal(0.01)

  @nn.compact
  def __call__(self, rep, action, next_rep):
    x = self.concat_input(rep, action, next_rep)
    x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)
    x = nn.relu(x)
    return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)

  def concat_input(self, rep, action, next_rep):
    onehot_action = jax.nn.one_hot(action, self.action_dim)
    return jnp.concatenate([rep, onehot_action, next_rep], axis=-1)

@flax.struct.dataclass
class ActorParams:
  base_params: flax.core.FrozenDict
  policy_head_params: flax.core.FrozenDict
  auxiliary_head_params: flax.core.FrozenDict
  auxiliary_advantage_head_params: Optional[flax.core.FrozenDict] = None
  dyna_head_params: Optional[flax.core.FrozenDict] = None

@flax.struct.dataclass
class CriticParams:
  base_params: flax.core.FrozenDict
  value_head_params: flax.core.FrozenDict
  dyna_head_params: Optional[flax.core.FrozenDict] = None

@flax.struct.dataclass
class AgentParams:
  actor_params: ActorParams
  critic_params: CriticParams

@flax.struct.dataclass
class AgentTrainStateWithTarget:
  actor_state: TrainState
  critic_state: TrainState
  actor_target_params: Optional[ActorParams] = None
  critic_target_params: Optional[CriticParams] = None

  def get_params(self):
    return AgentParams(
      actor_params=self.actor_state.params,
      critic_params=self.critic_state.params,
    )

  def get_target_params(self):
    return AgentParams(
      actor_params=self.actor_target_params,
      critic_params=self.critic_target_params,
    )

  def apply_gradients(self, grads):
    actor_state = self.actor_state.apply_gradients(grads=grads.actor_params)
    critic_state = self.critic_state.apply_gradients(grads=grads.critic_params)
    return AgentTrainStateWithTarget(actor_state, critic_state,
      actor_target_params=self.actor_target_params, critic_target_params=self.critic_target_params)

  def apply_gradients_actor(self, grads):
    actor_state = self.actor_state.apply_gradients(grads=grads.actor_params)
    return AgentTrainStateWithTarget(actor_state, self.critic_state,
      actor_target_params=self.actor_target_params, critic_target_params=self.critic_target_params)

  def apply_gradients_critic(self, grads):
    critic_state = self.critic_state.apply_gradients(grads=grads.critic_params)
    return AgentTrainStateWithTarget(self.actor_state, critic_state,
      actor_target_params=self.actor_target_params, critic_target_params=self.critic_target_params)

class PolPhaseStorage(NamedTuple):
  obs: list
  dones: list
  actions: list
  logprobs: list
  values: list
  env_ids: list
  rewards: list
  truncations: list
  terminations: list

class AuxPhaseStorage(NamedTuple):
  obs: list
  rewards: list
  actions: list
  target_values: list
  target_advantages: list
  dones: list

def get_global_step(policy_version, args):
  if policy_version < 0:
    return args.total_timesteps
  return (
      policy_version *                # past update cycles
      args.num_steps *                # env steps per update cycle (per env)
      args.local_num_envs *           # envs per actor thread
      args.num_actor_threads *        # actor threads on each device
      len(args.actor_device_ids) *    # devices on each jax process
      args.world_size                 # number of jax processes
  )

def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    initial_policy_version=0,
):
  assert initial_policy_version >= 0
  envs = make_env(
    args.env_id,
    args.seed + jax.process_index() + device_thread_id,
    args.local_num_envs,
    num_levels=args.num_train_levels,
  )()
  len_actor_device_ids = len(args.actor_device_ids)
  global_step = get_global_step(initial_policy_version, args)
  start_step = global_step
  start_time = time.time()
  
  if args.eval_every_n_log_cycles > 0:
    eval_envs = make_env(
      args.env_id,
      args.seed + jax.process_index() + device_thread_id,
      args.local_num_envs,
      start_level=args.num_train_levels
    )()
  else:
    eval_envs = None

  @jax.jit
  def get_model_outputs(
      params: AgentParams,
      obs: np.ndarray,
      key: jax.random.PRNGKey,
  ):
    obs = jnp.array(obs)
    actor_base = ResNetBase(args.channels, args.hiddens).apply(params.actor_params.base_params, obs)
    critic_base = ResNetBase(args.channels, args.hiddens).apply(params.critic_params.base_params, obs)
    logits = PolicyHead(envs.single_action_space.n).apply(params.actor_params.policy_head_params, actor_base)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    value_pred = ValueHead().apply(params.critic_params.value_head_params, critic_base)
    return obs, action, logprob, value_pred.squeeze(), key
  
  def step_envs_once(envs, action, comp_times):
    d2h_time_start = time.time()
    cpu_action = np.array(action)
    comp_times["d2h"] += time.time() - d2h_time_start

    env_send_time_start = time.time()
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

    comp_times["env_send"] += time.time() - env_send_time_start

    return next_obs, next_reward, next_done, next_terminated, next_truncated, next_ts, next_info, comp_times

  def update_rollout_stats(stats, next_info, term, trunc, cached_ep_returns, cached_ep_lengths):
    env_id = next_info["env_id"]
    cached_ep_returns[env_id] += next_info["reward"]
    stats["ep_returns"][env_id] = np.where(
      next_info["terminated"] + next_info["truncated"], cached_ep_returns[env_id], stats["ep_returns"][env_id]
    )
    cached_ep_returns[env_id] *= (1 - next_info["terminated"]) * (1 - next_info["truncated"])
    stats["levels_solved"][env_id] = np.where(
      next_info["terminated"] + next_info["truncated"], next_info["prev_level_complete"],
      stats["levels_solved"][env_id]
    )
    cached_ep_lengths[env_id] += 1
    stats["ep_lengths"][env_id] = np.where(
      next_info["terminated"] + next_info["truncated"], cached_ep_lengths[env_id], stats["ep_lengths"][env_id]
    )
    # accounts for extra envs.step() call
    cached_ep_lengths[env_id] *= (1 - next_info["terminated"]) * (1 - next_info["truncated"]) \
                                 * (1 - term) * (1 - trunc)

    return stats, cached_ep_returns, cached_ep_lengths

  cached_ep_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
  cached_ep_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
  rollout_stats = {
    "ep_returns": np.zeros((args.local_num_envs,), dtype=np.float32),
    "ep_lengths": np.zeros((args.local_num_envs,), dtype=np.int32),
    "levels_solved": np.zeros((args.local_num_envs,), dtype=np.bool_)
  }
  comp_times = {
    "param_queue_get": deque(maxlen=10),
    "rollout": deque(maxlen=10),
    "rollout_queue_put": deque(maxlen=10),
    "inference": 0,
    "storage": 0,
    "d2h": 0,
    "env_send": 0,
  }
  actor_policy_version = initial_policy_version
  next_obs = envs.reset()
  next_done = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
  next_terminated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
  next_truncated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
  next_ts = jnp.zeros(args.local_num_envs, dtype=jax.numpy.int32)
  if args.norm_and_clip_rewards:
    reward_wrapper = gym_util.RewardWrapper(
      normalize=True,
      clip=True,
      clip_coef=10,
      return_rms=gym_util.RunningMeanStd(np.zeros((), dtype=np.float32), np.ones((), dtype=np.float32), 1e-4),
      discounted_returns=np.zeros(args.local_num_envs),
      gamma=args.gamma,
    )

  if args.eval_every_n_log_cycles > 0:
    ev_cached_ep_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    ev_cached_ep_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    ev_rollout_stats = {
      "ep_returns": np.zeros((args.local_num_envs,), dtype=np.float32),
      "ep_lengths": np.zeros((args.local_num_envs,), dtype=np.int32),
      "levels_solved": np.zeros((args.local_num_envs,), dtype=np.bool_)
    }
    ev_comp_times = {
      "rollout": deque(maxlen=10),
      "inference": 0,
      "storage": 0,
      "d2h": 0,
      "env_send": 0,
    }
    ev_next_obs = eval_envs.reset()
    ev_next_terminated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
    ev_next_truncated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)

  @jax.jit
  def prepare_data(storage: List[PolPhaseStorage]) -> PolPhaseStorage:
    return jax.tree_map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

  for update in range(initial_policy_version + 1, args.num_updates + 2):
    update_time_start = time.time()
    # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
    # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
    # behind the learner's policy version
    params_queue_get_time_start = time.time()
    if args.concurrency:
      if update != initial_policy_version + 2:
        params = params_queue.get()
        # NOTE: block here is important because otherwise this thread will call
        # the jitted `get_model_outputs` function that hangs until the params are ready.
        # This blocks the `get_model_outputs` function in other actor threads.
        # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
        params.actor_params.base_params["params"]["Dense_0"][
          "kernel"
        ].block_until_ready()
        actor_policy_version += 1
    else:
      params = params_queue.get()
      actor_policy_version += 1
    comp_times["param_queue_get"].append(time.time() - params_queue_get_time_start)
    rollout_time_start = time.time()
    storage = []

    for _ in range(0, args.num_steps):
      cached_next_obs = next_obs
      cached_next_done = next_done
      cached_next_terminated = next_terminated
      cached_next_truncated = next_truncated
      global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size

      inference_time_start = time.time()
      cached_next_obs, action, logprob, value_pred, key = \
        get_model_outputs(params, cached_next_obs, key)
      comp_times["inference"] += time.time() - inference_time_start

      next_obs, next_reward, next_done, next_terminated, next_truncated, next_ts, next_info, comp_times = \
        step_envs_once(envs, action, comp_times)
      if args.norm_and_clip_rewards:
        reward_wrapper, next_reward = reward_wrapper.process_rewards(next_reward, next_terminated, next_truncated)

      storage_time_start = time.time()
      env_id = next_info["env_id"]
      storage.append(
        PolPhaseStorage(
          obs=cached_next_obs,
          dones=cached_next_done,
          actions=jax.lax.stop_gradient(action),
          logprobs=jax.lax.stop_gradient(logprob),
          values=jax.lax.stop_gradient(value_pred),
          env_ids=env_id,
          rewards=next_reward,
          truncations=cached_next_truncated,
          terminations=cached_next_terminated,
        )
      )
      comp_times["storage"] += time.time() - storage_time_start
      rollout_stats, cached_ep_returns, cached_ep_lengths = \
        update_rollout_stats(rollout_stats, next_info, cached_next_terminated, cached_next_truncated,
                             cached_ep_returns, cached_ep_lengths)

    comp_times["rollout"].append(time.time() - rollout_time_start)

    partitioned_storage = prepare_data(storage)
    sharded_storage = PolPhaseStorage(
      *list(map(lambda x: jax.device_put_sharded(x, devices=learner_devices), partitioned_storage))
    )
    # next_obs, next_done, next_terminated are still in the host
    sharded_last_obs = jax.device_put_sharded(np.split(next_obs, len(learner_devices)), devices=learner_devices)
    sharded_last_done = jax.device_put_sharded(np.split(next_done, len(learner_devices)), devices=learner_devices)
    sharded_last_term = jax.device_put_sharded(np.split(next_terminated, len(learner_devices)),
                                               devices=learner_devices)
    payload = (
      global_step,
      actor_policy_version,
      update,
      sharded_storage,
      sharded_last_obs,
      sharded_last_done,
      sharded_last_term,
      np.mean(comp_times["param_queue_get"]),
      device_thread_id,
    )
    rollout_queue_put_time_start = time.time()
    rollout_queue.put(payload)
    comp_times["rollout_queue_put"].append(time.time() - rollout_queue_put_time_start)

    if update % args.log_frequency == 0:
      avg_episodic_return = np.mean(rollout_stats['ep_returns'])
      avg_norm_episodic_return = avg_episodic_return / args.baseline_score if args.baseline_score else 0
      if device_thread_id == 0:
        print(
          f"global_step={global_step}, "
          f"avg_episodic_return={avg_episodic_return}, "
          f"rollout_time={np.mean(comp_times['rollout'])}"
        )
        print("SPS:", int((global_step - start_step) / (time.time() - start_time)))
      writer.add_scalar("stats/rollout_time", np.mean(comp_times["rollout"]), global_step)
      writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
      writer.add_scalar("charts/avg_norm_episodic_return", avg_norm_episodic_return, global_step)
      writer.add_scalar("charts/avg_nreward_per_ts", next_reward.mean(), global_step)
      writer.add_scalar("charts/max_nreward_diff", next_reward.max() - next_reward.min(), global_step)
      writer.add_scalar("charts/avg_reward_per_ts", next_info['reward'].mean(), global_step)
      writer.add_scalar("charts/max_reward_diff", next_info['reward'].max() - next_info['reward'].min(), global_step)
      writer.add_scalar("charts/avg_value_prediction", value_pred.mean(), global_step)

      writer.add_scalar("charts/avg_solved_rate", np.mean(rollout_stats["levels_solved"]), global_step)
      writer.add_scalar("charts/avg_episodic_length", np.mean(rollout_stats["ep_lengths"]), global_step)
      writer.add_scalar("stats/params_queue_get_time", np.mean(comp_times["param_queue_get"]), global_step)
      writer.add_scalar("stats/inference_time", comp_times["inference"], global_step)
      writer.add_scalar("stats/storage_time", comp_times["storage"], global_step)
      writer.add_scalar("stats/d2h_time", comp_times["d2h"], global_step)
      writer.add_scalar("stats/env_send_time", comp_times["env_send"], global_step)
      writer.add_scalar("stats/rollout_queue_put_time", np.mean(comp_times["rollout_queue_put"]), global_step)
      writer.add_scalar("charts/SPS", int((global_step - start_step) / (time.time() - start_time)), global_step)
      writer.add_scalar(
        "charts/SPS_update",
        int(
          args.local_num_envs
          * args.num_steps
          * len_actor_device_ids
          * args.num_actor_threads
          * args.world_size
          / (time.time() - update_time_start)
        ),
        global_step,
      )

      if args.eval_every_n_log_cycles > 0 and (update // args.log_frequency) % args.eval_every_n_log_cycles == 0:
        ev_rollout_time_start = time.time()
        for _ in range(0, args.num_steps):
          ev_cached_next_terminated = ev_next_terminated
          ev_cached_next_truncated = ev_next_truncated
          inference_time_start = time.time()
          ev_next_obs, action, _, _, key = get_model_outputs(params, ev_next_obs, key)
          ev_comp_times["inference"] += time.time() - inference_time_start

          ev_next_obs, _, _, ev_next_terminated, ev_next_truncated, _, ev_next_info, ev_comp_times = \
            step_envs_once(eval_envs, action, ev_comp_times)

          ev_rollout_stats, ev_cached_ep_returns, ev_cached_ep_lengths = \
            update_rollout_stats(ev_rollout_stats, ev_next_info, ev_cached_next_terminated, ev_cached_next_truncated,
                                 ev_cached_ep_returns, ev_cached_ep_lengths)

        ev_comp_times["rollout"].append(time.time() - ev_rollout_time_start)
        if device_thread_id == 0:
          print(
            f"global_step={global_step}, "
            f"avg_episodic_return_eval={np.mean(ev_rollout_stats['ep_returns'])}, "
            f"rollout_time_eval={np.mean(ev_comp_times['rollout'])}"
          )
        avg_episodic_return_eval = np.mean(ev_rollout_stats['ep_returns'])
        avg_norm_episodic_return_eval = avg_episodic_return_eval / args.baseline_score if args.baseline_score else 0
        writer.add_scalar("stats/eval_rollout_time", np.mean(ev_comp_times["rollout"]), global_step)
        writer.add_scalar("charts/avg_episodic_return_eval", avg_episodic_return_eval, global_step)
        writer.add_scalar("charts/avg_norm_episodic_return_eval", avg_norm_episodic_return_eval, global_step)
        writer.add_scalar("charts/avg_solved_rate_eval", np.mean(ev_rollout_stats["levels_solved"]), global_step)
        writer.add_scalar("charts/avg_episodic_length_eval", np.mean(ev_rollout_stats["ep_lengths"]), global_step)
        writer.add_scalar("stats/inference_time_eval", ev_comp_times["inference"], global_step)
        writer.add_scalar("stats/d2h_time_eval", ev_comp_times["d2h"], global_step)
        writer.add_scalar("stats/env_send_time_eval", ev_comp_times["env_send"], global_step)

  envs.close()
  eval_envs.close() if args.eval_every_n_log_cycles > 0 else None

def linear_schedule(count, update_epochs):
  # anneal learning rate linearly after one training iteration which contains
  # (args.num_minibatches) gradient updates
  frac = 1.0 - (count // (args.num_minibatches * update_epochs)) / args.num_updates
  return args.learning_rate * frac

def value_loss_fn(newvalue, target_value):
  v_loss = 0.5 * ((newvalue - target_value) ** 2).mean()
  return v_loss

def drA_loss_fn(logprob):
  return -logprob.mean()

def drC_loss_fn(newvalue, target_value):
  return value_loss_fn(newvalue, target_value)

if __name__ == "__main__":
  args = tyro.cli(Args)
  args.local_batch_size = int(
    args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
  args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
  assert (
      args.local_num_envs % len(args.learner_device_ids) == 0
  ), "local_num_envs must be divisible by len(learner_device_ids)"
  assert (
      int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
  ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
  assert (
      args.log_frequency % args.num_policy_phases == 0
  ), "log_frequency must be divisible by num_policy_phases"
  if args.distributed:
    jax.distributed.initialize(
      local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
    )
    print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))
  if args.no_cuda:
    jax.config.update('jax_platform_name', 'cpu')

  args.world_size = jax.process_count()
  args.local_rank = jax.process_index()
  args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
  args.batch_size = args.local_batch_size * args.world_size
  args.minibatch_size = args.local_minibatch_size * args.world_size
  args.num_rollouts_in_mb = args.minibatch_size // args.num_steps #note: only relevant when shuffle_rollouts() is used
  args.num_policy_phase_batches_for_auxiliary_phase = args.num_policy_phases \
    if args.num_policy_phase_batches_for_auxiliary_phase==-1 \
    else args.num_policy_phase_batches_for_auxiliary_phase
  args.num_sharded_rollouts_aux_phase = args.world_size * args.num_actor_threads * len(args.actor_device_ids) * \
                                        args.num_policy_phase_batches_for_auxiliary_phase
  args.aux_batch_rollouts = args.num_envs * args.num_policy_phase_batches_for_auxiliary_phase
  args.aux_batch_size = args.batch_size * args.num_policy_phase_batches_for_auxiliary_phase
  assert args.aux_batch_size == args.aux_batch_rollouts * args.num_steps, "check aux_batch_size calculation failed"
  args.aux_minibatch_size = args.num_rollouts_in_aux_mb * args.num_steps
  assert args.aux_batch_size % args.aux_minibatch_size == 0, "aux_batch_size must be divisible by aux_minibatch_size"
  args.num_aux_minibatches = args.aux_batch_size // args.aux_minibatch_size
  args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
  args.baseline_score = constants.PROCGEN_BASELINE_SCORES['ppo']['test_scores'][args.env_id]

  local_devices = jax.local_devices()
  global_devices = jax.devices()
  learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
  actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
  global_learner_devices = [
    global_devices[d_id + process_index * len(local_devices)]
    for process_index in range(args.world_size)
    for d_id in args.learner_device_ids
  ]
  print("global_learner_devices", global_learner_devices)
  args.global_learner_devices = [str(item) for item in global_learner_devices]
  args.actor_devices = [str(item) for item in actor_devices]
  args.learner_devices = [str(item) for item in learner_devices]
  pprint(args)

  config = vars(args)
  run_name, resume = job_util.generate_run_name(args)
  checkpoint_path = f"runs/{run_name}/{args.exp_name}.ckpt"

  slurm_metadata = job_util.gather_slurm_metadata()
  if slurm_metadata:
    print("SLURM METADATA:")
    pprint(slurm_metadata)
    config.update(slurm=slurm_metadata)

  # seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  key = jax.random.PRNGKey(args.seed)

  # agent setup
  dummy_env = make_env(args.env_id, args.seed, 1, num_levels=args.num_train_levels)()
  action_space = dummy_env.single_action_space
  observation_space = dummy_env.single_observation_space
  agent_state, model_modules, key = make_agent(args, dummy_env, key, print_model=True)
  arg_to_aug_fn = {"random_crop": partial(data_augmentation.batched_random_crop, padding=12)}
  aug_fn = arg_to_aug_fn[args.data_augmentation_fn]
  dummy_env.close()
  runstate = job_util.RunState(checkpoint_path, save_fn=job_util.save_agent_state)
  if resume:
    agent_state, _, runstate.metadata = job_util.restore_agent_state(checkpoint_path, (agent_state, None, runstate.metadata))
    if runstate.completed and runstate.post_eval_completed:
      print("Run already completed. Exiting.")
      sys.exit(0)

  if args.track and args.local_rank == 0:
    import wandb
    wandb.init(
      project=args.wandb_project_name,
      entity=args.wandb_entity,
      group=args.wandb_group,
      sync_tensorboard=True,
      config=config,
      name=run_name,
      id=run_name,
      resume="allow" if resume else None,
      monitor_gym=True,
      save_code=True,
    )

  initial_global_step = get_global_step(runstate.metadata.get('learner_policy_version', 0), args)
  writer = SummaryWriter(f"runs/{run_name}", purge_step=initial_global_step + 1, flush_secs=50)
  runstate.to_close = [writer]
  if initial_global_step == 0:
    writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

  agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
  learner_keys = jax.device_put_replicated(key, learner_devices)

  def update_target_params(target_p, online_p, tau=args.target_smoothing_coef):
    return jax.lax.stop_gradient((tau * online_p + (1 - tau) * target_p))

  def get_error_target_params(target_p, online_p):
    target_param_error = jax.tree_map(lambda x, y: jnp.abs(x - y).mean(), target_p, online_p)
    target_param_error = jax.tree_util.tree_leaves(target_param_error)
    return jnp.stack(target_param_error).mean().item()

  @jax.jit
  def get_representation(
      params,
      obs: np.ndarray,
  ):
    hidden = ResNetBase(args.channels, args.hiddens).apply(params.base_params, obs)
    return hidden

  @jax.jit
  def get_value_and_rep(
      critic_params: CriticParams,
      obs: np.ndarray,
  ):
    hidden = get_representation(critic_params, obs)
    value = ValueHead().apply(critic_params.value_head_params, hidden).squeeze(-1)
    return value, hidden

  @jax.jit
  def get_auxvalue_and_rep(
      actor_params: ActorParams,
      obs: np.ndarray,
  ):
    hidden = get_representation(actor_params, obs)
    aux_value = ValueHead().apply(actor_params.auxiliary_head_params, hidden).squeeze(-1)
    return aux_value, hidden

  @jax.jit
  def get_auxadv(
      actor_params: ActorParams,
      reps: np.ndarray,
      actions: np.ndarray,
  ):
    adv_pred = AdvantageHead(action_space.n).apply(actor_params.auxiliary_advantage_head_params, reps,
                                                               actions).squeeze(-1)
    return adv_pred

  @jax.jit
  def get_auxadv_and_rep(
      actor_params: ActorParams,
      obs: np.ndarray,
      actions: np.ndarray,
  ):
    hidden = get_representation(actor_params, obs)
    aux_value = get_auxadv(actor_params, hidden, actions)
    return aux_value, hidden

  @jax.jit
  def get_logprob_entropy_rep(
      actor_params: ActorParams,
      obs: np.ndarray,
      actions: np.ndarray,
  ):
    hidden = get_representation(actor_params, obs)
    logits = PolicyHead(action_space.n).apply(actor_params.policy_head_params, hidden)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    entropy = -p_log_p.sum(-1)
    return logprob, entropy, hidden

  @jax.jit
  def get_logits(
      actor_params: ActorParams,
      obs: np.ndarray,
  ):
    hidden = get_representation(actor_params, obs)
    logits = PolicyHead(action_space.n).apply(actor_params.policy_head_params, hidden)
    return logits

  @jax.jit
  def get_logits_auxvaluepred_rep(
      actor_params: ActorParams,
      obs: np.ndarray,
  ):
    hidden = get_representation(actor_params, obs)
    logits = PolicyHead(action_space.n).apply(actor_params.policy_head_params, hidden)
    aux_value_pred = ValueHead().apply(actor_params.auxiliary_head_params, hidden).squeeze(-1)
    return logits, aux_value_pred, hidden

  @jax.jit
  def get_dyna_head_logits(
      dyna_head_params,
      rep: np.ndarray,
      action: np.ndarray,
      next_rep: np.ndarray,
  ):
    return DynaHead(args.hiddens[-1], action_space.n).apply(dyna_head_params, rep, action, next_rep)

  def sample_ood_actions(actions, key):
    key, *subkeys = jax.random.split(key, actions.shape[0] + 1)
    onehot_actions = jax.nn.one_hot(actions, action_space.n)  #(batch_size, num_actions)
    action_count = jnp.sum(onehot_actions, axis=0)  # (num_actions)
    ood_count = jnp.tile(action_count, (actions.shape[0], 1))  # (batch_size, num_actions)
    ood_count = ood_count * (1 - onehot_actions)
    mask = ood_count.sum(axis=1, keepdims=True) == 0  # (batch_size, 1)
    ood_count = mask * (1 - onehot_actions) + (1 - mask) * ood_count
    ood_count = ood_count / ood_count.sum(axis=1, keepdims=True)  # probs
    ood_actions = jax.vmap(jax.random.choice)(jnp.stack(subkeys), jnp.tile(jnp.arange(action_space.n), (actions.shape[0], 1)), p=ood_count)
    return ood_actions, key  # (batch_size,)

  def ddcpg_dynamic_pred_loss_fn(dyna_head_params, rep, action, next_rep, key):
    # next_rep = jax.lax.stop_gradient(next_rep) # Interestingly DDCPG does not stop gradient on next_rep or shuffled next_rep.
    action = jax.lax.stop_gradient(action)
    ood_actions, key = sample_ood_actions(action, key)
    key, subkey = jax.random.split(key)
    ood_next_rep = jax.random.permutation(subkey, next_rep)
    logits_in = get_dyna_head_logits(dyna_head_params, rep, action, next_rep)
    logits_ood_a = get_dyna_head_logits(dyna_head_params, rep, ood_actions, next_rep)
    logits_ood_s = get_dyna_head_logits(dyna_head_params, rep, action, ood_next_rep)
    loss_in = jax.vmap(binary_cross_entropy_with_logits)(logits_in, jnp.ones_like(logits_in)).mean()
    loss_ood_a = jax.vmap(binary_cross_entropy_with_logits)(logits_ood_a, jnp.zeros_like(logits_ood_a)).mean()
    loss_ood_s = jax.vmap(binary_cross_entropy_with_logits)(logits_ood_s, jnp.zeros_like(logits_ood_s)).mean()
    dyna_loss = loss_in + 0.5 * loss_ood_a + loss_ood_s
    return dyna_loss, key

  def compute_gae_once(carry, inp, gamma, gae_lambda):
    advantages = carry
    nextdones, nextterm, nextvalues, curvalues, reward = inp

    delta = reward + gamma * nextvalues * (1.0 - nextterm) - curvalues
    advantages = delta + gamma * gae_lambda * (1.0 - nextdones) * advantages
    return advantages, advantages

  compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

  @jax.jit
  def compute_gae(
      value_preds: np.ndarray,
      last_value: np.ndarray,
      last_done: np.ndarray,
      last_term: np.ndarray,
      storage: PolPhaseStorage,
  ):
    last_value = jax.lax.stop_gradient(last_value)
    advantages = jnp.zeros_like(last_value)
    dones = jnp.concatenate([storage.dones, last_done[None, :]], axis=0)
    values = jnp.concatenate([value_preds, last_value[None, :]], axis=0)
    terms = jnp.concatenate([storage.terminations, last_term[None, :]], axis=0)
    _, advantages = jax.lax.scan(
      compute_gae_once, advantages, (dones[1:], terms[1:], values[1:], values[:-1], storage.rewards), reverse=True
    )
    # handle the extra timestep in envpool
    advantages = jnp.where(dones[:-1], 0, advantages)
    target_values = jnp.where(storage.terminations, 0, advantages + value_preds)

    return advantages, target_values

  @functools.partial(jax.jit, static_argnames=("n_minibatches",))
  def shuffle_rollouts(storage, last_obs, logits, key, n_minibatches):

    def shuffle(x):
      return jax.random.permutation(key, x, axis=1)

    # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
    def convert_data(x: jnp.ndarray):
      # make rollout axis the leading axis
      x = jnp.swapaxes(x, 0, 1)
      x = jnp.reshape(x, (n_minibatches * args.gradient_accumulation_steps, -1) + x.shape[2:])
      return x

    shuffled_storage = jax.tree_map(shuffle, storage)
    shuffled_last_obs = shuffle(last_obs.reshape(1, *last_obs.shape))
    shuffled_logits = shuffle(logits)
    shuffled_storage = jax.tree_map(convert_data, shuffled_storage)
    shuffled_last_obs = convert_data(shuffled_last_obs)
    shuffled_logits = convert_data(shuffled_logits)

    return shuffled_storage, shuffled_last_obs, shuffled_logits

  @functools.partial(jax.jit, static_argnames=("num_rollouts", "num_steps"))
  def reshape_to_rollout_dim(rewards, reps, target_reps, target_nreps, num_rollouts, num_steps):
    rewards = rewards.reshape(num_rollouts, num_steps)
    reps = reps.reshape(num_rollouts, num_steps, -1)
    target_reps = target_reps.reshape(num_rollouts, num_steps, -1)
    target_nreps = target_nreps.reshape(num_rollouts, num_steps, -1)
    return rewards, reps, target_reps, target_nreps

  def mico_reward_diff_loss_fn(rewards, reps, target_reps, target_nreps, remove_same_rollout_pairs=False):
    distance_fn = utils.cosine_distance
    diff_fn = utils.absolute_diff
    online_dist, norm_average, base_distances = \
      bisim_util.representation_distances(reps, target_reps, distance_fn,
                                          remove_same_rollout_pairs=remove_same_rollout_pairs,
                                          return_distance_components=True)
    target_dist, target_norm_average, target_base_distances, reward_diffs = \
      bisim_util.target_distances(target_nreps, rewards, diff_fn, distance_fn, args.gamma,
                                  remove_same_rollout_pairs=remove_same_rollout_pairs,
                                  return_distance_components=True)
    mico_loss = jnp.mean(jax.vmap(utils.huber_loss)(online_dist, target_dist))
    stats = {
      'mico_norm_average': norm_average.mean(),
      'mico_base_distances': base_distances.mean(),
      'mico_reward_diff': reward_diffs.mean(),
      'mico_norm_average_delta': jnp.abs(norm_average - target_norm_average).mean(),
      'mico_base_distances_delta': jnp.abs(base_distances - target_base_distances).mean(),
      'mico_online_dist': online_dist.mean(),
      'mico_target_dist': target_dist.mean(),
    }
    return mico_loss, stats

  def mico_policy_phase_loss_fn(params, target_params, obs, next_obs, rewards):
    # MICo loss critic
    if args.remove_same_rollout_pairs:
      reshape_fn = partial(reshape_to_rollout_dim, num_rollouts=args.num_rollouts_in_mb, num_steps=args.num_steps)
    else:
      reshape_fn = lambda rew, reps, treps, tnreps: (rew, reps, treps, tnreps)
    reps_c = get_representation(params.critic_params, obs)
    target_reps_c = get_representation(target_params.critic_params, obs)
    target_nreps_c = get_representation(target_params.critic_params, next_obs)
    mico_loss, mico_stats = mico_reward_diff_loss_fn(*reshape_fn(rewards, reps_c, target_reps_c, target_nreps_c),
                                                     remove_same_rollout_pairs=args.remove_same_rollout_pairs)

    # MICo loss actor
    if args.mico_coef_actor_pol_phase > 0.0:
      reps_a = get_representation(params.actor_params, obs)
      target_reps_a = get_representation(target_params.actor_params, obs)
      target_nreps_a = get_representation(target_params.actor_params, next_obs)
      mico_loss_a, mico_stats_a = mico_reward_diff_loss_fn(*reshape_fn(rewards, reps_a, target_reps_a, target_nreps_a),
                                                       remove_same_rollout_pairs=args.remove_same_rollout_pairs)
      mico_stats.update({f"{k}_a": v for k, v in mico_stats_a.items()})
    else:
      mico_loss_a = jnp.zeros_like(mico_loss)

    loss = args.mico_coef_critic_pol_phase * mico_loss + args.mico_coef_actor_pol_phase * mico_loss_a
    stats = {
      'losses/mico_loss': mico_loss,
      'losses/mico_loss_a': mico_loss_a,
    }
    stats.update({f"losses/{k}": v for k, v in mico_stats.items()})
    return loss, stats

  def policy_phase_loss_fn(params, target_params, obs, next_obs, rewards, actions, behavior_logprobs,
                           advantages, target_values):
    newlogprob, entropy, reps_a = get_logprob_entropy_rep(params.actor_params, obs, actions)
    logratio = newlogprob - behavior_logprobs
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    entropy_loss = entropy.mean()

    # Value loss
    if args.detach_value_grads_policy_phase:
      reps_c = get_representation(params.critic_params, obs)
      value_preds = ValueHead().apply(params.critic_params.value_head_params, jax.lax.stop_gradient(reps_c)).squeeze(-1)
    else:
      value_preds, reps_c = get_value_and_rep(params.critic_params, obs)
    v_loss = value_loss_fn(value_preds, target_values)

    new_target_p = jax.tree_map(update_target_params, target_params, params)

    loss = pg_loss - args.ent_coef * entropy_loss + v_loss
    stats = {
      'losses/policy_phase_loss': loss,
      'losses/policy_loss': pg_loss,
      'losses/entropy': entropy_loss,
      'losses/approx_kl': approx_kl,
      'losses/value_loss': v_loss,
    }
    return loss, (stats, new_target_p)

  @jax.jit
  def run_policy_phase_single_device(
      agent_state: AgentTrainStateWithTarget,
      sharded_storages: List,
      sharded_last_obs: List,
      sharded_last_done: List,
      sharded_last_term: List,
      key: jax.random.PRNGKey,
  ):
    storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
    last_obs = jnp.concatenate(sharded_last_obs)
    last_done = jnp.concatenate(sharded_last_done)
    last_term = jnp.concatenate(sharded_last_term)
    num_sharded_storages = len(sharded_storages)
    del sharded_storages, sharded_last_obs, sharded_last_done, sharded_last_term
    # Note: not the most memory efficient
    next_obs = jnp.concatenate([storage.obs[1:], last_obs[None, :]], axis=0)
    # mask next_obs when done to break spurious correlation between last and first obs of next episode
    next_obs_mask = jnp.broadcast_to(storage.dones.reshape((next_obs.shape[:2]) + (1, 1, 1)), next_obs.shape)
    next_obs = jnp.where(next_obs_mask, 0, next_obs)
    policy_phase_loss_grad_fn = jax.value_and_grad(policy_phase_loss_fn, has_aux=True)
    mico_policy_phase_loss_grad_fn = jax.value_and_grad(mico_policy_phase_loss_fn, has_aux=True)
    last_value, _ = get_value_and_rep(agent_state.get_params().critic_params, last_obs)
    advantages_raw, target_values = compute_gae(storage.values, last_value, last_done, last_term, storage)
    # NOTE: default for PPG is per-batch advantages normalization (PPO does per-minibatch normalization)
    if args.norm_adv == "batch":
      advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
    elif args.norm_adv == "minibatch": #Note: likely a bug in original cleanRL implt. normalising by norm here doesn't make sense because we shuffle data later
      advantages = advantages_raw.reshape(advantages_raw.shape[0], args.num_minibatches, -1)
      advantages = (advantages - advantages.mean((0, -1), keepdims=True)) / (
            advantages.std((0, -1), keepdims=True) + 1e-8)
      advantages = advantages.reshape(advantages.shape[0], -1)
    elif args.norm_adv == "none":
      advantages = advantages_raw

    def shuffle_storage(storage, advantages, target_values, next_obs, key):
      def flatten(x):
        return x.reshape((-1,) + x.shape[2:])

      # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
      def convert_data(x: jnp.ndarray):
        x = jax.random.permutation(key, x)
        x = jnp.reshape(x, (args.num_minibatches * args.gradient_accumulation_steps, -1) + x.shape[1:])
        return x

      flatten_storage = jax.tree_map(flatten, storage)
      flatten_advantages = flatten(advantages)
      flatten_target_values = flatten(target_values)
      flatten_next_obs = flatten(next_obs)

      shuffled_storage = jax.tree_map(convert_data, flatten_storage)
      shuffled_advantages = convert_data(flatten_advantages)
      shuffled_target_values = convert_data(flatten_target_values)
      shuffled_next_obs = convert_data(flatten_next_obs)

      return shuffled_storage, shuffled_advantages, shuffled_target_values, shuffled_next_obs

    def policy_phase_epoch(carry, _):
      def mico_minibatch(agent_state, minibatch):
        mb_obs, mb_next_obs, mb_rewards = minibatch

        (loss, stats_mico), grads = \
        mico_policy_phase_loss_grad_fn(
          agent_state.get_params(),
          agent_state.get_target_params(),
          mb_obs,
          mb_next_obs,
          mb_rewards,
        )
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        if args.mico_coef_actor_pol_phase > 0.0:
          agent_state = agent_state.apply_gradients(grads=grads)
        else:
          agent_state = agent_state.apply_gradients_critic(grads=grads)
        return agent_state, stats_mico

      def policy_phase_minibatch(agent_state, minibatch):
        mb_obs, mb_next_obs, mb_rewards, mb_actions, mb_behavior_logprobs, mb_advantages, mb_target_values = minibatch
        (loss, (stats_pol_phase, target_params)), grads = \
          policy_phase_loss_grad_fn(
            agent_state.get_params(),
            agent_state.get_target_params(),
            mb_obs,
            mb_next_obs,
            mb_rewards,
            mb_actions,
            mb_behavior_logprobs,
            mb_advantages,
            mb_target_values,
          )
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        agent_state = agent_state.apply_gradients(grads=grads)
        agent_state = agent_state.replace(actor_target_params=target_params.actor_params,
                                          critic_target_params=target_params.critic_params)
        return agent_state, stats_pol_phase

      agent_state, key = carry
      if args.enable_mico_pol_phase:
        key, subkey = jax.random.split(key)
        # Note: here replacing logits with next_obs is allowed.
        shuffled_r_storage, _, shuffled_r_next_obs = shuffle_rollouts(storage, last_obs, next_obs, key,
                                                                      args.num_minibatches)
        agent_state, stats_mico = \
          jax.lax.scan(
            mico_minibatch,
            agent_state,
            (
              shuffled_r_storage.obs,
              shuffled_r_next_obs,
              shuffled_r_storage.rewards,
            ),
          )
        del shuffled_r_storage, shuffled_r_next_obs
      else:
        stats_mico = {}

      key, subkey = jax.random.split(key)
      shuffled_storage, shuffled_advantages, shuffled_target_values, shuffled_next_obs = \
        shuffle_storage(storage, advantages, target_values, next_obs, subkey)

      agent_state, stats_pol_phase = \
        jax.lax.scan(
          policy_phase_minibatch,
          agent_state,
          (
            shuffled_storage.obs,
            shuffled_next_obs,
            shuffled_storage.rewards,
            shuffled_storage.actions,
            shuffled_storage.logprobs,
            shuffled_advantages,
            shuffled_target_values,
          ),
        )
      stats_pol_phase.update(stats_mico)
      return (agent_state, key), stats_pol_phase

    (agent_state, key), stats_pol_phase = \
      jax.lax.scan(policy_phase_epoch, (agent_state, key), (), length=args.policy_phase_epochs)
    stats_pol_phase = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="local_devices").mean(), stats_pol_phase)

    return agent_state, stats_pol_phase, jnp.hsplit(target_values, num_sharded_storages), jnp.hsplit(advantages_raw, num_sharded_storages), key

  def compute_bc_target_logits_once(obs, actor_params):
    return jax.lax.stop_gradient(get_logits(actor_params, obs))

  @jax.jit
  def compute_bc_target_logits(actor_params, storage):
    # prevents excessive memory usage.
    obs = storage.obs.reshape((-1, args.aux_minibatch_size) + storage.obs.shape[2:])
    #TODO (minor): compare two options to improve performance
    # logits = jnp.array([compute_bc_target_logits_once(ob, actor_params) for ob in obs]) #for loop, long compile time but faster execution
    compute_logit_fn = partial(compute_bc_target_logits_once, actor_params=actor_params)
    logits = jax.lax.map(compute_logit_fn, obs)
    return logits.reshape(storage.obs.shape[:2] + (logits.shape[-1],))

  @jax.jit
  def compute_bc_target_single_pass(actor_params, storage):
    obs = storage.obs.reshape((-1,) + storage.obs.shape[2:])
    logits = jax.lax.stop_gradient(get_logits(actor_params, obs))
    return logits.reshape(storage.obs.shape[:2] + (logits.shape[-1],))

  def auxiliary_phase_loss_fn(params, target_params, obs, next_obs, dones, rewards, target_values, target_logits,
                              actions, target_advantages, key):

    new_logits, aux_value_preds, reps_a = get_logits_auxvaluepred_rep(params.actor_params, obs)
    kl_policy_bc = kl_categorical_categorical(target_logits, new_logits).mean()
    value_preds, reps_c = get_value_and_rep(params.critic_params, obs)
    nreps_a = get_representation(params.actor_params, next_obs)
    nreps_c = get_representation(params.critic_params, next_obs)
    vloss = value_loss_fn(value_preds, target_values)
    if args.aux_vf_coef_aux_phase > 0.0:
      aux_vloss = value_loss_fn(aux_value_preds, target_values)
    else:
      aux_vloss = jnp.zeros_like(vloss)

    if args.drA_loss_coef > 0.0 or args.drC_loss_coef > 0.0:
      key, subkey = jax.random.split(key)
      base_batch_size = obs.shape[0]
      aug_obs = aug_fn(subkey, obs)
      aug_batch_size = aug_obs.shape[0]

    if args.drA_loss_coef > 0.0:
      aug_actions = jnp.concatenate([actions]*(aug_batch_size // base_batch_size), axis=0)
      aug_logprob, _, _ = get_logprob_entropy_rep(params.actor_params, aug_obs, aug_actions)
      drA_loss = drA_loss_fn(aug_logprob)
    else:
      drA_loss = jnp.zeros_like(vloss)

    if args.drC_loss_coef > 0.0:
      aug_values, _ = get_value_and_rep(params.critic_params, aug_obs)
      drC_loss = drC_loss_fn(aug_values, target_values)
    else:
      drC_loss = jnp.zeros_like(vloss)

    if args.adv_coef_aux_phase > 0.0:
      adv_pred = get_auxadv(params.actor_params, reps_a, actions)
      adv_loss = ((adv_pred - target_advantages) ** 2).mean()
    else:
      adv_loss = jnp.zeros_like(vloss)

    # MICo loss critic
    target_reps_c = get_representation(target_params.critic_params, obs)
    target_nreps_c = get_representation(target_params.critic_params, next_obs)
    if args.remove_same_rollout_pairs:
      reshape_fn = partial(reshape_to_rollout_dim, num_rollouts=args.num_rollouts_in_aux_mb, num_steps=args.num_steps)
    else:
      reshape_fn = lambda rew, reps, treps, tnreps: (rew, reps, treps, tnreps)
    mico_loss, mico_stats = mico_reward_diff_loss_fn(*reshape_fn(rewards, reps_c, target_reps_c, target_nreps_c),
                                                     remove_same_rollout_pairs=args.remove_same_rollout_pairs)

    # MICo loss actor
    if args.mico_coef_actor_aux_phase > 0.0:
      target_reps_a = get_representation(target_params.actor_params, obs)
      target_nreps_a = get_representation(target_params.actor_params, next_obs)
      mico_loss_a, mico_stats_a = mico_reward_diff_loss_fn(*reshape_fn(rewards, reps_a, target_reps_a, target_nreps_a),
                                                       remove_same_rollout_pairs=args.remove_same_rollout_pairs)
      mico_stats.update({f"{k}_a": v for k, v in mico_stats_a.items()})
    else:
      mico_loss_a = jnp.zeros_like(mico_loss)

    # dynamic prediction loss (ddcpg)
    if args.markov_coef_a > 0.0:
      dyna_loss_a, key = ddcpg_dynamic_pred_loss_fn(params.actor_params.dyna_head_params, reps_a, actions, nreps_a, key)
    else:
      dyna_loss_a = jnp.zeros_like(vloss)
    if args.markov_coef_c > 0.0:
      dyna_loss_c, key = ddcpg_dynamic_pred_loss_fn(params.critic_params.dyna_head_params, reps_c, actions, nreps_c, key)
    else:
      dyna_loss_c = jnp.zeros_like(vloss)

    aux_phase_loss = vloss + \
                     args.aux_vf_coef_aux_phase * aux_vloss + \
                     args.adv_coef_aux_phase * adv_loss + \
                     args.markov_coef_a * dyna_loss_a + \
                     args.markov_coef_c * dyna_loss_c + \
                     args.bc_coef * kl_policy_bc + \
                     args.mico_coef_critic_aux_phase * mico_loss + \
                     args.mico_coef_actor_aux_phase * mico_loss_a + \
                     args.drA_loss_coef * drA_loss + \
                     args.drC_loss_coef * drC_loss

    new_target_p = jax.tree_map(update_target_params, target_params, params)

    stats = {
      'losses/auxiliary_phase_loss': aux_phase_loss,
      'losses/auxiliary_head_value_loss': aux_vloss,
      'losses/auxiliary_phase_advantage_loss': adv_loss,
      'losses/kl_policy_bc': kl_policy_bc,
      'losses/auxiliary_phase_critic_value_loss': vloss,
      'losses/auxiliary_phase_critic_mico_loss': mico_loss,
      'losses/auxiliary_phase_actor_mico_loss': mico_loss_a,
      'losses/auxiliary_markov_loss_a': dyna_loss_a,
      'losses/auxiliary_markov_loss_c': dyna_loss_c,
      'losses/auxiliary_drA_loss': drA_loss,
      'losses/auxiliary_drC_loss': drC_loss,
    }
    stats.update({f"losses/auxiliary_phase_{k}": v for k, v in mico_stats.items()})

    # Compute representation stats
    key, subkey = jax.random.split(key)
    #Critic
    stats.update(
        repmetric_util.compute_nn_latent_stats(subkey, reps_c, nreps_c, dones, label='losses/z_critic_aux_phase'))
    stats.update(repmetric_util.compute_nn_latent_out_stats(reps_c, value_preds, metric_fn_out=utils.absolute_diff,
                                                            label='losses/z_value_l1_critic_aux_phase'))

    #Actor
    stats.update(
        repmetric_util.compute_nn_latent_stats(subkey, reps_a, nreps_a, dones, label='losses/z_actor_aux_phase'))
    stats.update(repmetric_util.compute_nn_latent_out_stats(reps_a, new_logits, metric_fn_out=utils.total_variation,
                                                            label='losses/z_logits_tv_actor_aux_phase'))
    stats.update(
        repmetric_util.compute_nn_latent_out_stats(reps_a, new_logits, metric_fn_out=utils.jsd_categorical_categorical,
                                                   label='losses/z_logits_jsd_actor_aux_phase'))
    if args.aux_vf_coef_aux_phase > 0.0:
      stats.update(
          repmetric_util.compute_nn_latent_out_stats(reps_a, aux_value_preds, metric_fn_out=utils.absolute_diff,
                                                     label='losses/z_aux_value_l1_actor_aux_phase'))
    if args.adv_coef_aux_phase > 0.0:
      stats.update(repmetric_util.compute_nn_latent_out_stats(reps_a, adv_pred, metric_fn_out=utils.absolute_diff,
                                                              label='losses/z_adv_l1_actor_aux_phase'))

    return aux_phase_loss, (stats, new_target_p, key)

  @jax.jit
  def run_auxiliary_phase_single_device(
      agent_state: AgentTrainStateWithTarget,
      sharded_storages: List,
      sharded_last_obs: List,
      key: jax.random.PRNGKey,
  ):

    storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
    del sharded_storages
    last_obs = jnp.concatenate(sharded_last_obs)
    old_pi_logits = compute_bc_target_logits(agent_state.get_params().actor_params, storage)
    # old_pi_logits = compute_bc_target_single_pass(agent_state.get_params().actor_params, storage) #faster, but memory inefficient
    aux_loss_grad_fn = jax.value_and_grad(auxiliary_phase_loss_fn, has_aux=True)
    if args.norm_adv == "batch":
      storage = storage._replace(
        target_advantages=(storage.target_advantages - storage.target_advantages.mean()) /
                          (storage.target_advantages.std() + 1e-8))
    elif args.norm_adv == "minibatch": #Note: likely a bug in original cleanRL implt. normalising by norm here doesn't make sense because we shuffle data later
      advantages = storage.target_advantages.reshape(storage.target_advantages.shape[0], args.num_aux_minibatches, -1)
      advantages = (advantages - advantages.mean((0, -1), keepdims=True)) / (
            advantages.std((0, -1), keepdims=True) + 1e-8)
      storage = storage._replace(target_advantages=advantages.reshape(advantages.shape[0], -1))

    def aux_phase_epoch(carry, _):
      agent_state, key = carry
      key, subkey = jax.random.split(key)
      shuffled_storage, shuffled_last_obs, shuffled_logits = shuffle_rollouts(storage, last_obs, old_pi_logits, subkey,
                                                                              args.num_aux_minibatches)
      def aux_phase_minibatch(carry, minibatch):
        agent_state, key = carry
        mb_obs, mb_last_obs, mb_dones, mb_rewards, mb_target_values, mb_target_logits, mb_actions, mb_target_adv = minibatch

        def obs_to_next_obs(obs, dones, last_obs):
          obs = obs.reshape((last_obs.shape[0], -1) + obs.shape[1:])
          next_obs = jnp.concatenate([obs[:,1:], last_obs[:, None, :]], axis=1)
          next_obs = next_obs.reshape((-1,) + next_obs.shape[2:])
          next_obs_mask = jnp.broadcast_to(dones.reshape((*dones.shape, 1, 1, 1)), next_obs.shape)
          next_obs = jnp.where(next_obs_mask, 0, next_obs)
          return next_obs

        mb_next_obs = obs_to_next_obs(mb_obs, mb_dones, mb_last_obs)

        (aux_phase_loss, (stats_aux_phase, target_params, key)), grads = \
          aux_loss_grad_fn(
            agent_state.get_params(),
            agent_state.get_target_params(),
            mb_obs,
            mb_next_obs,
            mb_dones,
            mb_rewards,
            mb_target_values,
            mb_target_logits,
            mb_actions,
            mb_target_adv,
            key,
          )
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        agent_state = agent_state.apply_gradients(grads=grads)
        agent_state = agent_state.replace(actor_target_params=target_params.actor_params,
                                          critic_target_params=target_params.critic_params)
        return (agent_state, key), stats_aux_phase

      (agent_state, key), stats_aux_phase = \
        jax.lax.scan(
          aux_phase_minibatch,
          (agent_state, key),
          (
            shuffled_storage.obs,
            shuffled_last_obs,
            shuffled_storage.dones,
            shuffled_storage.rewards,
            shuffled_storage.target_values,
            shuffled_logits,
            shuffled_storage.actions,
            shuffled_storage.target_advantages,
          ),
        )
      return (agent_state, key), stats_aux_phase

    (agent_state, key), stats_aux_phase = \
      jax.lax.scan(aux_phase_epoch, (agent_state, key), (), length=args.auxiliary_phase_epochs)
    stats_aux_phase = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="local_devices").mean(), stats_aux_phase)

    return agent_state, stats_aux_phase, key

  run_policy_phase_multi_device = jax.pmap(
    run_policy_phase_single_device,
    axis_name="local_devices",
    devices=global_learner_devices,
  )
  run_auxiliary_phase_multi_device = jax.pmap(
    run_auxiliary_phase_single_device,
    axis_name="local_devices",
    devices=global_learner_devices,
  )

  params_queues = []
  rollout_queues = []
  dummy_writer = SimpleNamespace()
  dummy_writer.add_scalar = lambda x, y, z: None

  if not runstate.metadata['training_completed']:
    unreplicated_params = flax.jax_utils.unreplicate(agent_state.get_params())
    for d_idx, d_id in enumerate(args.actor_device_ids):
      device_params = jax.device_put(unreplicated_params, local_devices[d_id])
      for thread_id in range(args.num_actor_threads):
        params_queues.append(queue.Queue(maxsize=1))
        rollout_queues.append(queue.Queue(maxsize=1))
        params_queues[-1].put(device_params)
        threading.Thread(
          target=rollout,
          args=(
            jax.device_put(key, local_devices[d_id]),
            args,
            rollout_queues[-1],
            params_queues[-1],
            writer if d_idx == 0 and thread_id == 0 else dummy_writer,
            learner_devices,
            d_idx * args.num_actor_threads + thread_id,
            runstate.metadata['learner_policy_version'],
          ),
          daemon=True
        ).start()

    rollout_queue_get_time = deque(maxlen=10)
    learner_policy_version = runstate.metadata['learner_policy_version']
    training_phase_start = time.time()
    last_checkpoint_time = time.time()
    sharded_storages_aux_phase = deque(maxlen=args.num_sharded_rollouts_aux_phase)
    sharded_last_obss_aux_phase = deque(maxlen=args.num_sharded_rollouts_aux_phase)
    while True:
      learner_policy_version += 1
      rollout_queue_get_time_start = time.time()
      sharded_storages_pol_phase = []
      sharded_last_obss_pol_phase = []
      sharded_last_dones_pol_phase = []
      sharded_last_terms_pol_phase = []
      for d_idx, d_id in enumerate(args.actor_device_ids):
        for thread_id in range(args.num_actor_threads):
          (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_last_obs,
            sharded_last_done,
            sharded_last_term,
            avg_params_queue_get_time,
            device_thread_id,
          ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
          sharded_storages_pol_phase.append(sharded_storage)
          sharded_last_obss_pol_phase.append(sharded_last_obs)
          sharded_last_dones_pol_phase.append(sharded_last_done)
          sharded_last_terms_pol_phase.append(sharded_last_term)
      rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
      training_time_start = time.time()
      (agent_state, stats_pol_phase, target_values, target_advantages, learner_keys) = \
        run_policy_phase_multi_device(
          agent_state,
          sharded_storages_pol_phase,
          sharded_last_obss_pol_phase,
          sharded_last_dones_pol_phase,
          sharded_last_terms_pol_phase,
          learner_keys,
      )
      sharded_storages_aux_phase.extend([
        AuxPhaseStorage(
          obs=sharded_storages_pol_phase[i].obs,
          rewards=sharded_storages_pol_phase[i].rewards,
          actions=sharded_storages_pol_phase[i].actions,
          target_values=target_values[i],
          target_advantages=target_advantages[i],
          dones=sharded_storages_pol_phase[i].dones,
        )
        for i in range(len(sharded_storages_pol_phase))
      ])
      sharded_last_obss_aux_phase.extend(sharded_last_obss_pol_phase)
      if learner_policy_version % args.num_policy_phases == 0:
        (agent_state, stats_aux_phase, learner_keys) = \
          run_auxiliary_phase_multi_device(
            agent_state,
            list(sharded_storages_aux_phase),
            list(sharded_last_obss_aux_phase),
            learner_keys,
        )
        gc.collect()
      unreplicated_params = flax.jax_utils.unreplicate(agent_state.get_params())
      for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
          params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

      writer.add_scalar("charts/target_param_error", get_error_target_params(agent_state.get_target_params(), agent_state.get_params()), global_step)
      # less frequent logging
      if learner_policy_version % args.log_frequency == 0:
        writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
        writer.add_scalar(
          "stats/rollout_params_queue_get_time_diff",
          np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
          global_step,
        )
        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
        writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
        print(
          global_step,
          f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
        )
        writer.add_scalar(
          "charts/learning_rate", agent_state.actor_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(),
          global_step
        )
        writer.add_scalar("charts/avg_value_target", jnp.concatenate(target_values).mean().item(), global_step)
        writer.add_scalar("charts/avg_adv_target", jnp.concatenate(target_advantages).mean().item(), global_step)
        for key, value in stats_pol_phase.items():
          writer.add_scalar(key, value[-1].item(), global_step)
        if learner_policy_version % args.num_policy_phases == 0:
          for key, value in stats_aux_phase.items():
            writer.add_scalar(key, value[-1].item(), global_step)
      if learner_policy_version >= args.num_updates:
        break
      # make sure we don't get extra updates due to checkpointing in middle of policy phase
      if learner_policy_version % args.num_policy_phases == 0:
        runstate.apply_signals(learner_policy_version, agent_state, args)
        if 0 < args.checkpoint_frequency < (time.time() - last_checkpoint_time):
          print(f"Saving periodic checkpoint at iteration {learner_policy_version}")
          runstate.save_state(agent_state, args)
          last_checkpoint_time = time.time()
      else:
        runstate.apply_signals(learner_policy_version, agent_state, args, save_model=False)
    runstate.after_training(agent_state, args)

  if args.distributed:
    jax.distributed.shutdown()
  if args.save_model and args.local_rank == 0 and not runstate.metadata['eval_completed']:
    from eval_utils.runstate_envpool_procgen_eval import evaluate

    test_episodic_returns = evaluate(
      checkpoint_path,
      make_env,
      make_agent,
      args,
      args.env_id,
      start_level=args.num_train_levels,
      eval_episodes=args.num_eval_episodes,
      run_name=f"{run_name}-eval-testset",
    )
    train_episodic_returns = evaluate(
      checkpoint_path,
      make_env,
      make_agent,
      args,
      args.env_id,
      num_levels=args.num_train_levels,
      start_level=0,
      eval_episodes=args.num_eval_episodes,
      run_name=f"{run_name}-eval-trainset",
    )

    runstate.apply_signals(-1, agent_state, args) #this is to make sure we don't have an unexpected job preemption while uploading evaluation data.
    global_step = args.total_timesteps # handles loading checkpoint post training but prior to eval.
    for idx, episodic_return in enumerate(test_episodic_returns):
      writer.add_scalar("eval/episodic_return_test", episodic_return, idx)
    writer.add_scalar("eval/avg_episodic_return_test", np.mean(test_episodic_returns), global_step)
    writer.add_scalar("eval/avg_norm_episodic_return_test", np.mean(test_episodic_returns)/args.baseline_score, global_step + 1)
    for idx, episodic_return in enumerate(train_episodic_returns):
      writer.add_scalar("eval/episodic_return_train", episodic_return, idx)
    writer.add_scalar("eval/avg_episodic_return_train", np.mean(train_episodic_returns), global_step)
    writer.add_scalar("eval/avg_norm_episodic_return_train", np.mean(train_episodic_returns)/args.baseline_score, global_step + 1)
    runstate.after_eval(agent_state, args)

  if args.save_model and args.local_rank == 0 and not runstate.metadata['post_eval_completed']:
    from eval_utils.mutual_info_procgen_eval import evaluate_mi, \
      compute_mi_levelset, compute_mi_markov, compute_mi_vf, compute_rep_stats

    eval_fns_test = [
      partial(compute_mi_markov, n_samples=args.mi_eval_downsample_to_n),
      partial(compute_mi_vf, n_samples=args.mi_eval_downsample_to_n, gamma=args.gamma),
      partial(compute_rep_stats, model_modules=model_modules, num_steps=args.num_steps,
              num_rollouts=args.mi_eval_downsample_to_n // args.num_steps)
    ]
    eval_fns_train = [
      partial(compute_mi_levelset, n_samples=args.mi_eval_downsample_to_n),
      partial(compute_mi_markov, n_samples=args.mi_eval_downsample_to_n),
      partial(compute_mi_vf, n_samples=args.mi_eval_downsample_to_n, gamma=args.gamma),
      partial(compute_rep_stats, model_modules=model_modules, num_steps=args.num_steps,
              num_rollouts=args.mi_eval_downsample_to_n // args.num_steps),
    ]
    print(f"Starting MI eval on test set")
    mi_stats_test = evaluate_mi(
      checkpoint_path,
      make_env,
      make_agent,
      eval_fns_test,
      args,
      seed=args.seed,
      total_timesteps=args.mi_eval_total_timesteps,
      num_envs=args.num_envs,
      start_level=args.num_train_levels
    )
    pprint(mi_stats_test)
    print(f"Starting MI eval on train set")
    mi_stats_train = evaluate_mi(
      checkpoint_path,
      make_env,
      make_agent,
      eval_fns_train,
      args,
      seed=args.seed,
      total_timesteps=args.mi_eval_total_timesteps,
      num_envs=args.num_envs,
      num_levels=args.num_train_levels
    )
    pprint(mi_stats_train)
    print(f"Uploading MI eval results to wandb/tensorboard")
    for k, v in mi_stats_test.items():
      if k.startswith('walltime'):
        writer.add_scalar(f"stats/{k}", v, args.total_timesteps + 2)
      else:
        writer.add_scalar(f"rep_eval_test/{k}", v, args.total_timesteps + 2)
    for k, v in mi_stats_train.items():
      if k.startswith('walltime'):
        writer.add_scalar(f"stats/{k}", v, args.total_timesteps + 2)
      else:
        writer.add_scalar(f"rep_eval_train/{k}", v, args.total_timesteps + 2)
    runstate.after_post_eval(agent_state, args)
    print(f"MI eval completed.")
    time.sleep(10)

  writer.close()
