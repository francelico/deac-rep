import os
import gc
import queue
import random
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Sequence
from functools import partial
from tqdm import tqdm

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro  # arg parsing

from flax.linen.initializers import orthogonal, lecun_normal, glorot_uniform
from jax.nn.initializers import constant, Initializer
from flax.training.train_state import TrainState

from rich.pretty import pprint
from tensorboardX import SummaryWriter
from brax.training.distribution import NormalTanhDistribution

from eval_utils.env_utils import make_pixel_brax, RewardWrapper, RunningMeanStd
from utils.utils import (
    set_layer_init_fn,
    binary_cross_entropy_with_logits,
    cosine_distance,
    absolute_diff,
)
import utils.job_util
import utils.repmetric_util
import utils.bisim_util

"""ARGS"""
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 12345
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
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    preemptible: bool = False
    "Whether to save the model on job preemption or timeout. SLURM only."
    start_from_checkpoint: bool = False
    "Begin training from checkpoint (if it exists). Set to True to allow resuming training on preemptible clusters."

    load_folder: str = ""
    "Folder from which to load .ckpt file"
    measure_mi: bool = False
    "Whether to measure the MI stats"
    mi_eval_downsample_to_n: int = 4096
    mi_eval_total_timesteps: int = 8128  # 32512 #65024

    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    checkpoint_frequency: int = 0
    "Save a checkpoint every n seconds. Default 0 means signal-based checkpointing only."
    log_frequency: int = 16
    "the logging frequency of the model performance (in terms of `updates`)"
    eval_every_n_log_cycles: int = 2
    "eval the model on the test environments every n log cycles"
    num_eval_episodes: int = 1000
    "the number of episodes to evaluate the model on at the end of training (both on training and testing levels)"

    recording_threshold: int = 99999
    "eval returns to break training to record video"
    do_recording: bool = False
    "do recording?"

    # Environment specific arguments
    env_id: str = "ant"
    "the id of the environment"
    backend: str = "spring"
    "the physics backend to use"
    action_repeat: int = 1
    "action repeat?"
    hw: int = 64
    "the height/width of the images"
    distractor: str = "none"
    "the distractor type to use"
    video_path: str = "none"
    "the path to the DAVIS videos; only for video distractors"
    float32_pixels: bool = False
    "the env returns float32 pixels? If False, then uint8"
    run_notes: str = "none"
    "notes to use for wandb logging. can be used for grouping purposes"

    # Algorithm specific arguments
    num_train_levels: int = 200
    "number of Procgen levels to use for training"
    total_timesteps: int = 25000000
    "total timesteps of the experiments"
    norm_and_clip_rewards: bool = False
    "whether to normalize and clip the rewards"
    learning_rate: float = 5e-4
    "the learning rate of the optimizer"
    adam_eps: float = 1e-8
    "the epsilon parameter of the Adam optimizer"
    local_num_envs: int = 32
    "the number of parallel game environments"
    num_actor_threads: int = 1
    "the number of actor threads to use"
    num_steps: int = 32
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
    num_policy_phases: int = 16
    "the number of policy phases to run before running an auxiliary phase"
    num_policy_phase_batches_for_auxiliary_phase: int = 1
    policy_phase_epochs: int = 1
    "the K epochs to run during the policy phase"
    auxiliary_phase_epochs: int = 1
    "the K epochs to run during the auxiliary phase"
    norm_adv: str = "batch"
    "Type of advantage normalization. Options: ['batch', 'minibatch', 'none']"
    clip_coef: float = 0.2
    "the surrogate clipping coefficient"
    clip_vf: bool = False
    "Toggles value clipping in the PPO loss (also uses clip_coef)"
    ent_coef: float = 0.01
    "coefficient of the entropy"
    vf_coef: float = 0.5
    "coefficient for the value prediction loss"
    aux_vf_loss_coef: float = 0.25
    "coefficient for the auxiliary head value prediction loss"
    aux_vf_loss_coef2: float = 0.25
    "coefficient for the auxiliary head value prediction loss"
    bc_coef: float = 0.0
    "coefficient for the behavior cloning loss during the auxiliary phase"
    aux_coef: float = 1.0
    "coefficient that scales the aux phase loss *in total*"
    max_grad_norm: float = 0.5
    "the maximum norm for the gradient clipping"
    channels: List[int] = field(default_factory=lambda: [16, 32, 32])
    "the channels of the CNN"
    hiddens: List[int] = field(default_factory=lambda: [256])
    "the hiddens size of the MLP"
    kernel_init_method: str = "ppo_cleanba"
    "which reference implementation to follow for weight initialization"

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

    target_smoothing_coef: float = 0.05
    "the coefficient for soft target network updates"

    aux_during_ppo: bool = False
    aux_during_aux: bool = False
    no_actor_grads_aux_phase: bool = False

    markov_coef_a_pol_phase: float = 0.0
    markov_coef_c_pol_phase: float = 0.0
    markov_coef_a_aux_phase: float = 0.0
    markov_coef_c_aux_phase: float = 0.0

    mico_coef_actor_pol_phase: float = 0.0
    mico_coef_critic_pol_phase: float = 0.0
    mico_coef_critic_aux_phase: float = 0.0
    mico_coef_actor_aux_phase: float = 0.0
    aux_vf_coef_aux_phase: float = 1.0
    adv_coef_aux_phase: float = 0.0

    do_aux_loss: bool = False

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
    global_learner_devices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    num_sharded_rollouts_aux_phase: int = 0
    learner_devices: Optional[List[str]] = None
    baseline_score: Optional[float] = (
        None  # the baseline score for the environment for reporting normalized scores
    )


"""AGENT NETWORKS"""


def make_agent(args, envs, key, print_model=False):
    (actor_params, critic_params), modules, key = make_ppg_model(
        args, envs, key, print_model=print_model
    )

    linear_schedule_fn = partial(
        linear_schedule, update_epochs=args.policy_phase_epochs
    )

    if args.start_from_checkpoint:
        file = __file__.split("/")[-1].replace(".py", "")
        print(f"Attempting to find file {file}.ckpt in {args.load_folder}")
        ckpt_file = f"{args.load_folder}/{file}.ckpt"
        params = job_util.load_params(ckpt_file)
        actor_params = params["src"]["actor"]
        critic_params = params["src"]["critic"]

    actor_state = TrainState.create(
        apply_fn=None, params=actor_params, tx=make_opt(args, linear_schedule_fn)
    )
    critic_state = TrainState.create(
        apply_fn=None, params=critic_params, tx=make_opt(args, linear_schedule_fn)
    )

    if not args.start_from_checkpoint:
        agent_state = AgentTrainStateWithTarget(
            actor_state,
            critic_state,
            actor_target_params=actor_params,
            critic_target_params=critic_params,
        )
    else:
        agent_state = AgentTrainStateWithTarget(
            actor_state,
            critic_state,
            actor_target_params=params["tgt"]["actor"],
            critic_target_params=params["tgt"]["critic"],
        )

    return agent_state, modules, key


def make_ppg_model(args, envs, key, print_model=False):
    (
        key,
        actor_base_key,
        policy_head_key,
        auxiliary_head_key,
        auxiliary_advantage_head_key,
        actor_projector_key,
        actor_predictor_key,
        critic_base_key,
        value_head_key,
        critic_projector_key,
        critic_predictor_key,
        dyna_head_actor_key,
        dyna_head_critic_key,
    ) = jax.random.split(key, 13)

    kernel_init_dict = set_layer_init_fn(args)

    actor_base = ResNetBase(args.channels, args.hiddens, kernel_init_dict)
    policy_head = PolicyHead(envs.action_size, kernel_init_dict["policy_head_dense"])
    dyna_head_actor = DynaHead(
        args.hiddens[-1], envs.action_size, kernel_init_dict["dyna_head_actor_dense"]
    )
    auxiliary_head = ValueHead(kernel_init_dict["auxiliary_head_dense"])
    auxiliary_advantage_head = AdvantageHead(
        envs.action_size, kernel_init_dict["auxiliary_advantage_head_dense"]
    )
    critic_base = ResNetBase(args.channels, args.hiddens, kernel_init_dict)
    value_head = ValueHead(kernel_init_dict["value_head_dense"])
    dyna_head_critic = DynaHead(
        args.hiddens[-1], envs.action_size, kernel_init_dict["dyna_head_critic_dense"]
    )

    actor_base_params = actor_base.init(
        actor_base_key, np.array([envs.observation_sample])
    )
    critic_base_params = critic_base.init(
        critic_base_key, np.array([envs.observation_sample])
    )
    hidden_sample_actor = actor_base.apply(
        actor_base_params, np.array([envs.observation_sample])
    )
    hidden_sample_critic = critic_base.apply(
        critic_base_params, np.array([envs.observation_sample])
    )

    actor_distribution = NormalTanhDistribution(envs.action_size)

    actor_params = ActorParams(
        actor_base_params,
        policy_head.init(policy_head_key, hidden_sample_actor),
        auxiliary_head.init(auxiliary_head_key, hidden_sample_actor),
        auxiliary_advantage_head.init(
            auxiliary_advantage_head_key,
            hidden_sample_actor,
            np.zeros(shape=(1, envs.action_size)),
        ),
        dyna_head_actor.init(
            dyna_head_actor_key,
            hidden_sample_actor,
            np.zeros(shape=(1, envs.action_size)),
            hidden_sample_actor,
        ),
    )
    critic_params = CriticParams(
        critic_base_params,
        value_head.init(value_head_key, hidden_sample_critic),
        dyna_head_critic.init(
            dyna_head_critic_key,
            hidden_sample_critic,
            np.zeros(shape=(1, envs.action_size)),
            hidden_sample_critic,
        ),
    )

    modules = {
        "actor_base": actor_base,
        "policy_head": policy_head,
        "auxiliary_head": auxiliary_head,
        "auxiliary_advantage_head": auxiliary_advantage_head,
        "critic_base": critic_base,
        "value_head": value_head,
        "actor_distribution": actor_distribution,
    }

    return (actor_params, critic_params), modules, key


def make_opt(args, lr_scheduler_fn):
    return optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=lr_scheduler_fn if args.anneal_lr else args.learning_rate,
                eps=args.adam_eps,
            ),
        ),
        every_k_schedule=args.gradient_accumulation_steps,
    )


class ResidualBlock(nn.Module):
    channels: int
    kernel_init_fn: Initializer

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn)(
            x
        )
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn)(
            x
        )
        return x + inputs


class ConvSequence(nn.Module):
    channels: int
    kernel_init_fn_conv: Initializer = lecun_normal()
    kernel_init_fn_resblock: Initializer = lecun_normal()

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels, kernel_size=(3, 3), kernel_init=self.kernel_init_fn_conv
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels, self.kernel_init_fn_resblock)(x)
        x = ResidualBlock(self.channels, self.kernel_init_fn_resblock)(x)
        return x


class ResNetBase(nn.Module):
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)
    kernel_init_dict: dict = field(
        default_factory=lambda: {
            "convsequence_conv": lecun_normal(),
            "convsequence_resblock": lecun_normal(),
            "resnet_dense": orthogonal(2**0.5),
        }
    )

    @nn.compact
    def __call__(self, x):
        if not args.float32_pixels:
            x = (x / 255.0).astype(jnp.float32)
        for channels in self.channels:
            x = ConvSequence(
                channels,
                kernel_init_fn_conv=self.kernel_init_dict["convsequence_conv"],
                kernel_init_fn_resblock=self.kernel_init_dict["convsequence_resblock"],
            )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(
                hidden,
                kernel_init=self.kernel_init_dict["resnet_dense"],
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        return x


class ValueHead(nn.Module):
    kernel_init_fn: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)


class AdvantageHead(nn.Module):
    action_dim: int
    kernel_init_fn: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, x, actions):
        gae_inputs = jnp.concatenate([x, actions], axis=1)
        return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(
            gae_inputs
        )


class PolicyHead(nn.Module):
    # TODO: we should be parameterizing some Gaussian. For SAC, we learn the covariance diagonal. Does PPG do that?
    action_dim: int
    kernel_init_fn: Initializer = orthogonal(0.01)
    logvar_min: float = -10.0
    logvar_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        mu = jnp.tanh(
            nn.Dense(
                self.action_dim,
                kernel_init=self.kernel_init_fn,
                bias_init=constant(0.0),
            )(x)
        )

        sigma = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0)
        )(x)
        sigma = jnp.clip(sigma, self.logvar_min, self.logvar_max)
        return mu, sigma


class DynaHead(nn.Module):
    hidden_dim: int
    action_dim: int
    kernel_init_fn: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, rep, action, next_rep):
        x = self.concat_input(rep, action, next_rep)
        x = nn.Dense(
            self.hidden_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim, kernel_init=self.kernel_init_fn, bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=self.kernel_init_fn, bias_init=constant(0.0))(x)

    def concat_input(self, rep, action, next_rep):
        return jnp.concatenate([rep, action, next_rep], axis=-1)


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
class AgentTrainState:
    actor_state: TrainState
    critic_state: TrainState

    def get_params(self):
        return AgentParams(
            actor_params=self.actor_state.params,
            critic_params=self.critic_state.params,
        )

    def apply_gradients(self, grads):
        actor_state = self.actor_state.apply_gradients(grads=grads.actor_params)
        critic_state = self.critic_state.apply_gradients(grads=grads.critic_params)
        return AgentTrainState(actor_state, critic_state)


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
        return AgentTrainStateWithTarget(
            actor_state,
            critic_state,
            actor_target_params=self.actor_target_params,
            critic_target_params=self.critic_target_params,
        )


class PolPhaseStorage(NamedTuple):
    obs: list
    dones: list
    actions: list
    logprobs: list
    values: list
    env_ids: jax.Array
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
        policy_version  # past update cycles
        * args.num_steps  # env steps per update cycle (per env)
        * args.local_num_envs  # envs per actor thread
        * args.num_actor_threads  # actor threads on each device
        * len(args.actor_device_ids)  # devices on each jax process
        * args.world_size  # number of jax processes
    )


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    envs,
    eval_envs,
    envs_step_fn,
    eval_envs_step_fn,
    # actor_distribution,
    initial_policy_version=0,
):
    assert initial_policy_version >= 0
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = get_global_step(initial_policy_version, args)
    start_step = global_step
    start_time = time.time()

    if args.eval_every_n_log_cycles > 0:
        pass
    else:
        eval_envs = None

    @partial(jax.jit, static_argnames=("eval",))
    def get_model_outputs(
        params: AgentParams,
        obs: np.ndarray,  # TODO: no longer an np.ndarray. it's the "state" object from brax env
        key: jax.random.PRNGKey,
        eval: bool,
    ):
        actor_base = ResNetBase(args.channels, args.hiddens).apply(
            params.actor_params.base_params, obs.pixels
        )

        critic_base = ResNetBase(args.channels, args.hiddens).apply(
            params.critic_params.base_params, obs.pixels
        )

        mu, sigma = PolicyHead(envs.action_size).apply(
            params.actor_params.policy_head_params, actor_base
        )
        dist = ACTOR_DISTRIBUTION.create_dist(jnp.concatenate([mu, sigma], -1))

        if not eval:
            action = dist.sample(seed=key)
        else:
            action = dist.loc
        _, key = jax.random.split(key, 2)

        logprob = dist.log_prob(action).sum(-1, keepdims=True)

        value_pred = ValueHead().apply(
            params.critic_params.value_head_params, critic_base
        )
        aux_value_pred = ValueHead().apply(
            params.actor_params.auxiliary_head_params, actor_base
        )
        return (
            obs,
            (action, mu, sigma, actor_base, critic_base, value_pred, aux_value_pred),
            logprob,
            value_pred.squeeze(),
            aux_value_pred.squeeze(),
            key,
        )

    def step_envs_once(step_fn, obs, action):
        action, mu, sigma, actor_base, critic_base, value_pred, aux_value_pred = action
        next_obs = step_fn(obs, action)
        next_reward = next_obs.reward

        next_done = next_obs.done
        next_truncated = next_obs.info["truncation"]
        next_ts = next_obs.info["steps"]
        next_info = next_obs.info

        next_terminated = next_done * (1 - next_truncated)
        next_info["terminated"] = next_terminated.copy()
        next_info["truncated"] = next_truncated.copy()
        next_info["reward"] = next_reward.copy()

        return (
            next_obs,
            next_reward,
            next_done,
            next_terminated,
            next_truncated,
            next_ts,
            next_info,
        )

    def update_rollout_stats(
        stats, next_info, term, trunc, cached_ep_returns, cached_ep_lengths
    ):
        cached_ep_returns += next_info["reward"]
        stats["ep_returns"] = jnp.where(
            next_info["terminated"] + next_info["truncated"],
            cached_ep_returns,
            stats["ep_returns"],
        )
        cached_ep_returns *= (1 - next_info["terminated"]) * (
            1 - next_info["truncated"]
        )

        stats["levels_solved"] = jnp.zeros_like(next_info["terminated"])
        cached_ep_lengths += 1
        stats["ep_lengths"] = jnp.where(
            next_info["terminated"] + next_info["truncated"],
            cached_ep_lengths,
            stats["ep_lengths"],
        )
        cached_ep_lengths *= (
            (1 - next_info["terminated"])
            * (1 - next_info["truncated"])
            * (1 - term)
            * (1 - trunc)
        )

        return stats, cached_ep_returns, cached_ep_lengths

    cached_ep_returns = jnp.zeros((args.local_num_envs,), dtype=jnp.float32)
    cached_ep_lengths = jnp.zeros((args.local_num_envs,), dtype=jnp.float32)
    rollout_stats = {
        "ep_returns": jnp.zeros((args.local_num_envs,), dtype=jnp.float32),
        "ep_lengths": jnp.zeros((args.local_num_envs,), dtype=jnp.int32),
        "levels_solved": jnp.zeros((args.local_num_envs,), dtype=jnp.bool_),
    }
    comp_times = {
        "param_queue_get": jnp.zeros((1,)),
        "rollout": jnp.zeros((1,)),
        "rollout_queue_put": jnp.zeros((1,)),
        "inference": jnp.zeros((1,)),
        "storage": jnp.zeros((1,)),
        "d2h": jnp.zeros((1,)),
        "env_send": jnp.zeros((1,)),
    }
    actor_policy_version = initial_policy_version
    next_obs = envs.reset(envs.seed)
    next_done = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
    next_terminated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
    next_truncated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
    next_ts = jnp.zeros(args.local_num_envs, dtype=jax.numpy.int32)

    if args.norm_and_clip_rewards:
        reward_wrapper = RewardWrapper(
            normalize=True,
            clip=False,
            clip_coef=10.0,
            return_rms=RunningMeanStd(
                jnp.zeros((), dtype=jnp.float32), jnp.ones((), dtype=jnp.float32), 1e-4
            ),
            discounted_returns=jnp.zeros((args.local_num_envs,)),
            gamma=args.gamma,
        )

    if eval_envs is not None:
        ev_cached_ep_returns = jnp.zeros((args.local_num_envs,), dtype=jnp.float32)
        ev_cached_ep_lengths = jnp.zeros((args.local_num_envs,), dtype=jnp.float32)
        ev_rollout_stats = {
            "ep_returns": jnp.zeros((args.local_num_envs,), dtype=jnp.float32),
            "ep_lengths": jnp.zeros((args.local_num_envs,), dtype=jnp.int32),
            "levels_solved": jnp.zeros((args.local_num_envs,), dtype=jnp.bool_),
        }
        
        ev_comp_times = {
            "rollout": jnp.zeros((1,)),
            "inference": jnp.zeros((1,)),
            "storage": jnp.zeros((1,)),
            "d2h": jnp.zeros((1,)),
            "env_send": jnp.zeros((1,)),
        }
        ev_next_obs = eval_envs.reset(eval_envs.seed)
        ev_next_terminated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)
        ev_next_truncated = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)

    @jax.jit
    def prepare_data(storage: List[PolPhaseStorage]) -> PolPhaseStorage:
        return jax.tree_map(
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    for update in range(initial_policy_version + 1, args.num_updates + 2):
        update_time_start = time.time()
        params_queue_get_time_start = time.time()

        if args.concurrency:
            if update != initial_policy_version + 2:
                params = params_queue.get()
                params.actor_params.base_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()
                actor_policy_version += 1

        else:
            params = params_queue.get()
            actor_policy_version += 1

        comp_times["param_queue_get"] += time.time() - params_queue_get_time_start
        rollout_time_start = time.time()
        storage = []

        times = []
        for _ in range(0, args.num_steps):
            cached_next_obs = next_obs
            cached_next_done = next_done
            cached_next_terminated = next_terminated
            cached_next_truncated = next_truncated
            global_step += (
                len(next_done)
                * args.num_actor_threads
                * len_actor_device_ids
                * args.world_size
            )

            inference_time_start = time.time()
            cached_next_obs, action, logprob, value_pred, gae_pred, key = (
                get_model_outputs(params, cached_next_obs, key, eval=False)
            )
            comp_times["inference"] += time.time() - inference_time_start

            (
                next_obs,
                next_reward,
                next_done,
                next_terminated,
                next_truncated,
                next_ts,
                next_info,
            ) = step_envs_once(envs_step_fn, cached_next_obs, action)

            if args.norm_and_clip_rewards:
                reward_wrapper, next_reward = reward_wrapper.process_rewards(
                    next_reward, next_terminated, next_truncated
                )

            storage_time_start = time.time()
            storage.append(
                PolPhaseStorage(
                    obs=cached_next_obs.pixels,
                    dones=cached_next_done,
                    actions=action[0],
                    logprobs=logprob,
                    values=value_pred,
                    env_ids=jnp.zeros((args.num_envs,)),
                    rewards=next_reward,
                    truncations=cached_next_truncated,
                    terminations=cached_next_terminated,
                )
            )

            comp_times["storage"] += time.time() - storage_time_start
            rollout_stats, cached_ep_returns, cached_ep_lengths = update_rollout_stats(
                rollout_stats,
                next_info,
                cached_next_terminated,
                cached_next_truncated,
                cached_ep_returns,
                cached_ep_lengths,
            )

        comp_times["rollout"] += time.time() - rollout_time_start
        partitioned_storage = prepare_data(storage)
        sharded_storage = PolPhaseStorage(
            *list(
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),
                    partitioned_storage,
                )
            )
        )
        # next_obs, next_done, next_terminated are still in the host
        sharded_last_obs = jax.device_put_sharded(
            jnp.split(next_obs.pixels, len(learner_devices)), devices=learner_devices
        )
        sharded_last_done = jax.device_put_sharded(
            jnp.split(next_done, len(learner_devices)), devices=learner_devices
        )
        sharded_last_term = jax.device_put_sharded(
            jnp.split(next_terminated, len(learner_devices)), devices=learner_devices
        )

        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_last_obs,
            sharded_last_done,
            sharded_last_term,
            comp_times["param_queue_get"],
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        comp_times["rollout_queue_put"] += time.time() - rollout_queue_put_time_start

        if update % args.log_frequency == 0:
            avg_episodic_return = jnp.mean(rollout_stats["ep_returns"])
            avg_norm_episodic_return = (
                avg_episodic_return / args.baseline_score if args.baseline_score else 0
            )
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, "
                    f"avg_episodic_return={avg_episodic_return}, "
                    f"rollout_time={np.mean(comp_times['rollout'])}"
                )
                print(
                    "SPS:", int((global_step - start_step) / (time.time() - start_time))
                )

            writer.add_scalar(
                "stats/rollout_time", np.mean(comp_times["rollout"]), global_step
            )
            writer.add_scalar(
                "charts/avg_episodic_return", avg_episodic_return, global_step
            )
            writer.add_scalar(
                "charts/avg_norm_episodic_return", avg_norm_episodic_return, global_step
            )
            writer.add_scalar(
                "charts/avg_solved_rate",
                np.mean(rollout_stats["levels_solved"]),
                global_step,
            )
            writer.add_scalar(
                "charts/avg_episodic_length",
                np.mean(rollout_stats["ep_lengths"]),
                global_step,
            )
            writer.add_scalar(
                "stats/params_queue_get_time",
                np.mean(comp_times["param_queue_get"]),
                global_step,
            )
            writer.add_scalar(
                "stats/inference_time", comp_times["inference"], global_step
            )
            writer.add_scalar("stats/storage_time", comp_times["storage"], global_step)
            writer.add_scalar("stats/d2h_time", comp_times["d2h"], global_step)
            writer.add_scalar(
                "stats/env_send_time", comp_times["env_send"], global_step
            )
            writer.add_scalar(
                "stats/rollout_queue_put_time",
                np.mean(comp_times["rollout_queue_put"]),
                global_step,
            )
            writer.add_scalar(
                "charts/SPS",
                int((global_step - start_step) / (time.time() - start_time)),
                global_step,
            )
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
            if (
                eval_envs is not None
                and (update // args.log_frequency) % args.eval_every_n_log_cycles == 0
            ):
                print(f"---- beginning evaluation ----")
                ev_rollout_time_start = time.time()

                for _ in range(1000 // args.action_repeat):
                    ev_cached_next_terminated = ev_next_terminated
                    ev_cached_next_truncated = ev_next_truncated
                    inference_time_start = time.time()
                    ev_next_obs, action, _, _, _, key = get_model_outputs(
                        params, ev_next_obs, key, eval=True
                    )

                    if jnp.isnan(action[0]).mean() > 0:
                        raise ValueError(f"NaN found in action")

                    if jnp.isinf(action[0]).mean() > 0:
                        raise ValueError(f"inf found in action")

                    ev_comp_times["inference"] += time.time() - inference_time_start

                    (
                        ev_next_obs,
                        _,
                        _,
                        ev_next_terminated,
                        ev_next_truncated,
                        _,
                        ev_next_info,
                    ) = step_envs_once(eval_envs_step_fn, ev_next_obs, action)

                    if jnp.isnan(ev_next_obs.pixels).mean() > 0:
                        raise ValueError(f"NaN found in pixels")

                    if jnp.isinf(ev_next_obs.pixels).mean() > 0:
                        raise ValueError(f"inf found in pixels")

                    if jnp.isinf(ev_next_info["reward"]).mean() > 0:
                        raise ValueError(f"inf found in reward")

                    ev_rollout_stats, ev_cached_ep_returns, ev_cached_ep_lengths = (
                        update_rollout_stats(
                            ev_rollout_stats,
                            ev_next_info,
                            ev_cached_next_terminated,
                            ev_cached_next_truncated,
                            ev_cached_ep_returns,
                            ev_cached_ep_lengths,
                        )
                    )

                if args.do_recording:
                    if (
                        jnp.mean(ev_rollout_stats["ep_returns"])
                        >= args.recording_threshold
                    ):
                        print(f"//// Doing Recording ////")

                        if args.float32_pixels:
                            frames = [
                                np.array(
                                    x.obs[0, :, :, 6:][None].transpose(0, 3, 1, 2) * 255
                                ).astype(np.uint8)
                                for x in storage
                            ]
                        else:
                            frames = [
                                np.array(x.obs[0, :, :, 6:][None].transpose(0, 3, 1, 2))
                                for x in storage
                            ]
                        wandb.log(
                            {"video": wandb.Video(np.concatenate(frames, 0), fps=30)}
                        )

                        print(f"returns: {np.sum([x.returns for x in storage])}")

                ev_comp_times["rollout"] += time.time() - ev_rollout_time_start
                if device_thread_id == 0:
                    print(
                        f"global_step={global_step}, "
                        f"avg_episodic_return_eval={jnp.mean(ev_rollout_stats['ep_returns'])}, "
                        f"rollout_time_eval={jnp.mean(ev_comp_times['rollout'])}"
                    )
                avg_episodic_return_eval = jnp.mean(ev_rollout_stats["ep_returns"])
                avg_norm_episodic_return_eval = (
                    avg_episodic_return_eval / args.baseline_score
                    if args.baseline_score
                    else 0
                )
                writer.add_scalar(
                    "stats/eval_rollout_time",
                    np.mean(ev_comp_times["rollout"]),
                    global_step,
                )
                writer.add_scalar(
                    "charts/avg_episodic_return_eval",
                    avg_episodic_return_eval,
                    global_step,
                )
                writer.add_scalar(
                    "charts/avg_norm_episodic_return_eval",
                    avg_norm_episodic_return_eval,
                    global_step,
                )
                writer.add_scalar(
                    "charts/avg_solved_rate_eval",
                    np.mean(ev_rollout_stats["levels_solved"]),
                    global_step,
                )
                writer.add_scalar(
                    "charts/avg_episodic_length_eval",
                    np.mean(ev_rollout_stats["ep_lengths"]),
                    global_step,
                )
                writer.add_scalar(
                    "stats/inference_time_eval", ev_comp_times["inference"], global_step
                )
                writer.add_scalar(
                    "stats/d2h_time_eval", ev_comp_times["d2h"], global_step
                )
                writer.add_scalar(
                    "stats/env_send_time_eval", ev_comp_times["env_send"], global_step
                )


def linear_schedule(count, update_epochs):
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches) gradient updates
    frac = 1.0 - (count // (args.num_minibatches * update_epochs)) / args.num_updates
    return args.learning_rate * frac


# as in https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/41332b78dfb50321c29bade65f9d244387f68a60/a2c_ppo_acktr/algo/ppo.py#L68
def clipped_value_loss_fn(newvalue, target_value, rollout_value, clip_coef):
    value_preds_clipped = rollout_value + jnp.clip(
        newvalue - rollout_value, -clip_coef, clip_coef
    )
    v_loss1 = 0.5 * (newvalue - target_value) ** 2
    v_loss2 = 0.5 * (value_preds_clipped - target_value) ** 2
    return jnp.maximum(v_loss1, v_loss2).mean(), (v_loss1.mean(), v_loss2.mean())


def value_loss_fn(newvalue, target_value, rollout_value):
    v_loss = 0.5 * ((newvalue - target_value) ** 2).mean()
    return v_loss, (jnp.zeros_like(v_loss), jnp.zeros_like(v_loss))


def huber_loss(
    targets: jnp.ndarray, predictions: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
    """Implementation of the Huber loss with threshold delta.

    Let `x = |targets - predictions|`, the Huber loss is defined as:
    `0.5 * x^2` if `x <= delta`
    `0.5 * delta^2 + delta * (x - delta)` otherwise.

    Args:
      targets: Target values.
      predictions: Prediction values.
      delta: Threshold.

    Returns:
      Huber loss.
    """
    x = jnp.abs(targets - predictions)
    return jnp.where(x <= delta, 0.5 * x**2, 0.5 * delta**2 + delta * (x - delta))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.local_batch_size = int(
        args.local_num_envs
        * args.num_steps
        * args.num_actor_threads
        * len(args.actor_device_ids)
    )
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids))
        * args.num_actor_threads
        % args.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    assert (
        args.log_frequency % args.num_policy_phases == 0
    ), "log_frequency must be divisible by num_policy_phases"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(
                len(args.learner_device_ids) + len(args.actor_device_ids)
            ),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))
    if args.no_cuda:
        jax.config.update("jax_platform_name", "cpu")

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = (
        args.local_num_envs
        * args.world_size
        * args.num_actor_threads
        * len(args.actor_device_ids)
    )
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size

    args.num_policy_phase_batches_for_auxiliary_phase = (
        args.num_policy_phases
        if args.num_policy_phase_batches_for_auxiliary_phase == -1
        else args.num_policy_phase_batches_for_auxiliary_phase
    )

    args.num_sharded_rollouts_aux_phase = (
        args.world_size
        * args.num_actor_threads
        * len(args.actor_device_ids)
        * args.num_policy_phase_batches_for_auxiliary_phase
    )

    args.aux_batch_rollouts = (
        args.num_envs * args.num_policy_phase_batches_for_auxiliary_phase
    )
    args.aux_batch_size = (
        args.batch_size * args.num_policy_phase_batches_for_auxiliary_phase
    )

    assert (
        args.aux_batch_size == args.aux_batch_rollouts * args.num_steps
    ), "check aux_batch_size calculation failed"
    args.aux_minibatch_size = args.num_rollouts_in_aux_mb * args.num_steps
    assert (
        args.aux_batch_size % args.aux_minibatch_size == 0
    ), "aux_batch_size must be divisible by aux_minibatch_size"
    args.num_aux_minibatches = args.aux_batch_size // args.aux_minibatch_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)

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
    run_name_no_uuid = f"{args.env_id}__{args.distractor}__{args.exp_name}__{args.seed}"
    prev_rundir = None
    resume = False
    
    if not resume:
        run_name = f"{run_name_no_uuid}__{uuid.uuid4()}"
        print(f"Starting new run: {run_name}")

    checkpoint_path = f"runs/{run_name}/{args.exp_name}.ckpt"

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    envs, _, _ = make_pixel_brax(
        backend=args.backend,
        env_name=args.env_id,
        n_envs=args.local_num_envs,
        seed=args.seed,
        hw=args.hw,
        distractor=args.distractor,
        video_path=args.video_path,
        video_set="train",
        return_float32=args.float32_pixels,
    )

    eval_envs, _, _ = make_pixel_brax(
        backend=args.backend,
        env_name=args.env_id,
        n_envs=args.local_num_envs,
        seed=args.seed + 1337,
        hw=args.hw,
        distractor=args.distractor,
        video_path=args.video_path,
        video_set="test",
        return_float32=args.float32_pixels,
    )

    envs_step_fn = jax.jit(envs.step)
    eval_envs_step_fn = jax.jit(eval_envs.step)

    if args.do_recording:
        rendering_env, _, _ = make_pixel_brax(
            backend=args.backend,
            env_name=args.env_id,
            n_envs=args.local_num_envs,
            seed=args.seed + 1337,
            hw=256,
            distractor=args.distractor,
            video_path=args.video_path,
            video_set="test",
            return_float32=args.float32_pixels,
        )

        rendering_envs_step_fn = jax.jit(rendering_env.step)

    agent_state, model_modules, key = make_agent(args, envs, key, print_model=True)
    ACTOR_DISTRIBUTION = model_modules["actor_distribution"]

    runstate = job_util.RunState(checkpoint_path, save_fn=job_util.save_agent_state)
    if resume:
        agent_state, runstate.metadata = job_util.restore_agent_state(
            checkpoint_path, agent_state, runstate.metadata
        )
        if runstate.completed:
            print("Run already completed. Exiting.")
            sys.exit(0)

    if args.track and args.local_rank == 0:
        import wandb
        import json

        """LOGGING"""
        with open("../../../wandb.txt", "r") as f:
            API_KEY = json.load(f)["api_key"]

        os.environ["WANDB_API_KEY"] = API_KEY
        os.environ["WANDB_DIR"] = "./wandb"
        os.environ["WANDB_CONFIG_DIR"] = "./wandb"
        os.environ["WANDB_DATA_DIR"] = "./wandb"

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            id=run_name,
            resume="allow" if resume else None,
            monitor_gym=False,
            save_code=False,
        )

    initial_global_step = get_global_step(
        runstate.metadata.get("learner_policy_version", 0), args
    )
    writer = SummaryWriter(
        f"runs/{run_name}", purge_step=initial_global_step + 1, flush_secs=50
    )
    runstate.to_close = [envs, writer]
    if initial_global_step == 0:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    learner_keys = jax.device_put_replicated(key, learner_devices)
    ppo_value_loss_fn = (
        partial(clipped_value_loss_fn, clip_coef=args.clip_coef)
        if args.clip_vf
        else value_loss_fn
    )

    @jax.jit
    def get_representation(
        params,
        obs: np.ndarray,
    ):
        hidden = ResNetBase(args.channels, args.hiddens).apply(params.base_params, obs)
        return hidden

    def update_target_params(target_p, online_p, tau=args.target_smoothing_coef):
        return jax.lax.stop_gradient((tau * online_p + (1 - tau) * target_p))

    def get_error_target_params(target_p, online_p):
        target_param_error = jax.tree_map(
            lambda x, y: jnp.abs(x - y).mean(), target_p, online_p
        )
        target_param_error = jax.tree_util.tree_leaves(target_param_error)
        return jnp.stack(target_param_error).mean().item()

    @jax.jit
    def get_value(
        critic_params: CriticParams,
        obs: np.ndarray,
    ):
        hidden = get_representation(critic_params, obs)
        value = ValueHead().apply(critic_params.value_head_params, hidden).squeeze(-1)
        return value

    @jax.jit
    def get_value_and_rep(
        critic_params: CriticParams,
        obs: np.ndarray,
    ):
        hidden = get_representation(critic_params, obs)
        value = ValueHead().apply(critic_params.value_head_params, hidden).squeeze(-1)
        return value, hidden

    @jax.jit
    def get_auxadv(actor_params: ActorParams, reps: jnp.ndarray, actions: jnp.ndarray):
        adv_pred = AdvantageHead(envs.action_size).apply(
            actor_params.auxiliary_advantage_head_params, reps, actions
        )
        return adv_pred.squeeze(-1)

    @jax.jit
    def get_logprob_entropy(
        actor_params: ActorParams,
        obs: jax.Array,
        actions: jax.Array,
    ):
        hidden = get_representation(actor_params, obs)
        mu, sigma = PolicyHead(envs.action_size).apply(
            actor_params.policy_head_params, hidden
        )
        dist = ACTOR_DISTRIBUTION.create_dist(jnp.concatenate([mu, sigma], -1))

        logprob = dist.log_prob(actions).sum(-1, keepdims=True)

        entropy = dist.entropy()

        return logprob, entropy

    @jax.jit
    def get_logprob_entropy_rep(
        actor_params: ActorParams,
        obs: jax.Array,
        actions: jax.Array,
    ):
        hidden = get_representation(actor_params, obs)
        mu, sigma = PolicyHead(envs.action_size).apply(
            actor_params.policy_head_params, hidden
        )
        dist = ACTOR_DISTRIBUTION.create_dist(jnp.concatenate([mu, sigma], -1))

        logprob = dist.log_prob(actions).sum(-1, keepdims=True)

        entropy = dist.entropy()

        return logprob, entropy, hidden

    @jax.jit
    def get_logits(
        actor_params: ActorParams,
        obs: np.ndarray,
    ):
        hidden = get_representation(actor_params, obs)
        logits = PolicyHead(envs.action_size).apply(
            actor_params.policy_head_params, hidden
        )
        return logits

    @jax.jit
    def get_logits_auxvaluepred(
        actor_params: ActorParams,
        obs: np.ndarray,
    ):
        hidden = get_representation(actor_params, obs)
        mu, sigma = PolicyHead(envs.action_size).apply(
            actor_params.policy_head_params, hidden
        )
        aux_value_pred = (
            ValueHead().apply(actor_params.auxiliary_head_params, hidden).squeeze(-1)
        )
        return mu, sigma, aux_value_pred

    @jax.jit
    def get_logits_auxvaluepred_rep(
        actor_params: ActorParams,
        obs: np.ndarray,
    ):
        hidden = get_representation(actor_params, obs)
        mu, sigma = PolicyHead(envs.action_size).apply(
            actor_params.policy_head_params, hidden
        )
        aux_value_pred = (
            ValueHead().apply(actor_params.auxiliary_head_params, hidden).squeeze(-1)
        )
        return mu, sigma, aux_value_pred, hidden

    @jax.jit
    def get_dyna_head_logits(
        dyna_head_params,
        rep: np.ndarray,
        action: np.ndarray,
        next_rep: np.ndarray,
    ):
        return DynaHead(args.hiddens[-1], envs.action_size).apply(
            dyna_head_params, rep, action, next_rep
        )

    def sample_ood_actions(actions, key):
        key, subkey = jax.random.split(key, 2)
        noise = jax.random.normal(key=key, shape=actions.shape) * 0.3
        ood_actions = actions + noise
        return jnp.clip(ood_actions, -1.0, 1.0), key

    def ddcpg_dynamic_pred_loss_fn(dyna_head_params, rep, action, next_rep, key):
        action = jax.lax.stop_gradient(action)
        ood_actions, key = sample_ood_actions(action, key)
        key, subkey = jax.random.split(key)
        ood_next_rep = jax.random.permutation(subkey, next_rep)
        logits_in = get_dyna_head_logits(dyna_head_params, rep, action, next_rep)
        logits_ood_a = get_dyna_head_logits(
            dyna_head_params, rep, ood_actions, next_rep
        )
        logits_ood_s = get_dyna_head_logits(dyna_head_params, rep, action, ood_next_rep)
        loss_in = jax.vmap(binary_cross_entropy_with_logits)(
            logits_in, jnp.ones_like(logits_in)
        ).mean()
        loss_ood_a = jax.vmap(binary_cross_entropy_with_logits)(
            logits_ood_a, jnp.zeros_like(logits_ood_a)
        ).mean()
        loss_ood_s = jax.vmap(binary_cross_entropy_with_logits)(
            logits_ood_s, jnp.zeros_like(logits_ood_s)
        ).mean()
        dyna_loss = loss_in + 0.5 * loss_ood_a + loss_ood_s
        return dyna_loss, key

    def aux_loss_fn(
        params,
        target_params,
        obs,
        next_obs,
        dones,
        rewards,
        target_values,
        target_mu,
        target_sigma,
        actions,
        target_advantages,
        key,
    ):
        # params can be obtained with agent_state.get_params()
        # first acquiring representations
        new_mu, new_sigma, aux_value_preds, reps_a = get_logits_auxvaluepred_rep(
            params.actor_params, obs
        )
        nreps_a = get_representation(params.actor_params, next_obs)
        nreps_c = get_representation(params.critic_params, next_obs)
        value_preds, reps_c = get_value_and_rep(params.critic_params, obs)
        vloss, _ = value_loss_fn(value_preds, target_values, None)

        # value loss
        if args.aux_vf_coef_aux_phase > 0.0:
            aux_vloss, _ = value_loss_fn(aux_value_preds, target_values, None)
        else:
            aux_vloss = 0.0

        if args.adv_coef_aux_phase > 0.0:
            adv_pred = get_auxadv(params.actor_params, reps_a, actions)
            adv_loss = ((adv_pred - target_advantages) ** 2).mean()
        else:
            adv_loss = jnp.zeros_like(vloss)

        if args.bc_coef > 0.0:
            # computing (estimating) the KL div
            p = ACTOR_DISTRIBUTION.create_dist(
                jnp.concatenate([target_mu, target_sigma], -1)
            )
            q = ACTOR_DISTRIBUTION.create_dist(jnp.concatenate([new_mu, new_sigma], -1))
            samples = p.sample(seed=key)
            kl_policy_bc = (
                p.log_prob(samples).sum(-1, keepdims=True)
                - q.log_prob(samples).sum(-1, keepdims=True)
            ).mean()
        else:
            kl_policy_bc = jnp.zeros_like(vloss)

        # dynamic prediction loss
        dyna_loss_a, key = ddcpg_dynamic_pred_loss_fn(
            params.actor_params.dyna_head_params, reps_a, actions, nreps_a, key
        )

        if args.markov_coef_c_aux_phase > 0.0:
            dyna_loss_c, key = ddcpg_dynamic_pred_loss_fn(
                params.critic_params.dyna_head_params, reps_c, actions, nreps_c, key
            )
        else:
            dyna_loss_c = jnp.zeros_like(vloss)

        if args.mico_coef_critic_aux_phase > 0.0:
            target_reps_c = get_representation(target_params.critic_params, obs)
            target_nreps_c = get_representation(target_params.critic_params, next_obs)
            mico_loss, mico_stats = mico_reward_diff_loss_fn(
                rewards, reps_c, target_reps_c, target_nreps_c
            )
        else:
            mico_loss, mico_stats = jnp.zeros_like(vloss), {}

        if args.mico_coef_actor_aux_phase > 0.0:
            target_reps_a = get_representation(target_params.actor_params, obs)
            target_nreps_a = get_representation(target_params.actor_params, next_obs)
            mico_loss_a, mico_stats_a = mico_reward_diff_loss_fn(
                rewards, reps_a, target_reps_a, target_nreps_a
            )
            mico_stats.update({f"{k}_a": v for k, v in mico_stats_a.items()})
        else:
            mico_loss_a = jnp.zeros_like(mico_loss)

        aux_phase_loss_actor = (
            args.bc_coef * kl_policy_bc
            + args.markov_coef_a_aux_phase * dyna_loss_a
            + args.mico_coef_actor_aux_phase * mico_loss_a
        )

        aux_phase_loss_critic = (
            vloss
            + args.mico_coef_critic_aux_phase * mico_loss
            + args.markov_coef_c_aux_phase * dyna_loss_c
            + args.adv_coef_aux_phase * adv_loss
            + args.aux_vf_coef_aux_phase * aux_vloss
        )
        aux_phase_loss = aux_phase_loss_actor + aux_phase_loss_critic

        new_target_p = jax.tree_map(update_target_params, target_params, params)

        stats = {
            "losses/auxiliary_phase_loss": aux_phase_loss,
            "losses/kl_policy_bc": kl_policy_bc,
            "losses/auxiliary_phase_critic_value_loss": vloss,
            "losses/auxiliary_phase_critic_mico_loss": mico_loss,
            "losses/auxiliary_phase_actor_mico_loss": mico_loss_a,
            "losses/auxiliary_markov_loss_a": dyna_loss_a,
            "losses/auxiliary_markov_loss_c": dyna_loss_c,
        }
        stats.update({f"losses/auxiliary_phase_{k}": v for k, v in mico_stats.items()})

        # Compute representation stats
        key, subkey = jax.random.split(key)

        # Critic
        stats.update(
            repmetric_util.compute_nn_latent_stats(
                subkey, reps_c, nreps_c, dones, label="losses/z_critic_aux_phase"
            )
        )
        stats.update(
            repmetric_util.compute_nn_latent_out_stats(
                reps_c,
                value_preds,
                metric_fn_out=absolute_diff,
                label="losses/z_value_l1_critic_aux_phase",
            )
        )

        # Actor
        stats.update(
            repmetric_util.compute_nn_latent_stats(
                subkey, reps_a, nreps_a, dones, label="losses/z_actor_aux_phase"
            )
        )

        stats.update(
            repmetric_util.compute_nn_latent_out_stats(
                reps_a,
                new_mu,
                metric_fn_out=bisim_util.tv_cont_lazy,
                label="losses/z_logits_tv_actor_aux_phase",
            )
        )
        
        return aux_phase_loss, (stats, new_target_p, key)

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdones, nextterm, nextvalues, curvalues, reward = inp

        delta = reward + gamma * nextvalues * (1.0 - nextterm) - curvalues
        advantages = delta + gamma * gae_lambda * (1.0 - nextdones) * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda
    )

    @jax.jit
    def compute_gae(
        agent_state: AgentTrainState,
        last_obs: np.ndarray,
        last_done: np.ndarray,
        last_term: np.ndarray,
        storage: PolPhaseStorage,
    ):
        last_value = get_value(agent_state.get_params().critic_params, last_obs)
        advantages = jnp.zeros_like(last_value)
        dones = jnp.concatenate([storage.dones, last_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, last_value[None, :]], axis=0)
        terms = jnp.concatenate([storage.terminations, last_term[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], terms[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        advantages = jnp.where(dones[:-1], 0, advantages)
        target_values = jnp.where(storage.terminations, 0, advantages + storage.values)

        return advantages, target_values

    def mico_reward_diff_loss_fn(
        rewards, reps, target_reps, target_nreps, remove_same_rollout_pairs=False
    ):
        distance_fn = cosine_distance
        diff_fn = absolute_diff
        online_dist, norm_average, base_distances = bisim_util.representation_distances(
            reps,
            target_reps,
            distance_fn,
            remove_same_rollout_pairs=remove_same_rollout_pairs,
            return_distance_components=True,
        )
        target_dist, target_norm_average, target_base_distances, reward_diffs = (
            bisim_util.target_distances(
                target_nreps,
                rewards,
                diff_fn,
                distance_fn,
                args.gamma,
                remove_same_rollout_pairs=remove_same_rollout_pairs,
                return_distance_components=True,
            )
        )
        mico_loss = jnp.mean(jax.vmap(huber_loss)(online_dist, target_dist))
        stats = {
            "mico_norm_average": norm_average.mean(),
            "mico_base_distances": base_distances.mean(),
            "mico_reward_diff": reward_diffs.mean(),
            "mico_norm_average_delta": jnp.abs(
                norm_average - target_norm_average
            ).mean(),
            "mico_base_distances_delta": jnp.abs(
                base_distances - target_base_distances
            ).mean(),
            "mico_online_dist": online_dist.mean(),
            "mico_target_dist": target_dist.mean(),
        }
        return mico_loss, stats

    def policy_phase_loss_fn(
        params,
        target_params,
        obs,
        next_obs,
        rewards,
        actions,
        behavior_logprobs,
        stored_value_preds,
        advantages,
        target_values,
        key,
    ):
        newlogprob, entropy, reps_a = get_logprob_entropy_rep(
            params.actor_params, obs, actions
        )
        logratio = newlogprob - behavior_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss1 = -jnp.expand_dims(advantages, 1) * ratio
        pg_loss2 = -jnp.expand_dims(advantages, 1) * jnp.clip(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()

        # Value loss
        value_preds, reps_c = get_value_and_rep(params.critic_params, obs)
        v_loss, (unclipped_v_loss, clipped_v_loss) = ppo_value_loss_fn(
            value_preds, target_values, stored_value_preds
        )
        # print(f'v_loss: {v_loss.shape}')

        # Thus begin the auxiliary losses
        if args.markov_coef_a_pol_phase > 0.0:
            nreps_a = get_representation(params.actor_params, next_obs)
            dyna_loss_a, key = ddcpg_dynamic_pred_loss_fn(
                params.actor_params.dyna_head_params, reps_a, actions, nreps_a, key
            )
        else:
            dyna_loss_a = jnp.zeros_like(pg_loss)

        if args.markov_coef_c_pol_phase > 0.0:
            nreps_c = get_representation(params.critic_params, next_obs)
            dyna_loss_c, key = ddcpg_dynamic_pred_loss_fn(
                params.critic_params.dyna_head_params, reps_c, actions, nreps_c, key
            )
        else:
            dyna_loss_c = jnp.zeros_like(v_loss)

        if args.mico_coef_critic_pol_phase > 0.0:
            target_reps_c = get_representation(target_params.critic_params, obs)
            target_nreps_c = get_representation(target_params.critic_params, next_obs)
            mico_loss, mico_stats = mico_reward_diff_loss_fn(
                rewards, reps_c, target_reps_c, target_nreps_c
            )
        else:
            mico_loss, mico_stats = jnp.zeros_like(v_loss), {}

        if args.mico_coef_actor_pol_phase > 0.0:
            target_reps_a = get_representation(target_params.actor_params, obs)
            target_nreps_a = get_representation(target_params.actor_params, next_obs)
            mico_loss_a, mico_stats_a = mico_reward_diff_loss_fn(
                rewards, reps_a, target_reps_a, target_nreps_a
            )
            mico_stats.update({f"{k}_a": v for k, v in mico_stats_a.items()})
        else:
            mico_loss_a = jnp.zeros_like(mico_loss)

        loss = (
            pg_loss
            - args.ent_coef * entropy_loss
            + v_loss * args.vf_coef
            + args.markov_coef_a_pol_phase * dyna_loss_a
            + args.markov_coef_c_pol_phase * dyna_loss_c
            + args.mico_coef_actor_pol_phase * mico_loss_a
            + args.mico_coef_critic_pol_phase * mico_loss
        )

        # PPG reference code does this regardless of aux losses in the policy phase
        new_target_p = jax.tree_map(update_target_params, target_params, params)

        stats = {
            "losses/policy_phase_loss": loss,
            "losses/policy_loss": pg_loss,
            "losses/entropy": entropy_loss,
            "losses/approx_kl": approx_kl,
            "losses/mico_loss_a": mico_loss_a,
            "losses/mico_loss_c": mico_loss,
            "losses/markov_loss_a": dyna_loss_a,
            "losses/markov_loss_c": dyna_loss_c,
        }

        return loss, (stats, new_target_p, key)

    @jax.jit
    def run_policy_phase_single_device(
        agent_state: AgentTrainState,
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

        next_obs = jnp.concatenate([storage.obs[1:], last_obs[None, :]], axis=0)

        policy_phase_loss_grad_fn = jax.value_and_grad(
            policy_phase_loss_fn, has_aux=True
        )
        advantages_raw, target_values = compute_gae(
            agent_state, last_obs, last_done, last_term, storage
        )

        if args.norm_adv == "batch":
            advantages = (advantages_raw - advantages_raw.mean()) / (
                advantages_raw.std() + 1e-8
            )
        elif args.norm_adv == "minibatch":
            advantages = advantages_raw.reshape(
                advantages_raw.shape[0], args.num_minibatches, -1
            )
            advantages = (advantages - advantages.mean((0, -1), keepdims=True)) / (
                advantages.std((0, -1), keepdims=True) + 1e-8
            )
            advantages = advantages.reshape(advantages.shape[0], -1)
        else:
            advantages = advantages_raw

        def shuffle_storage(storage, advantages, target_values, next_obs, key):
            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key, x)
                x = jnp.reshape(
                    x,
                    (args.num_minibatches * args.gradient_accumulation_steps, -1)
                    + x.shape[1:],
                )
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            flatten_advantages = flatten(advantages)
            flatten_target_values = flatten(target_values)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)
            shuffled_advantages = convert_data(flatten_advantages)
            shuffled_target_values = convert_data(flatten_target_values)

            flatten_next_obs = flatten(next_obs)
            next_obs_mask = jnp.broadcast_to(
                flatten_storage.dones.reshape(-1, 1, 1, 1), flatten_next_obs.shape
            )
            flatten_next_obs = jnp.where(next_obs_mask, 0, flatten_next_obs)

            shuffled_next_obs = convert_data(flatten_next_obs)

            return (
                shuffled_storage,
                shuffled_advantages,
                shuffled_target_values,
                shuffled_next_obs,
            )

        def policy_phase_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)
            (
                shuffled_storage,
                shuffled_advantages,
                shuffled_target_values,
                shuffled_next_obs,
            ) = shuffle_storage(storage, advantages, target_values, next_obs, subkey)

            def policy_phase_minibatch(agent_state, minibatch):
                agent_state, key = agent_state
                (
                    mb_obs,
                    mb_actions,
                    mb_rewards,
                    mb_next_obs,
                    mb_behavior_logprobs,
                    mb_value_preds,
                    mb_advantages,
                    mb_target_values,
                ) = minibatch
                (loss, (stats_pol_phase, target_params, key)), grads = (
                    policy_phase_loss_grad_fn(
                        agent_state.get_params(),
                        agent_state.get_target_params(),
                        mb_obs,
                        mb_next_obs,
                        mb_rewards,
                        mb_actions,
                        mb_behavior_logprobs,
                        mb_value_preds,
                        mb_advantages,
                        mb_target_values,
                        key,
                    )
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                agent_state = agent_state.replace(
                    actor_target_params=target_params.actor_params,
                    critic_target_params=target_params.critic_params,
                )

            agent_state, stats_pol_phase = jax.lax.scan(
                policy_phase_minibatch,
                (agent_state, key),
                (
                    shuffled_storage.obs,
                    shuffled_storage.actions,
                    shuffled_storage.rewards,
                    shuffled_next_obs,
                    shuffled_storage.logprobs,
                    shuffled_storage.values,
                    shuffled_advantages,
                    shuffled_target_values,
                ),
            )
            agent_state, key = agent_state
            return (agent_state, key), stats_pol_phase

        (agent_state, key), stats_pol_phase = jax.lax.scan(
            policy_phase_epoch, (agent_state, key), (), length=args.policy_phase_epochs
        )

        stats_pol_phase = jax.tree_map(
            lambda x: jax.lax.pmean(x, axis_name="local_devices").mean(),
            stats_pol_phase,
        )
        return (
            agent_state,
            stats_pol_phase,
            jnp.hsplit(target_values, num_sharded_storages),
            jnp.hsplit(advantages_raw, num_sharded_storages),
            key,
        )
        # return agent_state, stats_pol_phase, jnp.hsplit(target_values, num_sharded_storages), jnp.hsplit(advantages_raw, num_sharded_storages), key

    def compute_bc_target_logits_once(obs, actor_params):
        return jax.lax.stop_gradient(get_logits(actor_params, obs))

    @jax.jit
    def compute_bc_target_logits(actor_params, storage):
        # prevents excessive memory usage.
        # e.g., [10, 10, 54, 54, 9] -> [2, 50, 54, 54, 9]
        obs = storage.obs.reshape((-1, args.aux_minibatch_size) + storage.obs.shape[2:])
        # TODO (minor): compare two options to improve performance
        # logits = jnp.array([compute_bc_target_logits_once(ob, actor_params) for ob in obs]) #for loop, long compile time but faster execution
        compute_logit_fn = partial(
            compute_bc_target_logits_once, actor_params=actor_params
        )
        # [2, 50, 8]
        mu, sigma = jax.lax.map(compute_logit_fn, obs)
        return mu.reshape(storage.obs.shape[:2] + (mu.shape[-1],)), sigma.reshape(
            storage.obs.shape[:2] + (sigma.shape[-1],)
        )

    @jax.jit
    def compute_bc_target_single_pass(actor_params, storage):
        obs = storage.obs.reshape((-1,) + storage.obs.shape[2:])
        logits = jax.lax.stop_gradient(get_logits(actor_params, obs))
        return logits.reshape(storage.obs.shape[:2] + (logits.shape[-1],))

    @jax.jit
    def run_auxiliary_phase_single_device(
        agent_state: AgentTrainState,
        sharded_storages: List,
        sharded_last_obs: List,
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        old_pi_mu, old_pi_sigma = compute_bc_target_logits(
            agent_state.get_params().actor_params, storage
        )
        last_obs = jnp.concatenate(sharded_last_obs)
        # old_pi_logits = compute_bc_target_single_pass(agent_state.get_params().actor_params, storage) #faster, but memory inefficient
        # aux_loss_grad_fn = jax.value_and_grad(auxiliary_phase_loss_fn, has_aux=True)
        aux_loss_grad_fn = jax.value_and_grad(aux_loss_fn, has_aux=True)

        if args.norm_adv == "batch":
            storage = storage._replace(
                target_advantages=(
                    storage.target_advantages - storage.target_advantages.mean()
                )
                / (storage.target_advantages.std() + 1e-8)
            )
        elif args.norm_adv == "minibatch":
            raise NotImplementedError(f"This is probably the wrong choice to make.")

        def shuffle_rollouts(storage, last_obs, mu, sigma, key):
            def shuffle(x):
                return jax.random.permutation(key, x, axis=1)

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                # make rollout axis the leading axis
                # before; (50, 320, 64, 64, 9)
                # after swap: (320, 50, 64, 64, 9)                                                           │global_step=24544000, avg_episodic_return=-279.6080017089844, rollout_time=23711.068359375
                # after reshape: (80, 200, 64, 64, 9)
                x = jnp.swapaxes(x, 0, 1)
                x = jnp.reshape(
                    x,
                    (args.num_aux_minibatches * args.gradient_accumulation_steps, -1)
                    + x.shape[2:],
                )
                return x

            shuffled_storage = jax.tree_map(shuffle, storage)
            shuffled_last_obs = shuffle(last_obs.reshape(1, *last_obs.shape))
            shuffled_mu = shuffle(mu)
            shuffled_sigma = shuffle(sigma)
            shuffled_storage = jax.tree_map(convert_data, shuffled_storage)
            shuffled_last_obs = convert_data(shuffled_last_obs)
            shuffled_mu = convert_data(shuffled_mu)
            shuffled_sigma = convert_data(shuffled_sigma)

            return shuffled_storage, shuffled_last_obs, shuffled_mu, shuffled_sigma

        def aux_phase_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)
            shuffled_storage, shuffled_last_obs, shuffled_mu, shuffled_sigma = (
                shuffle_rollouts(storage, last_obs, old_pi_mu, old_pi_sigma, subkey)
            )

            def aux_phase_minibatch(carry, minibatch):
                agent_state, key = carry
                (
                    mb_obs,
                    mb_last_obs,
                    mb_dones,
                    mb_rewards,
                    mb_target_values,
                    mb_mu,
                    mb_sigma,
                    mb_actions,
                    mb_target_adv,
                ) = minibatch

                def obs_to_next_obs(obs, dones, last_obs):
                    obs = obs.reshape((last_obs.shape[0], -1) + obs.shape[1:])
                    next_obs = jnp.concatenate(
                        [obs[:, 1:], last_obs[:, None, :]], axis=1
                    )
                    next_obs = next_obs.reshape((-1,) + next_obs.shape[2:])
                    next_obs_mask = jnp.broadcast_to(
                        dones.reshape((*dones.shape, 1, 1, 1)), next_obs.shape
                    )
                    next_obs = jnp.where(next_obs_mask, 0, next_obs)
                    return next_obs

                mb_next_obs = obs_to_next_obs(mb_obs, mb_dones, mb_last_obs)

                (aux_phase_loss, (stats_aux_phase, target_params, key)), grads = (
                    aux_loss_grad_fn(
                        agent_state.get_params(),
                        agent_state.get_target_params(),
                        mb_obs,
                        mb_next_obs,
                        mb_dones,
                        mb_rewards,
                        mb_target_values,
                        mb_mu,
                        mb_sigma,
                        mb_actions,
                        mb_target_adv,
                        key,
                    )
                )

                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)

                # grads = jax.tree_map(lambda x: x * 0.0, grads)
                agent_state = agent_state.replace(
                    actor_target_params=target_params.actor_params,
                    critic_target_params=target_params.critic_params,
                )
                return (agent_state, key), stats_aux_phase

            (agent_state, key), stats_aux_phase = jax.lax.scan(
                aux_phase_minibatch,
                (agent_state, key),
                (
                    shuffled_storage.obs,
                    shuffled_last_obs,
                    shuffled_storage.dones,
                    shuffled_storage.rewards,
                    shuffled_storage.target_values,
                    shuffled_mu,
                    shuffled_sigma,
                    shuffled_storage.actions,
                    shuffled_storage.target_advantages,
                ),
            )

            return (agent_state, key), stats_aux_phase

        (agent_state, key), stats_aux_phase = jax.lax.scan(
            aux_phase_epoch, (agent_state, key), (), length=args.auxiliary_phase_epochs
        )

        stats_aux_phase = jax.tree_map(
            lambda x: jax.lax.pmean(x, axis_name="local_devices").mean(),
            stats_aux_phase,
        )

        # _, key = jax.random.split(key, 2)
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

    if not runstate.metadata["training_completed"] and not args.measure_mi:
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
                        envs,
                        eval_envs,
                        envs_step_fn,
                        eval_envs_step_fn,
                        # device_dist,
                        runstate.metadata["learner_policy_version"],
                    ),
                    daemon=False,
                ).start()

        print(f"Started the thread but moving on...")
        rollout_queue_get_time = jnp.zeros((1,))
        learner_policy_version = runstate.metadata["learner_policy_version"]
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
                    # print(f'Waiting for data from thread...')
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
            
            rollout_queue_get_time += time.time() - rollout_queue_get_time_start
            training_time_start = time.time()

            (
                agent_state,
                stats_pol_phase,
                target_values,
                target_advantages,
                learner_keys,
            ) = run_policy_phase_multi_device(
                agent_state,
                sharded_storages_pol_phase,
                sharded_last_obss_pol_phase,
                sharded_last_dones_pol_phase,
                sharded_last_terms_pol_phase,
                learner_keys,
            )
            sharded_storages_aux_phase.extend(
                [
                    AuxPhaseStorage(
                        obs=sharded_storages_pol_phase[i].obs,
                        rewards=sharded_storages_pol_phase[i].rewards,
                        actions=sharded_storages_pol_phase[i].actions,
                        target_values=target_values[i],
                        target_advantages=target_advantages[i],
                        dones=sharded_storages_pol_phase[i].dones,
                    )
                    for i in range(len(sharded_storages_pol_phase))
                ]
            )
            sharded_last_obss_aux_phase.extend(sharded_last_obss_pol_phase)
            if learner_policy_version % args.num_policy_phases == 0:
                if args.save_model:
                    aps = flax.jax_utils.unreplicate(
                        agent_state
                    )  # .get_params().actor_params
                    params = {
                        "src": {
                            "actor": aps.get_params().actor_params,
                            "critic": aps.get_params().critic_params,
                        },
                        "tgt": {
                            "actor": aps.get_target_params().actor_params,
                            "critic": aps.get_target_params().critic_params,
                        },
                    }

                    job_util.save_params(path=checkpoint_path, params=params)

                (agent_state, stats_aux_phase, learner_keys) = (
                    run_auxiliary_phase_multi_device(
                        agent_state,
                        list(sharded_storages_aux_phase),
                        list(sharded_last_obss_aux_phase),
                        learner_keys,
                    )
                )
                gc.collect()
                
            unreplicated_params = flax.jax_utils.unreplicate(agent_state.get_params())
            for d_idx, d_id in enumerate(args.actor_device_ids):
                device_params = jax.device_put(unreplicated_params, local_devices[d_id])
                for thread_id in range(args.num_actor_threads):
                    params_queues[d_idx * args.num_actor_threads + thread_id].put(
                        device_params
                    )

            # record rewards for plotting purposes
            writer.add_scalar(
                "charts/target_param_error",
                get_error_target_params(
                    agent_state.get_target_params(), agent_state.get_params()
                ),
                global_step,
            )
            # less frequent logging
            if learner_policy_version % args.log_frequency == 0:
                writer.add_scalar(
                    "stats/rollout_queue_get_time",
                    np.mean(rollout_queue_get_time),
                    global_step,
                )
                writer.add_scalar(
                    "stats/rollout_params_queue_get_time_diff",
                    np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                    global_step,
                )
                writer.add_scalar(
                    "stats/training_time",
                    time.time() - training_time_start,
                    global_step,
                )
                writer.add_scalar(
                    "stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step
                )
                writer.add_scalar(
                    "stats/params_queue_size", params_queues[-1].qsize(), global_step
                )
                print(
                    global_step,
                    f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
                )
                writer.add_scalar(
                    "charts/learning_rate",
                    agent_state.actor_state.opt_state[2][1]
                    .hyperparams["learning_rate"][-1]
                    .item(),
                    global_step,
                )
                writer.add_scalar(
                    "charts/avg_value_target",
                    jnp.concatenate(target_values).mean().item(),
                    global_step,
                )

                for key, value in stats_pol_phase.items():
                    writer.add_scalar(key, value[-1].item(), global_step)

                if learner_policy_version % args.num_policy_phases == 0:
                    for key, value in stats_aux_phase.items():
                        writer.add_scalar(key, value[-1].item(), global_step)

            if learner_policy_version >= args.num_updates:
                break
            # make sure we don't get extra updates due to checkpointing in middle of policy phase
            if learner_policy_version % args.num_policy_phases == 0:
                if 0 < args.checkpoint_frequency < (time.time() - last_checkpoint_time):
                    print(
                        f"Saving periodic checkpoint at iteration {learner_policy_version}"
                    )
                    runstate.save_state(agent_state, args)
                    last_checkpoint_time = time.time()

    if args.save_model:
        if args.distributed:
            jax.distributed.shutdown()

        aps = flax.jax_utils.unreplicate(agent_state)  # .get_params().actor_params
        
        params = {
            "src": {
                "actor": aps.get_params().actor_params,
                "critic": aps.get_params().critic_params,
            },
            "tgt": {
                "actor": aps.get_target_params().actor_params,
                "critic": aps.get_target_params().critic_params,
            },
        }

        job_util.save_params(path=checkpoint_path, params=params)
        

    if args.measure_mi:
        from mutual_info_brax import (
            evaluate_mi,
            compute_mi_levelset,
            compute_mi_markov,
            compute_mi_vf,
            compute_rep_stats,
        )

        eval_fns_test = [
            partial(compute_mi_markov, n_samples=args.mi_eval_downsample_to_n),
            partial(
                compute_mi_vf, n_samples=args.mi_eval_downsample_to_n, gamma=args.gamma
            ),
            partial(
                compute_rep_stats,
                model_modules=model_modules,
                num_steps=args.num_steps,
                num_rollouts=args.mi_eval_downsample_to_n // args.num_steps,
            ),
        ]

        eval_fns_train = [
            partial(compute_mi_levelset, n_samples=args.mi_eval_downsample_to_n),
            partial(compute_mi_markov, n_samples=args.mi_eval_downsample_to_n),
            partial(
                compute_mi_vf, n_samples=args.mi_eval_downsample_to_n, gamma=args.gamma
            ),
            partial(
                compute_rep_stats,
                model_modules=model_modules,
                num_steps=args.num_steps,
                num_rollouts=args.mi_eval_downsample_to_n // args.num_steps,
            ),
        ]

        mi_stats_test = evaluate_mi(
            envs,
            eval_envs,
            flax.jax_utils.unreplicate(agent_state),
            model_modules,
            eval_fns_test,
            seed=args.seed,
            total_timesteps=args.mi_eval_total_timesteps,
            num_envs=args.num_envs,
        )
        pprint(mi_stats_test)
        mi_stats_train = evaluate_mi(
            envs,
            eval_envs,
            flax.jax_utils.unreplicate(agent_state),
            model_modules,
            eval_fns_train,
            seed=args.seed,
            total_timesteps=args.mi_eval_total_timesteps,
            num_envs=args.num_envs,
            compute_mi_levels=True,
        )
        pprint(mi_stats_train)

        for k, v in mi_stats_test.items():
            if k.startswith("walltime"):
                writer.add_scalar(f"stats/{k}", v, args.total_timesteps + 2)
            else:
                writer.add_scalar(f"rep_eval_test/{k}", v, args.total_timesteps + 2)
        for k, v in mi_stats_train.items():
            if k.startswith("walltime"):
                writer.add_scalar(f"stats/{k}", v, args.total_timesteps + 2)
            else:
                writer.add_scalar(f"rep_eval_train/{k}", v, args.total_timesteps + 2)
        # runstate.after_post_eval(agent_state, args)
        time.sleep(4)

    # envs.close()
    writer.close()
