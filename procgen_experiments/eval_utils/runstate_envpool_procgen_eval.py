import os
from typing import Callable, Tuple, Any

import cv2
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def evaluate(
    ckpt_path: str,
    make_env: Callable,
    make_agent_fn: Callable,
    args: Any,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    capture_video: bool = True,
    seed=1,
    num_levels: int = 0,
    start_level: int = 0,
):
    envs = make_env(env_id, seed, num_envs=1, num_levels=num_levels, start_level=start_level)()
    key = jax.random.PRNGKey(seed)

    agent_state, model_modules, key = make_agent_fn(args=args, envs=envs, key=key, print_model=False)
    with open(ckpt_path, "rb") as f:
        agent_state, _, _ = \
            flax.serialization.from_bytes(
                (
                    agent_state,
                    None,
                    None,
                ),
                f.read(),
            )
    actor_params = agent_state.get_params().actor_params

    @jax.jit
    def get_action(
        actor_params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = model_modules['actor_base'].apply(actor_params.base_params, next_obs)
        logits = model_modules['policy_head'].apply(actor_params.policy_head_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    # a simple non-vectorized version
    episodic_returns = []
    for episode in range(eval_episodes):
        episodic_return = 0
        next_obs = envs.reset()
        if capture_video:
            recorded_frames = []
            # conversion from grayscale into rgb
            recorded_frames.append(np.moveaxis(next_obs[0], 0, 2))
        for _ in range(envs.spec.config.max_episode_steps):
            actions, key = get_action(actor_params, next_obs, key)
            next_obs, next_reward, next_done, infos = envs.step(np.array(actions))
            episodic_return += next_reward[0]
            done = next_done[0]
            if capture_video and episode == 0:
                recorded_frames.append(np.moveaxis(next_obs[0], 0, 2))
            if done:
                break

        print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
        episodic_returns.append(episodic_return)
        if capture_video and episode == 0:
            clip = ImageSequenceClip(recorded_frames, fps=24)
            os.makedirs(f"videos/{run_name}", exist_ok=True)
            clip.write_videofile(f"videos/{run_name}/{episode}.mp4", logger="bar")

    envs.close()
    return episodic_returns