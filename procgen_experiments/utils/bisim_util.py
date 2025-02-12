# coding=utf-8

"""Utilities for computing the MICo and KSMe losses.
Based on https://github.com/google-research/google-research/blob/master/mico/atari/metric_utils.py"""

import jax
import jax.numpy as jnp

from . import absolute_diff, squarify, total_variation, pairwise_similarities

#Operator used for discrete action spaces in https://arxiv.org/abs/2101.05265

#Operator used for continuous action spaces in https://arxiv.org/abs/2101.05265
def action_diff(a1, a2):
  return jnp.abs(a1 - a2)

def action_inequality(a1, a2):
  return jnp.where(a1 != a2, 1, 0)

def reward_kernel(r1, r2):
  return 1 - absolute_diff(r1, r2).squeeze() #if we normalize using 1/(Rmax-Rmin) then the 0.5 factor is not needed

def total_variation_kernel(logits1, logits2):
  return 1 - total_variation(logits1, logits2).squeeze()

def action_inequality_kernel(a1, a2):
  return 1 - action_inequality(a1, a2).squeeze()

def jnp_extract(condition_arr, arr, axis=0, size=None, fill_value=0):
  if condition_arr.ndim != 1:
    raise ValueError("condition must be a 1D array")
  arr = jnp.moveaxis(arr, axis, 0)
  condition_arr = condition_arr.astype(bool)
  condition_arr, extra = condition_arr[:arr.shape[0]], condition_arr[arr.shape[0]:]
  arr = arr[:condition_arr.shape[0]]
  mask = jnp.expand_dims(condition_arr, range(1, arr.ndim))
  arr = jnp.where(mask, arr, jnp.array(fill_value, dtype=arr.dtype))
  indices = jnp.argsort(condition_arr, axis=0)
  arr = arr[jax.lax.rev(indices, dimensions=[0])][:size]
  result = jax.lax.rev(arr, dimensions=[0])
  return jnp.moveaxis(result, 0, axis)

def representation_distances(first_representations, second_representations,
                             distance_fn, beta=0.1,
                             return_distance_components=False,
                             remove_same_rollout_pairs=False):
  """Compute distances between representations.

  This will compute the distances between two representations.

  Args:
    first_representations: first set of representations to use.
    second_representations: second set of representations to use.
    distance_fn: function to use for computing representation distances.
    beta: float, weight given to cosine distance between representations.
    return_distance_components: bool, whether to return the components used for
      computing the distance.
    remove_same_rollout_pairs: bool, whether to only compute distance between
      pairs taken from different rollouts. Input representations must be
      parsed with first two dimensions corresponding to (rollout_id, rollout_timestep)

  Returns:
    The distances between representations, combining the average of the norm of
    the representations and the distance given by distance_fn.
  """

  representation_dim = first_representations.shape[-1]
  if remove_same_rollout_pairs:
    rollout_dim, timestep_dim = first_representations.shape[0], first_representations.shape[1]
    batch_size = rollout_dim * timestep_dim
    first_representations = first_representations.reshape(batch_size, representation_dim)
    second_representations = second_representations.reshape(batch_size, representation_dim)
  else:
    batch_size = first_representations.shape[0]
  first_squared_reps = squarify(first_representations)
  second_squared_reps = squarify(second_representations)
  second_squared_reps = jnp.transpose(second_squared_reps, axes=[1, 0, 2])
  first_squared_reps = jnp.reshape(first_squared_reps,
                                   [batch_size**2, representation_dim])
  second_squared_reps = jnp.reshape(second_squared_reps,
                                    [batch_size**2, representation_dim])
  if remove_same_rollout_pairs:
    mask = jax.scipy.linalg.block_diag(*[jnp.ones((timestep_dim, timestep_dim)) for _ in range(rollout_dim)]).astype(bool)
    mask = ~mask
    newsize = (rollout_dim * timestep_dim) * (rollout_dim * timestep_dim - timestep_dim)
    first_squared_reps = jnp_extract(jnp.ravel(mask), first_squared_reps, size=newsize)
    second_squared_reps = jnp_extract(jnp.ravel(mask), second_squared_reps, size=newsize)
  base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(first_squared_reps,
                                                         second_squared_reps)
  norm_average = 0.5 * (jnp.sum(jnp.square(first_squared_reps), -1) +
                        jnp.sum(jnp.square(second_squared_reps), -1))
  if return_distance_components:
    return norm_average + beta * base_distances, \
           jax.lax.stop_gradient(norm_average), jax.lax.stop_gradient(base_distances)
  return norm_average + beta * base_distances

def target_distances(nrepresentations, data, diff_fn, distance_fn, cumulative_gamma, remove_same_rollout_pairs=False,
                     return_distance_components=False):
  """Target distance using the metric operator. Generalised from reward differences to any data and diff_fn.
  diff_fn should always output a shape of (C**2,1) or (C**2,) where B is the leading dim (batch size) of input data.
  """

  next_state_distances, norm_average, base_distances = representation_distances(
      nrepresentations, nrepresentations, distance_fn, remove_same_rollout_pairs=remove_same_rollout_pairs,
      return_distance_components=True)
  if remove_same_rollout_pairs:
    rollout_dim, timestep_dim = data.shape[0], data.shape[1]
    batch_size = rollout_dim * timestep_dim
    data = data.reshape(batch_size, -1)
  squared_data = squarify(data)
  squared_data_transp = jnp.swapaxes(squared_data, 0, 1)
  squared_data = squared_data.reshape((squared_data.shape[0]**2), -1)
  squared_data_transp = squared_data_transp.reshape(
      (squared_data_transp.shape[0]**2), -1)
  if remove_same_rollout_pairs:
    mask = jax.scipy.linalg.block_diag(*[jnp.ones((timestep_dim, timestep_dim)) for _ in range(rollout_dim)]).astype(
      bool)
    mask = ~mask
    newsize = (rollout_dim * timestep_dim) * (rollout_dim * timestep_dim - timestep_dim)
    squared_data = jnp_extract(jnp.ravel(mask), squared_data, size=newsize)
    squared_data_transp = jnp_extract(jnp.ravel(mask), squared_data_transp, size=newsize)
  diffs = diff_fn(squared_data, squared_data_transp).squeeze()
  if return_distance_components:
    return jax.lax.stop_gradient((diffs + cumulative_gamma * next_state_distances,
                                 norm_average, base_distances, diffs))
  return jax.lax.stop_gradient(diffs + cumulative_gamma * next_state_distances)

def target_similarities(nrepresentations, data, kernel_fn, similarity_fn, cumulative_gamma):
  """Target similarity using the metric operator. Generalised from reward differences to any data and diff_fn.
  diff_fn should always output a shape of (B**2,1) or (B**2,) where B is the leading dim (batch size) of input data.
  """

  next_state_similarities = pairwise_similarities(
      nrepresentations, nrepresentations, similarity_fn)
  squared_data = squarify(data)
  squared_data_transp = jnp.swapaxes(squared_data, 0, 1)
  squared_data = squared_data.reshape((squared_data.shape[0]**2), -1)
  squared_data_transp = squared_data_transp.reshape(
      (squared_data_transp.shape[0]**2), -1)
  kernel = kernel_fn(squared_data, squared_data_transp).squeeze()
  return (
      jax.lax.stop_gradient(
          kernel + cumulative_gamma * next_state_similarities))