import jax
import jax.numpy as jnp
import numpy as np

from utils.constants import EPSILON
from . import l2, absolute_diff, pairwise_similarities, cosine_similarity

# metrics reproduced from https://arxiv.org/abs/2203.15955

def complexity_reduction(Lrep, Lmax):
  """Computes the reduction in complexity of the value function due to the representation function.

  Args:
    Lrep: The mean lipschitz constant.
    Lmax: The max lipschitz constant.

  Returns:
    The reduction in complexity of the value function due to the representation function.
  """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = 1 - Lrep / (Lmax + EPSILON)
  return c

# @functools.partial(jax.jit, static_argnames=('metric_fn_reps','metric_fn_out',))
def lipschitz_complexity(reps, out, metric_fn_reps=l2, metric_fn_out=absolute_diff):
  """Computes the mean and max lipschitz constants between reps and outs=f(reps)
  projected under metric spaces defined by metric_fn_reps and metric_fn_out.

  Args:
    reps: The representations of the states.
    out: f(reps) for some function f.
    metric_fn_reps: The metric function for the representations.
    metric_fn_out: The metric function for the output.

  Returns:
    The reduction in complexity of the value function due to the representation function.
  """
  batch_size, representation_dim = reps.shape
  dist_reps = pairwise_similarities(reps, reps, metric_fn_reps).squeeze()
  dist_out = pairwise_similarities(out, out, metric_fn_out).squeeze()
  lipschitz = (dist_out / (dist_reps+EPSILON))
  Lrep = batch_size / (batch_size - 1) * jnp.mean(lipschitz)
  Lmax = batch_size / (batch_size - 1) * jnp.max(lipschitz)
  return Lrep, Lmax

# @functools.partial(jax.jit, static_argnames=('metric_fn',))
def dynamics_awareness(reps, next_reps, dones, key, metric_fn=l2):
  """Compute the dynamics awareness of the representation function.

  Args:
    reps: The representations of the states.
    key: The random key for shuffling the representations
    metric_fn: The metric function for the representations.

  Returns:
    The dynamics awareness of the representation function.
  """
  rand_reps = jax.random.permutation(key, reps)
  dist_nreps = jax.vmap(metric_fn, in_axes=(0, 0))(reps, next_reps)
  dist_randreps = jax.vmap(metric_fn, in_axes=(0, 0))(reps, rand_reps)
  dist_nreps = jnp.where(dones, 0.0, dist_nreps)
  dist_randreps = jnp.where(dones, 0.0, dist_randreps)
  dist_nreps = dist_nreps.sum() / (len(dones) - dones.sum())
  dist_randreps = dist_randreps.sum() / (len(dones) - dones.sum())
  return dist_nreps, dist_randreps

# @functools.partial(jax.jit, static_argnames=('metric_fn_reps','metric_fn_out',))
def diversity(reps, out, metric_fn_reps=l2, metric_fn_out=absolute_diff):
  """Compute the diversity of the representation function.

  Args:
    reps: The representations of the states.
    out: f(reps) for some function f.
    metric_fn_reps: The metric function for the representations.
    metric_fn_out: The metric function for the output.

  Returns:
    The diversity of the representation function.
  """
  batch_size, representation_dim = reps.shape
  dist_reps = pairwise_similarities(reps, reps, metric_fn_reps).squeeze()
  dist_out = pairwise_similarities(out, out, metric_fn_out).squeeze()
  dist_reps = dist_reps / (jnp.max(dist_reps) + EPSILON)
  dist_out = dist_out / (jnp.max(dist_out) + EPSILON)
  clipped_ratios = jnp.clip(dist_out / (dist_reps + EPSILON), a_max=1.0)
  clipped_ratios = jnp.reshape(clipped_ratios, (batch_size, batch_size))
  clipped_ratios = clipped_ratios - jnp.diag(jnp.diag(clipped_ratios))
  return 1 - batch_size / (batch_size - 1) * jnp.mean(clipped_ratios)

# @functools.partial(jax.jit, static_argnames=('metric_fn',))
def orthogonality(reps, metric_fn=cosine_similarity):
  """Compute the orthogonality of the representation function.

  Args:
    reps: The representations of the states.
    metric_fn: The metric function for the representations.

  Returns:
    The orthogonality of the representation function.
  """
  batch_size, representation_dim = reps.shape
  dist_reps = pairwise_similarities(reps, reps, metric_fn).squeeze()
  dist_reps = jnp.reshape(dist_reps, (batch_size, batch_size))
  dist_reps = dist_reps - jnp.diag(jnp.diag(dist_reps))
  return 1 - batch_size / (batch_size - 1) * jnp.mean(dist_reps)

# @functools.partial(jax.jit, static_argnames=('label',))
def compute_nn_latent_stats(key, reps, next_reps, dones, label):
  stats = {}
  reps, next_reps = jax.lax.stop_gradient((reps, next_reps))
  dist_nreps, dist_randreps = dynamics_awareness(reps, next_reps, dones, key, metric_fn=l2)
  stats[f'{label}:dist_nreps'] = dist_nreps
  stats[f'{label}:dist_randreps'] = dist_randreps
  stats[f'{label}:dynamics_awareness'] = (dist_randreps - dist_nreps) / (dist_randreps + EPSILON)
  stats[f'{label}:orthogonality'] = orthogonality(reps, metric_fn=cosine_similarity)
  stats[f'{label}:dormant:0.0'] = jax.vmap(lambda x: (jnp.abs(x) <= 0.0).mean())(reps).mean()
  l2norm = jax.vmap(lambda x: jnp.sqrt(jnp.sum(x ** 2)))(reps)
  stats[f'{label}:l2_norm_mean'] = l2norm.mean()
  stats[f'{label}:l2_norm_std'] = l2norm.std()
  stats[f'{label}:l2_norm_max'] = l2norm.max()
  stats[f'{label}:l2_norm_min'] = l2norm.min()
  return stats

# @functools.partial(jax.jit, static_argnames=('metric_fn_out','label',))
def compute_nn_latent_out_stats(reps, out, metric_fn_out, label):
  stats = {}
  reps, out = jax.lax.stop_gradient((reps, out))
  Lrep, Lmax = lipschitz_complexity(reps, out, metric_fn_reps=l2, metric_fn_out=metric_fn_out)
  stats[f'{label}:Lrep'] = Lrep
  stats[f'{label}:Lmax'] = Lmax
  stats[f'{label}:diversity'] = diversity(reps, out, metric_fn_reps=l2, metric_fn_out=metric_fn_out)
  return stats
