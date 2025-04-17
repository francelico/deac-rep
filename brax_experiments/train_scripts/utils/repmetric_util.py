import functools
import jax
import jax.numpy as jnp
from jax import custom_jvp


EPSILON = 1e-9


def squarify(x):
  batch_size = x.shape[0]
  if len(x.shape) > 1:
    representation_dim = x.shape[-1]
    return jnp.reshape(jnp.tile(x, batch_size),
                       (batch_size, batch_size, representation_dim))
  return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


# The following two functions were borrowed from
# https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py
# as they resolve the instabilities observed when using `jnp.arccos`.
@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.)


def l2(x, y):
  return _sqrt(jnp.sum(jnp.square(x - y)))


def absolute_diff(r1, r2):
  return jnp.abs(r1 - r2)


def dot(x, y):
  return jnp.dot(x, y)


def pairwise_similarities(first_representations,
                          second_representations,
                          similarity_fn=dot):
  """Compute similarities between representations.

  This will compute the similarities between two representations.

  Args:
    first_representations: first set of representations to use.
    second_representations: second set of representations to use.
    similarity_fn: function to use for computing representation similarities.

  Returns:
    The similarities between representations, combining the average of the norm of
    the representations and the similarity given by similarity_fn.
  """
  batch_size = first_representations.shape[0]
  # representation_dim = first_representations.shape[-1]
  first_squared_reps = squarify(first_representations)
  first_squared_reps = jnp.reshape(first_squared_reps,
                                   [batch_size**2, -1])
  second_squared_reps = squarify(second_representations)
  second_squared_reps = jnp.swapaxes(second_squared_reps, 0, 1)
  second_squared_reps = jnp.reshape(second_squared_reps,
                                    [batch_size**2, -1])
  similarities = jax.vmap(similarity_fn, in_axes=(0, 0))(
    first_squared_reps, second_squared_reps)
  return similarities

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


def cosine_similarity(x, y):
  numerator = jnp.sum(x * y)
  denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
  return numerator / (denominator + EPSILON)


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
