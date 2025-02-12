import functools
import jax
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import custom_jvp, numpy as jnp

from .constants import EPSILON

class DotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  __deepcopy__ = None

  def __init__(self, dct):
    for key, value in dct.items():
      if hasattr(value, 'keys'):
        value = DotDict(value)
      self[key] = value

  def __getstate__(self):
    return self

  def __setstate__(self, state):
    self.update(state)
    self.__dict__ = self

  def to_dict(self):
    d = {}
    for key, value in self.items():
      if key == '__dict__':
        continue
      if isinstance(value, DotDict):
        d[key] = value.to_dict()
      else:
        d[key] = value

    return d

@functools.partial(jax.jit, static_argnums=(1,2))
def mod_key_dict(d, prefix='', suffix=''):
  def mod_key(k, v):
    return f"{prefix}{str(k[0].key)}{suffix}", v
  return dict(jax.tree_util.tree_map_with_path(mod_key, d).values())

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

def l2sq(x, y):
  return jnp.sum(jnp.square(x - y))

def l2(x, y):
  return _sqrt(jnp.sum(jnp.square(x - y)))

def l2_normalized(x):
  return x / (jnp.sqrt(jnp.sum(x ** 2)) + EPSILON)
  # return x / (_sqrt(jnp.sum(jnp.square(x))) + EPSILON) # sqrt is used in l2(x,y) above, but that function is never used in code. It shouldn't be necessary

def cosine_similarity(x, y):
  numerator = jnp.sum(x * y)
  denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
  return numerator / (denominator + EPSILON)

def dot(x, y):
  return jnp.dot(x, y)

def absolute_diff(r1, r2):
  return jnp.abs(r1 - r2)

def cosine_distance(x, y):
  cos_similarity = cosine_similarity(x, y)
  return jnp.arctan2(_sqrt(1. - cos_similarity**2), cos_similarity)

def huber_loss(targets: jnp.array,
               predictions: jnp.array,
               delta: float = 1.0) -> jnp.ndarray:
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
  return jnp.where(x <= delta,
                   0.5 * x**2,
                   0.5 * delta**2 + delta * (x - delta))

def kl_categorical_categorical(logits_p, logits_q):
  logits_p = logits_p - logsumexp(logits_p, axis=-1, keepdims=True)
  logits_q = logits_q - logsumexp(logits_q, axis=-1, keepdims=True)
  probs_p = jax.nn.softmax(logits_p)
  probs_q = jax.nn.softmax(logits_q)
  kld = probs_p * (logits_p - logits_q)
  kld = jnp.where(probs_q == 0, jnp.inf, kld)
  kld = jnp.where(probs_p == 0, 0, kld)
  return kld.sum(-1)

def jsd_categorical_categorical(logits_p, logits_q):
  logits_p = logits_p - logsumexp(logits_p, axis=-1, keepdims=True)
  logits_q = logits_q - logsumexp(logits_q, axis=-1, keepdims=True)
  probs_p = jax.nn.softmax(logits_p)
  probs_q = jax.nn.softmax(logits_q)
  probs_m = 0.5 * (probs_p + probs_q)
  kld_pm = probs_p * (jnp.log(probs_p) - jnp.log(probs_m))
  kld_pm = jnp.where(probs_p == 0, 0, kld_pm)
  kld_pm = jnp.where(probs_m == 0, jnp.inf, kld_pm)
  kld_qm = probs_q * (jnp.log(probs_q) - jnp.log(probs_m))
  kld_qm = jnp.where(probs_q == 0, 0, kld_qm)
  kld_qm = jnp.where(probs_m == 0, jnp.inf, kld_qm)
  return 0.5 * (kld_pm.sum(-1) + kld_qm.sum(-1))

def binary_cross_entropy_with_logits(logits, labels):
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p

def squarify(x):
  batch_size = x.shape[0]
  if len(x.shape) > 1:
    representation_dim = x.shape[-1]
    return jnp.reshape(jnp.tile(x, batch_size),
                       (batch_size, batch_size, representation_dim))
  return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))

def total_variation(logits1, logits2):
  return jnp.max(jnp.abs(jax.nn.softmax(logits1) - jax.nn.softmax(logits2)), axis=-1)

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
