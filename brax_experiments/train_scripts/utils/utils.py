import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal, variance_scaling, lecun_normal, normal, zeros
from jax import custom_jvp
from functools import partial
from flax import linen as nn
import functools


EPSILON = 1e-9


def random_crop(key, img, padding):
  crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
  crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
  padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                       mode='edge')
  return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=12):
  keys = jax.random.split(key, imgs.shape[0])
  return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)

class RandomProjector(nn.Module):
  out_dim: int

  @nn.compact
  def __call__(self, x):
    return jax.lax.stop_gradient(nn.Dense(self.out_dim, kernel_init=normal(), bias_init=zeros)(x))

@functools.partial(jax.jit, static_argnames=('out_dim',))
def random_projection(x, key, out_dim: int):
  rand_projector = RandomProjector(out_dim)
  rand_projector_params = rand_projector.init(key, x)
  return rand_projector.apply(rand_projector_params, x)

def layer_init_normed_make_fn(scale, init_fn):

  def norm_rescale(key, shape, dtype):
    weight = init_fn(key, shape, dtype)
    l2_norm_fn = lambda x: l2_normalized(x)
    weight = jax.vmap(l2_norm_fn, in_axes=-1, out_axes=-1)(weight)
    return jax.lax.stop_gradient(weight * scale)

  return lambda key, shape, dtype: norm_rescale(key, shape, dtype)


def l2_normalized(x):
  return x / (jnp.sqrt(jnp.sum(x ** 2)) + EPSILON)


def layer_init_normed_make_fn(scale):

  def norm_rescale(key, shape, dtype):
    init_fn = variance_scaling(2.0, "fan_in", "uniform", in_axis=-2,
                            out_axis=-1, batch_axis=(), dtype=dtype)
    weight = init_fn(key, shape, dtype)
    l2_norm_fn = lambda x: l2_normalized(x)
    weight = jax.vmap(l2_norm_fn, in_axes=-1, out_axes=-1)(weight)
    return jax.lax.stop_gradient(weight * scale)

  return lambda key, shape, dtype: norm_rescale(key, shape, dtype)


def set_layer_init_fn(args):
  # ref: https://github.com/vwxyzjn/cleanrl/blob/65789babaae033433078504b4ff0b925d5e27b99/cleanrl/ppg_procgen.py
  if args.kernel_init_method == "ppg_cleanrl_procgen":
    init_fn_partial = partial(variance_scaling(2.0, "fan_in", "uniform", in_axis=-2,
                                               out_axis=-1, batch_axis=()))
    return {
      "convsequence_conv": layer_init_normed_make_fn(1.0),
      "convsequence_resblock": layer_init_normed_make_fn(1.0/(6**0.25)),
      "resnet_dense": layer_init_normed_make_fn(2**0.25),
      "value_head_dense": layer_init_normed_make_fn(0.1**0.5),
      "policy_head_dense": layer_init_normed_make_fn(0.1**0.5),
      "auxiliary_head_dense": layer_init_normed_make_fn(0.1**0.5),
      "auxiliary_advantage_head_dense": layer_init_normed_make_fn(0.1 ** 0.5, init_fn_partial),
      "dyna_head_actor_dense": layer_init_normed_make_fn(2**0.5, init_fn_partial),
      "dyna_head_critic_dense": layer_init_normed_make_fn(2**0.5, init_fn_partial)
    }
  # ref: https://github.com/vwxyzjn/cleanba/blob/9ff59b5b7ed17664e86217558c89b5132306474f/cleanba/cleanba_ppo.py
  elif args.kernel_init_method == "ppo_cleanba":
    return {
      # "convsequence_conv": lambda key, shape, dtype: lecun_normal()(key, shape, dtype),
      "convsequence_conv": lecun_normal(),
      "convsequence_resblock": lecun_normal(),
      "resnet_dense": orthogonal(2**0.5),
      "value_head_dense": orthogonal(1.0),
      "policy_head_dense": orthogonal(0.01),
      "auxiliary_head_dense": orthogonal(1.0),
      "auxiliary_advantage_head_dense": orthogonal(1.0),
      "dyna_head_actor_dense": orthogonal(2**0.5),
      "dyna_head_critic_dense": orthogonal(2**0.5)
    }


def binary_cross_entropy_with_logits(logits, labels):
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


# The following two functions were borrowed from
# https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py
# as they resolve the instabilities observed when using `jnp.arccos`.
@partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.)


def cosine_similarity(x, y):
  numerator = jnp.sum(x * y)
  denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
  return numerator / (denominator + EPSILON)


def cosine_distance(x, y):
  cos_similarity = cosine_similarity(x, y)
  return jnp.arctan2(_sqrt(1. - cos_similarity**2), cos_similarity)


def absolute_diff(r1, r2):
  return jnp.abs(r1 - r2)
