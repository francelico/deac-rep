import jax
import functools
from functools import partial
from flax.linen.initializers import orthogonal, variance_scaling, lecun_normal, normal, constant
import flax.linen as nn

from . import l2_normalized

class RandomProjector(nn.Module):
  out_dim: int

  @nn.compact
  def __call__(self, x):
    return jax.lax.stop_gradient(nn.Dense(self.out_dim, kernel_init=normal(), bias_init=constant(0.0))(x))

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

def set_layer_init_fn(args):
  # ref: https://github.com/vwxyzjn/cleanrl/blob/65789babaae033433078504b4ff0b925d5e27b99/cleanrl/ppg_procgen.py
  if args.kernel_init_method == "ppg_cleanrl_procgen":
    init_fn_partial = partial(variance_scaling(2.0, "fan_in", "uniform", in_axis=-2,
                     out_axis=-1, batch_axis=()))

    return {
      "convsequence_conv": layer_init_normed_make_fn(1.0, init_fn_partial),
      "convsequence_resblock": layer_init_normed_make_fn(1.0/(6**0.25), init_fn_partial),
      "resnet_dense": layer_init_normed_make_fn(2**0.25, init_fn_partial),
      "value_head_dense": layer_init_normed_make_fn(0.1**0.5, init_fn_partial),
      "policy_head_dense": layer_init_normed_make_fn(0.1**0.5, init_fn_partial),
      "auxiliary_head_dense": layer_init_normed_make_fn(0.1**0.5, init_fn_partial),
      "auxiliary_advantage_head_dense": layer_init_normed_make_fn(0.1**0.5, init_fn_partial),
      "dyna_head_actor_dense": layer_init_normed_make_fn(2**0.5, init_fn_partial),
      "dyna_head_critic_dense": layer_init_normed_make_fn(2**0.5, init_fn_partial)
    }
  # ref: https://github.com/vwxyzjn/cleanba/blob/9ff59b5b7ed17664e86217558c89b5132306474f/train_scripts/_ppo.py
  elif args.kernel_init_method == "ppo_cleanba":
    return {
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
  elif args.kernel_init_method == "ppg_orthogonal":
    init_fn_partial = partial(variance_scaling(2.0, "fan_in", "uniform", in_axis=-2,
                     out_axis=-1, batch_axis=()))
    return {
      "convsequence_conv": layer_init_normed_make_fn(1.0, init_fn_partial),
      "convsequence_resblock": layer_init_normed_make_fn(1.0/(6**0.25), init_fn_partial),
      "resnet_dense": layer_init_normed_make_fn(2**0.25, init_fn_partial),
      "value_head_dense": layer_init_normed_make_fn(0.1**0.5, orthogonal(1.0)),
      "policy_head_dense": layer_init_normed_make_fn(0.1**0.5, orthogonal(0.01)),
      "auxiliary_head_dense": layer_init_normed_make_fn(0.1**0.5, orthogonal(1.0)),
      "dyna_head_actor_dense": layer_init_normed_make_fn(2**0.5, orthogonal(1.0)),
      "dyna_head_critic_dense": layer_init_normed_make_fn(2**0.5, orthogonal(1.0))
    }