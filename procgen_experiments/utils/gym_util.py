import numpy as np
import flax

# adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
@flax.struct.dataclass
class RunningMeanStd:
  mean: np.ndarray
  var: np.ndarray
  count: float
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

  @staticmethod
  def update_mean_var_count_from_moments(
      mean, var, count, batch_mean, batch_var, batch_count
  ):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

  @classmethod
  def create(cls, x, **kwargs):
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    mean, var, count = cls.update_mean_var_count_from_moments(
        kwargs["mean"], kwargs["var"], kwargs["count"], batch_mean, batch_var, x.shape[0]
    )
    return cls(mean, var, count)

@flax.struct.dataclass
class RewardWrapper:
  normalize: bool
  clip: bool
  clip_coef: float
  discounted_returns: np.ndarray
  return_rms: RunningMeanStd
  gamma: float

  @classmethod
  def create(cls, rewards, next_terminated, next_truncated, **kwargs):
    discounted_returns = kwargs["discounted_returns"] * kwargs["gamma"] + rewards
    kwargs["return_rms"] = kwargs["return_rms"].create(discounted_returns, **kwargs["return_rms"].__dict__)
    kwargs["discounted_returns"] = np.where(next_terminated + next_truncated, 0, discounted_returns)
    return cls(**kwargs)

  def process_rewards(self, rewards, next_terminated, next_truncated):
    newcls = self.create(rewards, next_terminated, next_truncated, **self.__dict__)
    if self.normalize:
      rewards = self.normalize_rewards(rewards, newcls.return_rms)
    if self.clip:
      rewards = self.clip_rewards(rewards, self.clip_coef)
    return newcls, rewards

  @staticmethod
  def normalize_rewards(rewards, return_rms, epsilon=1e-8):
    return rewards / np.sqrt(return_rms.var + epsilon)

  @staticmethod
  def clip_rewards(rewards, clip_coef):
    return np.clip(rewards, -clip_coef, clip_coef)
