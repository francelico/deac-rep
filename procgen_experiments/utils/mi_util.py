from sklearn.preprocessing import scale
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.special import digamma
import numpy as np

def compression_efficiency(mi_in, mi_out):
  with np.errstate(divide='ignore', invalid='ignore'):
    comp = mi_out / mi_in
    comp = np.clip(comp, 0, 1)
  comp[mi_in == 0] = np.nan
  return comp

def learned_invariance(mi_in, mi_out):
  with np.errstate(divide='ignore', invalid='ignore'):
    inv = (mi_in - mi_out) / mi_in
    inv = np.clip(inv, 0, 1)
  inv[mi_in == 0] = np.nan
  return inv

def compute_total_correlation(X, n_neighbors=3, perturb=True, metric="euclidean"):
  """
  :param X: a vector of shape (n_samples, n_dim)
  :param n_neighbors:
  :param perturb:
  :return:
  """
  assert X.ndim == 2
  n_samples = X.shape[0]
  n_dim = X.shape[1]

  if perturb:
    means = np.maximum(1, np.mean(np.abs(X), axis=0))
    X += (
        1e-10
        * means
        * np.random.randn(*X.shape)
    )

  # Here we rely on NearestNeighbors to select the fastest algorithm.
  nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)

  nn.fit(X)
  radius = nn.kneighbors()[0]
  radius = np.nextafter(radius[:, -1], 0)

  def count_nns(xs):
    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    xs = xs.reshape(-1, 1)
    kd = KDTree(xs, metric=metric)
    nx = kd.query_radius(xs, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0
    return nx

  nns = [count_nns(X[:, i]) for i in range(X.shape[-1])]
  digamma_s = np.sum([np.mean(digamma(n)) for n in nns])

  mi = (
      digamma(n_samples) * (n_dim - 1)
      + digamma(n_neighbors)
      - digamma_s
  )

  return max(0, mi)

def compute_mi_cc(x, y, n_neighbors=3, metric="euclidean"):
  # First estimator from https://arxiv.org/pdf/cond-mat/0305641
  # Note: this estimator has ~2x smaller variance than using an estimator based on total correlation.
  # See test code below
  # def compute_mi_cc_from_total_correlation(x,y, n_neighbors):
  #   # Add small noise to continuous features as advised in Kraskov et. al.
  #   means = np.maximum(1, np.mean(np.abs(x), axis=0))
  #   x += (
  #       1e-10
  #       * means
  #       * np.random.randn(*x.shape)
  #   )
  #   means = np.maximum(1, np.mean(np.abs(y), axis=0))
  #   y += (
  #       1e-10
  #       * means
  #       * np.random.randn(*y.shape)
  #   )
  #   xy = np.hstack((x, y))
  #   Ix_tc = compute_total_correlation(x, n_neighbors, perturb=False)
  #   Iy_tc = compute_total_correlation(y, n_neighbors, perturb=False)
  #   Ixy_tc = compute_total_correlation(xy, n_neighbors, perturb=False)
  #   return max(0, Ixy_tc - Ix_tc - Iy_tc)
  #
  # X = np.random.randn(50, 500, 8)
  # Y = 2*np.random.randn(50, 500, 8)
  # MI, MI_tc = [], []
  # for x, y in zip(X,Y):
  #   mi = compute_mi_cc(x, y, n_neighbors)
  #   mi_tc = compute_mi_cc_from_total_correlation(x,y, n_neighbors)
  #   MI.append(mi)
  #   MI_tc.append(mi_tc)
  # MI = np.array(MI)
  # MI_tc = np.array(MI_tc)
  # print(MI.mean(), MI.std())
  # print(MI_tc.mean(), MI_tc.std())

  if not isinstance(x, np.ndarray):
    x = np.array(x)
  if not isinstance(y, np.ndarray):
    y = np.array(y)

  n_samples, _ = x.shape

  # make data have unit variance
  x = scale(x, with_mean=False, copy=True)
  y = scale(y, with_mean=False, copy=True)

  # Add small noise to continuous features as advised in Kraskov et. al.
  means = np.maximum(1, np.mean(np.abs(x), axis=0))
  x += (
      1e-10
      * means
      * np.random.randn(*x.shape)
  )
  means = np.maximum(1, np.mean(np.abs(y), axis=0))
  y += (
      1e-10
      * means
      * np.random.randn(*y.shape)
  )
  xy = np.hstack((x, y))

  # Here we rely on NearestNeighbors to select the fastest algorithm.
  nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)

  nn.fit(xy)
  radius = nn.kneighbors()[0]
  radius = np.nextafter(radius[:, -1], 0)

  # KDTree is explicitly fit to allow for the querying of number of
  # neighbors within a specified radius
  kd = KDTree(x, metric=metric)
  nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
  nx = np.array(nx) - 1.0

  kd = KDTree(y, metric=metric)
  ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
  ny = np.array(ny) - 1.0

  mi = (
      digamma(n_samples)
      + digamma(n_neighbors)
      - np.mean(digamma(nx + 1))
      - np.mean(digamma(ny + 1))
  )

  return max(0, mi)

def compute_mi_cd(c, d, n_neighbors=3, metric="euclidean"):
  """Compute mutual information between continuous and discrete variables.

  Parameters
  ----------
  c : ndarray, shape (n_samples,)
      Samples of a continuous random variable.

  d : ndarray, shape (n_samples,)
      Samples of a discrete random variable.

  n_neighbors : int
      Number of nearest neighbors to search for each point, see [1]_.

  Returns
  -------
  mi : float
      Estimated mutual information. If it turned out to be negative it is
      replace by 0.

  Notes
  -----
  True mutual information can't be negative. If its estimate by a numerical
  method is negative, it means (providing the method is adequate) that the
  mutual information is close to 0 and replacing it by 0 is a reasonable
  strategy.

  References
  ----------
  .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
     Data Sets". PLoS ONE 9(2), 2014.
  """

  if not isinstance(c, np.ndarray):
    c = np.array(c)
  if not isinstance(d, np.ndarray):
    d = np.array(d)

  n_samples = c.shape[0]
  # make data have unit variance
  c = scale(c, with_mean=False, copy=True)

  # Add small noise to continuous features as advised in Kraskov et. al.
  means = np.maximum(1, np.mean(np.abs(c), axis=0))
  c += (
      1e-10
      * means
      * np.random.randn(*c.shape)
  )

  radius = np.empty(n_samples)
  label_counts = np.empty(n_samples)
  k_all = np.empty(n_samples)
  nn = NearestNeighbors(metric=metric)
  for label in np.unique(d):
    mask = d == label
    count = np.sum(mask)
    if count > 1:
      k = min(n_neighbors, count - 1)
      nn.set_params(n_neighbors=k)
      nn.fit(c[mask])
      r = nn.kneighbors()[0]
      radius[mask] = np.nextafter(r[:, -1], 0)
      k_all[mask] = k
    label_counts[mask] = count

  # Ignore points with unique labels.
  mask = label_counts > 1
  n_samples = np.sum(mask)
  label_counts = label_counts[mask]
  k_all = k_all[mask]
  c = c[mask]
  radius = radius[mask]

  kd = KDTree(c, metric=metric)
  m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
  m_all = np.array(m_all)

  mi = (
      digamma(n_samples)
      + np.mean(digamma(k_all))
      - np.mean(digamma(label_counts))
      - np.mean(digamma(m_all))
  )

  return max(0, mi)