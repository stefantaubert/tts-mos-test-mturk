# ------------------------
# Calculation was taken from:
# Ribeiro, F., Florêncio, D., Zhang, C., & Seltzer, M. (2011). CrowdMOS: An approach for crowdsourcing mean opinion score studies. 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2416–2419. https://doi.org/10.1109/ICASSP.2011.5946971
# ------------------------

from math import sqrt

import numpy as np
from scipy.stats import t


def matlab_tinv(p: float, df: int) -> float:
  result = -t.isf(p, df)
  return result


def compute_alg_mos(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = compute_mos(ratings[algo_i])
  return result


def compute_alg_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty(n_algorithms, dtype=np.float32)
  for algo_i in range(n_algorithms):
    result[algo_i] = compute_ci95(ratings[algo_i])
  return result


def compute_alg_mos_ci95(ratings: np.ndarray) -> np.ndarray:
  n_algorithms = ratings.shape[0]
  result = np.empty((2, n_algorithms), dtype=np.float32)
  result[0, :] = compute_alg_mos(ratings)
  result[1, :] = compute_alg_ci95(ratings)
  return result


def compute_mos(Z: np.ndarray) -> float:
  mos = np.nanmean(Z)
  return mos


def compute_ci95(Z: np.ndarray) -> float:
  # Computes the 95% confidence interval using the sum of 3 Gaussian models.
  v_mu = mos_variance(Z)
  t = matlab_tinv(.5 * (1 + .95), min(Z.shape) - 1)
  ci95 = t * sqrt(v_mu)
  return ci95


def mos_variance(Z: np.ndarray) -> float:
  # Determines the variance of the mean opinion score, given a matrix of ratings.
  # Unknown ratings are represented by NaN.
  #
  # Z: a N-by-M matrix of ratings, where the rows are subjects and columns are sentences
  # We assume that
  #
  # 	Z_ij = mu + x_i + y_j + eps_ij, where
  #
  # mu is the mean opinion score (given by nanmean(Z(:)))
  # x_i ~ N(0, sigma_w^2), with sigma_w^2 modeling worker variation
  # y_j ~ N(0, sigma_s^2), with sigma_y^2 modeling sentence variation
  # eps_ij ~ N(0, sigma_u^2), with sigma_u^2 modeling worker uncertainty
  #
  # The returned value v_mu is Var[mu].

  # N, M = Z.shape
  W: np.ndarray = ~np.isnan(Z)
  Mi: np.ndarray = np.sum(W, axis=0)
  Mi = Mi.flatten()
  Nj: np.ndarray = np.sum(W, axis=1)
  Nj = Nj.flatten()
  T = np.sum(W.flatten())

  v_su = get_sentence_variance(Z.T)  # v_su  = v_s + v_u
  v_wu = get_sentence_variance(Z)  # v_wu  = v_w + v_u
  v_swu = np.nanvar(Z.flatten())  # v_swu = v_s + v_w + v_u

  v_mu: float
  if ~np.isnan(v_su) and ~np.isnan(v_wu):
    v = np.array([
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
    ])
    b = np.array([
      v_su,
      v_wu,
      v_swu,
    ], dtype=np.float32)
    v = np.linalg.solve(v, b)
    v_s = max(v[0], 0)
    v_w = max(v[1], 0)
    v_u = max(v[2], 0)
    v_mu = v_s * np.sum(Mi**2) / T**2 + v_w * sum(Nj**2) / T**2 + v_u / T
  elif np.isnan(v_su) and ~np.isnan(v_wu):
    v = np.array([
      [0, 1],
      [1, 1],
    ])
    b = np.array([
      v_wu,
      v_swu,
    ])
    v = np.linalg.solve(v, b)
    v_s = max(v[0], 0)
    v_wu = max(v[1], 0)
    v_mu = v_s * np.sum(Mi**2) / T**2 + v_wu / T
  elif ~np.isnan(v_su) and np.isnan(v_wu):
    v = np.array([
      [0, 1],
      [1, 1],
    ])
    b = np.array([
      v_su,
      v_swu,
    ])
    v = np.linalg.solve(v, b)
    v_w = max(v[0], 0)
    v_su = max(v[1], 0)
    v_mu = v_w * np.sum(Nj**2) / T**2 + v_su / T
  else:
    assert np.isnan(v_su) and np.isnan(v_wu)
    v_mu = v_swu / T
  return v_mu


def get_sentence_variance(Z: np.ndarray):
  n_files = Z.shape[1]
  v = []
  for file_i in range(n_files):
    if non_nan_count(Z[:, file_i]) >= 2:
      x = np.nanvar(Z[:, file_i])
      v.append(x)
  result = np.mean(v)
  return result


def non_nan_count(vec: np.ndarray):
  result = np.sum(~np.isnan(vec))
  return result
