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
  # Computes the 95% confidence interval for the mean, using a sum of 3 Gaussians model.
  # For more details, see
  #
  # F. Ribeiro, D. Florencio, C. Zhang and M. Seltzer, "crowdMOS: An Approach for
  # Crowdsourcing Mean Opinion Score Studies", submitted to ICASSP 2011.
  v_mu = mos_variance(Z)
  t = matlab_tinv(.5 * (1 + .95), min(Z.shape) - 1)
  ci95 = t * sqrt(v_mu)
  return ci95


def mos_variance(Z: np.ndarray) -> float:
  # Determines the variance of the mean opinion score, given a matrix of scores.
  # Unknown scores are represented by NaN.
  #
  # Z: a N-by-M matrix of scores, where the rows are subjects and columns are sentences
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

  v_su = vertical_var(Z.T)  # v_su  = v_s + v_u
  v_wu = vertical_var(Z)  # v_wu  = v_w + v_u
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

# def get_v_mu_from_v_su()


def vertical_var(Z: np.ndarray):
  v = []
  for i in range(Z.shape[1]):
    if nancount(Z[:i]) >= 2:
      x = np.nanvar(Z[:i])
      v.append(x)
  result = np.mean(v)
  return result


def nancount(vec: np.ndarray):
  x = ~np.isnan(vec)
  count_not_nan = np.sum(x)
  return count_not_nan
