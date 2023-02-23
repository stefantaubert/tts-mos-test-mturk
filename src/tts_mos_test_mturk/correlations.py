import warnings
from typing import Literal

import numpy as np
from mean_opinion_score import get_mos


def get_sentence_mos_correlation(worker: int, Z: np.ndarray) -> float:
  assert len(Z.shape) == 2
  n_workers = Z.shape[0]
  n_sentences = Z.shape[1]

  mos_ratings = np.empty((2, n_sentences))
  others = [w_i for w_i in range(n_workers) if w_i != worker]

  for sentence_i in range(n_sentences):
    mos_ratings[0, sentence_i] = Z[worker, sentence_i]
    mos_ratings[1, sentence_i] = get_mos(Z[others, sentence_i])

  return get_corrcoef(mos_ratings)


def get_sentence_mos_correlation_3dim(worker: int, ratings: np.ndarray) -> float:
  assert len(ratings.shape) == 3
  Z = np.concatenate(ratings, axis=1)
  return get_sentence_mos_correlation(worker, Z)


def get_sentence_mos_correlations_3dim(ratings: np.ndarray) -> np.ndarray:
  n_workers = ratings.shape[1]
  Z = np.concatenate(ratings, axis=1)

  correlations = np.empty(n_workers)
  for worker_i in range(n_workers):
    correlations[worker_i] = get_sentence_mos_correlation(worker_i, Z)

  return correlations


def get_algorithm_mos_correlation(worker: int, ratings: np.ndarray) -> float:
  assert len(ratings.shape) == 3
  n_alg = ratings.shape[0]
  n_workers = ratings.shape[1]

  mos_ratings = np.empty((2, n_alg))
  others = [w_i for w_i in range(n_workers) if w_i != worker]

  for alg_i in range(n_alg):
    mos_ratings[0, alg_i] = get_mos(ratings[alg_i, worker, :])
    mos_ratings[1, alg_i] = get_mos(ratings[alg_i, others, :])

  return get_corrcoef(mos_ratings)


def get_worker_mos_correlations(ratings: np.ndarray) -> np.ndarray:
  n_workers = ratings.shape[1]

  correlations = np.empty((2, n_workers))
  correlations[0, :] = get_algorithm_mos_correlations(ratings)
  correlations[1, :] = get_sentence_mos_correlations_3dim(ratings)

  # ignore warning "RuntimeWarning: Mean of empty slice" if both correlations are NaN
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    worker_correlations = np.nanmean(correlations, axis=0)
  return worker_correlations


def get_mos_correlations(ratings: np.ndarray, mode: Literal["sentence", "algorithm", "both"]) -> np.ndarray:
  if mode == "sentence":
    return get_sentence_mos_correlations_3dim(ratings)
  if mode == "algorithm":
    return get_algorithm_mos_correlations(ratings)
  if mode == "both":
    return get_worker_mos_correlations(ratings)
  raise NotImplementedError()


def get_algorithm_mos_correlations(ratings: np.ndarray) -> np.ndarray:
  n_workers = ratings.shape[1]

  correlations = np.empty(n_workers)
  for worker_i in range(n_workers):
    correlations[worker_i] = get_algorithm_mos_correlation(worker_i, ratings)

  return correlations


def get_corrcoef(v: np.ndarray) -> float:
  assert len(v.shape) == 2
  assert v.shape[0] == 2

  nan_rows_mask = np.any(np.isnan(v), axis=0)
  masked_v = v[:, ~nan_rows_mask]

  n_values_per_vec = masked_v.shape[1]
  if n_values_per_vec < 2:
    return np.nan

  sd0 = np.std(masked_v[0])
  sd1 = np.std(masked_v[1])

  if sd0 == 0 or sd1 == 0:
    return np.nan

  # ignore warning RuntimeWarnings if both correlations are NaN
  # with warnings.catch_warnings():
  #   warnings.simplefilter("ignore", category=RuntimeWarning)
  result = np.corrcoef(masked_v)
  result = result[0, 1]

  return result
