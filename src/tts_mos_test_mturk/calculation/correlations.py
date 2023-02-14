from typing import Literal

import numpy as np

from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_mos


def get_sentence_mos_correlation(worker: int, Z: np.ndarray) -> float:
  assert len(Z.shape) == 2
  n_workers = Z.shape[0]
  n_sentences = Z.shape[1]

  scores = np.empty((2, n_sentences))
  others = [w_i for w_i in range(n_workers) if w_i != worker]

  for sentence_i in range(n_sentences):
    scores[0, sentence_i] = Z[worker, sentence_i]
    scores[1, sentence_i] = compute_mos(Z[others, sentence_i])

  return get_corrcoef(scores)


def get_sentence_mos_correlation_3dim(worker: int, Zs: np.ndarray) -> float:
  assert len(Zs.shape) == 3
  Z = np.concatenate(Zs, axis=1)
  return get_sentence_mos_correlation(worker, Z)


def get_sentence_mos_correlations_3dim(opinion_scores: np.ndarray) -> np.ndarray:
  n_workers = opinion_scores.shape[1]
  Z = np.concatenate(opinion_scores, axis=1)

  correlations = np.empty(n_workers)
  for worker_i in range(n_workers):
    correlations[worker_i] = get_sentence_mos_correlation(worker_i, Z)

  return correlations


def get_algorithm_mos_correlation(worker: int, opinion_scores: np.ndarray) -> float:
  assert len(opinion_scores.shape) == 3
  n_alg = opinion_scores.shape[0]
  n_workers = opinion_scores.shape[1]
  # n_sentences = Zs.shape[2]

  scores = np.empty((2, n_alg))
  others = [w_i for w_i in range(n_workers) if w_i != worker]

  for alg_i in range(n_alg):
    scores[0, alg_i] = compute_mos(opinion_scores[alg_i, worker, :])
    scores[1, alg_i] = compute_mos(opinion_scores[alg_i, others, :])

  return get_corrcoef(scores)


def get_worker_mos_correlations(opinion_scores: np.ndarray) -> np.ndarray:
  n_workers = opinion_scores.shape[1]

  correlations = np.empty((2, n_workers))
  correlations[0, :] = get_algorithm_mos_correlations(opinion_scores)
  correlations[1, :] = get_sentence_mos_correlations_3dim(opinion_scores)

  # print(correlations)
  worker_correlations = np.nanmean(correlations, axis=0)
  # print(worker_correlations)
  return worker_correlations


def get_mos_correlations(opinion_scores: np.ndarray, mode: Literal["sentence", "algorithm", "both"]) -> np.ndarray:
  if mode == "sentence":
    return get_sentence_mos_correlations_3dim(opinion_scores)
  if mode == "algorithm":
    return get_algorithm_mos_correlations(opinion_scores)
  if mode == "both":
    return get_worker_mos_correlations(opinion_scores)
  raise NotImplementedError()


def get_algorithm_mos_correlations(opinion_scores: np.ndarray) -> np.ndarray:
  n_workers = opinion_scores.shape[1]

  correlations = np.empty(n_workers)
  for worker_i in range(n_workers):
    correlations[worker_i] = get_algorithm_mos_correlation(worker_i, opinion_scores)

  return correlations


def get_corrcoef(v: np.ndarray) -> float:
  assert len(v.shape) == 2
  assert v.shape[0] == 2

  nan_row_mask = np.any(np.isnan(v), axis=0)
  masked_v = v[:, ~nan_row_mask]

  result = np.corrcoef(masked_v)
  result = result[0, 1]

  return result
