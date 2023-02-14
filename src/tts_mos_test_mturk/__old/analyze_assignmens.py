import math
from logging import getLogger
from typing import List, Literal, Optional, Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_ci95, compute_mos
from tts_mos_test_mturk.calculation.etc import (get_workers_count, get_workers_percent,
                                                get_workers_percent_mask, mask_algos, mask_outliers)
from tts_mos_test_mturk.globals import LISTENING_TYPES
from tts_mos_test_mturk.types import EvaluationData

# used for finding outliers and computing bonuses


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


def compute_bonuses(Z_all: np.ndarray, workers: OrderedSet[str], all_audio_paths: OrderedSet[str], paths: OrderedSet[str], min_count_ass: int):
  logger = getLogger(__name__)
  alg_indices = list(mask_algos(all_audio_paths, paths))
  # TODO filter before
  fast_workers = set()
  bad_workers = set()

  n_workers = Z_all.shape[0]
  Z_workers = np.array(workers)
  bonuses = np.zeros(n_workers)

  # Ignore low ass count workers
  worker_audio_counts = np.nansum(~np.isnan(Z_all), axis=1)
  worker_assignment_counts = worker_audio_counts / 8
  min_count_mask: np.ndarray = worker_assignment_counts >= min_count_ass
  logger.info(
    f"{np.sum(min_count_mask)} / {Z_all.shape[0]} workers completed at least {min_count_ass} assignments!")

  all_workers = set(Z_workers)
  consider_workers = Z_workers[min_count_mask.nonzero()]

  worker_correlations = get_worker_correlations_old(Z_all, alg_indices)

  Z_all = Z_all[min_count_mask.nonzero()[0]]
  Z_workers = Z_workers[min_count_mask.nonzero()[0]]
  worker_correlations = worker_correlations[min_count_mask.nonzero()[0]]

  sorted_indices = np.argsort(worker_correlations)
  sorted_indices = np.array(list(reversed(sorted_indices)))
  correlations_sorted = worker_correlations[sorted_indices]
  workers_sorted = Z_workers[sorted_indices]
  n_finalists = len(workers_sorted)
  top_10_count = math.ceil(n_finalists * 0.1)
  top_50_count = math.ceil(n_finalists * 0.5)
  top_50_workers = workers_sorted[top_10_count:top_50_count]
  top_50_workers = set(top_50_workers)
  top_10_workers = workers_sorted[:top_10_count]
  top_10_workers = set(top_10_workers)

  remaining_workers = set(consider_workers) - top_50_workers - top_10_workers
  no_bonus_workers = all_workers - remaining_workers - top_50_workers - top_10_workers

  return fast_workers, bad_workers, no_bonus_workers, remaining_workers, top_50_workers, top_10_workers


def get_worker_mos_correlations(opinion_scores: np.ndarray) -> np.ndarray:
  n_workers = opinion_scores.shape[1]

  correlations = np.empty((2, n_workers))
  correlations[0, :] = get_algorithm_mos_correlations(opinion_scores)
  correlations[1, :] = get_sentence_mos_correlations_3dim(opinion_scores)

  # print(correlations)
  worker_correlations = np.nanmean(correlations, axis=0)
  # print(worker_correlations)
  return worker_correlations


def get_worker_correlations_old(Z_all: np.ndarray, alg_indices: List[List[int]]) -> np.ndarray:

  n_alg = len(alg_indices)
  n_workers = Z_all.shape[0]
  n_sentences = int(Z_all.shape[1] / n_alg)

  Zs = np.empty((n_alg, n_workers, n_sentences))
  for algo_i, indices in enumerate(alg_indices):
    Zs[algo_i] = Z_all[:, indices]

  correlations = np.empty((2, n_workers))
  for worker_i in range(n_workers):
    correlations[0, worker_i] = get_algorithm_mos_correlation(worker_i, Zs)
    correlations[1, worker_i] = get_sentence_mos_correlation(worker_i, Z_all)

  # print(correlations)
  worker_correlations = np.nanmean(correlations, axis=0)
  # print(worker_correlations)
  return worker_correlations


def worker_mask_to_assignment_mask(mask: np.ndarray, assignment_worker_matrix: np.ndarray):
  pass

# def get_Z_mask_for_workers(workers: np.ndarray, )


def get_os_count(Z: np.ndarray) -> int:
  result = np.sum(~np.isnan(Z.flatten()))
  return result


def ignore_bad_workers(data: EvaluationData, bad_worker_threshold: float):
  logger = getLogger(__name__)
  rejections = {}

  workers = np.array(data.workers)
  assignments = np.array(data.assignments)

  logger.info("--- Ignoring bad workers ---")
  worker_correlations = get_algorithm_mos_correlations(data.get_os())
  bad_worker_mask: np.ndarray = worker_correlations < bad_worker_threshold
  bad_workers = workers[bad_worker_mask.nonzero()[0]]
  logger.info(f"Ignored {len(bad_workers)} workers!")

  assignment_worker_matrix = data.get_assignment_worker_matrix()
  bad_assignments_mask = np.isin(assignment_worker_matrix, bad_workers)
  bad_assignments = assignments[bad_assignments_mask.nonzero()[0]]
  logger.info(f"Ignored {len(bad_assignments)} assignments!")
  for bad_assignment in bad_assignments:
    assert bad_assignment not in rejections
    rejections[bad_assignment] = "too bad"

  count_old = get_os_count(data.get_os())
  data.apply_ignore_assignments_mask(bad_assignments_mask)
  data.apply_ignore_os_mask(data.get_os_mask_from_assignments(bad_assignments))
  count_new = get_os_count(data.get_os())

  logger.info(
    f"Ignored {count_old - count_new} / {count_old} opinion scores, kept {count_new}!")
  logger.info("---------------------------")
  return rejections


def ignore_too_fast(data: EvaluationData, fast_worker_threshold: float):
  logger = getLogger(__name__)
  rejections = {}

  assignments = np.array(data.assignments)

  logger.info("--- Ignoring too fast assignments ---")
  assignment_work_times = data.get_assignment_work_times()
  # assignment_work_times = np.nan
  fast_assignments_mask = assignment_work_times < fast_worker_threshold
  fast_assignments = assignments[fast_assignments_mask.nonzero()[0]]
  logger.info(f"Ignored {len(fast_assignments)} assignments!")
  for fast_assignment in fast_assignments:
    assert fast_assignment not in rejections
    rejections[fast_assignment] = "too fast"

  count_old = get_os_count(data.get_os())
  data.apply_ignore_assignments_mask(fast_assignments_mask)
  data.apply_ignore_os_mask(data.get_os_mask_from_assignments(fast_assignments))
  count_new = get_os_count(data.get_os())

  logger.info(
    f"Ignored {count_old - count_new} / {count_old} opinion scores, kept {count_new}!")

  logger.info("---------------------------")


def ignore_outliers(data: EvaluationData, max_std_dev_diff: float, remove_workers_with_n_percent_of_diff: Optional[float]):
  logger = getLogger(__name__)
  rejections = {}

  logger.info("--- Ignoring outliers")
  os = data.get_os()
  count_old = get_os_count(data.get_os())
  outlier_mask = mask_outliers(os, max_std_dev_diff)
  data.apply_ignore_os_mask(outlier_mask)
  count_new = get_os_count(data.get_os())

  logger.info(
    f"Ignored {count_old - count_new} / {count_old} outlying opinion scores, kept {count_new}!")

  if remove_workers_with_n_percent_of_diff:

    workers = np.array(data.workers)
    assignments = np.array(data.assignments)

    count_old = count_new
    outlier_workers_count = get_workers_count(outlier_mask)
    outlier_workers_percent = get_workers_percent(outlier_mask)
    for w_i, worker in enumerate(data.workers):
      logger.info(
        f"Worker {worker} has {outlier_workers_percent[w_i]*100:.2f}% of outlying scores (#{outlier_workers_count[w_i]})")

    outlier_workers_mask = get_workers_percent_mask(
      outlier_mask, remove_workers_with_n_percent_of_diff)

    bad_workers = workers[outlier_workers_mask.nonzero()[0]]
    logger.info(f"Ignored {len(bad_workers)} workers!")

    assignment_worker_matrix = data.get_assignment_worker_matrix()
    outlier_assignments_mask = np.isin(assignment_worker_matrix, bad_workers)
    outlier_assignments = assignments[outlier_assignments_mask.nonzero()[0]]
    logger.info(f"Ignored {len(outlier_assignments)} assignments!")
    for bad_assignment in outlier_assignments:
      assert bad_assignment not in rejections
      rejections[bad_assignment] = "too bad"

    data.apply_ignore_assignments_mask(outlier_assignments_mask)
    data.apply_ignore_os_mask(data.get_os_mask_from_assignments(outlier_assignments))
    count_new = get_os_count(data.get_os())

    logger.info(
      f"Ignored {count_old - count_new} / {count_old} outlying opinion scores, kept {count_new}!")

  logger.info("---------------------------")


def analyze_v2(data: EvaluationData, fast_worker_threshold: float, bad_worker_threshold: float, lt: Set[int], bad_worker_threshold_2: float):
  logger = getLogger(__name__)
  rejections = {}

  workers = np.array(data.workers)
  assignments = np.array(data.assignments)

  ignore_bad_workers(data, bad_worker_threshold)
  ignore_too_fast(data, fast_worker_threshold)
  ignore_outliers(data, 1, 0.1)


def analyze(Z_all: np.ndarray, work_times_all: np.ndarray, listening_types_all: np.ndarray, workers: OrderedSet[str], all_audio_paths: OrderedSet[str], paths: OrderedSet[str], fast_worker_threshold: float, bad_worker_threshold: float, lt: Set[int], bad_worker_threshold_2: float):
  logger = getLogger(__name__)
  n_alg = len(paths)
  n_workers = Z_all.shape[0]
  n_sentences = int(Z_all.shape[1] / n_alg)

  alg_indices = list(mask_algos(all_audio_paths, paths))

  ignored_workers = []

  Z_workers = np.array(workers)

  # Ignore bad workers
  worker_correlations = get_worker_correlations_old(Z_all, alg_indices)
  bad_worker_mask: np.ndarray = worker_correlations < bad_worker_threshold
  logger.info(
    f"Ignored {np.sum(bad_worker_mask)} / {Z_all.shape[0]} bad workers, kept {Z_all.shape[0] - np.sum(bad_worker_mask)}!")
  old_count = np.nansum(Z_all.flatten() > 0)
  ignored_workers.extend(Z_workers[bad_worker_mask.nonzero()[0]])
  # ignored_workers.extend(bad_worker_mask.nonzero()[0])
  Z_all = Z_all[(~bad_worker_mask).nonzero()]
  Z_workers = Z_workers[(~bad_worker_mask).nonzero()]
  work_times_all = work_times_all[(~bad_worker_mask).nonzero()]
  listening_types_all = listening_types_all[(~bad_worker_mask).nonzero()]
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(
    f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}! (bad worker)")
  Zs = None
  correlations = None
  worker_correlations = None

  # Ignore fast workers
  fast_worker_mask: np.ndarray = np.nanmin(work_times_all, axis=1) < fast_worker_threshold
  logger.info(
    f"Ignored {np.sum(fast_worker_mask)} / {Z_all.shape[0]} too fast workers, kept {Z_all.shape[0] - np.sum(fast_worker_mask)}!")
  old_count = np.nansum(Z_all.flatten() > 0)
  ignored_workers.extend(Z_workers[fast_worker_mask.nonzero()[0]])
  Z_all = Z_all[(~fast_worker_mask).nonzero()]
  Z_workers = Z_workers[(~fast_worker_mask).nonzero()]
  work_times_all = work_times_all[(~fast_worker_mask).nonzero()]
  listening_types_all = listening_types_all[(~fast_worker_mask).nonzero()]
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(
    f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}! (fast worker")
  Zs = None
  correlations = None
  worker_correlations = None

  # Ignore lt
  allowed_listening_types_mask = np.isnan(listening_types_all)
  for l in lt:
    allowed_listening_types_mask |= (listening_types_all == LISTENING_TYPES[l])
  old_count = np.nansum(Z_all.flatten() > 0)
  Z_all[~allowed_listening_types_mask] = np.nan
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(
    f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}! (listening-type)")

  # Ignore bad workers part 2
  worker_correlations = get_worker_correlations_old(Z_all, alg_indices)
  bad_worker_mask: np.ndarray = worker_correlations < bad_worker_threshold_2
  logger.info(
    f"Ignored {np.sum(bad_worker_mask)} / {Z_all.shape[0]} bad workers, kept {Z_all.shape[0] - np.sum(bad_worker_mask)}!")
  old_count = np.nansum(Z_all.flatten() > 0)
  ignored_workers.extend(Z_workers[bad_worker_mask.nonzero()[0]])
  Z_all = Z_all[(~bad_worker_mask).nonzero()]
  Z_workers = Z_workers[(~bad_worker_mask).nonzero()]
  work_times_all = work_times_all[(~bad_worker_mask).nonzero()]
  listening_types_all = listening_types_all[(~bad_worker_mask).nonzero()]
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(
    f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}! (bad worker 2)")
  Zs = None
  correlations = None
  worker_correlations = None

  # Ignore low ass count workers
  worker_audio_counts = np.nansum(~np.isnan(Z_all), axis=1)
  worker_assignment_counts = worker_audio_counts / 8
  low_count_mask: np.ndarray = worker_assignment_counts < 20
  logger.info(
    f"Ignored {np.sum(low_count_mask)} / {Z_all.shape[0]} too fast workers, kept {Z_all.shape[0] - np.sum(low_count_mask)}!")
  old_count = np.nansum(Z_all.flatten() > 0)
  ignored_workers.extend(Z_workers[low_count_mask.nonzero()[0]])
  Z_all = Z_all[(~low_count_mask).nonzero()]
  Z_workers = Z_workers[(~low_count_mask).nonzero()]
  work_times_all = work_times_all[(~low_count_mask).nonzero()]
  listening_types_all = listening_types_all[(~low_count_mask).nonzero()]
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(
    f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}! (low assignment count")
  Zs = None
  correlations = None
  worker_correlations = None

  # # Ignore too short assignments
  # filter_mask = mask_smaller_than_val(work_times_all, min_worktime_s)
  # logger.info(
  #   f"Ignored {np.sum(filter_mask.flatten())} / {np.nansum(Z_all.flatten() > 0)} too short assignments!")
  # Z_all[filter_mask] = np.nan
  # work_times_all[filter_mask] = np.nan

  # # Ignore too deviate assignments from mean duration
  # filter_mask = mask_lower_outliers(work_times_all, 1.5)
  # logger.info(
  #   f"Ignored {np.sum(filter_mask.flatten())} / {np.nansum(Z_all.flatten() > 0)}  too different assignments based on time!")
  # Z_all[filter_mask] = np.nan

  # # Ignore too slow workers
  # worker_mask = mask_workers(filter_mask, 0.05)
  # logger.info(f"Ignored {np.sum(worker_mask)} / {Z_all.shape[0]} slow workers!")
  # old_count = np.nansum(Z_all.flatten() > 0)
  # Z_all = Z_all[(~worker_mask).nonzero()]
  # new_count = np.nansum(Z_all.flatten() > 0)
  # logger.info(
  #   f"Ignored {old_count - new_count} / {new_count} assignments!")

  scores = np.empty((n_alg, 2))

  for algo_i, indices in enumerate(alg_indices):
    Z = Z_all[:, indices]
    scores[algo_i][0] = compute_mos(Z)
    scores[algo_i][1] = compute_ci95(Z)

    # logger.info(f"MOS for alg{algo_i}: {mos} +- {ci95}")

    # std_ci95 = np.mean(calc_worker_std2(Z) * 1.95996)
    # logger.info(f"MOS for alg{algo_i}: {mos} +- {std_ci95}")
    # logger.info(f"MOS for alg{algo_i}: {mos} +- {std2}")

  res = pd.DataFrame(
    data=scores,
    columns=["MOS", "CI95"],
  )

  return res, ignored_workers
