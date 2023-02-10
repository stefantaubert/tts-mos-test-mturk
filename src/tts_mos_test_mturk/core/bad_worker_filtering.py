import math
from collections import OrderedDict
from logging import getLogger
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import get_algorithm_mos_correlations
from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_ci95, compute_mos
from tts_mos_test_mturk.calculation.etc import (get_workers_count, get_workers_percent,
                                                get_workers_percent_mask, mask_outliers)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.stats import print_stats_masks


def ignore_bad_workers(data: EvaluationData, mask_names: OrderedSet[str], threshold: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  logger.info("--- Ignoring bad workers ---")
  opinion_scores = data.get_os()
  opinion_scores_mask = factory.merge_masks_into_omask(masks)
  opinion_scores_mask.apply_by_nan(opinion_scores)

  worker_correlations = get_algorithm_mos_correlations(opinion_scores)
  bad_worker_np_mask = worker_correlations < threshold
  bad_worker_mask = factory.convert_ndarray_to_wmask(bad_worker_np_mask)
  data.add_or_update_mask(mask_name, bad_worker_mask)

  print_stats_masks(data, masks, [bad_worker_mask])


def ignore_bad_workers_percent(data: EvaluationData, mask_names: OrderedSet[str], from_percent_incl: float, to_percent_excl: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  logger.info("--- Ignoring bad workers ---")
  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  wcorrelations = get_algorithm_mos_correlations(os)

  windices = np.array(range(data.n_workers))
  wmask = factory.merge_masks_into_wmask(masks)
  sub_wcorrelations = wmask.apply_by_del(wcorrelations)
  sub_windices = wmask.apply_by_del(windices)

  # n_workers2 = np.sum(~np.isnan(worker_correlations))
  # n_workers = worker_mask.n_unmasked
  sub_n_workers = len(sub_windices)

  sub_sorted_indices = np.argsort(sub_wcorrelations)
  sub_sorted_indices = np.array(list(sub_sorted_indices))
  # correlations_sorted = sub_worker_correlations[sub_sorted_indices]
  sub_windices_sorted = sub_windices[sub_sorted_indices]
  # workers_sorted = data.workers[sub_worker_indices_sorted]

  res_wmask = factory.get_wmask()
  # TODO check is inclusive and exclusive
  from_position = math.ceil(sub_n_workers * from_percent_incl)
  to_position = math.ceil(sub_n_workers * to_percent_excl)
  sub_sel_windices = sub_windices_sorted[from_position:to_position]

  # workers_sorted_2 = data.workers[sub_sel_worker_indices]
  res_wmask.mask_indices(sub_sel_windices)
  res_wmask.combine_mask(wmask)

  data.add_or_update_mask(mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])


def ignore_too_fast_assignments(data: EvaluationData, mask_names: OrderedSet[str], threshold: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  logger.info("--- Ignoring fast assignments ---")
  worktimes = data.get_worktimes()
  worktimes_mask = factory.merge_masks_into_amask(masks)
  worktimes_mask.apply_by_nan(worktimes)

  too_fast_worktimes_np_mask = worktimes < threshold
  too_fast_worktimes_mask = factory.convert_ndarray_to_amask(too_fast_worktimes_np_mask)
  data.add_or_update_mask(mask_name, too_fast_worktimes_mask)

  print_stats_masks(data, masks, [too_fast_worktimes_mask])


def ignore_outlier_opinion_scores(data: EvaluationData, mask_names: OrderedSet[str], max_std_dev_diff: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  logger.info("--- Ignoring outliers ---")
  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  outlier_np_mask = mask_outliers(os, max_std_dev_diff)
  outlier_omask = factory.convert_ndarray_to_omask(outlier_np_mask)
  data.add_or_update_mask(mask_name, outlier_omask)

  print_stats_masks(data, masks, [outlier_omask])


def ignore_masked_count_opinion_scores(data: EvaluationData, mask_names: OrderedSet[str], ref_mask_name: str, percent: float, mask_name: str):
  logger = getLogger(__name__)
  factory = data.get_mask_factory()
  masks = [data.masks[mask_name] for mask_name in mask_names]
  ref_mask = data.masks[ref_mask_name]
  ref_omask = factory.convert_mask_to_omask(ref_mask)

  logger.info("--- Ignoring count masked ---")
  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  outlier_workers_count = get_workers_count(ref_omask.mask)
  outlier_workers_percent = get_workers_percent(ref_omask.mask)
  for w_i, worker in enumerate(data.workers):
    logger.info(
      f"Worker {worker} has {outlier_workers_percent[w_i]*100:.2f}% of outlying scores (#{outlier_workers_count[w_i]})")

  outlier_workers_np_mask = get_workers_percent_mask(ref_mask.mask, percent)
  outlier_wmask = factory.convert_ndarray_to_wmask(outlier_workers_np_mask)
  data.add_or_update_mask(mask_name, outlier_wmask)
  print_stats_masks(data, masks, [outlier_wmask])


def calc_mos(data: EvaluationData, mask_names: OrderedSet[str]) -> None:
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  logger.info("--- Ignoring count masked ---")
  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  scores: List[Dict] = []
  for algo_i, alg_name in enumerate(data.algorithms):
    row = OrderedDict((
      ("Algorithm", alg_name),
      ("MOS", compute_mos(os[algo_i])),
      ("CI95", compute_ci95(os[algo_i])),
    ))
    scores.append(row)

  res = pd.DataFrame(
    data=[x.values() for x in scores],
    columns=scores[0].keys(),
  )

  print(res)


def generate_approve_csv(data: EvaluationData, mask_names: OrderedSet[str], reason: Optional[str]) -> Optional[pd.DataFrame]:
  logger = getLogger(__name__)

  results: List[Dict[str, Any]] = []
  if reason is None:
    reason = "x"

  masks = [data.masks[mask_name] for mask_name in mask_names]
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("Approve", reason),
      ("Reject", ""),
    ))
    results.append(line)

  if len(results) == 0:
    return None
  result = pd.DataFrame(
    data=[x.values() for x in results],
    columns=results[0].keys(),
  )
  print(result)
  return result
