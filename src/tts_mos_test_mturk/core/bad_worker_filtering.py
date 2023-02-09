import math
from collections import OrderedDict
from logging import getLogger
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import get_algorithm_mos_correlations
from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_ci95, compute_mos
from tts_mos_test_mturk.calculation.etc import (get_workers_count, get_workers_percent,
                                                get_workers_percent_mask, mask_outliers)
from tts_mos_test_mturk.core.evaluation_data import (AssignmentMask, EvaluationData, MaskBase,
                                                     OpinionScoreMask, WorkerMask,
                                                     get_assignment_mask_from_masks,
                                                     get_opinion_score_mask,
                                                     get_opinion_score_mask_from_masks,
                                                     get_worker_mask_from_masks)
from tts_mos_test_mturk.core.stats import print_stats


def ignore_bad_workers(data: EvaluationData, mask_names: OrderedSet[str], threshold: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]

  logger.info("--- Ignoring bad workers ---")
  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)

  worker_correlations = get_algorithm_mos_correlations(opinion_scores)
  bad_worker_np_mask: np.ndarray = worker_correlations < threshold
  bad_worker_mask = WorkerMask(bad_worker_np_mask)
  data.add_mask(mask_name, bad_worker_mask)

  print_stats(data, masks, bad_worker_mask)


def ignore_too_fast_assignments(data: EvaluationData, mask_names: OrderedSet[str], threshold: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]

  logger.info("--- Ignoring fast workers ---")
  worktimes = data.get_worktimes()
  worktimes_mask = get_assignment_mask_from_masks(masks, data)
  worktimes_mask.apply_to(worktimes)

  too_fast_worktimes_np_mask: np.ndarray = worktimes < threshold
  too_fast_worktimes_mask = AssignmentMask(too_fast_worktimes_np_mask)
  data.add_mask(mask_name, too_fast_worktimes_mask)

  print_stats(data, masks, too_fast_worktimes_mask)


def ignore_outlier_opinion_scores(data: EvaluationData, mask_names: OrderedSet[str], max_std_dev_diff: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]

  logger.info("--- Ignoring outliers ---")
  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)

  outlier_np_mask = mask_outliers(opinion_scores, max_std_dev_diff)
  outlier_mask = OpinionScoreMask(outlier_np_mask)
  data.add_mask(mask_name, outlier_mask)

  print_stats(data, masks, outlier_mask)


def ignore_masked_count_opinion_scores(data: EvaluationData, mask_names: OrderedSet[str], ref_mask_name: str, percent: float, mask_name: str):
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]
  ref_mask = data.masks[ref_mask_name]

  logger.info("--- Ignoring count masked ---")
  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)

  ref_mask = get_opinion_score_mask(ref_mask, data)

  outlier_workers_count = get_workers_count(ref_mask.mask)
  outlier_workers_percent = get_workers_percent(ref_mask.mask)
  for w_i, worker in enumerate(data.workers):
    logger.info(
      f"Worker {worker} has {outlier_workers_percent[w_i]*100:.2f}% of outlying scores (#{outlier_workers_count[w_i]})")

  outlier_workers_np_mask = get_workers_percent_mask(ref_mask.mask, percent)

  outlier_workers_mask = WorkerMask(outlier_workers_np_mask)
  data.add_mask(mask_name, outlier_workers_mask)
  print_stats(data, masks, outlier_workers_mask)


def calc_mos(data: EvaluationData, mask_names: OrderedSet[str]) -> None:
  logger = getLogger(__name__)
  masks = [data.masks[mask_name] for mask_name in mask_names]

  logger.info("--- Ignoring count masked ---")
  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)

  scores: List[Dict] = []
  for algo_i, alg_name in enumerate(data.algorithms):
    row = OrderedDict((
      ("Algorithm", alg_name),
      ("MOS", compute_mos(opinion_scores[algo_i])),
      ("CI95", compute_ci95(opinion_scores[algo_i])),
    ))
    scores.append(row)

  res = pd.DataFrame(
    data=[x.values() for x in scores],
    columns=scores[0].keys(),
  )

  print(res)
