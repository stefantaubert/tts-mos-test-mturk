import math
from logging import getLogger
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import get_algorithm_mos_correlations
from tts_mos_test_mturk.core.evaluation_data import (AssignmentMask, EvaluationData, MaskBase,
                                                     WorkerMask, get_assignment_mask_from_masks,
                                                     get_opinion_score_mask_from_masks,
                                                     get_worker_mask_from_masks)


def print_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  if any(isinstance(m, WorkerMask) for m in added_masks):
    print_worker_stats(data, masks, added_masks)

  if any(isinstance(m, (WorkerMask, AssignmentMask)) for m in added_masks):
    print_assignment_stats(data, masks, added_masks)

  print_opinion_score_stats(data, masks, added_masks)


def print_opinion_score_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)

  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)
  old_count = np.sum(~np.isnan(opinion_scores))

  opinion_scores_mask = get_opinion_score_mask_from_masks(masks + added_masks, data)
  opinion_scores_mask.apply_to(opinion_scores)
  new_count = np.sum(~np.isnan(opinion_scores))

  logger.info(f"Ignored {old_count - new_count} / {old_count} opinion scores, kept {new_count}!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)

  assignments_mask_before = get_assignment_mask_from_masks(masks, data)
  assignments_mask_after = get_assignment_mask_from_masks(masks + added_masks, data)

  old_count = assignments_mask_before.n_unmasked
  new_count = assignments_mask_after.n_unmasked

  assignments_mask_new_ignored = np.logical_xor(
    assignments_mask_before.mask, assignments_mask_after.mask)
  ignored_assignments = sorted(data.assignments[assignments_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}!")
  if len(ignored_assignments) > 0:
    logger.info(f"Ignored assignments: {', '.join(ignored_assignments)}")


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)

  workers_mask_before = get_worker_mask_from_masks(masks, data)
  workers_mask_after = get_worker_mask_from_masks(masks + added_masks, data)

  old_count = workers_mask_before.n_unmasked
  new_count = workers_mask_after.n_unmasked

  workers_mask_new_ignored = np.logical_xor(workers_mask_before.mask, workers_mask_after.mask)
  ignored_workers = sorted(data.workers[workers_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} workers, kept {new_count}!")
  if len(ignored_workers) > 0:
    logger.info(f"Ignored workers: {', '.join(ignored_workers)}")
