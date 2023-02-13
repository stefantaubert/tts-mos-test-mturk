from typing import List, Set

import numpy as np

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.core.masks import AssignmentMask, MaskBase, WorkerMask


def print_stats(data: EvaluationData, mask_names: Set[str], added_mask_names: Set[str]) -> None:
  logger = get_logger()
  logger.info("--- Stats ---")
  masks = [data.masks[mask_name] for mask_name in mask_names]
  added_masks = [data.masks[mask_name] for mask_name in added_mask_names]
  print_stats_masks(data, masks, added_masks)
  logger.info("----------")


def print_stats_masks(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  if len(added_masks) == 0 or any(isinstance(m, WorkerMask) for m in added_masks):
    print_worker_stats(data, masks, added_masks)

  if len(added_masks) == 0 or any(isinstance(m, (WorkerMask, AssignmentMask)) for m in added_masks):
    print_assignment_stats(data, masks, added_masks)

  print_opinion_score_stats(data, masks, added_masks)


def print_opinion_score_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  factory = data.get_mask_factory()

  opinion_scores = data.get_os()

  opinion_scores_mask_before = factory.merge_masks_into_omask(masks)
  opinion_scores_mask_before.apply_by_nan(opinion_scores)
  old_count = np.sum(~np.isnan(opinion_scores))

  opinion_scores_mask_after = factory.merge_masks_into_omask(masks + added_masks)
  opinion_scores_mask_after.apply_by_nan(opinion_scores)
  new_count = np.sum(~np.isnan(opinion_scores))

  logger.info(f"Ignored {old_count - new_count} / {old_count} opinion scores, kept {new_count}!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  factory = data.get_mask_factory()

  assignments_mask_before = factory.merge_masks_into_amask(masks)
  assignments_mask_after = factory.merge_masks_into_amask(masks + added_masks)

  old_count = assignments_mask_before.n_unmasked
  new_count = assignments_mask_after.n_unmasked

  assignments_mask_new_ignored = np.logical_xor(
    assignments_mask_before.mask, assignments_mask_after.mask)
  ignored_assignments = sorted(data.assignments[assignments_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}!")
  if len(ignored_assignments) > 0:
    dlogger = get_detail_logger()
    dlogger.info(f"Ignored assignments: {', '.join(ignored_assignments)}")


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  factory = data.get_mask_factory()

  workers_mask_before = factory.merge_masks_into_wmask(masks)
  workers_mask_after = factory.merge_masks_into_wmask(masks + added_masks)

  old_count = workers_mask_before.n_unmasked
  new_count = workers_mask_after.n_unmasked

  workers_mask_new_ignored = np.logical_xor(workers_mask_before.mask, workers_mask_after.mask)
  ignored_workers = sorted(data.workers[workers_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} workers, kept {new_count}!")
  if len(ignored_workers) > 0:
    dlogger = get_detail_logger()
    dlogger.info(f"Ignored workers: {', '.join(ignored_workers)}")
