from typing import List, Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masks import AssignmentMask, MaskBase, WorkerMask


def print_stats(data: EvaluationData, mask_names: Set[str], added_mask_names: Set[str]) -> None:
  logger = get_logger()
  logger.info("--- Stats ---")
  masks = data.get_masks_from_names(mask_names)
  added_masks = data.get_masks_from_names(added_mask_names)
  print_stats_masks(data, masks, added_masks)
  logger.info("-------------")


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

  n_already_ignored = opinion_scores_mask_before.n_masked

  if data.n_opinion_scores > 0:
    logger.info("--- Opinion Score Statistics ---")
  if data.n_opinion_scores > 0:
    logger.info(
        f"{n_already_ignored} out of all {data.n_opinion_scores} opinion scores ({n_already_ignored/data.n_opinion_scores*100:.2f}%) were already masked (i.e., {data.n_opinion_scores - n_already_ignored} unmasked).")
  if old_count > 0:
    logger.info(
        f"Masked {old_count - new_count} out of the {old_count} unmasked opinion scores ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")
  if data.n_opinion_scores > 0:
    logger.info(f"Result: {opinion_scores_mask_after.n_masked} out of all {data.n_opinion_scores} opinion scores ({opinion_scores_mask_after.n_masked/data.n_opinion_scores*100:.2f}%) are masked now!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()

  factory = data.get_mask_factory()

  assignments_mask_before = factory.merge_masks_into_amask(masks)
  assignments_mask_after = factory.merge_masks_into_amask(masks + added_masks)

  old_count = assignments_mask_before.n_unmasked
  new_count = assignments_mask_after.n_unmasked

  already_ignored = sorted(data.assignments[assignments_mask_before.masked_indices])

  assignments_mask_new_ignored = np.logical_xor(
    assignments_mask_before.mask, assignments_mask_after.mask)
  newly_masked_assignments = sorted(data.assignments[assignments_mask_new_ignored.nonzero()[0]])

  if data.n_assignments > 0:
    logger.info("--- Assignment Statistics ---")

  if data.n_assignments > 0:
    logger.info(
      f"{len(already_ignored)} out of all {data.n_assignments} assignments ({len(already_ignored)/data.n_assignments*100:.2f}%) were already masked (i.e., {data.n_assignments - len(already_ignored)} unmasked).")

  if len(already_ignored) > 0:
    dlogger.info(f"Already masked assignments:  {', '.join(already_ignored)}")

  if old_count > 0:
    logger.info(
      f"Masked {old_count - new_count} out of the {old_count} unmasked assignments ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_assignments) > 0:
    dlogger.info(f"Masked assignments: {', '.join(newly_masked_assignments)}")

  # not_ignored = sorted(data.assignments[assignments_mask_after.unmasked_indices])
  # if len(not_ignored) > 0:
  #   dlogger.info(f"Unmasked assignments: {', '.join(not_ignored)}")

  if data.n_assignments > 0:
    logger.info(
      f"Result: {assignments_mask_after.n_masked} out of all {data.n_assignments} assignments ({assignments_mask_after.n_masked/data.n_assignments*100:.2f}%) are masked now!")


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()
  factory = data.get_mask_factory()

  workers_mask_before = factory.merge_masks_into_wmask(masks)
  workers_mask_after = factory.merge_masks_into_wmask(masks + added_masks)

  old_count = workers_mask_before.n_unmasked
  new_count = workers_mask_after.n_unmasked

  already_ignored = sorted(data.workers[workers_mask_before.masked_indices])

  workers_mask_new_ignored = np.logical_xor(workers_mask_before.mask, workers_mask_after.mask)
  newly_masked_workers = sorted(data.workers[workers_mask_new_ignored.nonzero()[0]])

  if data.n_workers > 0:
    logger.info("--- Worker Statistics ---")

  if data.n_workers > 0:
    logger.info(
      f"{len(already_ignored)} out of all {data.n_workers} workers ({len(already_ignored)/data.n_workers*100:.2f}%) were already masked (i.e., {data.n_workers - len(already_ignored)} unmasked).")

  if len(already_ignored) > 0:
    dlogger.info(f"Already masked workers:  {', '.join(already_ignored)}")

  if old_count > 0:
    logger.info(
      f"Masked {old_count - new_count} out of the {old_count} unmasked workers ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_workers) > 0:
    dlogger.info(f"Masked workers: {', '.join(newly_masked_workers)}")

  # not_ignored = sorted(data.workers[workers.unmasked_indices])
  # if len(not_ignored) > 0:
  #   dlogger.info(f"Unmasked workers: {', '.join(not_ignored)}")

  if data.n_workers > 0:
    logger.info(
      f"Result: {workers_mask_after.n_masked} out of all {data.n_workers} workers ({workers_mask_after.n_masked/data.n_workers*100:.2f}%) are masked now!")
