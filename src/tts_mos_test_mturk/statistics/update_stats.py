from typing import Generator, Iterator, List, Set

import numpy as np

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import AssignmentsMask, MaskBase, WorkersMask


def print_stats(data: EvaluationData, mask_names: Set[str], added_mask_names: Set[str]) -> None:
  logger = get_logger()
  logger.info("--- Stats ---")
  masks = data.get_masks_from_names(mask_names)
  added_masks = data.get_masks_from_names(added_mask_names)
  print_stats_masks(data, masks, added_masks)
  logger.info("-------------")


def print_stats_masks(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  if len(added_masks) == 0 or any(isinstance(m, WorkersMask) for m in added_masks):
    print_worker_stats(data, masks, added_masks)

  if len(added_masks) == 0 or any(isinstance(m, (WorkersMask, AssignmentsMask)) for m in added_masks):
    print_assignment_stats(data, masks, added_masks)

  print_rating_stats(data, masks, added_masks)


def print_rating_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  factory = MaskFactory(data)

  ratings = get_ratings(data)

  ratings_mask_before = factory.merge_masks_into_rmask(masks)
  ratings_mask_before.apply_by_nan(ratings)
  old_count = np.sum(~np.isnan(ratings))

  ratings_mask_after = factory.merge_masks_into_rmask(masks + added_masks)
  ratings_mask_after.apply_by_nan(ratings)
  new_count = np.sum(~np.isnan(ratings))

  n_already_ignored = ratings_mask_before.n_masked

  if data.n_ratings > 0:
    logger.info("--- Ratings Statistics ---")
  if data.n_ratings > 0:
    logger.info(
        f"{n_already_ignored} out of all {data.n_ratings} ratings ({n_already_ignored/data.n_ratings*100:.2f}%) were already masked (i.e., {data.n_ratings - n_already_ignored} unmasked).")
  if old_count > 0:
    logger.info(
        f"Masked {old_count - new_count} out of the {old_count} unmasked ratings ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")
  if data.n_ratings > 0:
    logger.info(
      f"Result: {ratings_mask_after.n_masked} out of all {data.n_ratings} ratings ({ratings_mask_after.n_masked/data.n_ratings*100:.2f}%) are masked now!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()

  factory = MaskFactory(data)

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
    dlogger.info(f"Already masked assignments:  {', '.join(to_quote(already_ignored))}")

  if old_count > 0:
    logger.info(
      f"Masked {old_count - new_count} out of the {old_count} unmasked assignments ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_assignments) > 0:
    dlogger.info(f"Masked assignments: {', '.join(to_quote(newly_masked_assignments))}")

  # not_ignored = sorted(data.assignments[assignments_mask_after.unmasked_indices])
  # if len(not_ignored) > 0:
  #   dlogger.info(f"Unmasked assignments: {', '.join(not_ignored)}")

  if data.n_assignments > 0:
    logger.info(
      f"Result: {assignments_mask_after.n_masked} out of all {data.n_assignments} assignments ({assignments_mask_after.n_masked/data.n_assignments*100:.2f}%) are masked now!")


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()
  factory = MaskFactory(data)

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
    dlogger.info(f"Already masked workers: {', '.join(to_quote(already_ignored))}")

  if old_count > 0:
    logger.info(
      f"Masked {old_count - new_count} out of the {old_count} unmasked workers ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_workers) > 0:
    dlogger.info(f"Masked workers: {', '.join(to_quote(newly_masked_workers))}")

  # not_ignored = sorted(data.workers[workers.unmasked_indices])
  # if len(not_ignored) > 0:
  #   dlogger.info(f"Unmasked workers: {', '.join(not_ignored)}")

  if data.n_workers > 0:
    logger.info(
      f"Result: {workers_mask_after.n_masked} out of all {data.n_workers} workers ({workers_mask_after.n_masked/data.n_workers*100:.2f}%) are masked now!")


def to_quote(strings: Iterator[str]) -> Generator[str, None, None]:
  for s in strings:
    yield f"\"{s}\""
