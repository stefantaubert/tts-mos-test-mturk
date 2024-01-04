from typing import Generator, Iterator, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import AssignmentsMask, MaskBase, WorkersMask

COL_MASK_TYPE = "Type"
COL_COUNT = "#"
COL_PREV_MASKED = "#PrevMasked"
COL_PREV_MASKED_PERCENT = "#PrevMasked %"
COL_PREV_UNMASKED = "#PrevUnmasked"
COL_NEWLY_MASKED = "#NewlyMasked"
COL_NEWLY_MASKED_PERCENT = "#NewlyMasked %"
COL_MASKED = "#Masked"
COL_MASKED_PERCENT = "#Masked %"
COL_UNMASKED = "#Unmasked"

COLS = [
  COL_MASK_TYPE,
  COL_COUNT,
  COL_PREV_MASKED,
  COL_NEWLY_MASKED,
  COL_MASKED,
  COL_PREV_MASKED_PERCENT,
  COL_NEWLY_MASKED_PERCENT,
  COL_MASKED_PERCENT,
  COL_PREV_UNMASKED,
  COL_UNMASKED,
]


def print_stats(data: EvaluationData, mask_names: Set[str], added_mask_names: Set[str]) -> None:
  logger = get_logger()
  logger.info("--- Stats ---")
  masks = data.get_masks_from_names(mask_names)
  added_masks = data.get_masks_from_names(added_mask_names)
  print_stats_masks(data, masks, added_masks)
  logger.info("-------------")


def print_stats_masks(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  all_stats = []
  if len(added_masks) == 0 or any(isinstance(m, WorkersMask) for m in added_masks):
    stats = print_worker_stats(data, masks, added_masks)
    all_stats.append(stats)

  if len(added_masks) == 0 or any(isinstance(m, (WorkersMask, AssignmentsMask)) for m in added_masks):
    stats = print_assignment_stats(data, masks, added_masks)
    all_stats.append(stats)

  stats = print_rating_stats(data, masks, added_masks)
  all_stats.append(stats)
  df = pd.DataFrame.from_records(all_stats, columns=COLS)
  print(df.to_string(index=False, max_cols=None))


def print_rating_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  factory = MaskFactory(data)

  ratings = get_ratings(data, data.rating_names)

  mask_before = factory.merge_masks_into_rmask(masks)
  mask_before.apply_by_nan(ratings)
  old_count = np.sum(~np.isnan(ratings))

  ratings_mask_after = factory.merge_masks_into_rmask(masks + added_masks)
  ratings_mask_after.apply_by_nan(ratings)
  new_count = np.sum(~np.isnan(ratings))

  stats = {}
  stats[COL_MASK_TYPE] = "Ratings"
  stats[COL_COUNT] = data.n_ratings
  stats[COL_PREV_MASKED] = mask_before.n_masked
  stats[COL_PREV_MASKED_PERCENT] = mask_before.n_masked / data.n_ratings * 100
  stats[COL_PREV_UNMASKED] = old_count
  stats[COL_NEWLY_MASKED] = old_count - new_count
  stats[COL_NEWLY_MASKED_PERCENT] = (old_count - new_count) / old_count * 100
  stats[COL_MASKED] = ratings_mask_after.n_masked
  stats[COL_MASKED_PERCENT] = ratings_mask_after.n_masked / data.n_ratings * 100
  stats[COL_UNMASKED] = new_count

  # if data.n_ratings > 0:
  #   logger.info("--- Ratings Statistics ---")
  # if data.n_ratings > 0:
  #   logger.info(
  #       f"{n_already_ignored} out of all {data.n_ratings} ratings ({n_already_ignored/data.n_ratings*100:.2f}%) were already masked (i.e., {data.n_ratings - n_already_ignored} unmasked).")
  # if old_count > 0:
  #   logger.info(
  #       f"Masked {old_count - new_count} out of the {old_count} unmasked ratings ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")
  # if data.n_ratings > 0:
  #   logger.info(
  #     f"Result: {ratings_mask_after.n_masked} out of all {data.n_ratings} ratings ({ratings_mask_after.n_masked/data.n_ratings*100:.2f}%) are masked now!")
  return stats


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()

  factory = MaskFactory(data)

  mask_before = factory.merge_masks_into_amask(masks)
  mask_after = factory.merge_masks_into_amask(masks + added_masks)

  old_count = mask_before.n_unmasked
  new_count = mask_after.n_unmasked

  already_ignored = sorted(data.assignments[mask_before.masked_indices])

  assignments_mask_new_ignored = np.logical_xor(
    mask_before.mask, mask_after.mask)
  newly_masked_assignments = sorted(data.assignments[assignments_mask_new_ignored.nonzero()[0]])

  # if data.n_assignments > 0:
  #   logger.info("--- Assignment Statistics ---")

  # if data.n_assignments > 0:
  #   logger.info(
  #     f"{len(already_ignored)} out of all {data.n_assignments} assignments ({len(already_ignored)/data.n_assignments*100:.2f}%) were already masked (i.e., {data.n_assignments - len(already_ignored)} unmasked).")

  if len(already_ignored) > 0:
    dlogger.info(f"Already masked assignments:  {', '.join(to_quote(already_ignored))}")

  # if old_count > 0:
  #   logger.info(
  #     f"Masked {old_count - new_count} out of the {old_count} unmasked assignments ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_assignments) > 0:
    dlogger.info(f"Masked assignments: {', '.join(to_quote(newly_masked_assignments))}")

  # # not_ignored = sorted(data.assignments[assignments_mask_after.unmasked_indices])
  # # if len(not_ignored) > 0:
  # #   dlogger.info(f"Unmasked assignments: {', '.join(not_ignored)}")

  # if data.n_assignments > 0:
  #   logger.info(
  #     f"Result: {mask_after.n_masked} out of all {data.n_assignments} assignments ({mask_after.n_masked/data.n_assignments*100:.2f}%) are masked now!")

  stats = {}
  stats[COL_MASK_TYPE] = "Assignments"
  stats[COL_COUNT] = data.n_assignments
  stats[COL_PREV_MASKED] = mask_before.n_masked
  stats[COL_PREV_MASKED_PERCENT] = mask_before.n_masked / data.n_assignments * 100
  stats[COL_PREV_UNMASKED] = old_count
  stats[COL_NEWLY_MASKED] = old_count - new_count
  stats[COL_NEWLY_MASKED_PERCENT] = (old_count - new_count) / old_count * 100
  stats[COL_MASKED] = mask_after.n_masked
  stats[COL_MASKED_PERCENT] = mask_after.n_masked / data.n_assignments * 100
  stats[COL_UNMASKED] = new_count
  return stats


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = get_logger()
  dlogger = get_detail_logger()
  factory = MaskFactory(data)

  mask_before = factory.merge_masks_into_wmask(masks)
  mask_after = factory.merge_masks_into_wmask(masks + added_masks)

  old_count = mask_before.n_unmasked
  new_count = mask_after.n_unmasked

  already_ignored = sorted(data.workers[mask_before.masked_indices])

  workers_mask_new_ignored = np.logical_xor(mask_before.mask, mask_after.mask)
  newly_masked_workers = sorted(data.workers[workers_mask_new_ignored.nonzero()[0]])

  # if data.n_workers > 0:
  #   logger.info("--- Worker Statistics ---")

  # if data.n_workers > 0:
  #   logger.info(
  #     f"{len(already_ignored)} out of all {data.n_workers} workers ({len(already_ignored)/data.n_workers*100:.2f}%) were already masked (i.e., {data.n_workers - len(already_ignored)} unmasked).")

  if len(already_ignored) > 0:
    dlogger.info(f"Already masked workers: {', '.join(to_quote(already_ignored))}")

  # if old_count > 0:
  #   logger.info(
  #     f"Masked {old_count - new_count} out of the {old_count} unmasked workers ({(old_count - new_count)/old_count*100:.2f}%), kept {new_count} unmasked!")

  if len(newly_masked_workers) > 0:
    dlogger.info(f"Masked workers: {', '.join(to_quote(newly_masked_workers))}")

  # # not_ignored = sorted(data.workers[workers.unmasked_indices])
  # # if len(not_ignored) > 0:
  # #   dlogger.info(f"Unmasked workers: {', '.join(not_ignored)}")

  # if data.n_workers > 0:
  #   logger.info(
  #     f"Result: {mask_after.n_masked} out of all {data.n_workers} workers ({mask_after.n_masked/data.n_workers*100:.2f}%) are masked now!")

  stats = {}
  stats[COL_MASK_TYPE] = "Workers"
  stats[COL_COUNT] = data.n_workers
  stats[COL_PREV_MASKED] = mask_before.n_masked
  stats[COL_PREV_MASKED_PERCENT] = mask_before.n_masked / data.n_workers * 100
  stats[COL_PREV_UNMASKED] = old_count
  stats[COL_NEWLY_MASKED] = old_count - new_count
  stats[COL_NEWLY_MASKED_PERCENT] = (old_count - new_count) / old_count * 100
  stats[COL_MASKED] = mask_after.n_masked
  stats[COL_MASKED_PERCENT] = mask_after.n_masked / data.n_workers * 100
  stats[COL_UNMASKED] = new_count
  return stats


def to_quote(strings: Iterator[str]) -> Generator[str, None, None]:
  for s in strings:
    yield f"\"{s}\""
