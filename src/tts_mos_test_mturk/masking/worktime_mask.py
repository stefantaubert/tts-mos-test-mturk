from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.masking.etc import sort_indices_after_values
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_worktime(data: EvaluationData, mask_names: Set[str], from_threshold_incl: int, to_threshold_excl: int, output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  amask = factory.merge_masks_into_amask(masks)

  if amask.n_masked > 0:
    dlogger.info('Already masked assignments:')
    for assignment_name in sorted(data.assignments[amask.masked_indices]):
      dlogger.info(f'- "{assignment_name}"')

  worktimes = get_worktimes(data)
  amask.apply_by_nan(worktimes)

  too_fast_worktimes_np_mask = get_worktime_mask(worktimes, from_threshold_incl, to_threshold_excl)
  too_fast_worktimes_mask = factory.convert_ndarray_to_amask(too_fast_worktimes_np_mask)

  aindices = np.arange(data.n_assignments)
  aindices_sorted = sort_indices_after_values(aindices, worktimes)
  masked_indices = too_fast_worktimes_mask.masked_indices

  dlogger.info("Assignment ranking by worktime:")
  already_masked_indices = amask.masked_indices
  for nr, a_i in enumerate(aindices_sorted, start=1):
    if a_i in already_masked_indices:
      break
    masked_str = " [masked]" if a_i in masked_indices else ""
    dlogger.info(
      f"{nr}. \"{data.assignments[a_i]}\": {int(worktimes[a_i])}{masked_str}")

  data.add_or_update_mask(output_mask_name, too_fast_worktimes_mask)

  print_stats_masks(data, masks, [too_fast_worktimes_mask])


def get_worktime_mask(worktimes: np.ndarray, from_threshold_incl: int, to_threshold_excl: int) -> np.ndarray:
  result = (from_threshold_incl <= worktimes) & (worktimes < to_threshold_excl)
  return result


def get_worktimes(data: EvaluationData) -> np.ndarray:
  worktimes = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float32,
  )
  for data_point in data.data:
    ass_i = data.assignments.get_loc(data_point.assignment_id)
    worktimes[ass_i] = data_point.worktime
  return worktimes
