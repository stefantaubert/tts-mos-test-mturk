from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_worktime(data: EvaluationData, mask_names: Set[str], threshold: float, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes = get_worktimes(data)
  worktimes_mask = factory.merge_masks_into_amask(masks)
  worktimes_mask.apply_by_nan(worktimes)

  too_fast_worktimes_np_mask = worktimes < threshold
  too_fast_worktimes_mask = factory.convert_ndarray_to_amask(too_fast_worktimes_np_mask)
  data.add_or_update_mask(output_mask_name, too_fast_worktimes_mask)

  print_stats_masks(data, masks, [too_fast_worktimes_mask])


def get_worktimes(data: EvaluationData) -> np.ndarray:
  worktimes = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float32,
  )
  for dp in data.data:
    ass_i = data.assignments.get_loc(dp.assignment_id)
    worktimes[ass_i] = dp.worktime
  return worktimes
