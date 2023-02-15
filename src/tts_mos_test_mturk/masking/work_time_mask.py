from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_work_time(data: EvaluationData, mask_names: Set[str], threshold: float, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  work_times = get_work_times(data)
  work_times_mask = factory.merge_masks_into_amask(masks)
  work_times_mask.apply_by_nan(work_times)

  too_fast_work_times_np_mask = get_work_time_mask(work_times, threshold)
  too_fast_work_times_mask = factory.convert_ndarray_to_amask(too_fast_work_times_np_mask)
  data.add_or_update_mask(output_mask_name, too_fast_work_times_mask)

  print_stats_masks(data, masks, [too_fast_work_times_mask])


def get_work_time_mask(work_times: np.ndarray, threshold: float) -> np.ndarray:
  result = work_times < threshold
  return result


def get_work_times(data: EvaluationData) -> np.ndarray:
  work_times = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float32,
  )
  for dp in data.data:
    ass_i = data.assignments.get_loc(dp.assignment_id)
    work_times[ass_i] = dp.work_time
  return work_times
