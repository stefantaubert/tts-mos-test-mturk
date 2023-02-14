from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_worktime(data: EvaluationData, mask_names: Set[str], threshold: float, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes = data.get_worktimes()
  worktimes_mask = factory.merge_masks_into_amask(masks)
  worktimes_mask.apply_by_nan(worktimes)

  too_fast_worktimes_np_mask = worktimes < threshold
  too_fast_worktimes_mask = factory.convert_ndarray_to_amask(too_fast_worktimes_np_mask)
  data.add_or_update_mask(output_mask_name, too_fast_worktimes_mask)

  print_stats_masks(data, masks, [too_fast_worktimes_mask])
