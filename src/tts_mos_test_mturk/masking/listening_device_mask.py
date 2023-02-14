
from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_listening_device(data: EvaluationData, mask_names: Set[str], listening_types: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  lts = data.get_listening_devices()
  amask = factory.merge_masks_into_amask(masks)
  amask.apply_by_nan(lts)

  res_amask = factory.get_amask()
  for lt in listening_types:
    lt_mask = lts == lt
    res_amask.combine_mask_np(lt_mask)

  data.add_or_update_mask(output_mask_name, res_amask)

  print_stats_masks(data, masks, [res_amask])
