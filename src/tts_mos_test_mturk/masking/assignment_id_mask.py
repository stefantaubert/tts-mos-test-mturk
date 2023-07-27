from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_id(data: EvaluationData, mask_names: Set[str], assignment_ids: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)

  res_amask = factory.get_amask()
  for i, assignment_id in enumerate(data.assignments):
    is_masked_already = amask.mask[i]
    if is_masked_already:
      continue
    if assignment_id in assignment_ids:
      res_amask.mask[i] = True

  data.add_or_update_mask(output_mask_name, res_amask)

  print_stats_masks(data, masks, [res_amask])
