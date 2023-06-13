
from typing import List, Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import AssignmentsMask, MaskBase, RatingsMask, WorkersMask
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def merge_masks(data: EvaluationData, mask_names: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)
  lowest_mask = get_lowest_mask(masks)
  if isinstance(lowest_mask, RatingsMask):
    new_mask = factory.merge_masks_into_rmask(masks)
  elif isinstance(lowest_mask, AssignmentsMask):
    new_mask = factory.merge_masks_into_amask(masks)
  else:
    assert isinstance(lowest_mask, WorkersMask)
    new_mask = factory.merge_masks_into_wmask(masks)

  data.add_or_update_mask(output_mask_name, new_mask)

  print_stats_masks(data, masks, [new_mask])


def get_lowest_mask(masks: List[MaskBase]) -> MaskBase:
  """get first lowest mask, rating is lowest, worker is highest"""
  lowest_mask = None
  for mask in masks:
    if isinstance(mask, RatingsMask):
      return mask
    if isinstance(mask, AssignmentsMask) and isinstance(lowest_mask, WorkersMask):
      lowest_mask = mask
    else:
      assert isinstance(mask, WorkersMask)
      if not isinstance(mask, AssignmentsMask):
        lowest_mask = mask
  return lowest_mask
