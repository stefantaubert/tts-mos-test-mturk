
from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def get_wmask_by_age_group(data: EvaluationData, age_groups: Set[str]):
  factory = MaskFactory(data)
  res_wmask = factory.get_wmask()
  for i, worker_id in enumerate(data.workers):
    worker_data = data.worker_data[worker_id]
    if worker_data.age_group in age_groups:
      res_wmask.mask[i] = True

  return res_wmask


def mask_workers_by_age_group(data: EvaluationData, mask_names: Set[str], age_groups: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  existing_wmask = factory.merge_masks_into_wmask(masks)
  new_wmask = get_wmask_by_age_group(data, age_groups)
  existing_wmask.apply_by_false(new_wmask.mask)

  data.add_or_update_mask(output_mask_name, new_wmask)

  print_stats_masks(data, masks, [new_wmask])
