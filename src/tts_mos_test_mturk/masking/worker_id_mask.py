from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_id(data: EvaluationData, mask_names: Set[str], worker_ids: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  wmask = factory.merge_masks_into_wmask(masks)

  res_wmask = factory.get_wmask()
  for i, worker_id in enumerate(data.workers):
    is_masked_already = wmask.mask[i]
    if is_masked_already:
      continue
    if worker_id in worker_ids:
      res_wmask.mask[i] = True

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])
