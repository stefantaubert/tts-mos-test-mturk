from typing import Set

from tts_mos_test_mturk.calculation.etc import (get_workers_count, get_workers_percent,
                                                get_workers_percent_mask)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_scores_by_masked_count(data: EvaluationData, mask_names: Set[str], ref_mask_name: str, percent: float, output_mask_name: str):
  logger = get_detail_logger()
  factory = data.get_mask_factory()
  masks = data.get_masks_from_names(mask_names)
  ref_mask = data.masks[ref_mask_name]
  ref_omask = factory.convert_mask_to_omask(ref_mask)

  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  outlier_workers_count = get_workers_count(ref_omask.mask)
  outlier_workers_percent = get_workers_percent(ref_omask.mask)
  for w_i, worker in enumerate(data.workers):
    logger.info(
      f"Worker {worker} has {outlier_workers_percent[w_i]*100:.2f}% of outlying scores (#{outlier_workers_count[w_i]})")

  outlier_workers_np_mask = get_workers_percent_mask(ref_mask.mask, percent)
  outlier_wmask = factory.convert_ndarray_to_wmask(outlier_workers_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_wmask)
  print_stats_masks(data, masks, [outlier_wmask])
