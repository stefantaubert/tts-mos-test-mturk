from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_ratings_by_masked_count(data: EvaluationData, mask_names: Set[str], ref_masks: Set[str], percent: float, output_mask_name: str):
  dlogger = get_detail_logger()
  factory = data.get_mask_factory()
  masks = data.get_masks_from_names(mask_names)
  ref_masks = data.get_masks_from_names(ref_masks)
  ref_rmask = factory.merge_masks_into_rmask(ref_masks)

  rmask = factory.merge_masks_into_rmask(masks)
  rmask.apply_by_false(ref_rmask.mask)

  outlier_workers_count = get_workers_masked_os_count(ref_rmask.mask)
  total_count = np.sum(outlier_workers_count)
  if total_count == 0:
    dlogger.info("No masked ratings exist!")
  else:
    for w_i, worker in enumerate(data.workers):
      dlogger.info(
        f"Worker {worker} has {outlier_workers_count[w_i]/total_count*100:.2f}% of outlying ratings (#{outlier_workers_count[w_i]}/{total_count})")

  outlier_workers_np_mask = get_workers_percent_mask(ref_rmask.mask, percent)
  outlier_wmask = factory.convert_ndarray_to_wmask(outlier_workers_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_wmask)
  print_stats_masks(data, masks, [outlier_wmask])


def get_workers_percent_mask(Z_mask: np.ndarray, p: float) -> np.ndarray:
  sums2 = get_workers_masked_os_count(Z_mask)
  total = np.sum(Z_mask.flatten())

  outlying_workers: np.ndarray = (total > 0) & (sums2 >= p * total)
  return outlying_workers


def get_workers_masked_os_count(Z_mask: np.ndarray) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)
  return sums2
