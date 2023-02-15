from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_scores_by_masked_count(data: EvaluationData, mask_names: Set[str], ref_mask_name: str, percent: float, output_mask_name: str):
  dlogger = get_detail_logger()
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
    dlogger.info(
      f"Worker {worker} has {outlier_workers_percent[w_i]*100:.2f}% of outlying scores (#{outlier_workers_count[w_i]})")

  outlier_workers_np_mask = get_workers_percent_mask(ref_mask.mask, percent)
  outlier_wmask = factory.convert_ndarray_to_wmask(outlier_workers_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_wmask)
  print_stats_masks(data, masks, [outlier_wmask])


def get_workers_percent(Z_mask: np.ndarray) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)
  total = np.sum(Z_mask.flatten())

  percent: np.ndarray = sums2 / total
  return percent


def get_workers_count(Z_mask: np.ndarray) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)

  percent: np.ndarray = sums2
  return percent


def get_workers_percent_mask(Z_mask: np.ndarray, p: float) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)
  total = np.sum(Z_mask.flatten())

  outlying_workers: np.ndarray = (total > 0) & (sums2 >= p * total)
  return outlying_workers
