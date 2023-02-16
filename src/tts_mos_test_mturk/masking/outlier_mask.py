from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_outliers(Z: np.ndarray, max_std_dev_diff: float) -> np.ndarray:
  assert len(Z.shape) == 2
  mu = np.nanmean(Z)
  s = np.nanstd(Z)

  mu_norm = abs(Z - mu) / s
  outlying_ratings: np.ndarray = mu_norm > max_std_dev_diff

  return outlying_ratings


def mask_outliers_alg(ratings: np.ndarray, max_std_dev_diff: float) -> np.ndarray:
  result = np.full_like(ratings, fill_value=False, dtype=bool)
  n_alg = ratings.shape[0]
  for alg_i in range(n_alg):
    Z = ratings[alg_i]
    result[alg_i, :] = mask_outliers(Z, max_std_dev_diff)
  return result


def mask_outlying_ratings(data: EvaluationData, mask_names: Set[str], max_std_dev_diff: float, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  ratings = data.get_ratings()
  rmask = factory.merge_masks_into_rmask(masks)
  rmask.apply_by_nan(ratings)

  outlier_np_mask = mask_outliers_alg(ratings, max_std_dev_diff)
  outlier_rmask = factory.convert_ndarray_to_rmask(outlier_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_rmask)

  print_stats_masks(data, masks, [outlier_rmask])
