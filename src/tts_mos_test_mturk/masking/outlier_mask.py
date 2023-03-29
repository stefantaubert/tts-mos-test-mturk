from collections import OrderedDict
from typing import Set

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame, concat

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_outlying_ratings(data: EvaluationData, mask_names: Set[str], min_std_dev_diff: float, max_std_dev_diff: float, output_mask_name: str, rating_names: Set[str]):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)

  ratings = get_ratings(data, rating_names)
  rmask.apply_by_nan(ratings)

  stats_df = mask_outliers_alg_stats_df(
    ratings, min_std_dev_diff, max_std_dev_diff, data.algorithms)
  log_full_df_info(stats_df, "Statistics:")

  outlier_np_mask = mask_outliers_alg(ratings, min_std_dev_diff, max_std_dev_diff)
  outlier_rmask = factory.convert_ndarray_to_rmask(outlier_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_rmask)

  print_stats_masks(data, masks, [outlier_rmask])


def mask_outliers_alg_stats_df(ratings: np.ndarray, min_std_dev_diff: float, max_std_dev_diff: float, algorithms: OrderedSet[str]) -> np.ndarray:
  col_alg = "Algorithm"
  col_min = "Min"
  col_mean = "Mean"
  col_median = "Median"
  col_max = "Max"
  col_masked = "# Masked"
  col_unmasked = "# Unmasked"
  col_all = "# All"

  n_alg = ratings.shape[0]
  lines = []
  for alg_i in range(n_alg):
    Z = ratings[alg_i]
    mu_norm = get_mu_norm(Z)
    masked = mask_values_in_boundary(mu_norm, min_std_dev_diff, max_std_dev_diff)
    lines.append(OrderedDict((
      (col_alg, algorithms[alg_i]),
      (col_min, np.nanmin(mu_norm)),
      (col_mean, np.nanmean(mu_norm)),
      (col_median, np.nanmedian(mu_norm)),
      (col_max, np.nanmax(mu_norm)),
      (col_masked, np.sum(masked)),
      (col_unmasked, np.sum(~np.isnan(Z)) - np.sum(masked)),
      (col_all, np.sum(~np.isnan(Z))),
    )))
  result = DataFrame.from_records(lines)
  row = {
    col_alg: "-ALL-",
    col_min: result[col_min].min(),
    col_mean: result[col_mean].mean(),
    col_median: result[col_median].median(),
    col_max: result[col_max].max(),
    col_masked: result[col_masked].sum(),
    col_unmasked: result[col_unmasked].sum(),
    col_all: result[col_all].sum(),
  }
  result.sort_values(by=[col_alg], inplace=True)
  result = concat([result, DataFrame.from_records([row])], ignore_index=True)
  return result


def get_mu_norm(Z: np.ndarray) -> np.ndarray:
  assert len(Z.shape) == 2
  mu = np.nanmean(Z)
  s = np.nanstd(Z)

  mu_norm = abs(Z - mu) / s
  return mu_norm


def mask_outliers(Z: np.ndarray, min_std_dev_diff: float, max_std_dev_diff: float) -> np.ndarray:
  mu_norm = get_mu_norm(Z)
  outlying_ratings = mask_values_in_boundary(mu_norm, min_std_dev_diff, max_std_dev_diff)
  return outlying_ratings


def mask_outliers_alg(ratings: np.ndarray, min_std_dev_diff: float, max_std_dev_diff: float) -> np.ndarray:
  result = np.full_like(ratings, fill_value=False, dtype=bool)
  n_alg = ratings.shape[0]
  for alg_i in range(n_alg):
    Z = ratings[alg_i]
    result[alg_i, :] = mask_outliers(Z, min_std_dev_diff, max_std_dev_diff)
  return result
