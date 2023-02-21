from collections import OrderedDict
from typing import Set

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame, concat

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


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


def mask_outliers_alg_stats_df(ratings: np.ndarray, min_std_dev_diff: float, max_std_dev_diff: float, algorithms: OrderedSet[str]) -> np.ndarray:
  n_alg = ratings.shape[0]
  lines = []
  for alg_i in range(n_alg):
    Z = ratings[alg_i]
    mu_norm = get_mu_norm(Z)
    masked = mask_values_in_boundary(mu_norm, min_std_dev_diff, max_std_dev_diff)
    lines.append(OrderedDict((
      ("Algorithm", algorithms[alg_i]),
      ("Min", np.nanmin(mu_norm)),
      ("Mean", np.nanmean(mu_norm)),
      ("Median", np.nanmedian(mu_norm)),
      ("Max", np.nanmax(mu_norm)),
      ("# Masked", np.sum(masked)),
      ("# Unmasked", np.sum(~np.isnan(Z)) - np.sum(masked)),
      ("# All", np.sum(~np.isnan(Z))),
    )))
  result = DataFrame.from_records(lines)
  row = {
    "Algorithm": "All",
    "Min": result["Min"].min(),
    "Mean": result["Mean"].mean(),
    "Median": result["Median"].median(),
    "Max": result["Max"].max(),
    "# Masked": result["# Masked"].sum(),
    "# Unmasked": result["# Unmasked"].sum(),
    "# All": result["# All"].sum(),
  }
  result = concat([result, DataFrame.from_records([row])], ignore_index=True)
  return result


def mask_outlying_ratings(data: EvaluationData, mask_names: Set[str], min_std_dev_diff: float, max_std_dev_diff: float, output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  rmask = factory.merge_masks_into_rmask(masks)

  ratings = data.get_ratings()
  rmask.apply_by_nan(ratings)

  df = mask_outliers_alg_stats_df(ratings, min_std_dev_diff, max_std_dev_diff, data.algorithms)
  dlogger.info(f'Statistics:\n{df}')

  outlier_np_mask = mask_outliers_alg(ratings, min_std_dev_diff, max_std_dev_diff)
  outlier_rmask = factory.convert_ndarray_to_rmask(outlier_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_rmask)

  print_stats_masks(data, masks, [outlier_rmask])
