from collections import OrderedDict
from typing import Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import RatingsMask
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_ratings_by_masked_count(data: EvaluationData, mask_names: Set[str], ref_masks: Set[str], from_percent_incl: float, to_percent_excl: float, output_mask_name: str):
  factory = MaskFactory(data)
  masks = data.get_masks_from_names(mask_names)
  ref_masks = data.get_masks_from_names(ref_masks)
  ref_rmask = factory.merge_masks_into_rmask(ref_masks)
  rmask = factory.merge_masks_into_rmask(masks)

  # it doesn't matter which ratings are taken
  ratings = get_ratings(data, data.rating_names)
  rmask.apply_by_false(ref_rmask.mask)
  rmask.apply_by_nan(ratings)

  outlier_wmask_np = get_wmask_percent(ref_rmask.mask, from_percent_incl, to_percent_excl)
  outlier_wmask = factory.convert_ndarray_to_wmask(outlier_wmask_np)
  outlying_worker_indices = outlier_wmask.masked_indices

  stats_df = get_stats_df(ref_rmask, data.workers, outlying_worker_indices, ratings)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, outlier_wmask)
  print_stats_masks(data, masks, [outlier_wmask])


def get_stats_df(ref_rmask: RatingsMask, workers: OrderedSet[str], outlier_indices: np.ndarray, ratings: np.ndarray) -> pd.DataFrame:
  col_worker = "Worker"
  col_outliers = "# Outliers"
  col_ratings = "# Ratings"
  col_percent = "%"
  col_percent_all = "% of all"
  col_masked = "Masked?"

  outlier_workers_count = get_workers_masked_ratings_count(ref_rmask.mask)
  total_count = np.sum(outlier_workers_count)

  lines = []
  for w_i, worker in enumerate(workers):
    lines.append(OrderedDict((
      (col_worker, worker),
      (col_outliers, outlier_workers_count[w_i]),
      (col_ratings, np.sum(~np.isnan(ratings[:, w_i, :]))),
      (col_percent, outlier_workers_count[w_i] /
       np.sum(~np.isnan(ratings[:, w_i, :])) * 100),
      (col_percent_all, outlier_workers_count[w_i] / total_count * 100),
      (col_masked, w_i in outlier_indices),
    )))

  result = pd.DataFrame.from_records(lines)
  row = {
    col_worker: "-ALL-",
    col_outliers: result[col_outliers].sum(),
    col_ratings: result[col_ratings].sum(),
    col_percent: result[col_percent].mean(),
    col_percent_all: result[col_percent_all].sum(),
    col_masked: result[col_masked].all(),
  }
  result.sort_values(by=[col_percent_all, col_worker], inplace=True)
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_wmask_percent(rmask: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  sums2 = get_workers_masked_ratings_count(rmask)
  total = np.sum(rmask)

  outlying_workers: np.ndarray = (total > 0) & mask_values_in_boundary(
    sums2, from_percent_incl * total, to_percent_excl * total)
  return outlying_workers


def get_workers_masked_ratings_count(rmask: np.ndarray) -> np.ndarray:
  sums = np.sum(rmask, axis=2)
  sums2 = np.sum(sums, axis=0)
  return sums2
