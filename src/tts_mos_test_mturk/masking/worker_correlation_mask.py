import math
from collections import OrderedDict
from typing import Literal, Set, Tuple

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.correlations import (get_algorithm_mos_correlations, get_mos_correlations,
                                             get_sentence_mos_correlations_3dim,
                                             get_worker_mos_correlations)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary, sort_indices_after_values
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_correlation(data: EvaluationData, mask_names: Set[str], from_threshold_incl: float, to_threshold_excl: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str, rating_names: Set[str], mask_nan: bool):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  # Note: it is not clever to merge all ratings to one
  ratings = get_ratings(data, rating_names)
  rmask.apply_by_nan(ratings)

  wcorrelations = get_mos_correlations(ratings, mode)
  res_wmask_np = mask_values_in_boundary(
    wcorrelations, from_threshold_incl, to_threshold_excl)
  if mask_nan:
    unmasked_nan_correlation = np.logical_xor(wmask.mask, np.isnan(wcorrelations))
    res_wmask_np = (res_wmask_np | unmasked_nan_correlation)
  res_wmask = factory.convert_ndarray_to_wmask(res_wmask_np)

  stats_df = get_stats_df(data.workers, ratings, res_wmask.masked_indices,
                          wcorrelations, wmask.masked_indices, mode, False)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])


def mask_workers_by_correlation_percent(data: EvaluationData, mask_names: Set[str], from_percent_incl: float, to_percent_excl: float, mode: Literal["sentence", "algorithm", "both"], consider_masked_workers: bool, output_mask_name: str, rating_names: Set[str]):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  # Note: it is not clever to merge all ratings to one
  ratings = get_ratings(data, rating_names)
  rmask.apply_by_nan(ratings)

  windices = np.arange(data.n_workers)
  wcorrelations = get_mos_correlations(ratings, mode)

  sub_wcorrelations = wcorrelations
  sub_windices = windices

  if consider_masked_workers:
    wmask.apply_by_nan(sub_wcorrelations)
    sub_wcorrelations[np.isnan(sub_wcorrelations)] = -2.0
  else:
    sub_wcorrelations = wmask.apply_by_del(sub_wcorrelations)
    sub_windices = wmask.apply_by_del(sub_windices)

  sub_sel_windices = get_indices(sub_windices, sub_wcorrelations,
                                 from_percent_incl, to_percent_excl)

  res_wmask = factory.get_wmask()
  res_wmask.mask_indices(sub_sel_windices)

  stats_df = get_stats_df(data.workers, ratings, res_wmask.masked_indices,
                          wcorrelations, wmask.masked_indices, mode, consider_masked_workers)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])


def get_stats_df(workers: OrderedSet[str], ratings: np.ndarray, masked_indices: np.ndarray, used_correlations: np.ndarray, already_masked_worker_indices: np.ndarray, mode: Literal["sentence", "algorithm", "both"], print_masked_workers: bool) -> pd.DataFrame:
  col_worker = "Worker"
  col_ratings = "# Ratings"
  col_sent_corr = "Sentence correlation"
  col_alg_corr = "Algorithm correlation"
  col_both_corr = "Both correlation"
  col_percent = "Ranking %"
  col_masked = "Masked?"
  col_used = "Used (tmp)"
  col_w_i = "w_i (tmp)"

  if mode == "sentence":
    col_sent_corr = f"[{col_sent_corr}]"
  elif mode == "algorithm":
    col_alg_corr = f"[{col_alg_corr}]"
  else:
    assert mode == "both"
    col_both_corr = f"[{col_both_corr}]"

  w_sent_corr = get_sentence_mos_correlations_3dim(ratings)
  w_algo_corr = get_algorithm_mos_correlations(ratings)
  w_both_corr = get_worker_mos_correlations(ratings)

  lines = []
  for w_i, worker in enumerate(workers):
    if w_i in already_masked_worker_indices and not print_masked_workers:
      continue
    lines.append(OrderedDict((
      (col_worker, worker),
      (col_ratings, np.sum(~np.isnan(ratings[:, w_i, :]))),
      (col_sent_corr, w_sent_corr[w_i]),
      (col_alg_corr, w_algo_corr[w_i]),
      (col_both_corr, w_both_corr[w_i]),
      (col_percent, 0),
      (col_masked, w_i in masked_indices),
      (col_w_i, w_i),
      (col_used, used_correlations[w_i]),
    )))

  result = pd.DataFrame.from_records(lines)

  if len(result.index) > 0:
    result.sort_values(by=[col_used, col_w_i], inplace=True)
    for nr, (i, row) in enumerate(result.iterrows(), start=1):
      result.at[i, col_percent] = nr / len(result.index) * 100

    result.drop(columns=[col_w_i, col_used], inplace=True)

    row = {
      col_worker: "-ALL-",
      col_ratings: result[col_ratings].sum(),
      col_sent_corr: result[col_sent_corr].mean(),
      col_alg_corr: result[col_alg_corr].mean(),
      col_both_corr: result[col_both_corr].mean(),
      col_percent: np.nan,
      col_masked: result[col_masked].all(),
    }
    result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_indices(sub_windices: np.ndarray, sub_wcorrelations: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  sub_windices_sorted = sort_indices_after_values(sub_windices, sub_wcorrelations)
  sub_sel_windices = get_range_percent(sub_windices_sorted, from_percent_incl, to_percent_excl)
  return sub_sel_windices


def get_range_percent(vec: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  if len(vec) == 0:
    return vec[0:0]
  a, b = get_range_start_end_percent(len(vec), from_percent_incl, to_percent_excl)
  if a == b:
    sub_sel_windices = [vec[a]]
  else:
    assert a < b
    sub_sel_windices = vec[a:b]
  return sub_sel_windices


def get_range_start_end_percent(n: int, start: float, end: float) -> Tuple[int, int]:
  end = max(start, end)

  a = get_range_start_incl_percent(n, start)
  b = get_range_end_excl_percent(n, end)
  return a, b


def get_range_start_incl_percent(n: int, p: float) -> int:
  p = max(p, 0)
  p = min(p, 1)
  if n == 0:
    raise ValueError("Argument 'n': Value needs to be greater than zero!")
  if p == 1.0:
    raise ValueError("Argument 'p': Value needs to be < 1")
  from_position = math.ceil(n * p)
  if from_position == 0:
    return 0
  return from_position - 1


def get_range_end_excl_percent(n: int, p: float) -> int:
  p = max(p, 0)
  p = min(p, 1)
  if n == 0:
    raise ValueError("Argument 'n': Value needs to be greater than zero!")
  if p == 0:
    raise ValueError("Argument 'p': Value needs to be > 0!")
  to = math.ceil(n * p)
  if to == 1:
    # raise ValueError(f"Argument 'p': needs to be greater than {1/n}!")
    return 1
  if to == n and p == 1.0:
    return n
  return to - 1
