import math
from typing import Literal, Set, Tuple

import numpy as np

from tts_mos_test_mturk.calculation.correlations import get_mos_correlations
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.masking.etc import sort_indices_after_values
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_correlation(data: EvaluationData, mask_names: Set[str], from_threshold_incl: float, to_threshold_excl: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  rmask = factory.merge_masks_into_rmask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  ratings = data.get_ratings()
  rmask.apply_by_nan(ratings)

  if wmask.n_masked > 0:
    dlogger.info('Already masked workers:')
    for worker_name in sorted(data.workers[wmask.masked_indices]):
      dlogger.info(f'- "{worker_name}"')

  wcorrelations = get_mos_correlations(ratings, mode)

  windices = np.arange(data.n_workers)
  windices_sorted = sort_indices_after_values(windices, wcorrelations)

  bad_worker_np_mask = (from_threshold_incl <= wcorrelations) & (wcorrelations < to_threshold_excl)
  bad_worker_mask = factory.convert_ndarray_to_wmask(bad_worker_np_mask)
  masked_indices = bad_worker_mask.masked_indices

  dlogger.info("Worker ranking:")
  for nr, w_i in enumerate(windices_sorted, start=1):
    if w_i in wmask.masked_indices:
      break
    masked_str = " [masked]" if w_i in masked_indices else ""
    dlogger.info(
      f"{nr}. \"{data.workers[w_i]}\": ({wcorrelations[w_i]}){masked_str}")

  data.add_or_update_mask(output_mask_name, bad_worker_mask)

  print_stats_masks(data, masks, [bad_worker_mask])


def mask_workers_by_correlation_percent(data: EvaluationData, mask_names: Set[str], from_percent_incl: float, to_percent_excl: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  rmask = factory.merge_masks_into_rmask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  ratings = data.get_ratings()
  rmask.apply_by_nan(ratings)

  wcorrelations = get_mos_correlations(ratings, mode)
  sub_wcorrelations = wmask.apply_by_del(wcorrelations)

  dlogger.info('Already masked workers:')
  for worker_name in sorted(data.workers[wmask.masked_indices]):
    dlogger.info(f'- "{worker_name}"')

  windices = np.arange(data.n_workers)
  sub_windices = wmask.apply_by_del(windices)

  sub_windices_sorted = sort_indices_after_values(sub_windices, sub_wcorrelations)
  sub_sel_windices = get_range_percent(sub_windices_sorted, from_percent_incl, to_percent_excl)

  # sub_sel_windices = get_indices(sub_windices, sub_wcorrelations,
  #                                from_percent_incl, to_percent_excl)

  dlogger.info("Worker ranking:")
  for nr, w_i in enumerate(sub_windices_sorted, start=1):
    masked_str = " [masked]" if w_i in sub_sel_windices else ""
    dlogger.info(
      f"{nr}. \"{data.workers[w_i]}\": {nr/len(sub_windices_sorted)*100:.2f}% ({wcorrelations[w_i]}){masked_str}")

  # workers_sorted_2 = data.workers[sub_sel_worker_indices]
  res_wmask = factory.get_wmask()
  res_wmask.mask_indices(sub_sel_windices)
  res_wmask.combine_mask(wmask)

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])


def get_indices(sub_windices: np.ndarray, sub_wcorrelations: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  sub_windices_sorted = sort_indices_after_values(sub_windices, sub_wcorrelations)
  sub_sel_windices = get_range_percent(sub_windices_sorted, from_percent_incl, to_percent_excl)
  return sub_sel_windices


# def get_range_percent(vec: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
#   to_percent_excl = max(to_percent_excl, from_percent_incl)
#   vec_len = len(vec)
#   from_position = math.floor(vec_len * from_percent_incl)
#   to_position = math.ceil(vec_len * to_percent_excl) - 1
#   if from_percent_incl == 0:
#     if to_position == 0 and len(vec) > 0:
#       to_position = 1
#   else:
#     if from_position == 0 and len(vec) > 0:
#       from_position = 1
#   sub_sel_windices = vec[from_position:to_position]
#   return sub_sel_windices


# def get_range_percent_np(vec: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
#   if len(vec) == 0:
#     return vec
#   a = np.quantile(vec, from_percent_incl, method="lower")
#   b = np.quantile(vec, to_percent_excl, method="lower")
#   res = vec[(vec >= a) & (vec <= b)]
#   return res


def get_range_percent(vec: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  a, b = get_range_start_end_percent(len(vec), from_percent_incl, to_percent_excl)
  # if from_percent_incl == 0:
  #   if to_position == 0 and len(vec) > 0:
  #     to_position = 1
  # else:
  #   if from_position == 0 and len(vec) > 0:
  #     from_position = 1
  sub_sel_windices = vec[a:b]
  return sub_sel_windices


def get_range_start_end_percent(n: int, start: float, end: float) -> Tuple[int, int]:
  end = max(start, end)

  a = get_range_start_incl_percent(n, start)
  b = get_range_end_excl_percent(n, end)
  return a, b


def get_range_start_incl_percent(n: int, p: float) -> int:
  p = max(p, 0.0)
  from_position = math.floor(n * p)
  return from_position


def get_range_end_excl_percent(n: int, p: float) -> int:
  p = min(1.0, p)
  if p == 1.0:
    to_position = n
  else:
    to_position = math.ceil(n * p) - 1
  return to_position
