import math
from typing import Literal, Set

import numpy as np

from tts_mos_test_mturk.calculation.correlations import get_mos_correlations
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_correlation(data: EvaluationData, mask_names: Set[str], threshold: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  ratings = data.get_ratings()
  ratings_mask = factory.merge_masks_into_rmask(masks)
  ratings_mask.apply_by_nan(ratings)

  wcorrelations = get_mos_correlations(ratings, mode)

  bad_worker_np_mask = wcorrelations < threshold
  bad_worker_mask = factory.convert_ndarray_to_wmask(bad_worker_np_mask)
  data.add_or_update_mask(output_mask_name, bad_worker_mask)

  print_stats_masks(data, masks, [bad_worker_mask])


def mask_workers_by_correlation_percent(data: EvaluationData, mask_names: Set[str], from_percent_incl: float, to_percent_excl: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  ratings = data.get_ratings()
  rmask = factory.merge_masks_into_rmask(masks)
  rmask.apply_by_nan(ratings)

  wcorrelations = get_mos_correlations(ratings, mode)

  windices = np.array(range(data.n_workers))
  wmask = factory.merge_masks_into_wmask(masks)
  sub_wcorrelations = wmask.apply_by_del(wcorrelations)
  sub_windices = wmask.apply_by_del(windices)

  sub_sel_windices = get_indices(sub_windices, sub_wcorrelations,
                                 from_percent_incl, to_percent_excl)

  # workers_sorted_2 = data.workers[sub_sel_worker_indices]
  res_wmask = factory.get_wmask()
  res_wmask.mask_indices(sub_sel_windices)
  res_wmask.combine_mask(wmask)

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])


def get_indices(sub_windices: np.ndarray, sub_wcorrelations: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  sub_windices_sorted = sort_indices_after_correlations(sub_windices, sub_wcorrelations)
  sub_sel_windices = get_range_percent(sub_windices_sorted, from_percent_incl, to_percent_excl)
  return sub_sel_windices


def sort_indices_after_correlations(sub_windices: np.ndarray, sub_wcorrelations: np.ndarray) -> np.ndarray:
  sub_sorted_indices = np.argsort(sub_wcorrelations)
  sub_sorted_indices = np.array(list(sub_sorted_indices))
  # correlations_sorted = sub_worker_correlations[sub_sorted_indices]
  sub_windices_sorted = sub_windices[sub_sorted_indices]
  # workers_sorted = data.workers[sub_worker_indices_sorted]
  return sub_windices_sorted


def get_range_percent(vec: np.ndarray, from_percent_incl: float, to_percent_excl: float) -> np.ndarray:
  to_percent_excl = max(to_percent_excl, from_percent_incl)
  vec_len = len(vec)
  from_position = math.ceil(vec_len * from_percent_incl)
  to_position = math.ceil(vec_len * to_percent_excl) - 1
  if to_position == 0 and len(vec) > 0:
    to_position = 1
  sub_sel_windices = vec[from_position:to_position]
  return sub_sel_windices
