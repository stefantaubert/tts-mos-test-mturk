import math
from typing import Literal, Set

import numpy as np

from tts_mos_test_mturk.analyze_assignmens import get_mos_correlations
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_correlation(data: EvaluationData, mask_names: Set[str], threshold: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  opinion_scores = data.get_os()
  opinion_scores_mask = factory.merge_masks_into_omask(masks)
  opinion_scores_mask.apply_by_nan(opinion_scores)

  wcorrelations = get_mos_correlations(opinion_scores, mode)

  bad_worker_np_mask = wcorrelations < threshold
  bad_worker_mask = factory.convert_ndarray_to_wmask(bad_worker_np_mask)
  data.add_or_update_mask(output_mask_name, bad_worker_mask)

  print_stats_masks(data, masks, [bad_worker_mask])


def ignore_bad_workers_percent(data: EvaluationData, mask_names: Set[str], from_percent_incl: float, to_percent_excl: float, mode: Literal["sentence", "algorithm", "both"], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  wcorrelations = get_mos_correlations(os, mode)

  windices = np.array(range(data.n_workers))
  wmask = factory.merge_masks_into_wmask(masks)
  sub_wcorrelations = wmask.apply_by_del(wcorrelations)
  sub_windices = wmask.apply_by_del(windices)

  # n_workers2 = np.sum(~np.isnan(worker_correlations))
  # n_workers = worker_mask.n_unmasked
  sub_n_workers = len(sub_windices)

  sub_sorted_indices = np.argsort(sub_wcorrelations)
  sub_sorted_indices = np.array(list(sub_sorted_indices))
  # correlations_sorted = sub_worker_correlations[sub_sorted_indices]
  sub_windices_sorted = sub_windices[sub_sorted_indices]
  # workers_sorted = data.workers[sub_worker_indices_sorted]

  res_wmask = factory.get_wmask()
  # TODO check is inclusive and exclusive
  from_position = math.ceil(sub_n_workers * from_percent_incl)
  to_position = math.ceil(sub_n_workers * to_percent_excl)
  sub_sel_windices = sub_windices_sorted[from_position:to_position]

  # workers_sorted_2 = data.workers[sub_sel_worker_indices]
  res_wmask.mask_indices(sub_sel_windices)
  res_wmask.combine_mask(wmask)

  data.add_or_update_mask(output_mask_name, res_wmask)

  print_stats_masks(data, masks, [res_wmask])
