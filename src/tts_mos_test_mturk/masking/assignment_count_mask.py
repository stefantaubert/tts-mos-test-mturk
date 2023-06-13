from collections import OrderedDict
from typing import Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_assignment_count(data: EvaluationData, mask_names: Set[str], from_threshold_incl: int, to_threshold_excl: int, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)

  aw_matrix = factory.get_assignments_worker_index_matrix()
  aw_matrix = amask.apply_by_del(aw_matrix)

  windices, a_counts = np.unique(aw_matrix, return_counts=True)
  matching_a_counts = mask_values_in_boundary(a_counts, from_threshold_incl, to_threshold_excl)
  masked_indices = matching_a_counts.nonzero()[0]
  matching_w = windices[masked_indices]

  stats_df = get_stats_df(data.workers, windices, a_counts, matching_w)
  log_full_df_info(stats_df, "Statistics:")

  wmask = factory.get_wmask()
  wmask.mask_indices(matching_w)

  data.add_or_update_mask(output_mask_name, wmask)

  print_stats_masks(data, masks, [wmask])


def get_stats_df(workers: OrderedSet[str], windices: np.ndarray, a_counts: np.ndarray, matching_w: np.ndarray) -> pd.DataFrame:
  col_worker = "Worker"
  col_count = "# Assignments"
  col_masked = "Masked?"
  lines = []
  for w_i, count in zip(windices, a_counts):
    lines.append(OrderedDict((
      (col_worker, workers[w_i]),
      (col_count, count),
      (col_masked, w_i in matching_w),
    )))

  result = pd.DataFrame.from_records(lines)
  row = {
    col_worker: "-ALL-",
    col_count: result[col_count].sum(),
    col_masked: result[col_masked].all(),
  }
  result.sort_values(by=[col_count, col_worker], inplace=True)
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result
