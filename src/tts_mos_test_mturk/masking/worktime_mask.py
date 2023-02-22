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


def mask_assignments_by_worktime(data: EvaluationData, mask_names: Set[str], from_threshold_incl: int, to_threshold_excl: int, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)

  worktimes = get_worktimes(data)
  amask.apply_by_nan(worktimes)

  too_fast_amask_np = mask_values_in_boundary(worktimes, from_threshold_incl, to_threshold_excl)
  too_fast_amask = factory.convert_ndarray_to_amask(too_fast_amask_np)

  stats_df = get_stats_df(data.assignments, worktimes, too_fast_amask.masked_indices)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, too_fast_amask)

  print_stats_masks(data, masks, [too_fast_amask])


def get_stats_df(assignments: OrderedSet[str], worktimes: np.ndarray, masked_indices: np.ndarray) -> pd.DataFrame:
  col_assignment = "Assignment"
  col_time = "Worktime (s)"
  col_masked = "Masked?"
  lines = []
  for a_i, worktime in zip(range(len(assignments)), worktimes):
    if np.isnan(worktime):
      continue
    lines.append(OrderedDict((
      (col_assignment, assignments[a_i]),
      (col_time, int(worktime)),
      (col_masked, a_i in masked_indices),
    )))

  result = pd.DataFrame.from_records(lines)
  row = {
    col_assignment: "All",
    col_time: result[col_time].sum(),
    col_masked: result[col_masked].all(),
  }
  result.sort_values(by=[col_time, col_assignment], inplace=True)
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_worktimes(data: EvaluationData) -> np.ndarray:
  worktimes = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float32,
  )

  for worker_data in data.worker_data.values():
    for assignment, assignment_data in worker_data.assignments.items():
      ass_i = data.assignments.get_loc(assignment)
      worktimes[ass_i] = assignment_data.worktime

  return worktimes
