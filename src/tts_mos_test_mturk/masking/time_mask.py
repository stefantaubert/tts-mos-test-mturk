from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.etc import mask_values_in_boundary
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_time(data: EvaluationData, mask_names: Set[str], from_threshold_incl: datetime, to_threshold_excl: datetime, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)

  times = get_times(data)
  amask.apply_by_nan(times)

  from_ticks = get_ticks(from_threshold_incl)
  to_ticks = get_ticks(to_threshold_excl)
  too_fast_amask_np = mask_values_in_boundary(times, from_ticks, to_ticks)
  too_fast_amask = factory.convert_ndarray_to_amask(too_fast_amask_np)

  times_str = get_times_str(data)
  for idx in amask.masked_indices:
    times_str[idx] = None
  stats_df = get_stats_df(data.assignments, times_str, too_fast_amask.masked_indices)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, too_fast_amask)

  print_stats_masks(data, masks, [too_fast_amask])


def get_stats_df(assignments: OrderedSet[str], times: List[Optional[datetime]], masked_indices: np.ndarray) -> pd.DataFrame:
  col_assignment = "AssignmentId"
  col_time = "Time"
  col_masked = "Masked?"
  lines = []
  for a_i, time in zip(range(len(assignments)), times):
    if time is None:
      continue
    lines.append(OrderedDict((
      (col_assignment, assignments[a_i]),
      (col_time, time.strftime("%Y-%m-%d %H:%M:%S")),
      (col_masked, a_i in masked_indices),
    )))

  result = pd.DataFrame.from_records(lines)
  row = {
    col_assignment: "-ALL-",
    col_time: "-",
    col_masked: result[col_masked].all(),
  }
  result.sort_values(by=[col_time, col_assignment], inplace=True)
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_ticks(dt: datetime) -> float:
  result = (dt - datetime(1, 1, 1)).total_seconds()
  return result


def get_times(data: EvaluationData) -> np.ndarray:
  times = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float64,
  )

  for worker_data in data.worker_data.values():
    for assignment, assignment_data in worker_data.assignments.items():
      ass_i = data.assignments.get_loc(assignment)
      times[ass_i] = get_ticks(assignment_data.time)

  return times


def get_times_str(data: EvaluationData) -> List[datetime]:
  times = [None] * data.n_assignments

  for worker_data in data.worker_data.values():
    for assignment, assignment_data in worker_data.assignments.items():
      ass_i = data.assignments.get_loc(assignment)
      times[ass_i] = assignment_data.time

  return times
