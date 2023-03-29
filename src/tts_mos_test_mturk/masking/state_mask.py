from collections import OrderedDict
from typing import Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_state(data: EvaluationData, mask_names: Set[str], states: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  lts = get_states(data)
  amask = factory.merge_masks_into_amask(masks)
  amask.apply_by_nan(lts)

  res_amask = get_states_amask(lts, states, factory)

  stats_df = get_stats_df(lts, states)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, res_amask)

  print_stats_masks(data, masks, [res_amask])


def get_stats_df(lts: np.ndarray, mask_states: Set[str]) -> pd.DataFrame:
  col_state = "State"
  col_count = "# Assignments"
  col_masked = "Masked?"

  states, ld_counts = np.unique(lts, return_counts=True)
  lines = []
  for state, count in (zip(states, ld_counts)):
    if state == "nan":
      continue
    lines.append(OrderedDict((
      (col_state, state),
      (col_count, count),
      (col_masked, state in mask_states),
    )))

  result = pd.DataFrame.from_records(lines)
  result.sort_values(by=[col_state], inplace=True)
  row = {
    col_state: "-ALL-",
    col_count: result[col_count].sum(),
    col_masked: result[col_masked].all(),
  }
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_states_amask(states: np.ndarray, mask_states: Set[str], factory: MaskFactory) -> None:
  res_amask = factory.get_amask()
  for state in mask_states:
    lt_mask = states == state
    res_amask.combine_mask_np(lt_mask)
  return res_amask


def get_states(data: EvaluationData) -> np.ndarray:
  states = [np.nan] * data.n_assignments
  for worker_data in data.worker_data.values():
    for assignment, assignment_data in worker_data.assignments.items():
      ass_i = data.assignments.get_loc(assignment)
      states[ass_i] = assignment_data.state
  states_np = np.array(states)
  return states_np
