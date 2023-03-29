from collections import OrderedDict
from typing import Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import log_full_df_info
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_listening_device(data: EvaluationData, mask_names: Set[str], listening_devices: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  lts = get_listening_devices(data)
  amask = factory.merge_masks_into_amask(masks)
  amask.apply_by_nan(lts)

  res_amask = get_listening_devices_amask(lts, listening_devices, factory)

  stats_df = get_stats_df(lts, listening_devices)
  log_full_df_info(stats_df, "Statistics:")

  data.add_or_update_mask(output_mask_name, res_amask)

  print_stats_masks(data, masks, [res_amask])


def get_stats_df(lts: np.ndarray, mask_devices: Set[str]) -> pd.DataFrame:
  col_device = "Device"
  col_count = "# Assignments"
  col_masked = "Masked?"

  listening_devices, ld_counts = np.unique(lts, return_counts=True)
  lines = []
  for device, count in (zip(listening_devices, ld_counts)):
    if device == "nan":
      continue
    lines.append(OrderedDict((
      (col_device, device),
      (col_count, count),
      (col_masked, device in mask_devices),
    )))

  result = pd.DataFrame.from_records(lines)
  result.sort_values(by=[col_device], inplace=True)
  row = {
    col_device: "-ALL-",
    col_count: result[col_count].sum(),
    col_masked: result[col_masked].all(),
  }
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def get_listening_devices_amask(listening_devices: np.ndarray, mask_listening_devices: Set[str], factory: MaskFactory) -> None:
  res_amask = factory.get_amask()
  for device in mask_listening_devices:
    lt_mask = listening_devices == device
    res_amask.combine_mask_np(lt_mask)
  return res_amask


def get_listening_devices(data: EvaluationData) -> np.ndarray:
  devices = [np.nan] * data.n_assignments
  for worker_data in data.worker_data.values():
    for assignment, assignment_data in worker_data.assignments.items():
      ass_i = data.assignments.get_loc(assignment)
      devices[ass_i] = assignment_data.device
  devices_np = np.array(devices)
  return devices_np
