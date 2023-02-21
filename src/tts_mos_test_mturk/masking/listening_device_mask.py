from collections import OrderedDict
from typing import Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def get_stats_df(lts: np.ndarray) -> pd.DataFrame:
  listening_devices, ld_counts = np.unique(lts, return_counts=True)
  lines = []
  for d, c in (zip(listening_devices, ld_counts)):
    if d == "nan":
      continue
    lines.append(OrderedDict((
      ("Device", d),
      ("Count", c),
    )))

  result = pd.DataFrame.from_records(lines)
  row = {
    "Device": "All",
    "Count": result["Count"].sum(),
  }
  result = pd.concat([result, pd.DataFrame.from_records([row])], ignore_index=True)
  return result


def mask_assignments_by_listening_device(data: EvaluationData, mask_names: Set[str], listening_devices: Set[str], output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  lts = get_listening_devices(data)
  amask = factory.merge_masks_into_amask(masks)
  amask.apply_by_nan(lts)

  dlogger.info(f"Statistics:\n{get_stats_df(lts)}")
  res_amask = get_listening_devices_amask(lts, listening_devices, factory)
  data.add_or_update_mask(output_mask_name, res_amask)

  print_stats_masks(data, masks, [res_amask])


def get_listening_devices_amask(listening_devices: np.ndarray, mask_listening_devices: Set[str], factory: MaskFactory) -> None:
  res_amask = factory.get_amask()
  for device in mask_listening_devices:
    lt_mask = listening_devices == device
    res_amask.combine_mask_np(lt_mask)
  return res_amask


def get_listening_devices(data: EvaluationData) -> np.ndarray:
  worktimes = [np.nan] * data.n_assignments

  for data_point in data.data:
    ass_i = data.assignments.get_loc(data_point.assignment_id)
    if worktimes[ass_i] != data_point.listening_device:
      worktimes[ass_i] = data_point.listening_device
  worktimes_np = np.array(worktimes)
  return worktimes_np
