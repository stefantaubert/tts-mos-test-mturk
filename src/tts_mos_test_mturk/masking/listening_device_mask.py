
from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_assignments_by_listening_device(data: EvaluationData, mask_names: Set[str], listening_devices: Set[str], output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  lts = get_listening_devices(data)
  amask = factory.merge_masks_into_amask(masks)
  amask.apply_by_nan(lts)

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
  work_times = [np.nan] * data.n_assignments

  for data_point in data.data:
    ass_i = data.assignments.get_loc(data_point.assignment_id)
    if work_times[ass_i] != data_point.listening_device:
      work_times[ass_i] = data_point.listening_device
  work_times_np = np.array(work_times)
  return work_times_np
