
from typing import List

import numpy as np
from ordered_set import OrderedSet

from tts_mos_test_mturk.data_point import DataPoint
from tts_mos_test_mturk.masking.masks import AssignmentMask, MaskBase, OpinionScoreMask, WorkerMask


class MaskFactory():
  def __init__(self, algorithms: OrderedSet[str], workers: OrderedSet[str], files: OrderedSet[str], assignments: OrderedSet[str], data_points: List[DataPoint]) -> None:
    self.algorithms = algorithms
    self.workers = workers
    self.files = files
    self.assignments = assignments
    self.data_points = data_points

  def get_omask(self) -> OpinionScoreMask:
    mask = np.full(
      (len(self.algorithms), len(self.workers), len(self.files)),
      fill_value=False,
      dtype=bool,
    )
    return OpinionScoreMask(mask)

  def get_amask(self) -> AssignmentMask:
    mask = np.full(
      len(self.assignments),
      fill_value=False,
      dtype=bool,
    )
    return AssignmentMask(mask)

  def get_wmask(self) -> WorkerMask:
    mask = np.full(
      len(self.workers),
      fill_value=False,
      dtype=bool,
    )
    return WorkerMask(mask)

  def convert_ndarray_to_omask(self, array: np.ndarray) -> OpinionScoreMask:
    if array.shape != (len(self.algorithms), len(self.workers), len(self.files)):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return OpinionScoreMask(array)

  def convert_ndarray_to_amask(self, array: np.ndarray) -> AssignmentMask:
    if array.shape != (len(self.assignments),):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return AssignmentMask(array)

  def convert_ndarray_to_wmask(self, array: np.ndarray) -> WorkerMask:
    if array.shape != (len(self.workers),):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return WorkerMask(array)

  def convert_mask_to_omask(self, mask: MaskBase) -> OpinionScoreMask:
    if isinstance(mask, WorkerMask):
      mask = self.convert_mask_to_amask(mask)
    if isinstance(mask, AssignmentMask):
      mask = self.convert_amask_to_omask(mask)
    assert isinstance(mask, OpinionScoreMask)
    return mask

  def merge_masks_into_omask(self, masks: List[MaskBase]) -> OpinionScoreMask:
    result = self.get_omask()
    for mask in masks:
      mask = self.convert_mask_to_omask(mask)
      result.combine_mask(mask)
    return result

  def get_opinion_scores_assignments_index_matrix(self) -> np.ndarray:
    res = np.full(
      (len(self.algorithms), len(self.workers), len(self.files)),
      fill_value=np.nan,
      dtype=np.float32,
    )

    for dp in self.data_points:
      alg_i = self.algorithms.get_loc(dp.algorithm)
      worker_i = self.workers.get_loc(dp.worker_id)
      file_i = self.files.get_loc(dp.file)
      ass_i = self.assignments.get_loc(dp.assignment_id)
      res[alg_i, worker_i, file_i] = ass_i
    return res

  def convert_amask_to_omask(self, amask: AssignmentMask) -> OpinionScoreMask:
    opinion_scores_assignments_index_matrix = self.get_opinion_scores_assignments_index_matrix()
    new_mask = np.isin(opinion_scores_assignments_index_matrix, amask.masked_indices)
    result = OpinionScoreMask(new_mask)
    return result

  def convert_mask_to_amask(self, mask: MaskBase) -> AssignmentMask:
    if isinstance(mask, OpinionScoreMask):
      raise ValueError("OpinionScoreMasks can't be converted to AssignmentMasks!")
    if isinstance(mask, WorkerMask):
      mask = self.convert_wmask_to_amask(mask)
    assert isinstance(mask, AssignmentMask)
    return mask

  def merge_masks_into_amask(self, masks: List[MaskBase]) -> AssignmentMask:
    result = self.get_amask()
    for mask in masks:
      try:
        mask = self.convert_mask_to_amask(mask)
      except ValueError:
        continue
      result.combine_mask(mask)
    return result

  def get_assignments_worker_index_matrix(self) -> np.ndarray:
    res = np.full(
      len(self.assignments),
      fill_value=-1,
      dtype=np.int32,
    )

    for dp in self.data_points:
      ass_i = self.assignments.get_loc(dp.assignment_id)
      worker_i = self.workers.get_loc(dp.worker_id)
      res[ass_i] = worker_i
    return res

  def convert_wmask_to_amask(self, wmask: WorkerMask) -> AssignmentMask:
    result = self.get_amask()
    assignments_worker_index_matrix = self.get_assignments_worker_index_matrix()
    result.mask = np.isin(assignments_worker_index_matrix, wmask.masked_indices)
    return result

  def convert_mask_to_wmask(self, mask: MaskBase) -> WorkerMask:
    if isinstance(mask, OpinionScoreMask):
      raise ValueError("OpinionScoreMasks can't be converted to WorkerMasks!")
    if isinstance(mask, AssignmentMask):
      raise ValueError("AssignmentMasks can't be converted to WorkerMasks!")
    assert isinstance(mask, WorkerMask)
    return mask

  def merge_masks_into_wmask(self, masks: List[WorkerMask]) -> WorkerMask:
    result = self.get_wmask()
    for mask in masks:
      try:
        mask = self.convert_mask_to_wmask(mask)
      except ValueError:
        continue
      result.combine_mask(mask)
    return result
