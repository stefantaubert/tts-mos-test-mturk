
from typing import List

import numpy as np
from ordered_set import OrderedSet

from tts_mos_test_mturk.data_point import DataPoint
from tts_mos_test_mturk.masking.masks import AssignmentsMask, MaskBase, RatingsMask, WorkersMask


class MaskFactory():
  def __init__(self, algorithms: OrderedSet[str], workers: OrderedSet[str], files: OrderedSet[str], assignments: OrderedSet[str], data_points: List[DataPoint]) -> None:
    self.algorithms = algorithms
    self.workers = workers
    self.files = files
    self.assignments = assignments
    self.data_points = data_points

  def get_rmask(self) -> RatingsMask:
    mask = np.full(
      (len(self.algorithms), len(self.workers), len(self.files)),
      fill_value=False,
      dtype=bool,
    )
    return RatingsMask(mask)

  def get_amask(self) -> AssignmentsMask:
    mask = np.full(
      len(self.assignments),
      fill_value=False,
      dtype=bool,
    )
    return AssignmentsMask(mask)

  def get_wmask(self) -> WorkersMask:
    mask = np.full(
      len(self.workers),
      fill_value=False,
      dtype=bool,
    )
    return WorkersMask(mask)

  def convert_ndarray_to_rmask(self, array: np.ndarray) -> RatingsMask:
    if array.shape != (len(self.algorithms), len(self.workers), len(self.files)):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return RatingsMask(array)

  def convert_ndarray_to_amask(self, array: np.ndarray) -> AssignmentsMask:
    if array.shape != (len(self.assignments),):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return AssignmentsMask(array)

  def convert_ndarray_to_wmask(self, array: np.ndarray) -> WorkersMask:
    if array.shape != (len(self.workers),):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return WorkersMask(array)

  def convert_mask_to_rmask(self, mask: MaskBase) -> RatingsMask:
    if isinstance(mask, WorkersMask):
      mask = self.convert_mask_to_amask(mask)
    if isinstance(mask, AssignmentsMask):
      mask = self.convert_amask_to_rmask(mask)
    assert isinstance(mask, RatingsMask)
    return mask

  def merge_masks_into_rmask(self, masks: List[MaskBase]) -> RatingsMask:
    result = self.get_rmask()
    for mask in masks:
      mask = self.convert_mask_to_rmask(mask)
      result.combine_mask(mask)
    return result

  def get_ratings_assignments_index_matrix(self) -> np.ndarray:
    res = np.full(
      (len(self.algorithms), len(self.workers), len(self.files)),
      fill_value=np.nan,
      dtype=np.float32,
    )

    for data_point in self.data_points:
      alg_i = self.algorithms.get_loc(data_point.algorithm)
      worker_i = self.workers.get_loc(data_point.worker_id)
      file_i = self.files.get_loc(data_point.file)
      ass_i = self.assignments.get_loc(data_point.assignment_id)
      res[alg_i, worker_i, file_i] = ass_i
    return res

  def convert_amask_to_rmask(self, amask: AssignmentsMask) -> RatingsMask:
    ratings_assignments_index_matrix = self.get_ratings_assignments_index_matrix()
    new_mask = np.isin(ratings_assignments_index_matrix, amask.masked_indices)
    result = RatingsMask(new_mask)
    return result

  def convert_mask_to_amask(self, mask: MaskBase) -> AssignmentsMask:
    if isinstance(mask, RatingsMask):
      raise ValueError("RatingsMasks can't be converted to AssignmentsMasks!")
    if isinstance(mask, WorkersMask):
      mask = self.convert_wmask_to_amask(mask)
    assert isinstance(mask, AssignmentsMask)
    return mask

  def merge_masks_into_amask(self, masks: List[MaskBase]) -> AssignmentsMask:
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

    for data_point in self.data_points:
      ass_i = self.assignments.get_loc(data_point.assignment_id)
      worker_i = self.workers.get_loc(data_point.worker_id)
      res[ass_i] = worker_i
    return res

  def convert_wmask_to_amask(self, wmask: WorkersMask) -> AssignmentsMask:
    result = self.get_amask()
    assignments_worker_index_matrix = self.get_assignments_worker_index_matrix()
    result.mask = np.isin(assignments_worker_index_matrix, wmask.masked_indices)
    return result

  def convert_mask_to_wmask(self, mask: MaskBase) -> WorkersMask:
    if isinstance(mask, RatingsMask):
      raise ValueError("RatingsMasks can't be converted to WorkersMasks!")
    if isinstance(mask, AssignmentsMask):
      raise ValueError("AssignmentsMasks can't be converted to WorkersMasks!")
    assert isinstance(mask, WorkersMask)
    return mask

  def merge_masks_into_wmask(self, masks: List[WorkersMask]) -> WorkersMask:
    result = self.get_wmask()
    for mask in masks:
      try:
        mask = self.convert_mask_to_wmask(mask)
      except ValueError:
        continue
      result.combine_mask(mask)
    return result
