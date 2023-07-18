from typing import List

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.masks import AssignmentsMask, MaskBase, RatingsMask, WorkersMask


class MaskFactory():
  def __init__(self, data: EvaluationData) -> None:
    self.__data = data

  def get_rmask(self) -> RatingsMask:
    mask = np.full(
      (self.__data.n_algorithms, self.__data.n_workers, self.__data.n_files),
      fill_value=False,
      dtype=bool,
    )
    return RatingsMask(mask)

  def get_amask(self) -> AssignmentsMask:
    mask = np.full(
      self.__data.n_assignments,
      fill_value=False,
      dtype=bool,
    )
    return AssignmentsMask(mask)

  def get_wmask(self) -> WorkersMask:
    mask = np.full(
      self.__data.n_workers,
      fill_value=False,
      dtype=bool,
    )
    return WorkersMask(mask)

  def get_wmask_by_gender(self, gender: str):
    res_wmask = self.get_wmask()
    for i, worker_id in enumerate(self.__data.workers):
      worker_data = self.__data.worker_data[worker_id]
      if worker_data.gender != gender:
        res_wmask.mask[i] = True
    return res_wmask

  def get_wmask_by_age_group(self, age_group: str):
    res_wmask = self.get_wmask()
    for i, worker_id in enumerate(self.__data.workers):
      worker_data = self.__data.worker_data[worker_id]
      if worker_data.age_group != age_group:
        res_wmask.mask[i] = True
    return res_wmask

  def get_wmask_by_worker_id(self, worker_id: str):
    res_wmask = self.get_wmask()
    for i, w_id in enumerate(self.__data.workers):
      if worker_id != w_id:
        res_wmask.mask[i] = True
    return res_wmask

  def convert_ndarray_to_rmask(self, array: np.ndarray) -> RatingsMask:
    if array.shape != (self.__data.n_algorithms, self.__data.n_workers, self.__data.n_files):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return RatingsMask(array)

  def convert_ndarray_to_amask(self, array: np.ndarray) -> AssignmentsMask:
    if array.shape != (self.__data.n_assignments,):
      raise ValueError("Invalid format!")
    if array.dtype != bool:
      raise ValueError("Invalid format!")
    return AssignmentsMask(array)

  def convert_ndarray_to_wmask(self, array: np.ndarray) -> WorkersMask:
    if array.shape != (self.__data.n_workers,):
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
      (self.__data.n_algorithms, self.__data.n_workers, self.__data.n_files),
      fill_value=np.nan,
      dtype=np.float32,
    )

    for worker, worker_data in self.__data.worker_data.items():
      worker_i = self.__data.workers.get_loc(worker)
      for assignment, assignment_data in worker_data.assignments.items():
        ass_i = self.__data.assignments.get_loc(assignment)
        for (alg_name, file_name) in assignment_data.ratings.keys():
          alg_i = self.__data.algorithms.get_loc(alg_name)
          file_i = self.__data.files.get_loc(file_name)
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
      self.__data.n_assignments,
      fill_value=-1,
      dtype=np.int32,
    )

    for worker, worker_data in self.__data.worker_data.items():
      worker_i = self.__data.workers.get_loc(worker)
      for assignment in worker_data.assignments.keys():
        ass_i = self.__data.assignments.get_loc(assignment)
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
