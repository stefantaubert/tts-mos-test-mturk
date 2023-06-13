from collections import OrderedDict
from pathlib import Path
from typing import List, Optional
from typing import OrderedDict as ODType
from typing import Set, cast

from ordered_set import OrderedSet

from tts_mos_test_mturk.io import load_obj, save_obj
from tts_mos_test_mturk.masking.masks import MaskBase, get_mask_name_and_reverse
from tts_mos_test_mturk.result import Result, Worker
from tts_mos_test_mturk.typing import MaskName


class EvaluationData():
  def __init__(self, result: Result):
    self.__result = result
    self.workers = OrderedSet(result.workers.keys())
    self.assignments = OrderedSet(
      assignment
      for worker in result.workers.values()
      for assignment in worker.assignments.keys()
    )
    self.n_ratings: int = sum(
      1
      for worker in result.workers.values()
      for assignment in worker.assignments.values()
      for _ in assignment.ratings.keys()
    )
    self.rating_names = OrderedSet(
      rating_name
      for worker in result.workers.values()
      for assignment in worker.assignments.values()
      for rating_data in assignment.ratings.values()
      for rating_name in rating_data.votes.keys()
    )

    self.masks: ODType[str, MaskBase] = OrderedDict()
    self.file_path: Optional[Path] = None

  @property
  def algorithms(self) -> OrderedSet[str]:
    return self.__result.algorithms

  @property
  def files(self) -> OrderedSet[str]:
    return self.__result.files

  @property
  def worker_data(self) -> ODType[str, Worker]:
    return self.__result.workers

  @classmethod
  def load(cls, path: Path):
    result = cast(EvaluationData, load_obj(path))
    result.file_path = path
    return result

  def save_to(self, path: Path) -> None:
    save_obj(self, path)

  def save(self) -> None:
    if self.file_path is None:
      raise ValueError("Project needs to be loaded from file before!")
    self.save_to(self.file_path)

  @property
  def n_assignments(self) -> int:
    return len(self.assignments)

  @property
  def n_algorithms(self) -> int:
    return len(self.algorithms)

  @property
  def n_workers(self) -> int:
    return len(self.workers)

  @property
  def n_files(self) -> int:
    return len(self.files)

  def get_mask(self, mask_name: MaskName) -> MaskBase:
    mask_name, reverse = get_mask_name_and_reverse(mask_name)
    if mask_name not in self.masks:
      raise ValueError(f"Mask \"{mask_name}\" doesn't exist!")
    mask = self.masks[mask_name]
    if reverse:
      new_mask = mask.clone()
      new_mask.reverse()
      return new_mask
    return mask

  def get_masks_from_names(self, mask_names: Set[MaskName]) -> List[MaskBase]:
    masks = [self.get_mask(mask_name) for mask_name in mask_names]
    return masks

  def get_mask_old(self, mask_name: MaskName) -> MaskBase:
    if mask_name not in self.masks:
      raise ValueError(f"Mask \"{mask_name}\" doesn't exist!")
    return self.masks[mask_name]

  def get_masks_from_names_old(self, mask_names: Set[MaskName]) -> List[MaskBase]:
    masks = [self.get_mask_old(mask_name) for mask_name in mask_names]
    return masks

  def add_or_update_mask(self, name: MaskName, mask: MaskBase) -> None:
    assert name is not None
    self.masks[name] = mask
