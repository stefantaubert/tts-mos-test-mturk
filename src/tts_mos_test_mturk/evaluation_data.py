from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
from typing import OrderedDict as ODType
from typing import Set

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.data_point import DataPoint, get_n_urls_per_assignment, parse_data_points
from tts_mos_test_mturk.io import load_obj, save_obj
from tts_mos_test_mturk.masks import MaskBase, MaskFactory


def get_file_dict_from_df(ground_truth_df: DataFrame) -> Dict[str, str]:
  ground_truth_dict = ground_truth_df.to_dict("index")
  file_dict: Dict[str, str] = {}

  for row in ground_truth_dict.values():
    audio_url = row["audio_url"]
    file_dict[audio_url] = row["file"]
  return file_dict


def get_alg_dict_from_df(ground_truth_df: DataFrame) -> Dict[str, str]:
  ground_truth_dict = ground_truth_df.to_dict("index")
  alg_dict: Dict[str, str] = {}

  for row in ground_truth_dict.values():
    audio_url = row["audio_url"]
    alg_dict[audio_url] = row["algorithm"]
  return alg_dict


class EvaluationData():
  def __init__(self, results_df: DataFrame, ground_truth_df: DataFrame):
    super().__init__()
    results_dict = results_df.to_dict("index")

    alg_dict = get_alg_dict_from_df(ground_truth_df)
    file_dict = get_file_dict_from_df(ground_truth_df)

    self.audio_urls = OrderedSet(sorted(alg_dict.keys()))
    self.algorithms = OrderedSet(sorted(alg_dict.values()))
    self.files = OrderedSet(sorted(file_dict.values()))
    self.data: List[DataPoint] = list(parse_data_points(results_dict, alg_dict, file_dict))
    self.workers = OrderedSet(sorted(set(dp.worker_id for dp in self.data)))
    self.assignments = OrderedSet(sorted(set(dp.assignment_id for dp in self.data)))
    self.n_urls_per_assignment = get_n_urls_per_assignment(self.data)
    self.masks: ODType[str, MaskBase] = OrderedDict()

  @classmethod
  def load(cls, path: Path):
    result = load_obj(path)
    return result

  def save(self, path: Path) -> None:
    save_obj(self, path)

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
  def n_opinion_scores(self) -> int:
    return len(self.data)

  @property
  def n_files(self) -> int:
    return len(self.files)

  def get_masks_from_names(self, mask_names: Set[str]) -> List[MaskBase]:
    masks = [self.masks[mask_name] for mask_name in mask_names]
    return masks

  def add_or_update_mask(self, name: str, mask: MaskBase) -> None:
    self.masks[name] = mask

  def get_mask_factory(self) -> MaskFactory:
    result = MaskFactory(self.algorithms, self.workers, self.files, self.assignments, self.data)
    return result

  def get_os(self) -> np.ndarray:
    Z = np.full(
      (self.n_algorithms, self.n_workers, self.n_files),
      fill_value=np.nan,
      dtype=np.float32
    )
    for dp in self.data:
      alg_i = self.algorithms.get_loc(dp.algorithm)
      worker_i = self.workers.get_loc(dp.worker_id)
      file_i = self.files.get_loc(dp.file)
      Z[alg_i, worker_i, file_i] = dp.opinion_score
    return Z
