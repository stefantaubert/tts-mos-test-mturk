from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
from typing import OrderedDict as ODType

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.core.data_point import (DataPoint, get_n_urls_per_assignment,
                                                parse_data_points)
from tts_mos_test_mturk.core.io import load_obj, save_obj
from tts_mos_test_mturk.core.masks import MaskBase, MaskFactory


class EvaluationData():
  def __init__(self, results_df: DataFrame, ground_truth_df: DataFrame):
    super().__init__()
    ground_truth_dict = ground_truth_df.to_dict("index")
    results_dict = results_df.to_dict("index")

    alg_dict: Dict[str, str] = {}
    file_dict: Dict[str, str] = {}
    for row in ground_truth_dict.values():
      audio_url = row["audio_url"]
      alg_dict[audio_url] = row["algorithm"]
      file_dict[audio_url] = row["file"]

    self.audio_urls = OrderedSet(sorted(alg_dict.keys()))
    self.algorithms = OrderedSet(sorted(alg_dict.values()))
    self.files = OrderedSet(sorted(file_dict.values()))
    self.data: List[DataPoint] = list(parse_data_points(results_dict, alg_dict, file_dict))
    self.workers = OrderedSet(sorted(set(dp.worker_id for dp in self.data)))
    self.assignments = OrderedSet(sorted(set(dp.assignment_id for dp in self.data)))
    self.n_urls_per_assignment = get_n_urls_per_assignment(self.data)
    # self.mask_factory = MaskFactory(self.n_algorithms, self.n_workers,
    #                                 self.n_files, self.n_assignments)

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
  def n_files(self) -> int:
    return len(self.files)

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

  def get_worktimes(self) -> np.ndarray:
    worktimes = np.full(
      self.n_assignments,
      fill_value=np.nan,
      dtype=np.float32,
    )
    for dp in self.data:
      ass_i = self.assignments.get_loc(dp.assignment_id)
      worktimes[ass_i] = dp.worktime
    return worktimes

  def get_listening_devices(self) -> np.ndarray:
    worktimes = [np.nan] * self.n_assignments

    for dp in self.data:
      ass_i = self.assignments.get_loc(dp.assignment_id)
      if worktimes[ass_i] != dp.listening_device:
        worktimes[ass_i] = dp.listening_device
    worktimes_np = np.array(worktimes)
    return worktimes_np
