import re
from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Generator, List, Literal
from typing import OrderedDict as ODType
from typing import Set, Union

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.globals import LISTENING_TYPES, STATE_TYPES


@dataclass
class DataPoint():
  opinion_score: Literal[1, 2, 3, 4, 5]
  worker_id: str
  assignment_id: str
  algorithm: str
  file: str
  audio_url: str
  listening_device: Literal["in-ear", "over-the-ear", "desktop", "laptop"]
  state: Literal["Accepted", "Rejected", "Approved"]
  worktime: float


class MaskBase():
  def __init__(self, mask: np.ndarray) -> None:
    self.mask = mask

  def add_mask(self, mask: "MaskBase") -> None:
    self.mask = (self.mask | mask.mask)

  def apply_to(self, data: np.ndarray) -> None:
    assert data.shape == self.mask.shape
    data[self.mask] = np.nan

  @property
  def n_masked(self) -> int:
    # result = np.sum(~np.isnan(self.mask))
    result = np.sum(self.mask)
    return result

  @property
  def n_unmasked(self) -> int:
    # result = np.sum(~np.isnan(self.mask))
    result = np.sum(~self.mask)
    return result


class OpinionScoreMask(MaskBase):
  def __init__(self, mask: np.ndarray) -> None:
    super().__init__(mask)


class AssignmentMask(MaskBase):
  def __init__(self, mask: np.ndarray) -> None:
    super().__init__(mask)

  def to_opinion_score_mask(self, factory: "MaskFactory", opinion_scores_assignments_index_matrix: np.ndarray) -> OpinionScoreMask:
    result = factory.get_opinion_score_mask()
    masked_assignments_indicies = self.mask.nonzero()[0]
    result.mask = np.isin(opinion_scores_assignments_index_matrix, masked_assignments_indicies)
    return result


class WorkerMask(MaskBase):
  def __init__(self, mask: np.ndarray) -> None:
    super().__init__(mask)

  def to_assignment_mask(self, factory: "MaskFactory", assignments_worker_index_matrix: np.ndarray) -> AssignmentMask:
    result = factory.get_assignment_mask()
    masked_worker_indices = self.mask.nonzero()[0]
    result.mask = np.isin(assignments_worker_index_matrix, masked_worker_indices)

    # result.mask[:, ] = True
    return result


class MaskFactory():
  def __init__(self, n_algorithms: int, n_workers: int, n_files: int, n_assignments: int) -> None:
    self.n_algorithms = n_algorithms
    self.n_workers = n_workers
    self.n_files = n_files
    self.n_assignments = n_assignments

  def get_opinion_score_mask(self) -> OpinionScoreMask:
    mask = np.full(
      (self.n_algorithms, self.n_workers, self.n_files),
      fill_value=False,
      dtype=bool,
    )
    return OpinionScoreMask(mask)

  def get_assignment_mask(self) -> AssignmentMask:
    mask = np.full(
      self.n_assignments,
      fill_value=False,
      dtype=bool,
    )
    return AssignmentMask(mask)

  def get_worker_mask(self) -> WorkerMask:
    mask = np.full(
      self.n_workers,
      fill_value=False,
      dtype=bool,
    )
    return WorkerMask(mask)


class EvaluationData():

  def __init__(self, results_df: DataFrame, ground_truth_df: DataFrame, base_mask: str):
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
    self.mask_factory = MaskFactory(self.n_algorithms, self.n_workers,
                                    self.n_files, self.n_assignments)

    self.masks: ODType[str, MaskBase] = OrderedDict((
      (base_mask, self.mask_factory.get_opinion_score_mask()),
    ))

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

  @property
  def n_files_per_hit(self) -> int:
    return get_n_urls_per_hit(self.data)

  def add_mask(self, name: str, mask: MaskBase) -> None:
    self.masks[name] = mask

  def get_opinion_scores(self) -> np.ndarray:
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

  # def get_opinion_scores_masked(self, masks: List[Union[WorkerMask, AssignmentMask, OpinionScoreMask]]) -> np.ndarray:
  #   opinion_scores = self.get_opinion_scores()
  #   mask = get_opinion_score_mask_from_masks(masks, self.data)
  #   opinion_scores[mask.mask] = np.nan
  #   return opinion_scores


def get_assignments_worker_index_matrix(data: EvaluationData) -> np.ndarray:
  res = np.full(
    data.n_assignments,
    fill_value=np.nan,
    dtype=np.float32,
  )

  for dp in data.data:
    ass_i = data.assignments.get_loc(dp.assignment_id)
    worker_i = data.workers.get_loc(dp.worker_id)
    res[ass_i] = worker_i
  return res


def get_opinion_scores_assignments_index_matrix(data: EvaluationData) -> np.ndarray:
  res = np.full(
    (data.n_algorithms, data.n_workers, data.n_files),
    fill_value=np.nan,
    dtype=np.float32,
  )

  for dp in data.data:
    alg_i = data.algorithms.get_loc(dp.algorithm)
    worker_i = data.workers.get_loc(dp.worker_id)
    file_i = data.files.get_loc(dp.file)
    ass_i = data.assignments.get_loc(dp.assignment_id)
    res[alg_i, worker_i, file_i] = ass_i
  return res


def get_assignments_worker_matrix(data: EvaluationData) -> np.ndarray:
  res = np.full(
    data.n_assignments,
    fill_value=np.nan,
    # dtype=np.float32,
    dtype="<U100"  # TODO
  )

  for dp in data.data:
    # worker_i = self.workers.get_loc(row['WorkerId'])
    ass_i = data.assignments.get_loc(dp.assignment_id)
    res[ass_i] = dp.worker_id
  return res


def get_opinion_score_mask_from_masks(masks: List[Union[WorkerMask, AssignmentMask, OpinionScoreMask]], data: EvaluationData) -> OpinionScoreMask:
  result = data.mask_factory.get_opinion_score_mask()
  for mask in masks:
    mask = get_opinion_score_mask(mask, data)
    result.add_mask(mask)
  return result


def get_opinion_score_mask(mask: MaskBase, data: EvaluationData) -> OpinionScoreMask:
  if isinstance(mask, WorkerMask):
    mask = mask.to_assignment_mask(data.mask_factory, get_assignments_worker_index_matrix(data))
  if isinstance(mask, AssignmentMask):
    mask = mask.to_opinion_score_mask(
      data.mask_factory, get_opinion_scores_assignments_index_matrix(data))
  assert isinstance(mask, OpinionScoreMask)
  return mask


def get_assignment_mask_from_masks(masks: List[Union[WorkerMask, AssignmentMask]], data: EvaluationData) -> AssignmentMask:
  result = data.mask_factory.get_assignment_mask()
  for mask in masks:
    if isinstance(mask, OpinionScoreMask):
      continue
    mask = get_assignment_mask(mask, data)
    result.add_mask(mask)
  return result


def get_assignment_mask(mask: MaskBase, data: AssignmentMask) -> OpinionScoreMask:
  if isinstance(mask, OpinionScoreMask):
    raise Exception()
  if isinstance(mask, WorkerMask):
    mask = mask.to_assignment_mask(data.mask_factory, get_assignments_worker_index_matrix(data))
  assert isinstance(mask, AssignmentMask)
  return mask


def get_worker_mask_from_masks(masks: List[WorkerMask], data: EvaluationData) -> WorkerMask:
  result = data.mask_factory.get_worker_mask()
  for mask in masks:
    if isinstance(mask, OpinionScoreMask):
      continue
    if isinstance(mask, AssignmentMask):
      continue
    assert isinstance(mask, WorkerMask)
    result.add_mask(mask)
  return result


def get_n_urls_per_hit(data: List[DataPoint]) -> int:
  tmp = {}
  for dp in data:
    if dp.assignment_id not in tmp:
      tmp[dp.assignment_id] = 0
    tmp[dp.assignment_id] += 1
  result = max(tmp.values())
  return result


def parse_data_points(results_dict: Dict, alg_dict: Dict, file_dict: Dict) -> Generator[DataPoint, None, None]:
  for row in results_dict.values():
    audios = parse_audio_files(row)
    mos = parse_mos_answers(row)
    lt = parse_listening_type(row)
    work_time = int(row["WorkTimeInSeconds"])

    for sample_nr, audio_url in audios.items():
      audio_alg = alg_dict[audio_url]
      audio_file = file_dict[audio_url]
      assert sample_nr in mos
      mos_rating = mos[sample_nr]
      data_point = DataPoint(
        assignment_id=row["AssignmentId"],
        worker_id=row["WorkerId"],
        algorithm=audio_alg,
        audio_url=audio_url,
        file=audio_file,
        listening_device=lt,
        opinion_score=mos_rating,
        state=row["AssignmentStatus"],
        worktime=work_time,
      )
      yield data_point


def parse_audio_files(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Input\.audio_url_(\d+)")
  result = {}
  for identifier, val in row.items():
    mos_match = re.match(pattern, identifier)
    if isinstance(mos_match, re.Match):
      sample_nr = mos_match.group(1)
      sample_nr = int(sample_nr)
      assert sample_nr not in result
      result[sample_nr] = val
  return result


def parse_mos_answers(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Answer\.(\d+)-mos-rating\.([1-5])")
  result = {}
  for identifier, val in row.items():
    if val:
      mos_match = re.match(pattern, identifier)
      if isinstance(mos_match, re.Match):
        sample_nr, mos_val = mos_match.groups()
        sample_nr = int(sample_nr)
        assert sample_nr not in result
        result[sample_nr] = int(mos_val)
  return result


def parse_listening_type(row: Dict[str, str]) -> str:
  pattern = re.compile(r"Answer\.listening-type\.(.+)")
  result = None
  for identifier, val in row.items():
    if val:
      mos_match = re.match(pattern, identifier)
      if isinstance(mos_match, re.Match):
        lt = mos_match.group(1)
        assert result is None
        result = lt
  return result
