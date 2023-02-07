import re
from logging import getLogger
from typing import Dict, Generator, List, Set

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.globals import LISTENING_TYPES, STATE_TYPES


class Evaluation():

  def __init__(self, results_df: DataFrame, ground_truth_df: DataFrame):
    super().__init__()
    self.__ground_truth_dict = ground_truth_df.to_dict("index")
    self.results_dict = results_df.to_dict("index")
    # can be found via csv
    # self.__n_files_per_hit

  def parse_results(self) -> None:
    self.alg_dict: Dict[str, str] = {}
    self.file_dict: Dict[str, str] = {}
    for row in self.__ground_truth_dict.values():
      audio_url = row["audio_url"]
      self.alg_dict[audio_url] = row["algorithm"]
      self.file_dict[audio_url] = row["file"]

    self.audio_urls = OrderedSet(sorted(self.alg_dict.keys()))
    self.algorithms = OrderedSet(sorted(self.alg_dict.values()))
    self.files = OrderedSet(sorted(self.file_dict.values()))
    self.workers = OrderedSet(sorted(set(row['WorkerId'] for row in self.results_dict.values())))
    self.assignments = OrderedSet(
      sorted(set(row['AssignmentId'] for row in self.results_dict.values())))

    self.n_audios = len(self.audio_urls)
    self.n_alg = len(self.algorithms)
    self.n_files = len(self.files)
    self.n_workers = len(self.workers)
    self.n_assignments = len(self.assignments)

    self.Z = np.full(
      (self.n_alg, self.n_workers, self.n_files),
      fill_value=np.nan,
      dtype=np.float32,
    )

    self.Z_ass = np.full_like(self.Z, fill_value=np.nan)

    n_urls_per_hit = 0
    for row in self.results_dict.values():
      worker_i = self.workers.get_loc(row['WorkerId'])
      ass_i = self.assignments.get_loc(row["AssignmentId"])
      audios = parse_audio_files(row)
      mos = parse_mos_answers(row)
      n_urls_per_hit = max(n_urls_per_hit, len(audios))

      for sample_nr, audio_url in audios.items():
        audio_alg = self.alg_dict[audio_url]
        audio_file = self.file_dict[audio_url]
        alg_i = self.algorithms.get_loc(audio_alg)
        file_i = self.files.get_loc(audio_file)
        assert sample_nr in mos
        mos_rating = mos[sample_nr]

        self.Z[alg_i, worker_i, file_i] = mos_rating
        self.Z_ass[alg_i, worker_i, file_i] = ass_i
    self.n_urls_per_hit = n_urls_per_hit

  @property
  def opinion_scores_count(self) -> int:
    os_count = np.nansum(~np.isnan(self.Z))
    return os_count

  def remove_opinion_scores(self, mask: np.ndarray) -> None:
    self.Z[mask] = np.nan


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
