import re
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Set

import boto3
import botocore
import numpy as np
import pandas as pd
import xmltodict
from mypy_boto3_mturk.type_defs import HITTypeDef
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_mos_ci95_3gaussian

MOS_PATTERN = re.compile(r"Answer\.(\d+)-mos-rating\.([1-5])")
LT_PATTERN = re.compile(r"Answer\.listening-type\.(.+)")
AUDIO_PATTERN = re.compile(r"Input\.audio_url_(\d+)")


def parse_audio_files(row: Dict[str, str]) -> Dict[str, int]:
  result = {}
  for identifier, val in row.items():
    mos_match = re.match(AUDIO_PATTERN, identifier)
    if isinstance(mos_match, re.Match):
      sample_nr = mos_match.group(1)
      sample_nr = int(sample_nr)
      assert sample_nr not in result
      result[sample_nr] = val
  return result


def parse_mos_answers(row: Dict[str, str]) -> Dict[str, int]:
  result = {}
  for identifier, val in row.items():
    if val:
      mos_match = re.match(MOS_PATTERN, identifier)
      if isinstance(mos_match, re.Match):
        sample_nr, mos_val = mos_match.groups()
        sample_nr = int(sample_nr)
        assert sample_nr not in result
        result[sample_nr] = int(mos_val)
  return result


def parse_listening_type(row: Dict[str, str]) -> str:
  result = None
  for identifier, val in row.items():
    if val:
      mos_match = re.match(LT_PATTERN, identifier)
      if isinstance(mos_match, re.Match):
        lt = mos_match.group(1)
        assert result is None
        result = lt
  return result


def parse_comment(row: Dict[str, str]) -> str:
  result = row["Answer.comments"]
  return result


def parse_df(input_df: DataFrame, output_df: DataFrame, consider_lt: Set[str], paths: OrderedSet[str], min_worktime_s: int):
  logger = getLogger(__name__)
  input_as_dict = input_df.to_dict("list")
  all_audio_paths = {
    audio_url
    for audio_urls in input_as_dict.values()
    for audio_url in audio_urls
  }
  all_audio_paths = OrderedSet(sorted(all_audio_paths))

  assert len(all_audio_paths) == 480

  mos_ratings: Dict[str, List[int]] = {
    url: [] for url in all_audio_paths
  }

  audio_files = 0

  ignored_assignments_count = 0
  kept_assignments_count = 0
  df_as_dict = output_df.to_dict("index")

  workers = {row['WorkerId'] for row in df_as_dict.values()}
  workers = OrderedSet(sorted(workers))

  result = np.full((len(workers), len(all_audio_paths)), fill_value=np.nan, dtype=np.float16)

  for i, row in df_as_dict.items():
    if row["AssignmentStatus"] == "Rejected":
      logger.info(f"Ignored rejected assignment: {row['AssignmentId']}")
      ignored_assignments_count += 1
      continue
    worktime = int(row["WorkTimeInSeconds"])
    if worktime < min_worktime_s:
      logger.info(f"Ignored too fast assignment: {row['AssignmentId']}")
      ignored_assignments_count += 1
      continue
    lt = parse_listening_type(row)
    if lt not in consider_lt:
      logger.info(f"Ignored invalid listening type assignment: {row['AssignmentId']}")
      ignored_assignments_count += 1
      continue
    worker_index = workers.get_loc(row['WorkerId'])
    audios = parse_audio_files(row)
    assert len(audios) == 8
    mos = parse_mos_answers(row)
    assert len(mos) == 8
    comment = parse_comment(row)
    for sample_nr, audio_url in audios.items():
      audio_index = all_audio_paths.get_loc(audio_url)
      assert sample_nr in mos
      mos_rating = mos[sample_nr]
      result[worker_index][audio_index] = mos_rating
      assert audio_url in mos_ratings
      mos_ratings[audio_url].append(mos_rating)
    kept_assignments_count += 1

  resulting_ratings = []
  for path in paths:
    ratings = {k[len(path):]: v for k, v in mos_ratings.items() if k.startswith(path)}
    resulting_ratings.append(ratings)
  total_assignment_count = len(df_as_dict)
  if total_assignment_count > 0:
    logger.info(
      f"Ignored {ignored_assignments_count} of {total_assignment_count} assignments ({ignored_assignments_count/total_assignment_count*100:.2f}%)!")
    logger.info(
      f"Considered {kept_assignments_count} of {total_assignment_count} assignments ({kept_assignments_count/total_assignment_count*100:.2f}%)!")

  algo_results = list(split_algos(result, all_audio_paths, paths))

  for algo_i, algo in enumerate(algo_results):
    mos, ci95 = compute_mos_ci95_3gaussian(algo)
    logger.info(f"MOS for alg{algo_i}: {mos} +- {ci95}")
    continue
    mos = np.mean(calc_worker_mos(algo))
    std = np.mean(calc_worker_std(algo))
    std_ci95 = np.mean(calc_worker_std(algo) * 1.95996)
    logger.info(f"MOS for alg{algo_i}: {mos} +- {std}")
    logger.info(f"MOS for alg{algo_i}: {mos} +- {std_ci95}")

  return resulting_ratings


def calc_worker_mos(array: np.array) -> np.ndarray:
  res = np.nanmean(array, axis=1)
  return res


def calc_worker_std(array: np.array) -> np.ndarray:
  # std = quality_ambiguity
  std = np.nanstd(array, axis=1)
  x = ~np.isnan(array)
  count_not_nan = np.nansum(x, axis=1)
  # std / sqrt(N), ignoring NaN
  mos_std = std / np.sqrt(count_not_nan)
  return mos_std


def split_algos(array: np.ndarray, audio_files: OrderedSet[str], split_paths: OrderedSet[str]) -> List[np.ndarray]:
  for split_path in split_paths:
    indices = [
      audio_files.get_loc(audio_path)
      for audio_path in audio_files
      if audio_path.startswith(split_path)
    ]
    result = array[:, indices]
    yield result
