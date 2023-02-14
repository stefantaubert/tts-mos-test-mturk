import re
from logging import getLogger
from typing import Dict, List, Set

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.globals import LISTENING_TYPES

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

  Z_all = np.full((len(workers), len(all_audio_paths)), fill_value=np.nan, dtype=np.float32)
  work_times_all = np.full((len(workers), len(all_audio_paths)),
                           fill_value=np.nan, dtype=np.float16)
  listening_types_all = np.full((len(workers), len(all_audio_paths)),
                                fill_value=np.nan, dtype=np.float16)

  worker_accepted_assignments: Dict[str, Set[str]] = {}

  for i, row in df_as_dict.items():
    lt = parse_listening_type(row)
    assert lt in LISTENING_TYPES
    assignment_id = row['AssignmentId']
    if row["AssignmentStatus"] == "Rejected":
      logger.info(f"Ignored rejected assignment: {assignment_id}")
      ignored_assignments_count += 1
      continue
    worktime = int(row["WorkTimeInSeconds"])
    # if worktime < min_worktime_s:
    #   logger.info(f"Ignored too fast assignment: {row['AssignmentId']}")
    #   ignored_assignments_count += 1
    #   continue
    # if lt not in consider_lt:
    #   logger.info(f"Ignored invalid listening type assignment: {row['AssignmentId']}")
    #   ignored_assignments_count += 1
    #   continue
    worker_id = row['WorkerId']
    worker_index = workers.get_loc(worker_id)
    if worker_id not in worker_accepted_assignments:
      worker_accepted_assignments[worker_id] = set()
    worker_accepted_assignments[worker_id].add(assignment_id)

    audios = parse_audio_files(row)
    assert len(audios) == 8
    mos = parse_mos_answers(row)
    assert len(mos) == 8
    comment = parse_comment(row)
    for sample_nr, audio_url in audios.items():
      audio_index = all_audio_paths.get_loc(audio_url)
      assert sample_nr in mos
      mos_rating = mos[sample_nr]
      Z_all[worker_index][audio_index] = mos_rating
      work_times_all[worker_index][audio_index] = worktime
      listening_types_all[worker_index][audio_index] = LISTENING_TYPES.get(lt)
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

  return Z_all, work_times_all, listening_types_all, workers, all_audio_paths, worker_accepted_assignments


def calc_worker_std2(array: np.array) -> np.ndarray:
  # std = quality_ambiguity
  std = np.nanstd(array.flatten())
  x = ~np.isnan(array.flatten())
  count_not_nan = np.nansum(x)
  # std / sqrt(N), ignoring NaN
  mos_std = std / np.sqrt(count_not_nan)
  return mos_std
