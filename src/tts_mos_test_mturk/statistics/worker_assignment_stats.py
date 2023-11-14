import datetime
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.correlations import (get_algorithm_mos_correlation,
                                             get_sentence_mos_correlation_3dim)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.typing import AlgorithmName, FileName, RatingName, RatingValue

DATE_FMT = "%Y-%m-%d %H:%M:%S"

COL_WORKER = "WorkerId"
COL_GENDER = "Gender"
COL_AGE_GROUP = "Age group"
COL_TOT_ASSIGNMENTS = "#Assignments"
COL_STATE = "#State"
COL_DEVICE = "#Device"
COL_FIRST_DEVICE = "First device"
COL_LAST_DEVICE = "Last device"
COL_SENT_CORR = "Sent. corr."
COL_ALGO_CORR = "Alg. corr."
COL_BOTH_CORR = "Corr."
COL_ALL_CORR = "All Corr."
COL_MASKED_ASSIGNMENTS = "#Masked assignments"
COL_MASKED = "Masked?"
COL_ALL = "-ALL-"
COL_LISTENED_FILE_COUNT = "#Listened files"
COL_FIRST_HIT_ACC_TIME = "First HIT acc."
COL_LAST_HIT_ACC_TIME = "Last HIT acc."


@dataclass
class WorkerEntry:
  age_group: str
  gender: str
  masked: bool = False
  masked_assignments: int = 0
  # Count of assignment traps which the worker fell into
  listened_file_count: int = 0
  statuses: List[str] = field(default_factory=list)
  accept_times: List[datetime.datetime] = field(default_factory=list)
  devices: List[str] = field(default_factory=list)
  sentence_correlations: Dict[str, float] = field(default_factory=dict)
  algorithm_correlations: Dict[str, float] = field(default_factory=dict)
  ratings: Dict[Tuple[AlgorithmName, FileName, RatingName],
                List[RatingValue]] = field(default_factory=dict)
  # difference_ratings: ODType[RatingName, List[float]] = field(default_factory=OrderedDict)

  @property
  def total_assignments(self) -> int:
    return len(self.statuses)

  def correlations_mean(self, rating_name: str) -> float:
    assert rating_name in self.algorithm_correlations
    assert rating_name in self.sentence_correlations

    if np.isnan(self.sentence_correlations[rating_name]):
      if np.isnan(self.algorithm_correlations[rating_name]):
        return np.nan
      return self.algorithm_correlations[rating_name]
    if np.isnan(self.algorithm_correlations[rating_name]):
      return self.sentence_correlations[rating_name]
    return np.mean([self.algorithm_correlations[rating_name], self.sentence_correlations[rating_name]])


def get_wass_stat_data(data: EvaluationData, masks: List[MaskBase]) -> Dict[str, WorkerEntry]:
  factory = MaskFactory(data)

  wmask = factory.merge_masks_into_wmask(masks)
  amask = factory.merge_masks_into_amask(masks)
  rmask = factory.merge_masks_into_rmask(masks)

  all_ratings = {}
  for rating_name in data.rating_names:
    ratings = get_ratings(data, {rating_name})
    rmask.apply_by_nan(ratings)
    all_ratings[rating_name] = ratings

  stats: Dict[str, WorkerEntry] = {}

  # init entries
  for worker, worker_data in data.worker_data.items():
    worker_entry = WorkerEntry(worker_data.age_group, worker_data.gender)
    for rating_name, ratings in all_ratings.items():
      worker_entry.algorithm_correlations[rating_name] = np.nan
      worker_entry.sentence_correlations[rating_name] = np.nan

    stats[worker] = worker_entry

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment, assignment_data in worker_data.assignments.items():
      worker_entry = stats[worker]

      skip = False
      w_is_masked = wmask.mask[w_i]
      if w_is_masked:
        worker_entry.masked = True
        skip = True
      a_i = data.assignments.get_loc(assignment)
      a_is_masked = amask.mask[a_i]
      if a_is_masked:
        worker_entry.masked_assignments += 1
        skip = True

      if skip:
        continue

      for (alg_name, file_name), rating_data in assignment_data.ratings.items():
        worker_entry.listened_file_count += 1
        for rating_name, rating_val in rating_data.votes.items():
          key = (alg_name, file_name, rating_name)
          if key not in worker_entry.ratings:
            worker_entry.ratings[key] = []
          worker_entry.ratings[key].append(rating_val)

      worker_entry.devices.append(assignment_data.device)
      worker_entry.statuses.append(assignment_data.state)
      worker_entry.accept_times.append(assignment_data.time)

  for worker, worker_data in data.worker_data.items():
    worker_entry = stats[worker]
    w_i = data.workers.get_loc(worker)
    for rating_name, ratings in all_ratings.items():
      worker_entry.algorithm_correlations[rating_name] = get_algorithm_mos_correlation(
        w_i, ratings)
      worker_entry.sentence_correlations[rating_name] = get_sentence_mos_correlation_3dim(
        w_i, ratings)

  return stats


def stats_to_df(stats: Dict[str, WorkerEntry]) -> pd.DataFrame:
  csv_data = []
  unique_statuses = sorted({
    status
    for worker_entry in stats.values()
    for status in worker_entry.statuses
  })

  unique_devices = sorted({
    device
    for worker_entry in stats.values()
    for device in worker_entry.devices
  })

  all_rating_names = OrderedSet((
    rating_name
    for worker_entry in stats.values()
    for rating_name in worker_entry.algorithm_correlations.keys()
  ))
  for worker, entry in stats.items():
    data_entry = OrderedDict()
    data_entry[COL_WORKER] = worker
    data_entry[COL_AGE_GROUP] = entry.age_group
    data_entry[COL_GENDER] = entry.gender

    status_counts = Counter(entry.statuses)
    for status in unique_statuses:
      key = f"{COL_STATE} \"{status}\""
      assert key not in data_entry
      data_entry[key] = status_counts.get(status, 0)
    data_entry[COL_TOT_ASSIGNMENTS] = entry.total_assignments

    data_entry[COL_LISTENED_FILE_COUNT] = entry.listened_file_count

    if len(entry.accept_times) == 0:
      data_entry[COL_FIRST_HIT_ACC_TIME] = np.nan
      data_entry[COL_FIRST_DEVICE] = np.nan
      data_entry[COL_LAST_HIT_ACC_TIME] = np.nan
      data_entry[COL_LAST_DEVICE] = np.nan
    else:
      tmp = zip(entry.accept_times, entry.devices)
      tmp_sorted = sorted(tmp, key=lambda x: x[0])
      first_accept_time, first_device = tmp_sorted[0]
      last_accept_time, last_device = tmp_sorted[-1]
      assert first_accept_time == min(entry.accept_times)
      assert last_accept_time == max(entry.accept_times)

      data_entry[COL_FIRST_HIT_ACC_TIME] = first_accept_time.strftime(DATE_FMT)
      data_entry[COL_FIRST_DEVICE] = first_device
      data_entry[COL_LAST_HIT_ACC_TIME] = max(entry.accept_times).strftime(DATE_FMT)
      data_entry[COL_LAST_DEVICE] = last_device

    device_counts = Counter(entry.devices)
    for device in unique_devices:
      key = f"{COL_DEVICE} \"{device}\""
      assert key not in data_entry
      data_entry[key] = device_counts.get(device, 0)

    all_corr_vals = []
    for rating_name in all_rating_names:
      col_append_str = "-ALL-" if rating_name is None else rating_name
      data_entry[f"{COL_SENT_CORR} ({col_append_str})"] = entry.sentence_correlations[rating_name]
      data_entry[f"{COL_ALGO_CORR} ({col_append_str})"] = entry.algorithm_correlations[rating_name]
      mn = entry.correlations_mean(rating_name)
      data_entry[f"{COL_BOTH_CORR} ({col_append_str})"] = mn
      if not np.isnan(mn):
        all_corr_vals.append(mn)
    all_corr = np.nan
    if len(all_corr_vals) > 0:
      all_corr = np.mean(all_corr_vals)
    data_entry[COL_ALL_CORR] = all_corr

    data_entry[COL_MASKED_ASSIGNMENTS] = entry.masked_assignments
    data_entry[COL_MASKED] = entry.masked

    csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_row(df: pd.DataFrame, stats: Dict[str, WorkerEntry]) -> pd.DataFrame:
  row = OrderedDict()
  row[COL_WORKER] = COL_ALL
  row[COL_GENDER] = ""
  row[COL_AGE_GROUP] = ""

  col: str
  for col in df.columns:
    if col.startswith(COL_STATE):
      assert col not in row
      row[col] = df[col].sum()
  row[COL_TOT_ASSIGNMENTS] = df[COL_TOT_ASSIGNMENTS].sum()

  row[COL_LISTENED_FILE_COUNT] = df[COL_LISTENED_FILE_COUNT].sum()

  all_accept_times = [
    time
    for entry in stats.values()
    for time in entry.accept_times
  ]

  row[COL_FIRST_HIT_ACC_TIME] = min(all_accept_times).strftime(DATE_FMT)
  row[COL_LAST_HIT_ACC_TIME] = max(all_accept_times).strftime(DATE_FMT)

  all_corr_vals = []
  for col in df.columns:
    if col.startswith(COL_DEVICE):
      assert col not in row
      row[col] = df[col].sum(skipna=True)
    if col.startswith(COL_SENT_CORR):
      assert col not in row
      row[col] = df[col].mean(skipna=True)
    if col.startswith(COL_ALGO_CORR):
      assert col not in row
      row[col] = df[col].mean(skipna=True)
    if col.startswith(COL_BOTH_CORR):
      assert col not in row
      row[col] = df[col].mean(skipna=True)
      all_corr_vals.append(row[col])
  row[COL_ALL_CORR] = mean(all_corr_vals)
  row[COL_MASKED_ASSIGNMENTS] = df[COL_MASKED_ASSIGNMENTS].sum()
  row[COL_MASKED] = df[COL_MASKED].all()
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)
  return df


def get_worker_assignment_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_wass_stat_data(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    # df.sort_values([COL_FIRST_HIT_ACC_TIME, COL_WORKER], inplace=True)
    df.sort_values([COL_GENDER, COL_AGE_GROUP, COL_ALL_CORR],
                   ascending=[False, True, False], inplace=True)
    df = add_all_row(df, stats)
  return df
