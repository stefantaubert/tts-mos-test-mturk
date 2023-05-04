import datetime
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from itertools import combinations, permutations
from statistics import mean
from typing import Dict, Generator, List
from typing import OrderedDict as ODType
from typing import Set, Tuple, Union

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
COL_COUNT_DUP_RATINGS = "#Duplicate ratings"
COL_AVG_DEV_DUP_RATINGS = "#Avg. dev. dupl."
COL_TOT_ASSIGNMENTS = "#Assignments"
COL_STATE = "#State"
COL_DEVICE = "#Device"
COL_FIRST_DEVICE = "First device"
COL_LAST_DEVICE = "Last device"
COL_SENT_CORR = "Sent. corr."
COL_ALGO_CORR = "Alg. corr."
COL_BOTH_CORR = "Corr."
COL_N_TRAPS = "#Traps"
COL_N_FALLEN_TRAPS = "#Fallen Traps"
COL_OVERLAPPING_HITS = "#HIT Overlaps"
COL_PAGE_HIDDEN_TOT_DUR = "Page hidden tot."
COL_PAGE_HIDDEN_AVG_DUR = "Page hidden avg."
COL_PAGE_HIDDEN_MIN_DUR = "Page hidden min."
COL_PAGE_HIDDEN_MAX_DUR = "Page hidden max."
COL_ACTIVE_SESSIONS_MIN = "#Active sessions min."
COL_ACTIVE_SESSIONS_AVG = "#Active sessions avg."
COL_ACTIVE_SESSIONS_MAX = "#Active sessions max."
COL_COUNT_BROWSERS = "#Browsers"
COL_BROWSERS = "Browsers"
COL_MIN_OVERLAP_DUR = "Min. overlap"
COL_AVG_OVERLAP_DUR = "Avg. overlap"
COL_MAX_OVERLAP_DUR = "Max. overlap"
COL_TOT_OVERLAP_DUR = "Tot. overlap"
COL_FIRST_HIT_ACC_TIME = "First HIT acc."
COL_FIRST_WORKTIME = "First worktime"
COL_FIRST_HIT_FIN_TIME = "First HIT fin."
COL_LAST_HIT_ACC_TIME = "Last HIT acc."
COL_LAST_WORKTIME = "Last worktime"
COL_LAST_HIT_FIN_TIME = "Last HIT fin."
COL_DIFF_WORKTIME_TIME = "Diff. first last worktime"
COL_MIN_WORKTIME = "Min. worktime"
COL_AVG_WORKTIME = "Avg. worktime"
COL_MAX_WORKTIME = "Max. worktime"
COL_TOT_WORKTIME = "Tot. worktime"
COL_COMMENTS = "Comments"
COL_MASKED_ASSIGNMENTS = "#Masked assignments"
COL_MASKED = "Masked?"
COL_ALL = "-ALL-"

COL_LISTENED_FILE_COUNT = "#Listened files"
COL_OVERPLAYED_MIN = "#Overplayed min."
COL_OVERPLAYED_AVG = "#Overplayed avg."
COL_OVERPLAYED_MAX = "#Overplayed max."
COL_OVERPLAYED_RATE = "Overplayed rate"
COL_OVERFULLPLAYED_MIN = "#Overfullplayed min."
COL_OVERFULLPLAYED_AVG = "#Overfullplayed avg."
COL_OVERFULLPLAYED_MAX = "#Overfullplayed max."
COL_OVERFULLPLAYED_RATE = "Overfullplayed rate"
COL_STOPPED_MIN = "#Stopped min."
COL_STOPPED_AVG = "#Stopped avg."
COL_STOPPED_MAX = "#Stopped max."
COL_STOPPED_RATE = "Stopped rate"


@dataclass
class WorkerEntry:
  age_group: str
  gender: str
  masked: bool = False
  masked_assignments: int = 0
  # Count of assignment traps which the worker fell into
  fallen_traps: int = 0
  total_traps: int = 0
  page_hidden: List[int] = field(default_factory=list)
  active_sessions: List[int] = field(default_factory=list)
  browser_strings: List[str] = field(default_factory=list)
  # normal is one play
  overplayed_counts: List[int] = field(default_factory=list)
  # normal is no stop
  stopped_counts: List[int] = field(default_factory=list)
  # normal is one fullplay
  overfullplayed_counts: List[int] = field(default_factory=list)
  listened_file_count: int = 0
  statuses: List[str] = field(default_factory=list)
  worktimes: List[Union[int, float]] = field(default_factory=list)
  accept_times: List[datetime.datetime] = field(default_factory=list)
  devices: List[str] = field(default_factory=list)
  sentence_correlations: Dict[str, float] = field(default_factory=dict)
  algorithm_correlations: Dict[str, float] = field(default_factory=dict)
  comments: List[str] = field(default_factory=list)
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
    return mean([self.algorithm_correlations[rating_name], self.sentence_correlations[rating_name]])


def get_data(data: EvaluationData, masks: List[MaskBase]):
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

  for worker, worker_data in data.worker_data.items():
    stats[worker] = WorkerEntry(worker_data.age_group, worker_data.gender)

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

      for rating_name, ratings in all_ratings.items():
        worker_entry.algorithm_correlations[rating_name] = np.nan
        worker_entry.sentence_correlations[rating_name] = np.nan

      if skip:
        continue

      fell_in_trap = any(
        x != 0
        for x in assignment_data.traps.values()
      )

      worker_entry.total_traps += len(assignment_data.traps)

      for (alg_name, file_name), rating_data in assignment_data.ratings.items():
        worker_entry.overplayed_counts.append(rating_data.played_count - 1)
        worker_entry.overfullplayed_counts.append(rating_data.full_play_count - 1)
        worker_entry.stopped_counts.append(rating_data.stopped_count)
        worker_entry.listened_file_count += 1
        for rating_name, rating_val in rating_data.votes.items():
          key = (alg_name, file_name, rating_name)
          if key not in worker_entry.ratings:
            worker_entry.ratings[key] = []
          worker_entry.ratings[key].append(rating_val)

      if fell_in_trap:
        worker_entry.fallen_traps += 1

      worker_entry.devices.append(assignment_data.device)
      worker_entry.statuses.append(assignment_data.state)
      worker_entry.worktimes.append(assignment_data.worktime)
      worker_entry.accept_times.append(assignment_data.time)
      worker_entry.comments.append(assignment_data.comments)
      worker_entry.browser_strings.append(assignment_data.browser_info)
      worker_entry.active_sessions.append(assignment_data.active_sessions_count)
      worker_entry.page_hidden.append(assignment_data.time_page_hidden_sec)

      for rating_name, ratings in all_ratings.items():
        worker_entry.algorithm_correlations[rating_name] = get_algorithm_mos_correlation(
          w_i, ratings)
        worker_entry.sentence_correlations[rating_name] = get_sentence_mos_correlation_3dim(
          w_i, ratings)

  return stats


def get_overlap_time(time1: Tuple[datetime.datetime, datetime.datetime], time_overlapping: Tuple[datetime.datetime, datetime.datetime]) -> datetime.timedelta:
  a1, b1 = time1
  a2, b2 = time_overlapping
  if a1 < a2 < b1 and a1 < b2 < b1:
    return b2 - a2
  if a1 < a2 < b1:
    return b1 - a2
  if a1 < b2 < b1:
    return b2 - a1
  return datetime.timedelta(seconds=0)


def get_overlaps(timings: Set[Tuple[datetime.datetime, datetime.datetime]]) -> Generator[Tuple[Tuple[datetime.datetime, datetime.datetime], Tuple[datetime.datetime, datetime.datetime]], None, None]:
  all_combinations = list(combinations(sorted(timings), 2))
  for permutation in all_combinations:
    (a1, b1), (a2, b2) = permutation
    if a1 < a2 < b1 or a1 < b2 < b1:
      yield (a1, b1), (a2, b2)
    elif a2 < a1 < b2 or a2 < b1 < b2:
      yield (a2, b2), (a1, b1)


def get_max_deviation(values: List[RatingValue]) -> float:
  for v in values:
    assert v >= 0

  result = max(
    abs(a - b)
    for a, b in combinations(values, 2)
  )
  return result


def get_entry_overlap_times(entry: WorkerEntry) -> Generator[datetime.timedelta, None, None]:
  timings = {
    (accept_time, accept_time + datetime.timedelta(seconds=worktime))
    for accept_time, worktime in zip(entry.accept_times, entry.worktimes)
  }
  for overlap in get_overlaps(timings):
    time1, time2 = overlap
    yield get_overlap_time(time1, time2)


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

    duplicate_ratings_count = sum(
      len(rating_values_unique_file) - 1
      for rating_values_unique_file in entry.ratings.values()
      if len(rating_values_unique_file) > 1
    )

    status_counts = Counter(entry.statuses)
    for status in unique_statuses:
      key = f"{COL_STATE} \"{status}\""
      assert key not in data_entry
      data_entry[key] = status_counts.get(status, 0)
    data_entry[COL_TOT_ASSIGNMENTS] = entry.total_assignments
    data_entry[COL_N_TRAPS] = entry.total_traps
    data_entry[COL_N_FALLEN_TRAPS] = entry.fallen_traps

    data_entry[COL_PAGE_HIDDEN_MIN_DUR] = min(entry.page_hidden)
    data_entry[COL_PAGE_HIDDEN_AVG_DUR] = mean(entry.page_hidden)
    data_entry[COL_PAGE_HIDDEN_MAX_DUR] = max(entry.page_hidden)
    data_entry[COL_PAGE_HIDDEN_TOT_DUR] = sum(entry.page_hidden)
    data_entry[COL_ACTIVE_SESSIONS_MIN] = min(entry.active_sessions)
    data_entry[COL_ACTIVE_SESSIONS_AVG] = mean(entry.active_sessions)
    data_entry[COL_ACTIVE_SESSIONS_MAX] = max(entry.active_sessions)
    data_entry[COL_COUNT_BROWSERS] = len(set(entry.browser_strings))

    data_entry[COL_LISTENED_FILE_COUNT] = entry.listened_file_count
    data_entry[COL_OVERPLAYED_MIN] = min(entry.overplayed_counts)
    data_entry[COL_OVERPLAYED_AVG] = mean(entry.overplayed_counts)
    data_entry[COL_OVERPLAYED_MAX] = max(entry.overplayed_counts)
    data_entry[COL_OVERPLAYED_RATE] = sum(
      1 for x in entry.overplayed_counts if x > 0
    ) / entry.listened_file_count * 100

    data_entry[COL_OVERFULLPLAYED_MIN] = min(entry.overfullplayed_counts)
    data_entry[COL_OVERFULLPLAYED_AVG] = mean(entry.overfullplayed_counts)
    data_entry[COL_OVERFULLPLAYED_MAX] = max(entry.overfullplayed_counts)
    data_entry[COL_OVERFULLPLAYED_RATE] = sum(
      1 for x in entry.overfullplayed_counts if x > 0
    ) / entry.listened_file_count * 100

    data_entry[COL_STOPPED_MIN] = min(entry.stopped_counts)
    data_entry[COL_STOPPED_AVG] = mean(entry.stopped_counts)
    data_entry[COL_STOPPED_MAX] = max(entry.stopped_counts)
    data_entry[COL_STOPPED_RATE] = sum(
      1 for x in entry.stopped_counts if x > 0
    ) / entry.listened_file_count * 100

    max_deviations = [
      get_max_deviation(rating_values_unique_file)
      for rating_values_unique_file in entry.ratings.values()
      if len(rating_values_unique_file) > 1
    ]

    average_max_deviation = mean(max_deviations) if len(max_deviations) > 0 else np.nan

    data_entry[COL_COUNT_DUP_RATINGS] = duplicate_ratings_count
    data_entry[COL_AVG_DEV_DUP_RATINGS] = average_max_deviation

    overlap_times = list(get_entry_overlap_times(entry))
    data_entry[COL_OVERLAPPING_HITS] = len(overlap_times)
    if len(overlap_times) > 0:
      data_entry[COL_MIN_OVERLAP_DUR] = str(min(overlap_times))
      data_entry[COL_AVG_OVERLAP_DUR] = str(datetime.timedelta(
          seconds=mean(x.total_seconds() for x in overlap_times)))
      data_entry[COL_MAX_OVERLAP_DUR] = str(max(overlap_times))
    else:
      data_entry[COL_MIN_OVERLAP_DUR] = np.nan
      data_entry[COL_AVG_OVERLAP_DUR] = np.nan
      data_entry[COL_MAX_OVERLAP_DUR] = np.nan
    data_entry[COL_TOT_OVERLAP_DUR] = str(datetime.timedelta(
      seconds=sum(x.total_seconds() for x in overlap_times)))
    if len(entry.accept_times) == 0:
      data_entry[COL_FIRST_HIT_ACC_TIME] = np.nan
      data_entry[COL_FIRST_WORKTIME] = np.nan
      data_entry[COL_FIRST_HIT_FIN_TIME] = np.nan
      data_entry[COL_FIRST_DEVICE] = np.nan
      data_entry[COL_LAST_HIT_ACC_TIME] = np.nan
      data_entry[COL_LAST_WORKTIME] = np.nan
      data_entry[COL_LAST_HIT_FIN_TIME] = np.nan
      data_entry[COL_LAST_DEVICE] = np.nan
      data_entry[COL_DIFF_WORKTIME_TIME] = np.nan
      data_entry[COL_MIN_WORKTIME] = np.nan
      data_entry[COL_AVG_WORKTIME] = np.nan
      data_entry[COL_MAX_WORKTIME] = np.nan
      data_entry[COL_TOT_WORKTIME] = np.nan
    else:
      tmp = zip(entry.accept_times, entry.worktimes, entry.devices)
      tmp_sorted = sorted(tmp, key=lambda x: x[0])
      first_accept_time, first_worktime, first_device = tmp_sorted[0]
      last_accept_time, last_worktime, last_device = tmp_sorted[-1]
      assert first_accept_time == min(entry.accept_times)
      assert last_accept_time == max(entry.accept_times)

      data_entry[COL_FIRST_HIT_ACC_TIME] = first_accept_time.strftime(DATE_FMT)
      data_entry[COL_FIRST_WORKTIME] = str(datetime.timedelta(seconds=first_worktime))
      data_entry[COL_FIRST_HIT_FIN_TIME] = (
        first_accept_time + datetime.timedelta(seconds=first_worktime)).strftime(DATE_FMT)
      data_entry[COL_FIRST_DEVICE] = first_device
      data_entry[COL_LAST_HIT_ACC_TIME] = max(entry.accept_times).strftime(DATE_FMT)
      data_entry[COL_LAST_WORKTIME] = str(datetime.timedelta(seconds=last_worktime))
      data_entry[COL_LAST_HIT_FIN_TIME] = (
        last_accept_time + datetime.timedelta(seconds=last_worktime)).strftime(DATE_FMT)
      data_entry[COL_LAST_DEVICE] = last_device
      prepend_minus = "-" if first_worktime > last_worktime else "+"
      if first_worktime == last_worktime:
        prepend_minus = "±"
      data_entry[COL_DIFF_WORKTIME_TIME] = prepend_minus + str(
        datetime.timedelta(seconds=abs(last_worktime - first_worktime)))
      data_entry[COL_MIN_WORKTIME] = str(datetime.timedelta(seconds=min(entry.worktimes)))
      data_entry[COL_AVG_WORKTIME] = str(
        datetime.timedelta(seconds=mean(entry.worktimes)))
      data_entry[COL_MAX_WORKTIME] = str(datetime.timedelta(seconds=max(entry.worktimes)))
      data_entry[COL_TOT_WORKTIME] = str(datetime.timedelta(seconds=sum(entry.worktimes)))

    device_counts = Counter(entry.devices)
    for device in unique_devices:
      key = f"{COL_DEVICE} \"{device}\""
      assert key not in data_entry
      data_entry[key] = device_counts.get(device, 0)

    for rating_name in all_rating_names:
      col_append_str = "-ALL-" if rating_name is None else rating_name
      data_entry[f"{COL_SENT_CORR} ({col_append_str})"] = entry.sentence_correlations[rating_name]
      data_entry[f"{COL_ALGO_CORR} ({col_append_str})"] = entry.algorithm_correlations[rating_name]
      data_entry[f"{COL_BOTH_CORR} ({col_append_str})"] = entry.correlations_mean(rating_name)
    data_entry[COL_MASKED_ASSIGNMENTS] = entry.masked_assignments
    data_entry[COL_MASKED] = entry.masked

    browser_counter = Counter(entry.browser_strings)
    browser_strings = []
    for browser_string, count in browser_counter.most_common():
      browser_strings.append(f"{count}x \"{browser_string}\"")
    data_entry[COL_BROWSERS] = "; ".join(browser_strings)

    comment_counter = Counter(entry.comments)
    if len(comment_counter) == 1 and "" in comment_counter.keys():
      data_entry[COL_COMMENTS] = ""
    else:
      comments = []
      for comment, count in comment_counter.most_common():
        if comment == "":
          comments.append(f"{count}x empty")
        else:
          comments.append(f"{count}x \"{comment}\"")
      data_entry[COL_COMMENTS] = "; ".join(comments)

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

  row[COL_COUNT_DUP_RATINGS] = df[COL_COUNT_DUP_RATINGS].sum()
  row[COL_AVG_DEV_DUP_RATINGS] = df[COL_AVG_DEV_DUP_RATINGS].mean()

  row[COL_N_TRAPS] = df[COL_N_TRAPS].sum()
  row[COL_N_FALLEN_TRAPS] = df[COL_N_FALLEN_TRAPS].sum()

  row[COL_PAGE_HIDDEN_MIN_DUR] = df[COL_PAGE_HIDDEN_MIN_DUR].min()
  row[COL_PAGE_HIDDEN_AVG_DUR] = df[COL_PAGE_HIDDEN_AVG_DUR].mean()
  row[COL_PAGE_HIDDEN_MAX_DUR] = df[COL_PAGE_HIDDEN_MAX_DUR].max()
  row[COL_PAGE_HIDDEN_TOT_DUR] = df[COL_PAGE_HIDDEN_TOT_DUR].sum()
  row[COL_ACTIVE_SESSIONS_MIN] = df[COL_ACTIVE_SESSIONS_MIN].min()
  row[COL_ACTIVE_SESSIONS_AVG] = df[COL_ACTIVE_SESSIONS_AVG].mean()
  row[COL_ACTIVE_SESSIONS_MAX] = df[COL_ACTIVE_SESSIONS_MAX].max()
  row[COL_COUNT_BROWSERS] = df[COL_ACTIVE_SESSIONS_MAX].mean()
  row[COL_BROWSERS] = np.nan

  row[COL_LISTENED_FILE_COUNT] = df[COL_LISTENED_FILE_COUNT].sum()
  row[COL_OVERPLAYED_MIN] = df[COL_OVERPLAYED_MIN].min()
  row[COL_OVERPLAYED_AVG] = df[COL_OVERPLAYED_AVG].mean()
  row[COL_OVERPLAYED_MAX] = df[COL_OVERPLAYED_MAX].max()
  row[COL_OVERPLAYED_RATE] = df[COL_OVERPLAYED_RATE].mean()
  row[COL_OVERFULLPLAYED_MIN] = df[COL_OVERFULLPLAYED_MIN].min()
  row[COL_OVERFULLPLAYED_AVG] = df[COL_OVERFULLPLAYED_AVG].mean()
  row[COL_OVERFULLPLAYED_MAX] = df[COL_OVERFULLPLAYED_MAX].max()
  row[COL_OVERFULLPLAYED_RATE] = df[COL_OVERFULLPLAYED_RATE].mean()
  row[COL_STOPPED_MIN] = df[COL_STOPPED_MIN].min()
  row[COL_STOPPED_AVG] = df[COL_STOPPED_AVG].mean()
  row[COL_STOPPED_MAX] = df[COL_STOPPED_MAX].max()
  row[COL_STOPPED_RATE] = df[COL_STOPPED_RATE].mean()

  all_accept_times = [
    time
    for entry in stats.values()
    for time in entry.accept_times
  ]

  all_worktimes = [
    time
    for entry in stats.values()
    for time in entry.worktimes
  ]

  row[COL_OVERLAPPING_HITS] = df[COL_OVERLAPPING_HITS].sum()
  row[COL_MIN_OVERLAP_DUR] = np.nan  # TODO
  row[COL_AVG_OVERLAP_DUR] = np.nan  # TODO
  row[COL_MAX_OVERLAP_DUR] = np.nan  # TODO
  row[COL_TOT_OVERLAP_DUR] = np.nan  # TODO
  row[COL_FIRST_HIT_ACC_TIME] = min(all_accept_times).strftime(DATE_FMT)
  row[COL_FIRST_WORKTIME] = np.nan
  row[COL_FIRST_HIT_FIN_TIME] = np.nan
  row[COL_LAST_HIT_ACC_TIME] = max(all_accept_times).strftime(DATE_FMT)
  row[COL_LAST_WORKTIME] = np.nan
  row[COL_LAST_HIT_FIN_TIME] = np.nan
  row[COL_MIN_WORKTIME] = str(datetime.timedelta(seconds=min(all_worktimes)))
  row[COL_AVG_WORKTIME] = str(datetime.timedelta(seconds=mean(all_worktimes)))
  row[COL_MAX_WORKTIME] = str(datetime.timedelta(seconds=max(all_worktimes)))
  row[COL_TOT_WORKTIME] = str(datetime.timedelta(seconds=sum(all_worktimes)))
  row[COL_DIFF_WORKTIME_TIME] = np.nan

  for col in df.columns:
    if col.startswith(COL_DEVICE):
      assert col not in row
      row[col] = df[col].sum()
    if col.startswith(COL_SENT_CORR):
      assert col not in row
      row[col] = df[col].mean()
    if col.startswith(COL_ALGO_CORR):
      assert col not in row
      row[col] = df[col].mean()
    if col.startswith(COL_BOTH_CORR):
      assert col not in row
      row[col] = df[col].mean()
  row[COL_MASKED_ASSIGNMENTS] = df[COL_MASKED_ASSIGNMENTS].sum()
  row[COL_MASKED] = df[COL_MASKED].all()
  row[COL_COMMENTS] = np.nan
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)
  return df


def get_worker_assignment_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_data(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    df.sort_values([COL_FIRST_HIT_FIN_TIME, COL_WORKER], inplace=True)
    df = add_all_row(df, stats)
  return df
