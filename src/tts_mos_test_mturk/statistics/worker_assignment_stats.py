from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.correlations import (get_algorithm_mos_correlation,
                                             get_sentence_mos_correlation_3dim)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase

COL_WORKER = "WorkerId"
COL_TOT_ASSIGNMENTS = "#Assignments"
COL_STATE = "#S="
COL_DEVICE = "#D="
COL_SENT_CORR = "Sent. corr."
COL_ALGO_CORR = "Alg. corr."
COL_BOTH_CORR = "Corr."
COL_AVG_WORKTIME = "Avg. worktime (s)"
COL_TOT_WORKTIME = "Tot. worktime (min)"
COL_MASKED_ASSIGNMENTS = "#Masked assignments"
COL_MASKED = "Masked?"
COL_ALL = "ALL"


@dataclass
class WorkerEntry:
  masked: bool = False
  masked_assignments: int = 0
  statuses: List[str] = field(default_factory=list)
  worktimes: List[Union[int, float]] = field(default_factory=list)
  devices: List[str] = field(default_factory=list)
  sentence_correlations: Dict[str, float] = field(default_factory=dict)
  algorithm_correlations: Dict[str, float] = field(default_factory=dict)

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

  for worker in data.workers:
    stats[worker] = WorkerEntry()

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment, assignment_data in worker_data.assignments.items():
      entry = stats[worker]

      skip = False
      w_is_masked = wmask.mask[w_i]
      if w_is_masked:
        entry.masked = True
        skip = True
      a_i = data.assignments.get_loc(assignment)
      a_is_masked = amask.mask[a_i]
      if a_is_masked:
        entry.masked_assignments += 1
        skip = True

      if skip:
        continue

      entry.devices.append(assignment_data.device)
      entry.statuses.append(assignment_data.state)
      entry.worktimes.append(assignment_data.worktime)
      for rating_name, ratings in all_ratings.items():
        entry.algorithm_correlations[rating_name] = get_algorithm_mos_correlation(w_i, ratings)
        entry.sentence_correlations[rating_name] = get_sentence_mos_correlation_3dim(w_i, ratings)

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

    status_counts = Counter(entry.statuses)
    for status in unique_statuses:
      key = f"{COL_STATE}{status}"
      assert key not in data_entry
      data_entry[key] = status_counts.get(status, 0)
    data_entry[COL_TOT_ASSIGNMENTS] = entry.total_assignments

    data_entry[COL_AVG_WORKTIME] = np.mean(entry.worktimes)
    data_entry[COL_TOT_WORKTIME] = np.sum(entry.worktimes) / 60

    device_counts = Counter(entry.devices)
    for device in unique_devices:
      key = f"{COL_DEVICE}{device}"
      assert key not in data_entry
      data_entry[key] = device_counts.get(device, 0)

    for rating_name in all_rating_names:
      col_append_str = "ALL" if rating_name is None else rating_name
      data_entry[f"{COL_SENT_CORR} ({col_append_str})"] = entry.sentence_correlations[rating_name]
      data_entry[f"{COL_ALGO_CORR} ({col_append_str})"] = entry.algorithm_correlations[rating_name]
      data_entry[f"{COL_BOTH_CORR} ({col_append_str})"] = entry.correlations_mean(rating_name)
    data_entry[COL_MASKED_ASSIGNMENTS] = entry.masked_assignments
    data_entry[COL_MASKED] = entry.masked
    csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_row(df: pd.DataFrame) -> pd.DataFrame:
  row = OrderedDict()
  row[COL_WORKER] = COL_ALL

  col: str
  for col in df.columns:
    if col.startswith(COL_STATE):
      assert col not in row
      row[col] = df[col].sum()
  row[COL_TOT_ASSIGNMENTS] = df[COL_TOT_ASSIGNMENTS].sum()

  row[COL_AVG_WORKTIME] = df[COL_AVG_WORKTIME].mean()
  row[COL_TOT_WORKTIME] = df[COL_TOT_WORKTIME].sum()

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
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)
  return df


def get_worker_assignment_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_data(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    df = add_all_row(df)
    df.sort_values([COL_WORKER], inplace=True)
  return df
