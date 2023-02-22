from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

from tts_mos_test_mturk.calculation.correlations import (get_algorithm_mos_correlation,
                                                         get_sentence_mos_correlation_3dim)
from tts_mos_test_mturk.common import get_ratings
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
  sentence_corr: float = None
  algorithm_corr: float = None

  @property
  def total_assignments(self) -> int:
    return len(self.statuses)

  @property
  def correlation_mean(self) -> float:
    if self.sentence_corr is None:
      if self.algorithm_corr is None:
        return None
      return self.algorithm_corr
    if self.algorithm_corr is None:
      return self.sentence_corr
    return np.mean([self.sentence_corr, self.algorithm_corr])


def get_data(data: EvaluationData, masks: List[MaskBase]):
  factory = MaskFactory(data)

  wmask = factory.merge_masks_into_wmask(masks)
  amask = factory.merge_masks_into_amask(masks)
  rmask = factory.merge_masks_into_rmask(masks)

  ratings = get_ratings(data)
  rmask.apply_by_nan(ratings)

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

      if entry.algorithm_corr is None:
        entry.algorithm_corr = get_algorithm_mos_correlation(w_i, ratings)
      if entry.sentence_corr is None:
        entry.sentence_corr = get_sentence_mos_correlation_3dim(w_i, ratings)

  return stats


def stats_to_df(stats: Dict[str, WorkerEntry]) -> pd.DataFrame:
  csv_data = []
  unique_statuses = sorted({
    s
    for x in stats.values()
    for s in x.statuses
  })

  unique_devices = sorted({
    d
    for x in stats.values()
    for d in x.devices
  })

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

    data_entry[COL_SENT_CORR] = entry.sentence_corr
    data_entry[COL_ALGO_CORR] = entry.algorithm_corr
    data_entry[COL_BOTH_CORR] = entry.correlation_mean
    data_entry[COL_MASKED_ASSIGNMENTS] = entry.masked_assignments
    data_entry[COL_MASKED] = entry.masked
    csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_row(df: pd.DataFrame) -> pd.DataFrame:
  row = OrderedDict()
  row[COL_WORKER] = COL_ALL

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

  row[COL_SENT_CORR] = df[COL_SENT_CORR].mean()
  row[COL_ALGO_CORR] = df[COL_ALGO_CORR].mean()
  row[COL_BOTH_CORR] = df[COL_BOTH_CORR].mean()
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
