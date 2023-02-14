from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.analyze_assignmens import (get_algorithm_mos_correlation,
                                                   get_sentence_mos_correlation_3dim)
from tts_mos_test_mturk.data_point import (DEVICE_DESKTOP, DEVICE_IN_EAR, DEVICE_LAPTOP,
                                           DEVICE_ON_EAR, STATE_ACCEPTED, STATE_APPROVED,
                                           STATE_REJECTED)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masks import MaskBase


@dataclass
class WorkerEntry:
  masked: bool = False
  masked_assignments: int = 0
  rejected_assignments: int = 0
  accepted_assignments: int = 0
  approved_assignments: int = 0
  worktimes: List[float] = field(default_factory=list)
  in_ear: int = 0
  over_ear: int = 0
  laptop: int = 0
  desktop: int = 0
  sentence_corr: float = None
  algorithm_corr: float = None

  @property
  def total_assignments(self) -> int:
    return self.rejected_assignments + self.accepted_assignments + self.approved_assignments

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
  factory = data.get_mask_factory()

  wmask = factory.merge_masks_into_wmask(masks)
  amask = factory.merge_masks_into_amask(masks)
  omask = factory.merge_masks_into_omask(masks)

  os = data.get_os()
  omask.apply_by_nan(os)

  stats: Dict[str, WorkerEntry] = {}

  for worker in data.workers:
    stats[worker] = WorkerEntry()

  for dp in data.data:
    entry = stats[dp.worker_id]

    skip = False
    w_i = data.workers.get_loc(dp.worker_id)
    w_is_masked = wmask.mask[w_i]
    if w_is_masked:
      entry.masked = True
      skip = True
    a_i = data.assignments.get_loc(dp.assignment_id)
    a_is_masked = amask.mask[a_i]
    if a_is_masked:
      entry.masked_assignments += 1
      skip = True

    if skip:
      continue

    entry = stats[dp.worker_id]
    if dp.listening_device == DEVICE_IN_EAR:
      entry.in_ear += 1
    elif dp.listening_device == DEVICE_ON_EAR:
      entry.over_ear += 1
    elif dp.listening_device == DEVICE_LAPTOP:
      entry.laptop += 1
    else:
      assert dp.listening_device == DEVICE_DESKTOP
      entry.desktop += 1

    if dp.state == STATE_ACCEPTED:
      entry.accepted_assignments += 1
    elif dp.state == STATE_REJECTED:
      entry.rejected_assignments += 1
    else:
      assert dp.state == STATE_APPROVED
      entry.approved_assignments += 1
    entry.worktimes.append(dp.worktime)

    if entry.algorithm_corr is None:
      entry.algorithm_corr = get_algorithm_mos_correlation(w_i, os)
    if entry.sentence_corr is None:
      entry.sentence_corr = get_sentence_mos_correlation_3dim(w_i, os)

  return stats


def stats_to_df(stats: Dict[str, WorkerEntry]) -> pd.DataFrame:
  csv_data = []
  for worker, entry in stats.items():
    data_entry = OrderedDict((
      ("WorkerId", worker),
      ("Total Assignments", entry.total_assignments),
      ("Rejected Assignments", entry.rejected_assignments),
      ("Approved Assignments", entry.approved_assignments),
      ("Accepted Assignments", entry.accepted_assignments),
      ("Average worktime (s)", np.mean(entry.worktimes)),
      ("Total worktime (min)", np.sum(entry.worktimes) / 60),
      (DEVICE_IN_EAR, entry.in_ear),
      (DEVICE_ON_EAR, entry.over_ear),
      (DEVICE_LAPTOP, entry.laptop),
      (DEVICE_DESKTOP, entry.desktop),
      ("Sentence correlation", entry.sentence_corr),
      ("Algorithm correlation", entry.algorithm_corr),
      ("Correlation", entry.correlation_mean),
      ("Masked Assignments", entry.masked_assignments),
      ("Masked", entry.masked),
    ))
    csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_row(df: pd.DataFrame) -> pd.DataFrame:
  row = OrderedDict((
    ("WorkerId", "All"),
    ("Total Assignments", df["Total Assignments"].sum()),
    ("Rejected Assignments", df["Rejected Assignments"].sum()),
    ("Approved Assignments", df["Approved Assignments"].sum()),
    ("Accepted Assignments", df["Accepted Assignments"].sum()),
    ("Average worktime (s)", df["Average worktime (s)"].mean()),
    ("Total worktime (min)", df["Total worktime (min)"].sum()),
    (DEVICE_IN_EAR, df[DEVICE_IN_EAR].sum()),
    (DEVICE_ON_EAR, df[DEVICE_ON_EAR].sum()),
    (DEVICE_LAPTOP, df[DEVICE_LAPTOP].sum()),
    (DEVICE_DESKTOP, df[DEVICE_DESKTOP].sum()),
    ("Sentence correlation", df["Sentence correlation"].mean()),
    ("Algorithm correlation", df["Algorithm correlation"].mean()),
    ("Correlation", df["Correlation"].mean()),
    ("Masked Assignments", df["Masked Assignments"].sum()),
    ("Masked", df["Masked"].sum()),
  ))
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)
  return df


def get_worker_assignment_stats(data: EvaluationData, mask_names: Set[str]):
  masks = data.get_masks_from_names(mask_names)
  stats = get_data(data, masks)
  df = stats_to_df(stats)
  if df is None:
    return None
  df = add_all_row(df)
  df.sort_values(["WorkerId"], inplace=True)
  return df
