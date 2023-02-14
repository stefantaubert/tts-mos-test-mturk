from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.core.data_point import (DEVICE_DESKTOP, DEVICE_IN_EAR, DEVICE_LAPTOP,
                                                DEVICE_ON_EAR)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.masks import MaskBase


@dataclass
class WorkerEntry:
  opinion_scores: List[float] = field(default_factory=list)
  in_ear: int = 0
  over_ear: int = 0
  laptop: int = 0
  desktop: int = 0
  masked: int = 0

  @property
  def min_os(self) -> float:
    if len(self.opinion_scores) == 0:
      return np.nan
    return np.min(self.opinion_scores)

  @property
  def max_os(self) -> float:
    if len(self.opinion_scores) == 0:
      return np.nan
    return np.max(self.opinion_scores)

  @property
  def mean_os(self) -> float:
    if len(self.opinion_scores) == 0:
      return np.nan
    return np.mean(self.opinion_scores)

  @property
  def std_os(self) -> float:
    if len(self.opinion_scores) == 0:
      return np.nan
    return np.std(self.opinion_scores)


def get_worker_stats(data: EvaluationData, masks: List[MaskBase]):
  factory = data.get_mask_factory()

  omask = factory.merge_masks_into_omask(masks)

  os = data.get_os()
  omask.apply_by_nan(os)

  stats: Dict[str, Dict[str, WorkerEntry]] = {}

  for algorithm in data.algorithms:
    stats[algorithm] = {}
    for worker in data.workers:
      stats[algorithm][worker] = WorkerEntry()

  for dp in data.data:
    entry = stats[dp.algorithm][dp.worker_id]

    w_i = data.workers.get_loc(dp.worker_id)
    a_i = data.algorithms.get_loc(dp.algorithm)
    f_i = data.files.get_loc(dp.file)
    o_is_masked = omask.mask[a_i, w_i, f_i]
    if o_is_masked:
      entry.masked += 1
      continue

    if dp.listening_device == DEVICE_IN_EAR:
      entry.in_ear += 1
    elif dp.listening_device == DEVICE_ON_EAR:
      entry.over_ear += 1
    elif dp.listening_device == DEVICE_LAPTOP:
      entry.laptop += 1
    else:
      assert dp.listening_device == DEVICE_DESKTOP
      entry.desktop += 1

    entry.opinion_scores.append(dp.opinion_score)

  return stats


def stats_to_df(stats: Dict[str, Dict[str, WorkerEntry]]) -> pd.DataFrame:
  csv_data = []
  for algorithm, xx in stats.items():
    for worker, entry in xx.items():
      mos_counts = Counter(entry.opinion_scores)
      data_entry = OrderedDict((
        ("Algorithm", algorithm),
        ("WorkerId", worker),
        ("Min", entry.min_os),
        ("Max", entry.max_os),
        ("MOS", entry.mean_os),
        ("STD", entry.std_os),
        (DEVICE_IN_EAR, entry.in_ear),
        (DEVICE_ON_EAR, entry.over_ear),
        (DEVICE_LAPTOP, entry.laptop),
        (DEVICE_DESKTOP, entry.desktop),
        ("Score 1", mos_counts.get(1, 0)),
        ("Score 2", mos_counts.get(2, 0)),
        ("Score 3", mos_counts.get(3, 0)),
        ("Score 4", mos_counts.get(4, 0)),
        ("Score 5", mos_counts.get(5, 0)),
        ("Masked", entry.masked),
      ))
      csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_to_df(df: pd.DataFrame) -> pd.DataFrame:
  algorithms = df["Algorithm"].unique()

  row = OrderedDict((
    ("Algorithm", "All"),
    ("WorkerId", "All"),
    ("Min", df["Min"].min()),
    ("Max", df["Max"].max()),
    ("MOS", df["MOS"].mean()),
    ("STD", df["STD"].mean()),
    (DEVICE_IN_EAR, df[DEVICE_IN_EAR].sum()),
    (DEVICE_ON_EAR, df[DEVICE_ON_EAR].sum()),
    (DEVICE_LAPTOP, df[DEVICE_LAPTOP].sum()),
    (DEVICE_DESKTOP, df[DEVICE_DESKTOP].sum()),
    ("Score 1", df["Score 1"].sum()),
    ("Score 2", df["Score 2"].sum()),
    ("Score 3", df["Score 3"].sum()),
    ("Score 4", df["Score 4"].sum()),
    ("Score 5", df["Score 5"].sum()),
    ("Masked", df["Masked"].sum()),
  ))
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  for algorithm in algorithms:
    subset: pd.DataFrame = df.loc[df['Algorithm'] == algorithm]
    row = OrderedDict((
      ("Algorithm", algorithm),
      ("WorkerId", "All"),
      ("Min", subset["Min"].min()),
      ("Max", subset["Max"].max()),
      ("MOS", subset["MOS"].mean()),
      ("STD", subset["STD"].mean()),
      (DEVICE_IN_EAR, subset[DEVICE_IN_EAR].sum()),
      (DEVICE_ON_EAR, subset[DEVICE_ON_EAR].sum()),
      (DEVICE_LAPTOP, subset[DEVICE_LAPTOP].sum()),
      (DEVICE_DESKTOP, subset[DEVICE_DESKTOP].sum()),
      ("Score 1", subset["Score 1"].sum()),
      ("Score 2", subset["Score 2"].sum()),
      ("Score 3", subset["Score 3"].sum()),
      ("Score 4", subset["Score 4"].sum()),
      ("Score 5", subset["Score 5"].sum()),
      ("Masked", subset["Masked"].sum()),
    ))
    df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  return df


def get_worker_algorithm_stats(data: EvaluationData, mask_names: Set[str]):
  masks = data.get_masks_from_names(mask_names)
  stats = get_worker_stats(data, masks)
  df = stats_to_df(stats)
  if df is None:
    return None
  df = add_all_to_df(df)
  df.sort_values(["Algorithm", "WorkerId"], inplace=True)
  return df
