from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.statistics.globals import (DEVICE_DESKTOP, DEVICE_IN_EAR, DEVICE_LAPTOP,
                                                   DEVICE_ON_EAR)


@dataclass
class WorkerEntry:
  ratings: List[float] = field(default_factory=list)
  in_ear: int = 0
  over_ear: int = 0
  laptop: int = 0
  desktop: int = 0
  masked: int = 0

  @property
  def min_ratings(self) -> float:
    if len(self.ratings) == 0:
      return np.nan
    return np.min(self.ratings)

  @property
  def max_ratings(self) -> float:
    if len(self.ratings) == 0:
      return np.nan
    return np.max(self.ratings)

  @property
  def mean_ratings(self) -> float:
    if len(self.ratings) == 0:
      return np.nan
    return np.mean(self.ratings)

  @property
  def std_ratings(self) -> float:
    if len(self.ratings) == 0:
      return np.nan
    return np.std(self.ratings)


def get_worker_stats(data: EvaluationData, masks: List[MaskBase]):
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)

  ratings = data.get_ratings()
  rmask.apply_by_nan(ratings)

  stats: Dict[str, Dict[str, WorkerEntry]] = {}

  for algorithm in data.algorithms:
    stats[algorithm] = {}
    for worker in data.workers:
      stats[algorithm][worker] = WorkerEntry()

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment_data in worker_data.assignments.values():
      for rating_data in assignment_data.ratings:
        alg_i = data.algorithms.get_loc(rating_data.algorithm)
        file_i = data.files.get_loc(rating_data.file)
        entry = stats[rating_data.algorithm][worker]

        o_is_masked = rmask.mask[alg_i, w_i, file_i]
        if o_is_masked:
          entry.masked += 1
          continue

        if assignment_data.device == DEVICE_IN_EAR:
          entry.in_ear += 1
        elif assignment_data.device == DEVICE_ON_EAR:
          entry.over_ear += 1
        elif assignment_data.device == DEVICE_LAPTOP:
          entry.laptop += 1
        else:
          assert assignment_data.device == DEVICE_DESKTOP
          entry.desktop += 1

        entry.ratings.append(rating_data.rating)

  return stats


def stats_to_df(stats: Dict[str, Dict[str, WorkerEntry]]) -> pd.DataFrame:
  csv_data = []
  for algorithm, xx in stats.items():
    for worker, entry in xx.items():
      mos_counts = Counter(entry.ratings)
      data_entry = OrderedDict((
        ("Algorithm", algorithm),
        ("WorkerId", worker),
        ("Min", entry.min_ratings),
        ("Max", entry.max_ratings),
        ("MOS", entry.mean_ratings),
        ("STD", entry.std_ratings),
        (DEVICE_IN_EAR, entry.in_ear),
        (DEVICE_ON_EAR, entry.over_ear),
        (DEVICE_LAPTOP, entry.laptop),
        (DEVICE_DESKTOP, entry.desktop),
        ("Rating 1", mos_counts.get(1, 0)),
        ("Rating 2", mos_counts.get(2, 0)),
        ("Rating 3", mos_counts.get(3, 0)),
        ("Rating 4", mos_counts.get(4, 0)),
        ("Rating 5", mos_counts.get(5, 0)),
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
    ("Rating 1", df["Rating 1"].sum()),
    ("Rating 2", df["Rating 2"].sum()),
    ("Rating 3", df["Rating 3"].sum()),
    ("Rating 4", df["Rating 4"].sum()),
    ("Rating 5", df["Rating 5"].sum()),
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
      ("Rating 1", subset["Rating 1"].sum()),
      ("Rating 2", subset["Rating 2"].sum()),
      ("Rating 3", subset["Rating 3"].sum()),
      ("Rating 4", subset["Rating 4"].sum()),
      ("Rating 5", subset["Rating 5"].sum()),
      ("Masked", subset["Masked"].sum()),
    ))
    df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  return df


def get_worker_algorithm_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_worker_stats(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    df = add_all_to_df(df)
    df.sort_values(["Algorithm", "WorkerId"], inplace=True)
  return df
