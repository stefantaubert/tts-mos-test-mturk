from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List
from typing import OrderedDict as ODType
from typing import Set, Union

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase

COL_ALG = "Alg."
COL_WORKER = "WorkerId"
COL_MIN = "Min"
COL_MAX = "Max"
COL_MOS = "Avg"
COL_STD = "Std"
COL_DEVICE = "#D="
COL_RATING = "#R="
COL_MASKED = "#Masked"
COL_ALL = "-ALL-"


@dataclass
class WorkerEntry:
  ratings: ODType[str, List[Union[int, float]]] = field(default_factory=OrderedDict)
  devices: List[str] = field(default_factory=list)
  masked: int = 0

  def min_ratings(self, rating_name: str) -> Union[int, float]:
    if rating_name not in self.ratings or len(self.ratings[rating_name]) == 0:
      return np.nan
    return np.min(self.ratings[rating_name])

  def max_ratings(self, rating_name: str) -> Union[int, float]:
    if rating_name not in self.ratings or len(self.ratings[rating_name]) == 0:
      return np.nan
    return np.max(self.ratings[rating_name])

  def mean_ratings(self, rating_name: str) -> float:
    if rating_name not in self.ratings or len(self.ratings[rating_name]) == 0:
      return np.nan
    return np.mean(self.ratings[rating_name])

  def std_ratings(self, rating_name: str) -> float:
    if rating_name not in self.ratings or len(self.ratings[rating_name]) == 0:
      return np.nan
    return np.std(self.ratings[rating_name])


def get_worker_stats(data: EvaluationData, masks: List[MaskBase]):
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)
  stats: Dict[str, Dict[str, WorkerEntry]] = {}

  for algorithm in data.algorithms:
    stats[algorithm] = {}
    for worker in data.workers:
      stats[algorithm][worker] = WorkerEntry()

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment_data in worker_data.assignments.values():
      for (alg_name, file_name), ass_ratings in assignment_data.ratings.items():
        alg_i = data.algorithms.get_loc(alg_name)
        file_i = data.files.get_loc(file_name)
        entry = stats[alg_name][worker]

        o_is_masked = rmask.mask[alg_i, w_i, file_i]
        if o_is_masked:
          entry.masked += 1
          continue

        entry.devices.append(assignment_data.device)
        for rating_name, rating in ass_ratings.votes.items():
          if rating_name not in entry.ratings:
            entry.ratings[rating_name] = []
          entry.ratings[rating_name].append(rating)

  return stats


def stats_to_df(stats: Dict[str, Dict[str, WorkerEntry]]) -> pd.DataFrame:
  csv_data = []
  rating_names = sorted({
    k
    for x in stats.values()
    for y in x.values()
    for k in y.ratings.keys()
  })

  unique_ratings = OrderedDict()
  for rating_name in rating_names:
    tmp = sorted({
      rating
      for worker_entries in stats.values()
      for worker_entry in worker_entries.values()
      for rating in worker_entry.ratings.get(rating_name, [])
    })
    unique_ratings[rating_name] = tmp

  unique_devices = sorted({
    d
    for x in stats.values()
    for y in x.values()
    for d in y.devices
  })

  for algorithm, xx in stats.items():
    for worker, entry in xx.items():
      device_counts = Counter(entry.devices)
      data_entry = OrderedDict()
      data_entry[COL_ALG] = algorithm
      data_entry[COL_WORKER] = worker

      for device in unique_devices:
        key = f"{COL_DEVICE}{device}"
        assert key not in data_entry
        data_entry[key] = device_counts.get(device, 0)

      for rating_name, unique_vals in unique_ratings.items():
        data_entry[f"{COL_MIN}({rating_name})"] = entry.min_ratings(rating_name)
        data_entry[f"{COL_MAX}({rating_name})"] = entry.max_ratings(rating_name)
        data_entry[f"{COL_MOS}({rating_name})"] = entry.mean_ratings(rating_name)
        data_entry[f"{COL_STD}({rating_name})"] = entry.std_ratings(rating_name)
        if rating_name in entry.ratings:
          rating_counts = Counter(entry.ratings[rating_name])
        else:
          rating_counts = Counter()
        for unique_val in unique_vals:
          key = f"{COL_RATING}({rating_name})={unique_val}"
          assert key not in data_entry
          data_entry[key] = rating_counts.get(unique_val, 0)
      data_entry[COL_MASKED] = entry.masked
      csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_to_df(df: pd.DataFrame) -> pd.DataFrame:
  algorithms = df[COL_ALG].unique()

  row = {}
  row[COL_ALG] = COL_ALL
  row[COL_WORKER] = COL_ALL

  col: str
  for col in df.columns:
    if col.startswith(COL_DEVICE):
      assert col not in row
      row[col] = df[col].sum()

  for col in df.columns:
    if col.startswith(COL_MIN):
      assert col not in row
      row[col] = df[col].min()
    if col.startswith(COL_MAX):
      assert col not in row
      row[col] = df[col].max()
    if col.startswith(COL_MOS):
      assert col not in row
      row[col] = df[col].mean()
    if col.startswith(COL_STD):
      assert col not in row
      row[col] = df[col].mean()
    if col.startswith(COL_RATING):
      assert col not in row
      row[col] = df[col].sum()

  row[COL_MASKED] = df[COL_MASKED].sum()
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  for algorithm in algorithms:
    subset: pd.DataFrame = df.loc[df[COL_ALG] == algorithm]
    row = {
      COL_ALG: algorithm,
      COL_WORKER: COL_ALL,
    }

    for col in subset.columns:
      if col.startswith(COL_DEVICE):
        row[col] = subset[col].sum()

    for col in subset.columns:
      if col.startswith(COL_MIN):
        assert col not in row
        row[col] = subset[col].min()
      if col.startswith(COL_MAX):
        assert col not in row
        row[col] = subset[col].max()
      if col.startswith(COL_MOS):
        assert col not in row
        row[col] = subset[col].mean()
      if col.startswith(COL_STD):
        assert col not in row
        row[col] = subset[col].mean()
      if col.startswith(COL_RATING):
        assert col not in row
        row[col] = subset[col].sum()

    row[COL_MASKED] = subset[COL_MASKED].sum()
    df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  return df


def get_worker_algorithm_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_worker_stats(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    df = add_all_to_df(df)
    df.sort_values([COL_ALG, COL_WORKER], inplace=True)
  return df
