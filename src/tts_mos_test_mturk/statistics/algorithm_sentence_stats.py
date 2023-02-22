from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase

COL_ALG = "Alg."
COL_SENT = "Sent."
COL_MIN = "Min"
COL_MAX = "Max"
COL_MOS = "Avg"
COL_STD = "Std"
COL_DEVICE = "#D="
COL_RATING = "#R="
COL_MASKED = "#Masked"
COL_ALL = "ALL"


@dataclass
class FileEntry:
  ratings: List[Union[int, float]] = field(default_factory=list)
  devices: List[str] = field(default_factory=list)
  masked: int = 0

  @property
  def min_ratings(self) -> Union[int, float]:
    if len(self.ratings) == 0:
      return np.nan
    return np.min(self.ratings)

  @property
  def max_ratings(self) -> Union[int, float]:
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

  ratings = get_ratings(data)
  rmask.apply_by_nan(ratings)

  stats: Dict[str, Dict[str, FileEntry]] = {}

  for algorithm in data.algorithms:
    stats[algorithm] = {}
    for file in data.files:
      stats[algorithm][file] = FileEntry()

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment_data in worker_data.assignments.values():
      for rating_data in assignment_data.ratings:
        alg_i = data.algorithms.get_loc(rating_data.algorithm)
        file_i = data.files.get_loc(rating_data.file)
        entry = stats[rating_data.algorithm][rating_data.file]

        o_is_masked = rmask.mask[alg_i, w_i, file_i]
        if o_is_masked:
          entry.masked += 1
          continue

        entry.devices.append(assignment_data.device)
        entry.ratings.append(rating_data.rating)

  return stats


def stats_to_df(stats: Dict[str, Dict[str, FileEntry]]) -> pd.DataFrame:
  csv_data = []
  unique_ratings = sorted({
    r
    for x in stats.values()
    for y in x.values()
    for r in y.ratings
  })

  unique_devices = sorted({
    d
    for x in stats.values()
    for y in x.values()
    for d in y.devices
  })

  for algorithm, xx in stats.items():
    for file, entry in xx.items():
      rating_counts = Counter(entry.ratings)
      device_counts = Counter(entry.devices)
      data_entry = OrderedDict((
        (COL_ALG, algorithm),
        (COL_SENT, file),
        (COL_MIN, entry.min_ratings),
        (COL_MAX, entry.max_ratings),
        (COL_MOS, entry.mean_ratings),
        (COL_STD, entry.std_ratings),
      ))
      for device in unique_devices:
        key = f"{COL_DEVICE}{device}"
        assert key not in data_entry
        data_entry[key] = device_counts.get(device, 0)
      for rating in unique_ratings:
        key = f"{COL_RATING}{rating}"
        assert key not in data_entry
        data_entry[key] = rating_counts.get(rating, 0)
      data_entry[COL_MASKED] = entry.masked
      csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data)
  return result


def add_all_to_df(df: pd.DataFrame) -> pd.DataFrame:
  algorithms = df[COL_ALG].unique()

  row = {
    COL_ALG: COL_ALL,
    COL_SENT: COL_ALL,
    COL_MIN: df[COL_MIN].min(),
    COL_MAX: df[COL_MAX].max(),
    COL_MOS: df[COL_MOS].mean(),
    COL_STD: df[COL_STD].mean(),
  }

  for col in df.columns:
    if col.startswith(COL_DEVICE):
      assert col not in row
      row[col] = df[col].sum()
    if col.startswith(COL_RATING):
      assert col not in row
      row[col] = df[col].sum()

  row[COL_MASKED] = df[COL_MASKED].sum()
  df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  for algorithm in algorithms:
    subset: pd.DataFrame = df.loc[df[COL_ALG] == algorithm]
    row = {
      COL_ALG: algorithm,
      COL_SENT: COL_ALL,
      COL_MIN: subset[COL_MIN].min(),
      COL_MAX: subset[COL_MAX].max(),
      COL_MOS: subset[COL_MOS].mean(),
      COL_STD: subset[COL_STD].mean(),
    }

    for col in subset.columns:
      if col.startswith(COL_DEVICE):
        row[col] = subset[col].sum()
      if col.startswith(COL_RATING):
        row[col] = subset[col].sum()

    row[COL_MASKED] = subset[COL_MASKED].sum()
    df = pd.concat([df, pd.DataFrame.from_records([row])], ignore_index=True)

  return df


def get_algorithm_sentence_stats(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  stats = get_worker_stats(data, masks)
  df = stats_to_df(stats)
  if len(df.index) > 0:
    df = add_all_to_df(df)
    df.sort_values([COL_ALG, COL_SENT], inplace=True)
  return df
