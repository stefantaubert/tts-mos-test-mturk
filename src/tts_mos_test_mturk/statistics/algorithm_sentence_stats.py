from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.statistics.globals import (DEVICE_DESKTOP, DEVICE_IN_EAR, DEVICE_LAPTOP,
                                                   DEVICE_ON_EAR)

COL_ALG = "Algorithm"
COL_SENT = "Sentence"
COL_MIN = "Min"
COL_MAX = "Max"
COL_MOS = "MOS"
COL_STD = "STD"
COL_INEAR = DEVICE_IN_EAR
COL_ONEAR = DEVICE_ON_EAR
COL_LAPTOP = DEVICE_LAPTOP
COL_DESKTOP = DEVICE_DESKTOP
COL_MOS1 = "Rating 1"
COL_MOS2 = "Rating 2"
COL_MOS3 = "Rating 3"
COL_MOS4 = "Rating 4"
COL_MOS5 = "Rating 5"
COL_MASKED = "Masked"
COL_ALL = "All"


COLS = [
  COL_ALG,
  COL_SENT,
  COL_MIN,
  COL_MAX,
  COL_MOS,
  COL_STD,
  COL_INEAR,
  COL_ONEAR,
  COL_LAPTOP,
  COL_DESKTOP,
  COL_MOS1,
  COL_MOS2,
  COL_MOS3,
  COL_MOS4,
  COL_MOS5,
  COL_MASKED,
]


@dataclass
class FileEntry:
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


def stats_to_df(stats: Dict[str, Dict[str, FileEntry]]) -> pd.DataFrame:
  csv_data = []
  for algorithm, xx in stats.items():
    for file, entry in xx.items():
      rating_counts = Counter(entry.ratings)
      data_entry = {
        COL_ALG: algorithm,
        COL_SENT: file,
        COL_MIN: entry.min_os,
        COL_MAX: entry.max_os,
        COL_MOS: entry.mean_os,
        COL_STD: entry.std_os,
        COL_INEAR: entry.in_ear,
        COL_ONEAR: entry.over_ear,
        COL_LAPTOP: entry.laptop,
        COL_DESKTOP: entry.desktop,
        COL_MOS1: rating_counts.get(1, 0),
        COL_MOS2: rating_counts.get(2, 0),
        COL_MOS3: rating_counts.get(3, 0),
        COL_MOS4: rating_counts.get(4, 0),
        COL_MOS5: rating_counts.get(5, 0),
        COL_MASKED: entry.masked,
      }
      csv_data.append(data_entry)
  result = pd.DataFrame.from_records(csv_data, columns=COLS)
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
    COL_INEAR: df[COL_INEAR].sum(),
    COL_ONEAR: df[COL_ONEAR].sum(),
    COL_LAPTOP: df[COL_LAPTOP].sum(),
    COL_DESKTOP: df[COL_DESKTOP].sum(),
    COL_MOS1: df[COL_MOS1].sum(),
    COL_MOS2: df[COL_MOS2].sum(),
    COL_MOS3: df[COL_MOS3].sum(),
    COL_MOS4: df[COL_MOS4].sum(),
    COL_MOS5: df[COL_MOS5].sum(),
    COL_MASKED: df[COL_MASKED].sum(),
  }
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
      COL_INEAR: subset[COL_INEAR].sum(),
      COL_ONEAR: subset[COL_ONEAR].sum(),
      COL_LAPTOP: subset[COL_LAPTOP].sum(),
      COL_DESKTOP: subset[COL_DESKTOP].sum(),
      COL_MOS1: subset[COL_MOS1].sum(),
      COL_MOS2: subset[COL_MOS2].sum(),
      COL_MOS3: subset[COL_MOS3].sum(),
      COL_MOS4: subset[COL_MOS4].sum(),
      COL_MOS5: subset[COL_MOS5].sum(),
      COL_MASKED: subset[COL_MASKED].sum(),
    }
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
