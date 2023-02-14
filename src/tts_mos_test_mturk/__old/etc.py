from typing import Generator, List

import numpy as np
from ordered_set import OrderedSet


def get_assignment_count(assignments: np.ndarray) -> int:
  result = np.sum(~np.isnan(assignments))
  return result


def compute_bonuses(min_assignments: int):
  pass



def mask_smaller_than_val(Z: np.ndarray, val: float) -> np.ndarray:
  result = Z < val
  return result


def mask_lower_outliers(Z: np.ndarray, max_std_dev_diff: float) -> np.ndarray:
  N, M = Z.shape

  mu = np.nanmean(Z)
  s = np.nanstd(Z)

  tmp = Z - np.tile(mu, (N, 1))
  mu_norm = -tmp / np.tile(s, (N, 1))
  outlying_scores: np.ndarray = mu_norm > max_std_dev_diff

  return outlying_scores


def mask_workers(Z_mask: np.ndarray, p: float) -> np.ndarray:
  # remove subjects which have more than 'bad_score_percent' * 100 % of outlying scores
  sums = np.sum(Z_mask, axis=1)
  total = np.sum(Z_mask.flatten())
  outlying_workers: np.ndarray = sums >= p * total
  return outlying_workers


def mask_workers_percent(Z_mask: np.ndarray, p: float) -> np.ndarray:
  # remove subjects which have more than 'bad_score_percent' * 100 % of outlying scores
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)
  total = np.sum(Z_mask.flatten())
  outlying_workers: np.ndarray = sums2 >= p * total
  # x = np.copy(Z_mask)
  Z_mask[:, outlying_workers, :] = True
  return Z_mask



def get_workers_percent(Z_mask: np.ndarray) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)
  total = np.sum(Z_mask.flatten())

  percent: np.ndarray = sums2 / total
  return percent


def get_workers_count(Z_mask: np.ndarray) -> np.ndarray:
  sums = np.sum(Z_mask, axis=2)
  sums2 = np.sum(sums, axis=0)

  percent: np.ndarray = sums2
  return percent

# def split_algos(array: np.ndarray, audio_files: OrderedSet[str], split_paths: OrderedSet[str]) -> List[np.ndarray]:
#   for split_path in split_paths:
#     indices = [
#       audio_files.get_loc(audio_path)
#       for audio_path in audio_files
#       if audio_path.startswith(split_path)
#     ]
#     result = array[:, indices]
#     yield result


def mask_algos(audio_files: OrderedSet[str], split_paths: OrderedSet[str]) -> Generator[List[int], None, None]:
  for split_path in split_paths:
    indices = [
      audio_files.get_loc(audio_path)
      for audio_path in audio_files
      if audio_path.startswith(split_path)
    ]
    yield indices
