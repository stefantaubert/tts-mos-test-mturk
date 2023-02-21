
import numpy as np


def sort_indices_after_values(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
  sub_sorted_indices = np.argsort(values)
  sub_sorted_indices = np.array(list(sub_sorted_indices))
  # correlations_sorted = sub_worker_correlations[sub_sorted_indices]
  sub_windices_sorted = indices[sub_sorted_indices]
  return sub_windices_sorted


def mask_values_in_boundary(a: np.ndarray, min_incl: float, max_excl: float):
  outlying_ratings = (min_incl <= a) & (a < max_excl)
  return outlying_ratings
