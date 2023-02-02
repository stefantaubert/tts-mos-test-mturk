import numpy as np
from scipy.stats import norm, t


def get_assignment_count(assignments: np.ndarray) -> int:
  result = np.sum(~np.isnan(assignments))
  return result


def compute_bonuses(min_assignments: int):
  pass


def matlab_tinv(percentile: float, deg_of_freedom: int) -> float:
  result = -t.isf(0.5 * (1 + percentile), deg_of_freedom)
  return result
