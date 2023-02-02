import numpy as np


def corrcoef(u: np.ndarray, v: np.ndarray) -> float:
  u = u.flatten()
  v = v.flatten()
  N = len(u)

  u = u - np.nanmean(u)
  v = v - np.nanmean(v)

  c = (np.nansum(u * v) / (N - 1)) / (np.nanstd(u) * np.nanstd(v))
  return c
