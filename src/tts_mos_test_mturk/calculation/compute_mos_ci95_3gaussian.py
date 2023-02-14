from math import sqrt

import numpy as np
from scipy.stats import t

from tts_mos_test_mturk.calculation.mos_variance import mos_variance


def matlab_tinv(p: float, df: int) -> float:
  result = -t.isf(p, df)
  return result


def compute_mos(Z: np.ndarray) -> float:
  mos = np.nanmean(Z.flatten())
  return mos


def compute_ci95(Z: np.ndarray) -> float:
  v_mu = mos_variance(Z)
  t = matlab_tinv(.5 * (1 + .95), min(Z.shape) - 1)
  ci95 = t * sqrt(v_mu)
  return ci95


# def compute_mos_ci95_3gaussian(Z: np.ndarray) -> Tuple[float, float]:
#   # Computes the MOS and 95% confidence interval for the mean, using a sum of 3
#   # Gaussians model.
#   #
#   # For more details, see
#   #
#   # F. Ribeiro, D. Florencio, C. Zhang and M. Seltzer, "crowdMOS: An Approach for
#   # Crowdsourcing Mean Opinion Score Studies", submitted to ICASSP 2011.

#   # Z = remove_outliers(Z)
#   mos = np.nanmean(Z.flatten())

#   v_mu = mos_variance(Z)
#   t = matlab_tinv(.5 * (1 + .95), min(Z.shape) - 1)
#   ci95 = t * sqrt(v_mu)
#   return mos, ci95
