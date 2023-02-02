from math import sqrt
from typing import Tuple

import numpy as np
import scipy
from scipy.stats import norm, t

from tts_mos_test_mturk.calculation.etc import matlab_tinv
from tts_mos_test_mturk.calculation.mos_variance import mos_variance
from tts_mos_test_mturk.calculation.remove_outliers import remove_outliers


def compute_mos_ci95_3gaussian(Z: np.ndarray) -> Tuple[float, float]:
  # Computes the MOS and 95% confidence interval for the mean, using a sum of 3
  # Gaussians model.
  #
  # For more details, see
  #
  # F. Ribeiro, D. Florencio, C. Zhang and M. Seltzer, "crowdMOS: An Approach for
  # Crowdsourcing Mean Opinion Score Studies", submitted to ICASSP 2011.

  # norm.ppf(0.95)
  Z = remove_outliers(Z)
  mos = np.nanmean(Z.flatten())

  mos_var = mos_variance(Z)
  ci95 = matlab_tinv(0.95, min(Z.shape) - 1) * sqrt(mos_var)
  return mos, ci95
