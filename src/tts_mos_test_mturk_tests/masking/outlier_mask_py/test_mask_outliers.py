from math import inf

import numpy as np

from tts_mos_test_mturk.masking.outlier_mask import mask_outliers

_ = np.nan


def test_component_simple():
  worktimes = np.array([
    [4, 5, _],
    [4, 4, 4],
    [_, 1, 5],
    [_, _, _],
  ])

  res = mask_outliers(worktimes, 1, inf)

  np.testing.assert_equal(res, [
    [False, False, False],
    [False, False, False],
    [False, True, False],
    [False, False, False],
  ])


def test_component():
  worktimes = np.array([
    [4, 5, _, 1, 2, _],
    [4, 4, 4, 1, 1, 1],
    [_, 1, 5, _, 2, 5],
    [_, _, _, _, _, _],
  ])

  res = mask_outliers(worktimes, 1, inf)

  # all 1's and 5's are outliers
  np.testing.assert_equal(res, [
    [False, True, False, True, False, False],
    [False, False, False, True, True, True],
    [False, True, True, False, False, True],
    [False, False, False, False, False, False],
  ])
