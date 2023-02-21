from math import inf

import numpy as np

from tts_mos_test_mturk.masking.outlier_mask import mask_outliers_alg

_ = np.nan


def test_component():
  worktimes = np.array([
    [
      [4, 5, _],
      [4, 4, 4],
      [_, 1, 5],
      [_, _, _],
    ],
    [
      [1, 2, _],
      [1, 1, 1],
      [_, 2, 5],
      [_, _, _],
    ]
  ])

  res = mask_outliers_alg(worktimes, 1, inf)
  np.testing.assert_equal(res, [
    [
      [False, False, False],
      [False, False, False],
      [False, True, False],
      [False, False, False],
    ],
    [
      [False, False, False],
      [False, False, False],
      [False, False, True],
      [False, False, False],
    ]
  ])
