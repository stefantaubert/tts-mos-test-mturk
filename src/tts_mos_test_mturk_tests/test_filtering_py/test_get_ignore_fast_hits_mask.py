import numpy as np

from tts_mos_test_mturk.filtering_old import get_too_fast_hits_mask


def test_comp():
  worktimes = np.array([
    21,  # 0
    22,  # 1
    23,  # 2
    24,  # 3
  ])
  Z_ass = np.full((2, 3, 4), fill_value=np.nan)
  Z_ass[0, 0, 0] = 0
  Z_ass[0, 1, 1] = 2
  Z_ass[1, 2, 2] = 1
  res = get_too_fast_hits_mask(worktimes, Z_ass, 22.5)
  np.testing.assert_equal(res, [
    [
      [True, False, False, False],
      [False, False, False, False],
      [False, False, False, False]
    ],
    [
      [False, False, False, False],
      [False, False, False, False],
      [False, False, True, False]
     ]
  ])
