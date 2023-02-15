import numpy as np

from tts_mos_test_mturk.masking.work_time_mask import get_work_time_mask


def test_component():
  worktimes = np.array([
    21,  # 0
    22,  # 1
    23,  # 2
    24,  # 3
    np.nan,
  ])

  res = get_work_time_mask(worktimes, 22.5)
  np.testing.assert_equal(res, [
    True,
    True,
    False,
    False,
    False,
  ])
