import numpy as np

from tts_mos_test_mturk.masking.worktime_mask import get_worktime_mask


def test_component():
  worktimes = np.array([
    21,
    22,
    23,
    24,
    np.nan,
  ])

  res = get_worktime_mask(worktimes, 22.5)
  np.testing.assert_equal(res, [
    True,
    True,
    False,
    False,
    False,
  ])
