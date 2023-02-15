import numpy as np

from tts_mos_test_mturk.masking.work_time_mask import get_work_time_mask


def test_component():
  work_times = np.array([
    21,
    22,
    23,
    24,
    np.nan,
  ])

  res = get_work_time_mask(work_times, 22.5)
  np.testing.assert_equal(res, [
    True,
    True,
    False,
    False,
    False,
  ])
