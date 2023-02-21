import numpy as np

from tts_mos_test_mturk.masking.etc import mask_values_in_boundary


def test_component():
  worktimes = np.array([
    12,
    20,
    21,
    22,
    23,
    24,
    np.nan,
  ])

  res = mask_values_in_boundary(worktimes, 20, 23)

  np.testing.assert_equal(res, [
    False,
    True,
    True,
    True,
    False,
    False,
    False,
  ])
