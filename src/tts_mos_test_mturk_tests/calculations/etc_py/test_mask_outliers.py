import numpy as np

from tts_mos_test_mturk.__old.etc import mask_outliers


def test_component():
  Z = np.array([
    [np.nan, 8, 7, 15],
    [7, 8, np.nan, np.nan],
    [np.nan, np.nan, 4, 8],
  ])
  result = mask_outliers(Z, max_std_dev_diff=1.0)
  np.testing.assert_array_equal(result, [
    [False, False, False, True],
    [False, False, False, False],
    [False, False, True, False],
  ])


def test_empty__returns_empty():
  Z = np.empty((0, 0))
  result = mask_outliers(Z, max_std_dev_diff=1.0)
  np.testing.assert_array_equal(result, np.empty((0, 0)))
