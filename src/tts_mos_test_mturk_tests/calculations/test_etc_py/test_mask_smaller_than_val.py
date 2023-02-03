import numpy as np

from tts_mos_test_mturk.calculation.etc import mask_smaller_than_val


def test_component():
  Z = np.array([
    [np.nan, 8, 4, np.nan],
    [4, 8, np.nan, np.nan],
    [np.nan, np.nan, 4, 8],
  ])
  result = mask_smaller_than_val(Z, val=5)
  np.testing.assert_array_equal(result, [
    [False, False, True, False],
    [True, False, False, False],
    [False, False, True, False],
  ])


def test_empty__returns_empty():
  Z = np.array([])
  result = mask_smaller_than_val(Z, val=5)
  np.testing.assert_array_equal(result, [])
