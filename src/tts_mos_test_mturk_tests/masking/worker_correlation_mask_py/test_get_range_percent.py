import numpy as np

from tts_mos_test_mturk.masking.worker_correlation_mask import get_range_percent


def test_0__0_100__returns_empty():
  indices = np.array([])
  result = get_range_percent(indices, 0, 1.00)
  np.testing.assert_equal(result, [])


def test_0__0_25__returns_empty():
  indices = np.array([])
  result = get_range_percent(indices, 0, 0.25)
  np.testing.assert_equal(result, [])


def test_4__0_25__returns_0():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0, 0.25)
  np.testing.assert_equal(result, [0])


def test_4__0_50__returns_0():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0, 0.5)
  np.testing.assert_equal(result, [0])


def test_4__0_51__returns_0_1():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0, 0.51)
  np.testing.assert_equal(result, [0, 1])


def test_4__0_100__returns_0_1_2_3():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0, 1.00)
  np.testing.assert_equal(result, [0, 1, 2, 3])


def test_4__0_101__returns_0_1_2_3():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0, 1.01)
  np.testing.assert_equal(result, [0, 1, 2, 3])


def test_4__25_51__returns_0_1():
  indices = np.array([0, 1, 2, 3])
  result = get_range_percent(indices, 0.25, 0.51)
  np.testing.assert_equal(result, [0, 1])


def test_2__0_25__returns_empty():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 0.25)
  np.testing.assert_equal(result, [0])


def test_2__0_50__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 0.5)
  np.testing.assert_equal(result, [0])


def test_2__0_51__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 0.51)
  np.testing.assert_equal(result, [0])


def test_2__0_100__returns_0_1():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 1.00)
  np.testing.assert_equal(result, [0, 1])


def test_2__0_101__returns_0_1():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 1.01)
  np.testing.assert_equal(result, [0, 1])


def test_2__25_101__returns_0_1():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.25, 1.01)
  np.testing.assert_equal(result, [0, 1])


def test_2__50_101__returns_1():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.5, 1.01)
  np.testing.assert_equal(result, [0, 1])


def test_2__50_50__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.5, 0.5)
  np.testing.assert_equal(result, [0])


def test_2__49_50__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.49, 0.5)
  np.testing.assert_equal(result, [0])


def test_2__0_33__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0, 0.33)
  np.testing.assert_equal(result, [0])


def test_2__33_33__returns_0():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.33, 0.33)
  np.testing.assert_equal(result, [0])


def test_2__33_101__returns_0_1():
  indices = np.array([0, 1])
  result = get_range_percent(indices, 0.33, 1.01)
  np.testing.assert_equal(result, [0, 1])


def test_7__25_101__returns_0():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0.25, 1.01)
  np.testing.assert_equal(result, [1, 2, 3, 4, 5, 6])


def test_7__0_25__returns_0():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0, 0.25)
  np.testing.assert_equal(result, [0])


def test_7__0_30__returns_0_1():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0, 0.3)
  np.testing.assert_equal(result, [0, 1])


def test_7__0_1_0_1__returns_0():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0.1, 0.1)
  np.testing.assert_equal(result, [0])


def test_7__0_1_0_14__returns_0():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0.1, 0.14)
  np.testing.assert_equal(result, [0])


def test_7__0_1_0_15__returns_0():
  indices = np.array([0, 1, 2, 3, 4, 5, 6])
  result = get_range_percent(indices, 0.1, 0.15)
  np.testing.assert_equal(result, [0])


def test_10__0_9_1_0__returns_8_9():
  indices = np.arange(10)
  result = get_range_percent(indices, 0.9, 1.0)
  np.testing.assert_equal(result, [8, 9])


def test_10__0_2_0_8__returns_2_to_7():
  indices = np.arange(1, 11, 1)
  result = get_range_percent(indices, 0.2, 0.8)
  np.testing.assert_equal(result, [2, 3, 4, 5, 6, 7])


def test_10__0_21_0_8__returns_3_to_7():
  indices = np.arange(1, 11, 1)
  result = get_range_percent(indices, 0.21, 0.8)
  np.testing.assert_equal(result, [3, 4, 5, 6, 7])
