
import numpy as np

from tts_mos_test_mturk.calculation.mos_variance import get_custom_variance


def test_1dim_len_0__returns_nan():
  vec = np.array([])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_1dim_len_1__returns_nan():
  vec = np.array([1])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_1dim_len_2__returns_var():
  vec = np.array([1, 2])
  result = get_custom_variance(vec)
  assert result == 0.25


def test_1dim_len_3__returns_var():
  vec = np.array([1, 2, 3])
  result = get_custom_variance(vec)
  assert result == 0.6666666666666666


def test_1dim_len_2_1_nan__returns_nan():
  vec = np.array([1, np.nan])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_1dim_len_2_2_nan__returns_nan():
  vec = np.array([np.nan, np.nan])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_1dim_len_3_1_nan__returns_var():
  vec = np.array([1, 2, np.nan])
  result = get_custom_variance(vec)
  assert result == 0.25


def test_2dim_len_0__returns_nan():
  vec = np.array([[]])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_2dim_len_1__returns_nan():
  vec = np.array([[1]])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_2dim_len_2__returns_var():
  vec = np.array([[1, 2]])
  result = get_custom_variance(vec)
  assert result == 0.25


def test_2dim_len_3__returns_var():
  vec = np.array([[1, 2, 3]])
  result = get_custom_variance(vec)
  assert result == 0.6666666666666666


def test_2dim_len_3_sep__returns_var():
  vec = np.array([[1], [2], [3]])
  result = get_custom_variance(vec)
  assert result == 0.6666666666666666


def test_2dim_len_2_1_nan__returns_nan():
  vec = np.array([[1, np.nan]])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_2dim_len_2_2_nan__returns_nan():
  vec = np.array([[np.nan, np.nan]])
  result = get_custom_variance(vec)
  assert np.isnan(result)


def test_2dim_len_3_1_nan__returns_var():
  vec = np.array([[1, 2, np.nan]])
  result = get_custom_variance(vec)
  assert result == 0.25