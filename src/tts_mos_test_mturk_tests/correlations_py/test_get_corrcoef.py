import numpy as np

from tts_mos_test_mturk.correlations import get_corrcoef

_ = np.nan


def test_0_999999999():
  v = np.array([
    [
      1,
      2,
      np.nan,
      np.nan,
      4,
    ],
    [
      np.nan,
      1,
      3,
      np.nan,
      5,
    ],
  ])

  res = get_corrcoef(v)

  assert res == 0.9999999999999999


def test_component():
  v = np.array([
    [5, 4, 1, 2, _, 5, 1, 3, 4, 4, _, 1, _, 4, 5, 2, 1, 4, _, 2, 2, 4, 5, 3, _, _],
    [5, 5, 1, 1, 5, _, 1, 3, 5, 5, 2, _, _, 5, _, 1, 2, 4, _, 1, _, 4, 4, 2, _, _],
  ])

  result = get_corrcoef(v)

  assert result == 0.8856578155269456


def test_2x0__returns_nan():
  v = np.array([
    [],
    [],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x1__returns_nan():
  v = np.array([
    [1],
    [2],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x2_1_2__1_2__returns_1():
  v = np.array([
    [1, 2],
    [1, 2],
  ])

  result = get_corrcoef(v)

  assert result == 0.9999999999999999


def test_2x2_1_nan__1_nan__returns_nan():
  v = np.array([
    [1, _],
    [1, _],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x2_nan_1__1_nan__returns_nan():
  v = np.array([
    [1, _],
    [_, 1],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x3_1_1_1__nan_1_1__returns_nan():
  v = np.array([
    [1, 1, 2],
    [_, 1, 2],
  ])

  result = get_corrcoef(v)

  assert result == 0.9999999999999999


def test_2x2_1_1__2_3__returns_nan():
  v = np.array([
    [1, 1],
    [2, 3],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x2_2_3__1_1__returns_nan():
  v = np.array([
    [2, 3],
    [1, 1],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x2_1_1__1_1__returns_nan():
  v = np.array([
    [1, 1],
    [1, 1],
  ])

  result = get_corrcoef(v)

  assert np.isnan(result)


def test_2x2_2_3__2_3__returns_1():
  v = np.array([
    [2, 3],
    [2, 3],
  ])

  result = get_corrcoef(v)

  assert result == 0.9999999999999999
