import numpy as np

from tts_mos_test_mturk.correlations import get_algorithm_mos_correlation


def test_component():
  _ = np.nan
  ratings = np.array([
    [
      [5, 4, 1],
      [4, 5, 2],
      [5, 5, 1],
      [3, _, 1],
      [_, 4, _],
      [_, _, _],
    ],
    [
      [2, _, 5],
      [1, 4, _],
      [1, 5, _],
      [2, 4, _],
      [_, _, _],
      [_, _, _],
    ],
    [
      [1, 3, 4],
      [2, 2, 4],
      [1, 3, 5],
      [1, _, 3],
      [_, 2, _],
      [_, _, _],
    ],
    [
      [4, _, 1],
      [5, 3, _],
      [5, 2, _],
      [4, 2, _],
      [_, _, _],
      [_, _, _],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)
  r3 = get_algorithm_mos_correlation(2, ratings)
  r4 = get_algorithm_mos_correlation(3, ratings)
  r5 = get_algorithm_mos_correlation(4, ratings)
  r6 = get_algorithm_mos_correlation(5, ratings)

  assert r1 == -0.11503946170860961
  assert r2 == 0.33583319917566934
  assert r3 == 0.7646444421810364
  assert r4 == 0.024551430807011734
  assert r5 == 0.9999999999999999
  assert np.isnan(r6)


def test_one_algorithm_return_nan_for_each_worker():
  ratings = np.array([
    [
      [5, 4, 4],
      [1, 2, 1],
      [2, 2, 1],
      [1, 2, 2],
    ]
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)
  r3 = get_algorithm_mos_correlation(2, ratings)
  r4 = get_algorithm_mos_correlation(3, ratings)

  assert np.isnan(r1)
  assert np.isnan(r2)
  assert np.isnan(r3)
  assert np.isnan(r4)


def test_all_nan_return_for_each_worker():
  ratings = np.array([
    [
      [np.nan, np.nan],
      [np.nan, np.nan],
    ],
    [
      [np.nan, np.nan],
      [np.nan, np.nan],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)

  assert np.isnan(r1)
  assert np.isnan(r2)


def test_positive_correlation():
  ratings = np.array([
    [
      [5, 4, 4],
      [4, 5, 5],
    ],
    [
      [2, 2, 1],
      [1, 2, 2],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)

  assert r1 == 1.0
  assert r2 == 0.9999999999999999


def test_negative_correlation():
  ratings = np.array([
    [
      [5, 4, 4],
      [1, 2, 2],
    ],
    [
      [2, 2, 1],
      [5, 5, 4],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)

  assert r1 == -1.0
  assert r2 == -0.9999999999999999


def test_one_negative_two_positive_correlation():
  ratings = np.array([
    [
      [5, 4, 4, 4],
      [1, 2, 2, 1],
      [2, 1, 2, 1],
    ],
    [
      [2, 2, 1, 2],
      [5, 5, 4, 4],
      [4, 5, 4, 4],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)
  r3 = get_algorithm_mos_correlation(2, ratings)

  assert r1 == -1.0
  assert r2 == 1.0
  assert r3 == 1.0


def test_one_zero_two_negative_correlation():
  ratings = np.array([
    [
      [5, 4, 4, 4],
      [1, 2, 2, 1],
      [3, 3, 3, 2],
    ],
    [
      [2, 2, 1, 2],
      [5, 5, 4, 4],
      [1, 2, 2, 1],
    ],
    [
      [5, 4, 4, 4],
      [3, 2, 3, 2],
      [3, 3, 3, 2],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, ratings)
  r2 = get_algorithm_mos_correlation(1, ratings)
  r3 = get_algorithm_mos_correlation(2, ratings)

  assert r1 == -0.8219949365267863
  assert r2 == -0.9449111825230679
  assert r3 == 0.0
