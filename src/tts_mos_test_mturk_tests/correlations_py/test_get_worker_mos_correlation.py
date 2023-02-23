import numpy as np

from tts_mos_test_mturk.correlations import get_worker_mos_correlations


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

  r = get_worker_mos_correlations(ratings)

  assert r[0] == 0.39630561749956444
  assert r[1] == 0.6054909989887703
  assert r[2] == 0.8693238858358626
  assert r[3] == 0.4704998146752205
  assert r[4] == 1.0
  assert np.isnan(r[5])
  assert len(r) == 6


def test_one_algorithm_return_nan_for_each_worker():
  ratings = np.array([
    [
      [5, 4, 4],
      [1, 2, 1],
      [2, 2, 1],
      [1, 2, 2],
    ]
  ])

  r = get_worker_mos_correlations(ratings)

  assert r[0] == -0.49999999999999994
  assert r[1] == 0.5000000000000001
  assert r[2] == 0.5000000000000001
  assert r[3] == -0.5000000000000001
  assert len(r) == 4


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

  r = get_worker_mos_correlations(ratings)

  assert np.isnan(r[0])
  assert np.isnan(r[1])
  assert len(r) == 2


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

  r = get_worker_mos_correlations(ratings)

  assert r[0] == 0.9122424289477238
  assert r[1] == 0.9122424289477238
  assert len(r) == 2


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

  r = get_worker_mos_correlations(ratings)

  assert r[0] == -0.949719013397517
  assert r[1] == -0.9497190133975169
  assert len(r) == 2


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

  r = get_worker_mos_correlations(ratings)

  assert r[0] == -0.9444444444444444
  assert r[1] == 0.6012739367083666
  assert r[2] == 0.7434322477800739
  assert len(r) == 3


def test_one_almost_zero_two_negative_correlation():
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

  r = get_worker_mos_correlations(ratings)

  assert r[0] == -0.6545894593638613
  assert r[1] == -0.8608101960290726
  assert r[2] == 0.06741998624632421
  assert len(r) == 3
