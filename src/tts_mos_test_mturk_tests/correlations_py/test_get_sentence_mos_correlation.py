import numpy as np

from tts_mos_test_mturk.correlations import get_sentence_mos_correlation


def test_component():
  _ = np.nan
  Zs = np.array([
    [5, 4, 1, 2, _, 5, 1, 3, 4, 4, _, 1],
    [4, 5, 2, 1, 4, _, 2, 2, 4, 5, 3, _],
    [5, 5, 1, 1, 5, _, 1, 3, 5, 5, 2, _],
    [3, _, 1, 2, 4, _, 1, _, 3, 4, 2, _],
    [_, 4, _, _, _, _, _, 2, _, _, _, _],
    [_, _, _, _, _, _, _, _, _, _, _, _],
  ])

  r1 = get_sentence_mos_correlation(0, Zs)
  r2 = get_sentence_mos_correlation(1, Zs)
  r3 = get_sentence_mos_correlation(2, Zs)
  r4 = get_sentence_mos_correlation(3, Zs)
  r5 = get_sentence_mos_correlation(4, Zs)
  r6 = get_sentence_mos_correlation(5, Zs)

  assert r1 == 0.9076506967077385
  assert r2 == 0.8751487988018714
  assert r3 == 0.9740033294906888
  assert r4 == 0.9164481985434293
  assert r5 == 1
  assert np.isnan(r6)


def test_zero_corr():
  Zs = np.array([
    [5, 4, 4],
    [1, 2, 1],
    [2, 2, 1],
  ])

  r = get_sentence_mos_correlation(0, Zs)
  assert r == 0


def test_get_high_corr():
  Zs = np.array([
    [5, 4, 1, 2],
    [4, 5, 2, 1],
    [5, 5, 1, 1],
    [3, 4, 1, 2],
  ])

  r = get_sentence_mos_correlation(0, Zs)
  assert r == 0.9024731748508101


def test_get_low_corr():
  Zs = np.array([
    [5, 4, 1, 2],
    [1, 2, 4, 5],
    [1, 1, 5, 5],
    [2, 1, 3, 4],
  ])

  r = get_sentence_mos_correlation(0, Zs)
  assert r == -0.9024731748508104


def test_get_almost_zero_corr():
  Zs = np.array([
    [5, 4, 1, 2],
    [3, 2, 3, 3],
    [2, 2, 2, 3],
    [3, 2, 2, 3],
  ])

  r = get_sentence_mos_correlation(0, Zs)
  assert r == -0.14142135623730975
