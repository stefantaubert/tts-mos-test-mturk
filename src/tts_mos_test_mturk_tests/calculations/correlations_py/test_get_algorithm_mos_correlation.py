import numpy as np

from tts_mos_test_mturk.calculation.correlations import get_algorithm_mos_correlation


def test_component():
  _ = np.nan
  opinion_scores = np.array([
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

  r1 = get_algorithm_mos_correlation(0, opinion_scores)
  r2 = get_algorithm_mos_correlation(1, opinion_scores)
  r3 = get_algorithm_mos_correlation(2, opinion_scores)
  r4 = get_algorithm_mos_correlation(3, opinion_scores)
  r5 = get_algorithm_mos_correlation(4, opinion_scores)
  r6 = get_algorithm_mos_correlation(5, opinion_scores)

  assert r1 == -0.11503946170860961
  assert r2 == 0.33583319917566934
  assert r3 == 0.7646444421810364
  assert r4 == 0.024551430807011734
  assert r5 == 0.9999999999999999
  assert np.isnan(r6)
