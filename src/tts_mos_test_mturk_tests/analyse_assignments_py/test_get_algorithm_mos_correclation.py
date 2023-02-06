import numpy as np

from tts_mos_test_mturk.analyze_assignmens import get_algorithm_mos_correlation


def test_component():
  _ = np.nan
  Zs = np.array([
    [
      [5, 4, 1, 2, _, 5],
      [4, 5, 2, 1, 4, _],
      [5, 5, 1, 1, 5, _],
      [3, _, 1, 2, 4, _],
      [_, 4, _, _, _, _],
      [_, _, _, _, _, _],
    ],
    [
      [1, 3, 4, 4, _, 1],
      [2, 2, 4, 5, 3, _],
      [1, 3, 5, 5, 2, _],
      [1, _, 3, 4, 2, _],
      [_, 2, _, _, _, _],
      [_, _, _, _, _, _],
    ],
  ])

  r1 = get_algorithm_mos_correlation(0, Zs)
  r2 = get_algorithm_mos_correlation(1, Zs)
  r3 = get_algorithm_mos_correlation(2, Zs)
  r4 = get_algorithm_mos_correlation(3, Zs)
  r5 = get_algorithm_mos_correlation(4, Zs)
  r6 = get_algorithm_mos_correlation(5, Zs)

  assert r1 == 0.9076506967077385
  assert r2 == 0.8751487988018714
  assert r3 == 0.9740033294906888
  assert r4 == 0.9164481985434293
  assert r5 == 1
  assert np.isnan(r6)
