import numpy as np

from tts_mos_test_mturk.analyze_assignmens import get_corrcoef


def test_component():
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

  get_corrcoef(v)
