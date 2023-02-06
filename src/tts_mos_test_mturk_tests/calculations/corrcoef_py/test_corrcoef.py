import numpy as np


def test_component():
  result = np.corrcoef(
    np.array([
      [0, 0],
    ]),
    np.array([
      [0, 0],
    ])
  )
  res = np.nanmean(result)
  assert result == 0


def test_component2():
  result = np.corrcoef(
    np.array([
      [0, 1],
    ]),
    np.array([
      [0, 1],
    ])
  )
  res = np.nanmean(result)
  assert result == 0