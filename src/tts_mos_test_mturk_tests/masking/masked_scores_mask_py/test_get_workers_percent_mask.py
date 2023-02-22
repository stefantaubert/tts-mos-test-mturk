import numpy as np

from tts_mos_test_mturk.masking.masked_count_mask import get_wmask_percent


def test_component_25():
  mask = np.array([
    [
      [False, False, False, False],
      [False, False, False, False],  # 1
      [False, False, False, False],
      [True, False, False, False],  # 2
      [True, False, False, False],  # 1
    ],
    [
      [False, False, False, False],
      [True, False, False, False],  # 1
      [False, False, False, False],
      [True, False, False, False],  # 2
      [False, False, False, False],  # 1
    ]
  ])

  result = get_wmask_percent(mask, 0.25, 1.0)
  np.testing.assert_array_equal(result, [
    False,
    True,
    False,
    True,
    True,
  ])


def test_all_false():
  mask = np.array([
    [
      [False, False, False, False],
      [False, False, False, False],  # 1
      [False, False, False, False],
      [False, False, False, False],  # 2
      [False, False, False, False],  # 1
    ],
    [
      [False, False, False, False],
      [False, False, False, False],  # 1
      [False, False, False, False],
      [False, False, False, False],  # 2
      [False, False, False, False],  # 1
    ]
  ])

  result = get_wmask_percent(mask, 0.25, 1.0)
  np.testing.assert_array_equal(result, [
    False,
    False,
    False,
    False,
    False,
  ])


def test_component_50():
  mask = np.array([
    [
      [False, False, False, False],
      [True, False, False, False],  # 1
      [False, False, False, False],
      [True, True, False, False],  # 2
      [True, False, False, False],  # 1
    ]
  ])

  result = get_wmask_percent(mask, 0.5, 1.0)
  np.testing.assert_array_equal(result, [
    False,
    False,
    False,
    True,
    False,
  ])


def test_component_51():
  mask = np.array([
    [
      [False, False, False, False],
      [True, False, False, False],  # 1
      [False, False, False, False],
      [True, True, False, False],  # 2
      [True, False, False, False],  # 1
    ]
  ])

  result = get_wmask_percent(mask, 0.51, 1.0)
  np.testing.assert_array_equal(result, [
    False,
    False,
    False,
    False,
    False,
  ])


def test_empty__returns_empty():
  mask = np.empty((0, 0, 0))
  result = get_wmask_percent(mask, 0.5, 1.0)
  np.testing.assert_array_equal(result, np.empty((0,)))
