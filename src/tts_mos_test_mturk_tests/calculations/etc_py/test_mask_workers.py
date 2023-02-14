import numpy as np

from tts_mos_test_mturk.__old.etc import mask_workers


def test_component_25():
  mask = np.array([
    [False, False, False, False],
    [True, False, False, False],  # 1
    [False, False, False, False],
    [True, True, False, False],  # 2
    [True, False, False, False],  # 1
  ])

  result = mask_workers(mask, 0.25)
  np.testing.assert_array_equal(result, [
    False,
    True,
    False,
    True,
    True,
  ])


def test_component_50():
  mask = np.array([
    [False, False, False, False],
    [True, False, False, False],  # 1
    [False, False, False, False],
    [True, True, False, False],  # 2
    [True, False, False, False],  # 1
  ])

  result = mask_workers(mask, 0.5)
  np.testing.assert_array_equal(result, [
    False,
    False,
    False,
    True,
    False,
  ])


def test_component_51():
  mask = np.array([
    [False, False, False, False],
    [True, False, False, False],  # 1
    [False, False, False, False],
    [True, True, False, False],  # 2
    [True, False, False, False],  # 1
  ])

  result = mask_workers(mask, 0.51)
  np.testing.assert_array_equal(result, [
    False,
    False,
    False,
    False,
    False,
  ])


def test_empty__returns_empty():
  mask = np.empty((0, 0))
  result = mask_workers(mask, p=0.5)
  np.testing.assert_array_equal(result, np.empty((0,)))
