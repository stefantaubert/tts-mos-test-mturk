import numpy as np

from tts_mos_test_mturk.masking.worker_correlation_mask import get_indices


def test_component():
  indices = np.array([1, 3, 4, 7])
  correlations = np.array([-0.3, 0.8, 0.2, 0.4])
  result = get_indices(indices, correlations, 0.25, 0.76)
  np.testing.assert_equal(result, [1, 4, 7])


def test_component_nan():
  indices = np.array([1, 3, 4, 5, 7])
  correlations = np.array([-0.3, 0.8, 0.2, np.nan, 0.4])
  result = get_indices(indices, correlations, 0.20, 0.81)
  np.testing.assert_equal(result, [1, 4, 7, 3])


def test_component_nan2():
  indices = np.array([1, 3, 4, 5, 7])
  correlations = np.array([-0.3, 0.8, 0.2, np.nan, 0.4])
  result = get_indices(indices, correlations, 0.0, 0.81)
  np.testing.assert_equal(result, [1, 4, 7, 3])
