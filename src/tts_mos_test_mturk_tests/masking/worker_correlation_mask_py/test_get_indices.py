import numpy as np

from tts_mos_test_mturk.masking.worker_correlation_mask import get_indices


def test_component():
  indices = np.array([1, 3, 4, 7])
  correlations = np.array([-0.3, 0.8, 0.2, 0.4])
  result = get_indices(indices, correlations, 0.25, 0.76)
  np.testing.assert_equal(result, [1, 4, 7])
