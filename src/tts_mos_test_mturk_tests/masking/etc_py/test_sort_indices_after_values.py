import numpy as np

from tts_mos_test_mturk.masking.etc import sort_indices_after_values


def test_sorting_with_same_values_takes_input_sorting():
  inp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19])
  vals = np.array([-2., 0.44981464, 0.47163612, 0.52961867, -2.,
                   0.42201074, 0.45084383, 0.46545402, -2., 0.30659144,
                   0.34573972, -2., -2., -2., -2.,
                   -2., -2., -0.41610076, -0.16052528, -2.])
  res = sort_indices_after_values(inp, vals)
  np.testing.assert_equal(res, [
    0, 4, 8, 11, 12, 13, 14, 15, 16, 19, 17, 18, 9, 10, 5, 1, 6,
      7, 2, 3
      ])
