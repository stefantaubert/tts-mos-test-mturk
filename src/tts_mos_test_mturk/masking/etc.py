import math
from typing import Literal, Set, Tuple

import numpy as np

from tts_mos_test_mturk.calculation.correlations import get_mos_correlations
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def sort_indices_after_values(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
  sub_sorted_indices = np.argsort(values)
  sub_sorted_indices = np.array(list(sub_sorted_indices))
  # correlations_sorted = sub_worker_correlations[sub_sorted_indices]
  sub_windices_sorted = indices[sub_sorted_indices]
  return sub_windices_sorted
