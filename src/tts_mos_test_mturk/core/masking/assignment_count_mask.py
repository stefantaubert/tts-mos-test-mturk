from typing import Set

import numpy as np

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.statistics.update_stats import print_stats_masks


def mask_workers_by_assignment_count(data: EvaluationData, mask_names: Set[str], threshold: int, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  x = factory.get_assignments_worker_index_matrix()

  amask = factory.merge_masks_into_amask(masks)
  x = amask.apply_by_del(x)

  unique_w, a_counts = np.unique(x, return_counts=True)

  matching_a_counts = a_counts < threshold
  indices = matching_a_counts.nonzero()[0]
  matching_w = unique_w[indices]

  wmask = factory.get_wmask()
  wmask.mask_indices(matching_w)

  data.add_or_update_mask(output_mask_name, wmask)

  print_stats_masks(data, masks, [wmask])
