from typing import Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger
from tts_mos_test_mturk.masking.etc import sort_indices_after_values
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_workers_by_assignment_count(data: EvaluationData, mask_names: Set[str], from_threshold_incl: int, to_threshold_excl: int, output_mask_name: str):
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  if wmask.n_masked > 0:
    dlogger.info('Already masked workers:')
    for worker_name in sorted(data.workers[wmask.masked_indices]):
      dlogger.info(f'- "{worker_name}"')

  aw_matrix = factory.get_assignments_worker_index_matrix()
  aw_matrix = amask.apply_by_del(aw_matrix)

  windices, a_counts = np.unique(aw_matrix, return_counts=True)

  matching_a_counts = (from_threshold_incl <= a_counts) & (a_counts < to_threshold_excl)
  masked_indices = matching_a_counts.nonzero()[0]
  matching_w = windices[masked_indices]

  windices_sorted = sort_indices_after_values(windices, a_counts)
  a_counts_sorted = sort_indices_after_values(a_counts, a_counts)

  dlogger.info("Worker ranking by assignment count:")
  for nr, (w_i, count) in enumerate(zip(windices_sorted, a_counts_sorted), start=1):
    masked_str = " [masked]" if w_i in matching_w else ""
    dlogger.info(
      f"{nr}. \"{data.workers[w_i]}\": {count}{masked_str}")

  wmask = factory.get_wmask()
  wmask.mask_indices(matching_w)

  data.add_or_update_mask(output_mask_name, wmask)

  print_stats_masks(data, masks, [wmask])
