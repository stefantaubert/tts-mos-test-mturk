import math
from logging import getLogger
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import get_algorithm_mos_correlations
from tts_mos_test_mturk.core.evaluation_data import (AssignmentMask, EvaluationData, MaskBase,
                                                     WorkerMask, get_assignment_mask_from_masks,
                                                     get_opinion_score_mask_from_masks,
                                                     get_worker_mask_from_masks)


def print_stats(data: EvaluationData, masks: List[MaskBase], added_mask: MaskBase) -> None:
  if isinstance(added_mask, WorkerMask):
    print_worker_stats(data, masks, added_mask)

  if isinstance(added_mask, (WorkerMask, AssignmentMask)):
    print_assignment_stats(data, masks, added_mask)

  print_opinion_score_stats(data, masks, added_mask)


def print_opinion_score_stats(data: EvaluationData, masks: List[MaskBase], added_mask: MaskBase) -> None:
  logger = getLogger(__name__)

  opinion_scores = data.get_opinion_scores()
  opinion_scores_mask = get_opinion_score_mask_from_masks(masks, data)
  opinion_scores_mask.apply_to(opinion_scores)
  old_count = np.sum(~np.isnan(opinion_scores))

  opinion_scores_mask = get_opinion_score_mask_from_masks(masks + [added_mask], data)
  opinion_scores_mask.apply_to(opinion_scores)
  new_count = np.sum(~np.isnan(opinion_scores))

  logger.info(f"Ignored {old_count - new_count} / {old_count} opinion scores, kept {new_count}!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_mask: MaskBase) -> None:
  logger = getLogger(__name__)

  assignments_mask = get_assignment_mask_from_masks(masks, data)
  old_count = assignments_mask.n_unmasked

  assignments_mask = get_assignment_mask_from_masks(masks + [added_mask], data)
  new_count = assignments_mask.n_unmasked

  logger.info(f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}!")


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_mask: MaskBase) -> None:
  logger = getLogger(__name__)

  workers_mask = get_worker_mask_from_masks(masks, data)
  old_count = workers_mask.n_unmasked

  workers_mask = get_worker_mask_from_masks(masks + [added_mask], data)
  new_count = workers_mask.n_unmasked

  logger.info(f"Ignored {old_count - new_count} / {old_count} workers, kept {new_count}!")
