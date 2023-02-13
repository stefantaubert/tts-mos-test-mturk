from collections import OrderedDict
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from tts_mos_test_mturk.analyze_assignmens import (get_algorithm_mos_correlation,
                                                   get_sentence_mos_correlation,
                                                   get_sentence_mos_correlation_3dim)
from tts_mos_test_mturk.core.data_point import (DEVICE_DESKTOP, DEVICE_IN_EAR, DEVICE_LAPTOP,
                                                DEVICE_ON_EAR, STATE_ACCEPTED, STATE_APPROVED,
                                                STATE_REJECTED)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.masks import AssignmentMask, MaskBase, WorkerMask


def print_stats(data: EvaluationData, mask_names: Set[str], added_mask_names: Set[str]) -> None:
  logger = getLogger(__name__)
  logger.info("--- Stats ---")
  masks = [data.masks[mask_name] for mask_name in mask_names]
  added_masks = [data.masks[mask_name] for mask_name in added_mask_names]
  print_stats_masks(data, masks, added_masks)
  logger.info("----------")


def print_stats_masks(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  if len(added_masks) == 0 or any(isinstance(m, WorkerMask) for m in added_masks):
    print_worker_stats(data, masks, added_masks)

  if len(added_masks) == 0 or any(isinstance(m, (WorkerMask, AssignmentMask)) for m in added_masks):
    print_assignment_stats(data, masks, added_masks)

  print_opinion_score_stats(data, masks, added_masks)


def print_opinion_score_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)
  factory = data.get_mask_factory()

  opinion_scores = data.get_os()

  opinion_scores_mask_before = factory.merge_masks_into_omask(masks)
  opinion_scores_mask_before.apply_by_nan(opinion_scores)
  old_count = np.sum(~np.isnan(opinion_scores))

  opinion_scores_mask_after = factory.merge_masks_into_omask(masks + added_masks)
  opinion_scores_mask_after.apply_by_nan(opinion_scores)
  new_count = np.sum(~np.isnan(opinion_scores))

  logger.info(f"Ignored {old_count - new_count} / {old_count} opinion scores, kept {new_count}!")


def print_assignment_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)
  factory = data.get_mask_factory()

  assignments_mask_before = factory.merge_masks_into_amask(masks)
  assignments_mask_after = factory.merge_masks_into_amask(masks + added_masks)

  old_count = assignments_mask_before.n_unmasked
  new_count = assignments_mask_after.n_unmasked

  assignments_mask_new_ignored = np.logical_xor(
    assignments_mask_before.mask, assignments_mask_after.mask)
  ignored_assignments = sorted(data.assignments[assignments_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} assignments, kept {new_count}!")
  if len(ignored_assignments) > 0:
    # logger.info(f"Ignored assignments: {', '.join(ignored_assignments)}")
    pass


def print_worker_stats(data: EvaluationData, masks: List[MaskBase], added_masks: List[MaskBase]) -> None:
  logger = getLogger(__name__)
  factory = data.get_mask_factory()

  workers_mask_before = factory.merge_masks_into_wmask(masks)
  workers_mask_after = factory.merge_masks_into_wmask(masks + added_masks)

  old_count = workers_mask_before.n_unmasked
  new_count = workers_mask_after.n_unmasked

  workers_mask_new_ignored = np.logical_xor(workers_mask_before.mask, workers_mask_after.mask)
  ignored_workers = sorted(data.workers[workers_mask_new_ignored.nonzero()[0]])

  logger.info(f"Ignored {old_count - new_count} / {old_count} workers, kept {new_count}!")
  if len(ignored_workers) > 0:
    # logger.info(f"Ignored workers: {', '.join(ignored_workers)}")
    pass

