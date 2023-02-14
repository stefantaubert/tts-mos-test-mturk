from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import compute_ci95, compute_mos
from tts_mos_test_mturk.core.evaluation_data import EvaluationData


def get_mos_df(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  scores: List[Dict] = []
  for algo_i, alg_name in enumerate(data.algorithms):
    row = OrderedDict((
      ("Algorithm", alg_name),
      ("MOS", compute_mos(os[algo_i])),
      ("CI95", compute_ci95(os[algo_i])),
    ))
    scores.append(row)
  result = pd.DataFrame.from_records(scores)
  return result


def generate_approve_csv(data: EvaluationData, mask_names: Set[str], reason: Optional[str]) -> Optional[pd.DataFrame]:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  if reason is None:
    reason = "x"

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("Approve", reason),
      ("Reject", ""),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result


def generate_reject_csv(data: EvaluationData, mask_names: Set[str], reason: str) -> Optional[pd.DataFrame]:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.masked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("Approve", ""),
      ("Reject", reason),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result


def generate_bonus_csv(data: EvaluationData, mask_names: Set[str], bonus: float, reason: str) -> Optional[pd.DataFrame]:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("BonusAmount", bonus),
      ("Reason", reason),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result


def generate_ground_truth_table(data: EvaluationData, mask_names: Set[str]) -> Optional[pd.DataFrame]:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  omask = factory.merge_masks_into_omask(masks)

  results: List[Dict[str, Any]] = []
  for dp in data.data:
    w_i = data.workers.get_loc(dp.worker_id)
    a_i = data.algorithms.get_loc(dp.algorithm)
    s_i = data.files.get_loc(dp.file)
    is_masked = omask.mask[a_i, w_i, s_i]
    line = OrderedDict((
        ("WorkerId", dp.worker_id),
        ("Algorithm", dp.algorithm),
        ("File", dp.file),
        ("Score", dp.opinion_score),
        ("AssignmentWorktime (s)", dp.worktime),
        ("Device", dp.listening_device),
        ("AssignmentState", dp.state),
        ("AssignmentId", dp.assignment_id),
        ("Masked", is_masked),
        ("Audio-URL", dp.audio_url),
      ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result
