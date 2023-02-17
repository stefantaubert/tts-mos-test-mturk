from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from tts_mos_test_mturk.calculation.mos_variance import compute_alg_mos_ci95
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_logger


def get_mos_df(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  ratings = data.get_ratings()
  rmask = factory.merge_masks_into_rmask(masks)
  rmask.apply_by_nan(ratings)

  alg_mos_ci95 = compute_alg_mos_ci95(ratings)

  scores: List[Dict] = []
  for algo_i, alg_name in enumerate(data.algorithms):
    row = OrderedDict((
      ("Algorithm", alg_name),
      ("MOS", alg_mos_ci95[0, algo_i]),
      ("CI95", alg_mos_ci95[1, algo_i]),
    ))
    scores.append(row)
  result = pd.DataFrame.from_records(scores)
  return result


def generate_approve_csv(data: EvaluationData, mask_names: Set[str], reason: Optional[str], approval_cost: Optional[float]) -> pd.DataFrame:
  logger = get_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be approved: {len(assignment_indices)}")

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
  if approval_cost is not None:
    logger.info(
      f"Estimated costs ({len(assignment_indices)} assignments x {approval_cost:.2f}$): {len(assignment_indices) * approval_cost:.2f}$")
  return result


def generate_reject_csv(data: EvaluationData, mask_names: Set[str], reason: str) -> pd.DataFrame:
  logger = get_logger()

  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.masked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be rejected: {len(assignment_indices)}")

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


def generate_bonus_csv(data: EvaluationData, mask_names: Set[str], bonus: float, reason: str) -> pd.DataFrame:
  logger = get_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  worktimes_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = worktimes_amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be paid a bonus: {len(assignment_indices)}")

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
  logger = get_logger()
  logger.info(
    f"Estimated costs ({len(assignment_indices)} assignments x {bonus:.2f}$): {len(assignment_indices) * bonus:.2f}$")
  return result


def generate_ground_truth_table(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  rmask = factory.merge_masks_into_rmask(masks)

  results: List[Dict[str, Any]] = []
  for data_point in data.data:
    w_i = data.workers.get_loc(data_point.worker_id)
    a_i = data.algorithms.get_loc(data_point.algorithm)
    s_i = data.files.get_loc(data_point.file)
    is_masked = rmask.mask[a_i, w_i, s_i]
    line = OrderedDict((
        ("WorkerId", data_point.worker_id),
        ("Algorithm", data_point.algorithm),
        ("File", data_point.file),
        ("Rating", data_point.rating),
        ("AssignmentWorktime (s)", data_point.worktime),
        ("Device", data_point.listening_device),
        ("AssignmentState", data_point.state),
        ("AssignmentId", data_point.assignment_id),
        ("Masked", is_masked),
        ("Audio-URL", data_point.audio_url),
      ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result
