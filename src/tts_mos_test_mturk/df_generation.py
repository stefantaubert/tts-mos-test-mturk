from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from tts_mos_test_mturk.calculation.mos_variance import compute_alg_mos_ci95
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_logger


def get_mos_df(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  alg_mos_ci95 = compute_alg_mos_ci95(os)

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
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  work_times_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = work_times_amask.unmasked_indices
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
  if approval_cost is not None:
    logger = get_logger()
    logger.info(
      f"Estimated costs ({len(assignment_indices)} assignments x {approval_cost:.2f}$): {len(assignment_indices) * approval_cost:.2f}$")
  return result


def generate_reject_csv(data: EvaluationData, mask_names: Set[str], reason: str) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  work_times_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = work_times_amask.masked_indices
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


def generate_bonus_csv(data: EvaluationData, mask_names: Set[str], bonus: float, reason: str) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  work_times_amask = factory.merge_masks_into_amask(masks)
  assignment_indices = work_times_amask.unmasked_indices
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
  logger = get_logger()
  logger.info(
    f"Estimated costs ({len(assignment_indices)} assignments x {bonus:.2f}$): {len(assignment_indices) * bonus:.2f}$")
  return result


def generate_ground_truth_table(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  omask = factory.merge_masks_into_omask(masks)

  results: List[Dict[str, Any]] = []
  for data_point in data.data:
    w_i = data.workers.get_loc(data_point.worker_id)
    a_i = data.algorithms.get_loc(data_point.algorithm)
    s_i = data.files.get_loc(data_point.file)
    is_masked = omask.mask[a_i, w_i, s_i]
    line = OrderedDict((
        ("WorkerId", data_point.worker_id),
        ("Algorithm", data_point.algorithm),
        ("File", data_point.file),
        ("Score", data_point.opinion_score),
        ("AssignmentWorktime (s)", data_point.work_time),
        ("Device", data_point.listening_device),
        ("AssignmentState", data_point.state),
        ("AssignmentId", data_point.assignment_id),
        ("Masked", is_masked),
        ("Audio-URL", data_point.audio_url),
      ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result
