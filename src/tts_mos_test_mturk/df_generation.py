from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from mean_opinion_score import get_ci95, get_mos

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory


def get_mos_df(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  logger = get_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  ratings = get_ratings(data)
  all_ratings_count = np.sum(~np.isnan(ratings))

  rmask = factory.merge_masks_into_rmask(masks)
  rmask.apply_by_nan(ratings)

  ratings_count = np.sum(~np.isnan(ratings))

  logger.info(
    f"Count of ratings (unmasked/all): {ratings_count}/{all_ratings_count} -> on average {round(ratings_count/data.n_algorithms)}/{round(all_ratings_count/data.n_algorithms)} per algorithm")

  scores: List[Dict] = []
  for algo_i, alg_name in enumerate(data.algorithms):
    alg_ratings = ratings[algo_i]
    mos = get_mos(alg_ratings)
    ci = get_ci95(alg_ratings)
    row = OrderedDict((
      ("Algorithm", alg_name),
      ("MOS", mos),
      ("CI95", ci),
    ))
    scores.append(row)
  result = pd.DataFrame.from_records(scores)
  return result


def generate_approve_csv(data: EvaluationData, mask_names: Set[str], reason: Optional[str], approval_cost: Optional[float], amazon_fee: Optional[float]) -> pd.DataFrame:
  logger = get_logger()
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers (will be approved):")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (will not be approved):")
    for nr, w in enumerate(sorted(masked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No masked workers exist.")

  assignment_indices = amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be approved: {len(assignment_indices)}")

  if reason is None:
    reason = "x"

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("HITId", hit_id),
      ("Approve", reason),
      ("Reject", ""),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  if approval_cost is not None:
    if amazon_fee is None:
      amazon_fee = 0.0
    assert amazon_fee >= 0
    costs = len(assignment_indices) * approval_cost
    fees = costs * amazon_fee
    # Ex.:
    # Estimated Total Reward: 480 assignments x $0.15 = $30
    # Estimated Fees to Mechanical Turk: 20.00% of $30 = $2
    # Estimated Cost: Total Reward + Fees = $32
    logger.info(
      f"Estimated Total Reward: {len(assignment_indices)} assignments x ${approval_cost:.2f} = ${costs:.2f}")
    logger.info(
      f"Estimated Fees to Mechanical Turk: {amazon_fee*100:.2f}% of ${costs:.2f} = ${fees:.2f}")
    logger.info(
      f"Estimated Cost: Total Reward + Fees = ${costs + fees:.2f}")
  return result


def generate_reject_csv(data: EvaluationData, mask_names: Set[str], reason: str) -> pd.DataFrame:
  logger = get_logger()
  dlogger = get_detail_logger()

  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (will be rejected):")
    for nr, w in enumerate(sorted(masked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No masked workers exist.")

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers (will not be rejected):")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  assignment_indices = amask.masked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be rejected: {len(assignment_indices)}")

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("HITId", hit_id),
      ("Approve", ""),
      ("Reject", reason),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result


def generate_bonus_csv(data: EvaluationData, mask_names: Set[str], bonus: float, reason: str, amazon_fee: float) -> pd.DataFrame:
  assert amazon_fee >= 0
  logger = get_logger()
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers (will be paid a bonus):")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (will not be paid a bonus):")
    for nr, w in enumerate(sorted(masked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No masked workers exist.")

  assignment_indices = amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be paid a bonus: {len(assignment_indices)}")

  results: List[Dict[str, Any]] = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("AssignmentId", assignment_id),
      ("WorkerId", worker_id),
      ("HITId", hit_id),
      ("BonusAmount", bonus),
      ("Reason", reason),
    ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  logger = get_logger()
  costs = len(assignment_indices) * bonus
  fees = costs * amazon_fee
  # Ex.:
  # Estimated Total Bonus: 480 assignments x $0.15 = $30
  # Estimated Fees to Mechanical Turk: 20.00% of $30 = $2
  # Estimated Cost: Total Bonus + Fees = $32
  logger.info(
    f"Estimated Total Bonus: {len(assignment_indices)} assignments x ${bonus:.2f} = ${costs:.2f}")
  logger.info(
    f"Estimated Fees to Mechanical Turk: {amazon_fee*100:.2f}% of ${costs:.2f} = ${fees:.2f}")
  logger.info(
    f"Estimated Cost: Total Reward + Fees = ${costs + fees:.2f}")
  return result


def generate_ground_truth_table(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)

  results: List[Dict[str, Any]] = []

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment, assignment_data in worker_data.assignments.items():
      for rating_data in assignment_data.ratings:
        alg_i = data.algorithms.get_loc(rating_data.algorithm)
        file_i = data.files.get_loc(rating_data.file)
        is_masked = rmask.mask[alg_i, w_i, file_i]
        line = OrderedDict((
            ("Worker", worker),
            ("Algorithm", rating_data.algorithm),
            ("File", rating_data.file),
            ("Rating", rating_data.rating),
            ("Worktime (s)", assignment_data.worktime),
            ("Device", assignment_data.device),
            ("State", assignment_data.state),
            ("HIT", assignment_data.hit_id),
            ("Assignment", assignment),
            ("Masked?", is_masked),
          ))
    results.append(line)
  result = pd.DataFrame.from_records(results)
  return result
