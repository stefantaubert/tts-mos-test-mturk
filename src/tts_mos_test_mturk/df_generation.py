import datetime
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from typing import OrderedDict as ODType
from typing import Set, Tuple

import numpy as np
import pandas as pd
from mean_opinion_score import get_ci95, get_ci95_default, get_mos
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.typing import MaskName

ALL_CELL_CONTENT = "-ALL-"

def get_row(row_template: ODType, ratings: np.ndarray, ratings_masked: np.ndarray, algo_i: int, masks: List[MaskBase]) -> ODType:
  row = row_template.copy()
  current_ratings_masked = ratings_masked.copy()
  for mask in masks:
    mask.apply_by_nan(current_ratings_masked)
  row["MOS"] = get_mos(current_ratings_masked[algo_i])
  row["CI95"] = get_ci95(current_ratings_masked[algo_i])
  row["CI95 (default)"] = get_ci95_default(current_ratings_masked[algo_i])
  row["STD"] = np.nanstd(current_ratings_masked[algo_i])
  row["Min. #Ratings per Sentence"] = np.min(
    np.sum(~np.isnan(current_ratings_masked[algo_i]), axis=0))
  row["Max. #Ratings per Sentence"] = np.max(
    np.sum(~np.isnan(current_ratings_masked[algo_i]), axis=0))
  row["Avg. #Ratings per Sentence"] = np.mean(
    np.sum(~np.isnan(current_ratings_masked[algo_i]), axis=0))
  row["#Sentences"] = current_ratings_masked.shape[0]
  row["#Workers"] = np.sum(np.any(~np.isnan(current_ratings_masked[algo_i]), axis=1))
  row["#Workers (all)"] = current_ratings_masked.shape[1]
  row["#Ratings"] = np.sum(~np.isnan(current_ratings_masked[algo_i]))
  row["#Ratings (all)"] = np.sum(~np.isnan(ratings[algo_i]))
  if row["#Ratings (all)"] == 0:
    row["%"] = np.nan
  else:
    row["%"] = row["#Ratings"] / row["#Ratings (all)"] * 100
  return row

def get_mos_df(data: EvaluationData, mask_names: Set[MaskName]) -> List[ODType[str, Any]]:
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)
  rmask = factory.merge_masks_into_rmask(masks)

  all_genders = OrderedSet(sorted({x.gender for x in data.worker_data.values()}))
  all_age_groups = OrderedSet(sorted({x.age_group for x in data.worker_data.values()}))
  all_workers = data.workers

  gender_masks = {
    gender: factory.convert_mask_to_rmask(factory.get_wmask_by_gender(gender))
    for gender in all_genders
  }

  age_group_masks = {
    age_group: factory.convert_mask_to_rmask(factory.get_wmask_by_age_group(age_group))
    for age_group in all_age_groups
  }

  worker_masks = {
    worker_id: factory.convert_mask_to_rmask(factory.get_wmask_by_worker_id(worker_id))
    for worker_id in all_workers
  }

  rows = []
  for rating_name in data.rating_names:
    row_template = OrderedDict()
    row_template["Rating"] = rating_name
    current_ratings = get_ratings(data, {rating_name})
    current_ratings_masked = current_ratings.copy()
    rmask.apply_by_nan(current_ratings_masked)

    for algo_i, alg_name in enumerate(data.algorithms):
      row_template["Algorithm"] = alg_name

      for gender in all_genders:
        row_template["Gender"] = gender
        gender_mask = gender_masks[gender]
        for age_group in all_age_groups:
          row_template["AgeGroup"] = age_group
          age_group_mask = age_group_masks[age_group]
          for worker_id, worker_info in data.worker_data.items():
            if worker_info.age_group != age_group:
              continue
            if worker_info.gender != gender:
              continue
            row_template["WorkerId"] = worker_id
            worker_mask = worker_masks[worker_id]
            rows.append(get_row(row_template, current_ratings,
                        current_ratings_masked, algo_i, [worker_mask]))
          # All workers row
          row_template["WorkerId"] = ALL_CELL_CONTENT
          rows.append(get_row(row_template, current_ratings,
                      current_ratings_masked, algo_i, [gender_mask, age_group_mask]))
        row_template["AgeGroup"] = ALL_CELL_CONTENT
        rows.append(get_row(row_template, current_ratings,
                    current_ratings_masked, algo_i, [gender_mask]))
      row_template["Gender"] = ALL_CELL_CONTENT
      rows.append(get_row(row_template, current_ratings,
                  current_ratings_masked, algo_i, []))

      for age_group in all_age_groups:
        row_template["AgeGroup"] = age_group
        age_group_mask = age_group_masks[age_group]
        rows.append(get_row(row_template, current_ratings,
                    current_ratings_masked, algo_i, [age_group_mask]))
  rows.sort(key=lambda row: (row["Gender"], row["AgeGroup"],
            row["WorkerId"], row["Rating"], row["Algorithm"]))
  return rows


def generate_approve_csv(data: EvaluationData, mask_names: Set[str], reason: Optional[str], approval_cost: Optional[float], amazon_fee: Optional[float]) -> Tuple[pd.DataFrame, List[Dict]]:
  logger = get_logger()
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers:")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (none assignment will be approved):")
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
  results_dict = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("WorkerId", worker_id),
      ("AssignmentId", assignment_id),
      ("HITId", hit_id),
      ("Approve", reason),
      ("Reject", ""),
    ))
    results.append(line)
    # if worker_id not in results_dict:
    #   results_dict[worker_id] = []
    # worker_data: List[str] = results_dict[worker_id]
    # assert assignment_id not in worker_data
    # worker_data.append(assignment_id)
    # assignment_meta = OrderedDict()
    # assignment_meta["HITId"] = hit_id
    # assignment_meta["ApprovalReason"] = reason
    # assignment_meta["Reject"] = ""
    # assignment_meta["Costs"] = approval_cost
    # assignment_meta["Fee"] = amazon_fee
    # worker_data[assignment_id] = assignment_meta
    results_dict.append({
      "WorkerId": worker_id,
      "AssignmentId": assignment_id,
      "Gender": data.worker_data[worker_id].gender,
      "AgeGroup": data.worker_data[worker_id].age_group,
      "HITId": data.worker_data[worker_id].assignments[assignment_id].hit_id,
      "Device": data.worker_data[worker_id].assignments[assignment_id].device,
      "State": data.worker_data[worker_id].assignments[assignment_id].state,
      "Time": datetime.datetime.strftime(data.worker_data[worker_id].assignments[assignment_id].time, "%d.%m.%Y %H:%M:%S"),
    })

  result = pd.DataFrame.from_records(results)
  if len(result.index) > 0:
    result.sort_values(["WorkerId", "AssignmentId"], inplace=True)
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
  return result, results_dict


def generate_reject_csv(data: EvaluationData, mask_names: Set[str], reject_mask_names: Set[str], reason: str) -> Tuple[pd.DataFrame, List[Dict]]:
  logger = get_logger()
  dlogger = get_detail_logger()

  factory = MaskFactory(data)
  masks = data.get_masks_from_names(mask_names)
  reject_masks = data.get_masks_from_names(reject_mask_names)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  reject_amask = factory.merge_masks_into_amask(reject_masks)
  reject_wmask = factory.merge_masks_into_wmask(reject_masks)

  # unmasked no-reject -> False
  # unmasked reject -> True
  # masked no-reject -> False
  # masked reject -> False
  amask.mask = ~amask.mask & reject_amask.mask
  wmask.mask = ~wmask.mask & reject_wmask.mask

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (all assignments will be rejected):")
    for nr, w in enumerate(sorted(masked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No masked workers exist.")

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers:")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  assignment_indices = amask.masked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be rejected: {len(assignment_indices)}")

  results: List[Dict[str, Any]] = []
  results_dict = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("WorkerId", worker_id),
      ("AssignmentId", assignment_id),
      ("HITId", hit_id),
      ("Approve", ""),
      ("Reject", reason),
    ))
    results.append(line)
    # if worker_id not in results_dict:
    #   results_dict[worker_id] = []
    # worker_data: List[str] = results_dict[worker_id]
    # assert assignment_id not in worker_data
    # worker_data.append(assignment_id)
    results_dict.append({
      "WorkerId": worker_id,
      "AssignmentId": assignment_id,
      "Gender": data.worker_data[worker_id].gender,
      "AgeGroup": data.worker_data[worker_id].age_group,
      "HITId": data.worker_data[worker_id].assignments[assignment_id].hit_id,
      "Device": data.worker_data[worker_id].assignments[assignment_id].device,
      "State": data.worker_data[worker_id].assignments[assignment_id].state,
      "Time": datetime.datetime.strftime(data.worker_data[worker_id].assignments[assignment_id].time, "%d.%m.%Y %H:%M:%S"),
    })
  result = pd.DataFrame.from_records(results)
  if len(result.index) > 0:
    result.sort_values(["WorkerId", "AssignmentId"], inplace=True)
  return result, results_dict


def generate_bonus_csv(data: EvaluationData, mask_names: Set[str], bonus: float, reason: str, amazon_fee_percent: float) -> Tuple[pd.DataFrame, List[Dict]]:
  assert 0 <= amazon_fee_percent <= 1
  logger = get_logger()
  dlogger = get_detail_logger()
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  amask = factory.merge_masks_into_amask(masks)
  wmask = factory.merge_masks_into_wmask(masks)

  unmasked_workers = data.workers[wmask.unmasked_indices]
  if len(unmasked_workers) > 0:
    dlogger.info("Unmasked workers:")
    for nr, w in enumerate(sorted(unmasked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No unmasked workers exist.")

  masked_workers = data.workers[wmask.masked_indices]
  if len(masked_workers) > 0:
    dlogger.info("Masked workers (none assignment will be paid a bonus):")
    for nr, w in enumerate(sorted(masked_workers), start=1):
      dlogger.info(f"{nr}. \"{w}\"")
  else:
    dlogger.info("No masked workers exist.")

  assignment_indices = amask.unmasked_indices
  assignments_worker_matrix = factory.get_assignments_worker_index_matrix()

  logger.info(f"Count of assignments that will be paid a bonus: {len(assignment_indices)}")

  results: List[Dict[str, Any]] = []
  results_dict = []
  for assignment_index in sorted(assignment_indices):
    assignment_id = data.assignments[assignment_index]
    worker_index = assignments_worker_matrix[assignment_index]
    worker_id = data.workers[worker_index]
    hit_id = data.worker_data[worker_id].assignments[assignment_id].hit_id
    line = OrderedDict((
      ("WorkerId", worker_id),
      ("AssignmentId", assignment_id),
      ("HITId", hit_id),
      ("BonusAmount", bonus),
      ("BonusAmountWithFee", bonus + (bonus * amazon_fee_percent)),
      ("Reason", reason),
    ))
    results.append(line)
    results_dict.append({
      "WorkerId": worker_id,
      "AssignmentId": assignment_id,
      "Gender": data.worker_data[worker_id].gender,
      "AgeGroup": data.worker_data[worker_id].age_group,
      "HITId": data.worker_data[worker_id].assignments[assignment_id].hit_id,
      "Device": data.worker_data[worker_id].assignments[assignment_id].device,
      "State": data.worker_data[worker_id].assignments[assignment_id].state,
      "Time": datetime.datetime.strftime(data.worker_data[worker_id].assignments[assignment_id].time, "%d.%m.%Y %H:%M:%S"),
      "BonusAmount": bonus,
      "BonusAmountWithFee": bonus + (bonus * amazon_fee_percent),
      "Reason": reason,
    })
  result = pd.DataFrame.from_records(results)
  if len(result.index) > 0:
    result.sort_values(["WorkerId", "AssignmentId"], inplace=True)
  logger = get_logger()
  costs = len(assignment_indices) * bonus
  fees = costs * amazon_fee_percent
  # Ex.:
  # Estimated Total Bonus: 480 assignments x $0.15 = $30
  # Estimated Fees to Mechanical Turk: 20.00% of $30 = $2
  # Estimated Cost: Total Bonus + Fees = $32
  logger.info(
    f"Estimated Total Reward: {len(assignment_indices)} assignments x ${bonus:.2f} = ${costs:.2f}")
  logger.info(
    f"Estimated Fees to Mechanical Turk: {amazon_fee_percent*100:.2f}% of ${costs:.2f} = ${fees:.2f}")
  logger.info(f"Total Reward + Fees = ${costs + fees:.2f}")
  # TODO tax as param
  logger.info(f"19% Tax of ${costs + fees:.2f} = ${(costs + fees)*0.19:.2f}")
  logger.info(f"Estimated Cost: Total Reward + Fees + Tax = ${(costs + fees)*1.19:.2f}")
  return result, results_dict


def generate_ground_truth_table(data: EvaluationData, mask_names: Set[str]) -> pd.DataFrame:
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)

  rmask = factory.merge_masks_into_rmask(masks)

  results: List[Dict[str, Any]] = []

  for worker, worker_data in data.worker_data.items():
    w_i = data.workers.get_loc(worker)
    for assignment, assignment_data in worker_data.assignments.items():
      for (alg_name, file_name), ass_ratings in assignment_data.ratings.items():
        alg_i = data.algorithms.get_loc(alg_name)
        file_i = data.files.get_loc(file_name)
        is_masked = rmask.mask[alg_i, w_i, file_i]
        line = OrderedDict()
        line["WorkerId"] = worker
        line["Algorithm"] = alg_name
        line["File"] = file_name
        for rating_name, rating in ass_ratings.votes.items():
          line[f"Rating \"{rating_name}\""] = rating
        # line["AcceptTime"] = assignment_data.time
        # line["FinishTime"] = assignment_data.time + datetime.timedelta(seconds=assignment_data.worktime)
        # line["Worktime"] = str(datetime.timedelta(seconds=assignment_data.worktime))
        # line["Worktime (s)"] = assignment_data.worktime
        line["Device"] = assignment_data.device
        line["State"] = assignment_data.state
        line["HITId"] = assignment_data.hit_id
        line["AssignmentId"] = assignment
        line["Masked?"] = is_masked
        # line["Comments"] = assignment_data.comments
        results.append(line)

  result = pd.DataFrame.from_records(results)
  if len(result.index) > 0:
    result.sort_values(["WorkerId", "Algorithm", "File"], inplace=True)
  return result
