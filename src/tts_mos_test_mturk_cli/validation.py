from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.masks import get_mask_name_and_reverse
from tts_mos_test_mturk_cli.types import CLIError


def ensure_mask_exists(data: EvaluationData, mask: str) -> None:
  mask, _ = get_mask_name_and_reverse(mask)
  if mask not in data.masks:
    raise CLIError(f"Mask \"{mask}\" doesn't exist!")


def ensure_masks_exist(data: EvaluationData, masks: Set[str]) -> None:
  for mask in masks:
    ensure_mask_exists(data, mask)


def ensure_ratings_exist(data: EvaluationData, rating_names: Set[str]) -> None:
  for rating_name in rating_names:
    ensure_rating_exists(data, rating_name)


def ensure_rating_exists(data: EvaluationData, rating_name: str) -> None:
  if rating_name not in data.rating_names:
    raise CLIError(f"Rating \"{rating_name}\" doesn't exist!")


def ensure_workers_exist(data: EvaluationData, worker_ids: Set[str]) -> None:
  for worker_id in worker_ids:
    ensure_worker_exists(data, worker_id)


def ensure_worker_exists(data: EvaluationData, worker_id: str) -> None:
  if worker_id not in data.workers:
    raise ValueError(f"Worker \"{worker_id}\" doesn't exist!")


def ensure_assignments_exists(data: EvaluationData, assignment_ids: Set[str]) -> None:
  for assignment_id in assignment_ids:
    ensure_assignment_exists(data, assignment_id)


def ensure_assignment_exists(data: EvaluationData, assignment_id: str) -> None:
  if assignment_id not in data.assignments:
    raise ValueError(f"Assignment \"{assignment_id}\" doesn't exist!")


def ensure_age_groups_exist(data: EvaluationData, age_groups: Set[str]) -> None:
  for age_group in age_groups:
    ensure_age_group_exists(data, age_group)


def ensure_age_group_exists(data: EvaluationData, age_group: str) -> None:
  all_age_groups = {x.age_group for x in data.worker_data.values()}
  if age_group not in all_age_groups:
    raise ValueError(
      f"Age group \"{age_group}\" doesn't exist! Please select from: {', '.join(sorted(all_age_groups))}")
