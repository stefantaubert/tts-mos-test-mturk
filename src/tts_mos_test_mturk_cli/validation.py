from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.types import CLIError


def ensure_mask_exists(data: EvaluationData, mask: str) -> None:
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
