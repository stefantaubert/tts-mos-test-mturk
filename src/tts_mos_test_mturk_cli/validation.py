from typing import Optional, Set

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
