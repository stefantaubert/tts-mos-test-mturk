
from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.types import CLIError


def ensure_mask_exists(data: EvaluationData, mask: str) -> None:
  if mask not in data.masks:
    raise CLIError(f"Mask \"{mask}\" doesn't exist!")


def ensure_masks_exist(data: EvaluationData, masks: Set[str]) -> None:
  for mask in masks:
    ensure_mask_exists(data, mask)
