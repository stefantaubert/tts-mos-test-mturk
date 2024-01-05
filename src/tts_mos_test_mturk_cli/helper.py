import json
import os
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.globals import DISPLAY_PRECISION
from tts_mos_test_mturk_cli.logging_configuration import get_cli_logger
from tts_mos_test_mturk_cli.types import CLIError


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def save_project(project: EvaluationData):
  logger = get_cli_logger()
  try:
    project.save()
  except Exception as ex:
    raise CLIError(f"Project couldn't be saved to \"{project.file_path.absolute()}\"!") from ex
  logger.info(f"Updated project at: \"{project.file_path.absolute()}\"")


def log_full_df(df: pd.DataFrame) -> None:
  logger = get_cli_logger()
  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    "display.width", None,
    "display.precision", DISPLAY_PRECISION):
    logger.info(f"\n{df.to_string(index=False)}")


def save_csv(path: Path, df: pd.DataFrame, output_name: str = "output") -> None:
  logger = get_cli_logger()
  try:
    df.to_csv(path, index=False)
  except Exception as ex:
    raise CLIError(f"CSV-file couldn't be saved to: \"{path.absolute()}\"!") from ex
  logger.info(f"Written {output_name} to: \"{path.absolute()}\"")


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def save_json(path: Path, data: Dict) -> None:
  try:
    with open(path, "w", encoding="utf8") as f:
      json.dump(data, f, indent=2, cls=NumpyEncoder)
    logger = get_cli_logger()
    logger.info(f"Written output to: \"{path.absolute()}\"")
  except Exception as ex:
    raise CLIError("Output file couldn't be created!") from ex
