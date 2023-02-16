
import os
from logging import Logger
from pathlib import Path
from typing import Generator

import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.logging_configuration import get_cli_logger, get_file_logger
from tts_mos_test_mturk_cli.types import CLIError


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def save_project(project: EvaluationData):
  logger = get_cli_logger()
  flogger = get_file_logger()

  try:
    project.save()
  except Exception as ex:
    flogger.debug(ex)
    raise CLIError(f"Project couldn't be saved to \"{project.file_path.absolute()}\"!") from ex
  logger.info(f"Updated project at: \"{project.file_path.absolute()}\"")


def print_full_df(df: pd.DataFrame) -> None:
  with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None):
    print(df)


def save_output_csv(path: Path, df: pd.DataFrame, logger: Logger, flogger: Logger) -> bool:
  try:
    df.to_csv(path, index=False)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Output couldn't be saved to: \"{path.absolute()}\"!")
    return False
  logger.info(f"Written output to: \"{path.absolute()}\"")
  return True
