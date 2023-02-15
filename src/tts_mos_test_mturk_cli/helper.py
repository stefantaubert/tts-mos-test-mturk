
import os
from logging import Logger
from pathlib import Path
from typing import Generator

from tts_mos_test_mturk.evaluation_data import EvaluationData


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def save_project(project: EvaluationData, logger: Logger, flogger: Logger) -> bool:
  try:
    project.save()
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project couldn't be saved to \"{project.file_path.absolute()}\"!")
    return False
  logger.info(f"Updated project at: \"{project.file_path.absolute()}\"")
  return True