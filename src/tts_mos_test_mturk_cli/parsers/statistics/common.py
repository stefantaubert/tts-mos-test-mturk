from argparse import ArgumentParser
from logging import Logger
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd

from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, ConvertToSetAction,
                                                    get_optional, parse_codec, parse_existing_file,
                                                    parse_non_empty_or_whitespace, parse_path,
                                                    parse_positive_integer, parse_project)


def add_optional_output_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--output", type=get_optional(parse_path),
                      help="write results to this CSV", metavar="OUTPUT-CSV", default=None)


def add_silent_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-s", "--silent", action="store_true",
                      help="don't print table to console")


def save_output_csv(path: Path, df: pd.DataFrame, logger: Logger, flogger: Logger) -> bool:
  try:
    df.to_csv(path, index=False)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Output CSV \"{path.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Written output to: \"{path.absolute()}\"")
  return True
