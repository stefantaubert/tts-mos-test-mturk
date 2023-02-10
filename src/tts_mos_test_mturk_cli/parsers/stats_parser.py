from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd

from tts_mos_test_mturk.core.bad_worker_filtering import calc_mos
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.stats import print_stats
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, get_optional,
                                                    parse_existing_file,
                                                    parse_non_empty_or_whitespace, parse_path)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_stats_parser(parser: ArgumentParser):
  parser.description = "Calculate MOS for each algorithm"
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToOrderedSetAction)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  print_stats(project, set(), ns.masks)

  return True
