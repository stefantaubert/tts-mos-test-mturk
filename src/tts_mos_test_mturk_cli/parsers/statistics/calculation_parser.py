from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd

from tts_mos_test_mturk.df_generation import get_mos_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.parsers.statistics.common import (add_optional_output_argument, add_silent_argument,
                                                              save_output_csv)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_calculation_parser(parser: ArgumentParser):
  parser.description = "Calculate MOS for each algorithm"
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  result_df = get_mos_df(ns.project, ns.masks)
  
  if not ns.silent:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None):
      print(result_df)

  success = True
  if ns.output:
    success = save_output_csv(ns.output, result_df, logger, flogger)
  return success
