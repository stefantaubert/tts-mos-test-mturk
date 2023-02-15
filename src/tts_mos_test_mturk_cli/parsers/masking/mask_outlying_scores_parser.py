from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.masking.outlier_mask import mask_outlying_scores
from tts_mos_test_mturk_cli.argparse_helper import parse_positive_float
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_outlying_scores_parser(parser: ArgumentParser):
  parser.description = "Ignore outliers."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="THRESHOLD",
                      help="ignore opinion scores with that amount of standard deviations away")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_outlying_scores(ns.project, ns.masks, ns.threshold, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
