from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.assignment_count_mask import mask_workers_by_assignment_count
from tts_mos_test_mturk_cli.argparse_helper import parse_positive_integer
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_workers_by_assignment_count_parser(parser: ArgumentParser):
  parser.description = "Reject workers with too few assignments"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("count", type=parse_positive_integer, metavar="COUNT",
                      help="ignore all which have fewer assignments than COUNT")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_workers_by_assignment_count(ns.project, ns.masks, ns.count, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
