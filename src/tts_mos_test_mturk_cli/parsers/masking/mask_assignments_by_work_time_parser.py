from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.masking.work_time_mask import mask_assignments_by_work_time
from tts_mos_test_mturk_cli.argparse_helper import parse_positive_integer
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_assignments_by_work_time_parser(parser: ArgumentParser):
  parser.description = "Reject too fast assignments"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_integer, metavar="THRESHOLD",
                      help="ignore all assignments, which have a work_time smaller than THRESHOLD")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_assignments_by_work_time(ns.project, ns.masks, ns.threshold, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
