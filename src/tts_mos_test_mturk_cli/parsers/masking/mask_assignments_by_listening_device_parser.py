from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.listening_device_mask import mask_assignments_by_listening_device
from tts_mos_test_mturk_cli.argparse_helper import ConvertToSetAction
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_assignments_by_listening_device_parser(parser: ArgumentParser):
  parser.description = "Reject assignments with listening type"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="ignore all assignments with DEVICE", action=ConvertToSetAction)
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_assignments_by_listening_device(ns.project, ns.masks, ns.devices, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
