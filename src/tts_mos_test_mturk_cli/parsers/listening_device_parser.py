from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.masking.listening_device_mask import \
  mask_assignments_by_listening_device
from tts_mos_test_mturk_cli.argparse_helper import ConvertToSetAction
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_listening_device_parser(parser: ArgumentParser):
  parser.description = "Reject assignments with listening type"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="ignore all assignments with DEVICE", action=ConvertToSetAction)
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  mask_assignments_by_listening_device(project, ns.masks, ns.devices, ns.output_mask)

  if ns.dry:
    return True

  try:
    project.save(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Updated project at: \"{ns.project.absolute()}\"")

  return True
