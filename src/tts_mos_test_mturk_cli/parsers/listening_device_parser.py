from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.bad_worker_filtering import mask_assignments_by_lt
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, ConvertToSetAction,
                                                    parse_existing_file,
                                                    parse_non_empty_or_whitespace)
from tts_mos_test_mturk_cli.default_args import add_dry_argument
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_listening_device_parser(parser: ArgumentParser):
  parser.description = "Reject assignments with listening type"
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToOrderedSetAction)
  parser.add_argument("output_mask", type=parse_non_empty_or_whitespace,
                      metavar="OUTPUT-MASK", help="name of the output mask")
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="ignore all assignments with DEVICE", action=ConvertToSetAction)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  mask_assignments_by_lt(project, ns.masks, ns.devices, ns.output_mask)

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
