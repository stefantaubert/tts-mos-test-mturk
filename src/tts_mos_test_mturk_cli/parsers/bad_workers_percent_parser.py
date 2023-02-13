from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.bad_worker_filtering import ignore_bad_workers_percent
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_file,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float, parse_positive_float)
from tts_mos_test_mturk_cli.default_args import add_dry_argument
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_bad_workers_percent_parser(parser: ArgumentParser):
  parser.description = "Reject bad workers percent-wise."
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToOrderedSetAction)
  parser.add_argument("from_percent", type=parse_non_negative_float, metavar="FROM-PERCENT",
                      help="inclusive lower boundary")
  parser.add_argument("to_percent", type=parse_positive_float, metavar="TO-PERCENT",
                      help="exclusive top boundary")
  parser.add_argument("output_mask", type=parse_non_empty_or_whitespace,
                      metavar="OUTPUT-MASK", help="name of the output mask")
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  ignore_bad_workers_percent(project, ns.masks, ns.from_percent, ns.to_percent, ns.output_mask)

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
