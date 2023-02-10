from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd

from tts_mos_test_mturk.core.bad_worker_filtering import (calc_mos, ignore_bad_workers,
                                                          ignore_masked_count_opinion_scores,
                                                          ignore_outlier_opinion_scores,
                                                          ignore_too_fast_assignments)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, get_optional,
                                                    parse_existing_file,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_integer, parse_path,
                                                    parse_positive_float, parse_positive_integer)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_os_count_parser(parser: ArgumentParser):
  parser.description = "Ignore workers who have at least an amount of masked opinion scores."
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToOrderedSetAction)

  parser.add_argument("ref_mask", type=parse_non_empty_or_whitespace,
                      metavar="REF-MASK", help="name of the mask on which the masked opinion scores should be counted")
  parser.add_argument("percent", type=parse_positive_float, metavar="PERCENT",
                      help=f"ignore workers who have at least PERCENT % of all masked opinion scores")
  parser.add_argument("output_mask", type=parse_non_empty_or_whitespace,
                      metavar="OUTPUT-MASK", help="name of the output mask")
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  ignore_masked_count_opinion_scores(project, ns.masks, ns.ref_mask, ns.percent, ns.output_mask)

  try:
    project.save(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Updated project at: \"{ns.project.absolute()}\"")

  return True
