from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.worker_correlation_mask import mask_workers_by_correlation
from tts_mos_test_mturk_cli.argparse_helper import parse_positive_float
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_bad_workers_parser(parser: ArgumentParser):
  parser.description = "Reject bad workers"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="THRESHOLD",
                      help="ignore all assignments, which have a work_time smaller than THRESHOLD")
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="Mode to calculate the correlations")
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

  mask_workers_by_correlation(project, ns.masks, ns.threshold, ns.mode, ns.output_mask)

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
