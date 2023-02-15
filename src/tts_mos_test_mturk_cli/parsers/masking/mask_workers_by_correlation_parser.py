from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.worker_correlation_mask import (mask_workers_by_correlation,
                                                                mask_workers_by_correlation_percent)
from tts_mos_test_mturk_cli.argparse_helper import parse_non_negative_float, parse_positive_float
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_workers_by_correlation_parser(parser: ArgumentParser):
  parser.description = "Reject bad workers"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="THRESHOLD",
                      help="ignore all assignments, which have a work_time smaller than THRESHOLD")
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="Mode to calculate the correlations")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return mask_workers_by_correlation_ns


def mask_workers_by_correlation_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_workers_by_correlation(ns.project, ns.masks, ns.threshold, ns.mode, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success


def get_mask_workers_by_correlation_percent_parser(parser: ArgumentParser):
  parser.description = "Reject bad workers percent-wise."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("from_percent", type=parse_non_negative_float, metavar="FROM-PERCENT",
                      help="inclusive lower boundary")
  parser.add_argument("to_percent", type=parse_positive_float, metavar="TO-PERCENT",
                      help="exclusive top boundary")
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="Mode to calculate the correlations")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return mask_workers_by_correlation_percent_ns


def mask_workers_by_correlation_percent_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_workers_by_correlation_percent(ns.project, ns.masks, ns.from_percent,
                                      ns.to_percent, ns.mode, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
