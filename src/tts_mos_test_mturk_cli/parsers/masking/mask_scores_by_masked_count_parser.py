from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.masking.masked_count_mask import mask_scores_by_masked_count
from tts_mos_test_mturk_cli.argparse_helper import (parse_non_empty_or_whitespace,
                                                    parse_positive_float)
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_mask_scores_by_masked_count_parser(parser: ArgumentParser):
  parser.description = "Ignore workers who have at least an amount of masked opinion scores."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("ref_mask", type=parse_non_empty_or_whitespace,
                      metavar="REF-MASK", help="name of the mask on which the masked opinion scores should be counted")
  parser.add_argument("percent", type=parse_positive_float, metavar="PERCENT",
                      help=f"ignore workers who have at least PERCENT % of all masked opinion scores")
  add_output_mask_argument(parser)
  add_dry_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  mask_scores_by_masked_count(ns.project, ns.masks, ns.ref_mask, ns.percent, ns.output_mask)

  if ns.dry:
    return True

  success = save_project(ns.project, logger, flogger)
  return success
