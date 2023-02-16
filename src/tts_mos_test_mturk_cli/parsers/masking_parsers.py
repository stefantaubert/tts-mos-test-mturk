from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.masking.assignment_count_mask import mask_workers_by_assignment_count
from tts_mos_test_mturk.masking.listening_device_mask import mask_assignments_by_listening_device
from tts_mos_test_mturk.masking.masked_count_mask import mask_scores_by_masked_count
from tts_mos_test_mturk.masking.outlier_mask import mask_outlying_scores
from tts_mos_test_mturk.masking.work_time_mask import mask_assignments_by_work_time
from tts_mos_test_mturk.masking.worker_correlation_mask import (mask_workers_by_correlation,
                                                                mask_workers_by_correlation_percent)
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float, parse_positive_float,
                                                    parse_positive_integer)
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def get_mask_assignments_by_listening_device_parser(parser: ArgumentParser):
  parser.description = "Reject assignments with listening type"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="ignore all assignments with DEVICE", action=ConvertToSetAction)
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_listening_device(ns.project, ns.masks, ns.devices, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


def get_mask_assignments_by_work_time_parser(parser: ArgumentParser):
  parser.description = "Reject too fast assignments"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_integer, metavar="THRESHOLD",
                      help="ignore all assignments, which have a work_time smaller than THRESHOLD")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_work_time(ns.project, ns.masks, ns.threshold, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


def get_mask_outlying_scores_parser(parser: ArgumentParser):
  parser.description = "Ignore outliers."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="THRESHOLD",
                      help="ignore opinion scores with that amount of standard deviations away")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_outlying_scores(ns.project, ns.masks, ns.threshold, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_scores_by_masked_count(ns.project, ns.masks, ns.ref_mask, ns.percent, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


def get_mask_workers_by_assignment_count_parser(parser: ArgumentParser):
  parser.description = "Reject workers with too few assignments"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("count", type=parse_positive_integer, metavar="COUNT",
                      help="ignore all which have fewer assignments than COUNT")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_assignment_count(ns.project, ns.masks, ns.count, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation(ns.project, ns.masks, ns.threshold, ns.mode, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)
  return main


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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation_percent(ns.project, ns.masks, ns.from_percent,
                                        ns.to_percent, ns.mode, ns.output_mask)

    if ns.dry:
      return True

    return save_project(ns.project)

  return main
