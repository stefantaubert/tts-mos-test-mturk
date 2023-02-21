import math
from argparse import ArgumentParser, Namespace

from tts_mos_test_mturk.masking.assignment_count_mask import mask_workers_by_assignment_count
from tts_mos_test_mturk.masking.listening_device_mask import mask_assignments_by_listening_device
from tts_mos_test_mturk.masking.masked_count_mask import mask_ratings_by_masked_count
from tts_mos_test_mturk.masking.outlier_mask import mask_outlying_ratings
from tts_mos_test_mturk.masking.worker_correlation_mask import (mask_workers_by_correlation,
                                                                mask_workers_by_correlation_percent)
from tts_mos_test_mturk.masking.worktime_mask import mask_assignments_by_worktime
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float,
                                                    parse_non_negative_integer,
                                                    parse_positive_float, parse_positive_integer)
from tts_mos_test_mturk_cli.default_args import (add_opt_dry_argument, add_opt_masks_argument,
                                                 add_req_output_mask_argument,
                                                 add_req_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def get_mask_assignments_by_device_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their listening devices."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="mask all assignments that were done on DEVICE", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_listening_device(ns.project, ns.masks, ns.devices, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_assignments_by_worktime_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their worktime."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-time", type=parse_non_negative_integer, metavar="FROM-TIME",
                      help="mask all assignments, which have a worktime greater than or equal to FROM-TIME (inclusive); in [0; inf)", default=0)
  parser.add_argument("to_time", type=parse_positive_integer, metavar="TO-TIME",
                      help="mask all assignments, which have a worktime smaller than TO-TIME (exclusive); in (0; inf)")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_worktime(ns.project, ns.masks, ns.from_time, ns.to_time, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_rating_outliers_parser(parser: ArgumentParser):
  parser.description = "Mask outlying ratings of each algorithm."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("from_threshold", type=parse_non_negative_float, metavar="FROM-N-STD",
                      help="mask outlying ratings that lie at least FROM-N-STD standard deviations away from the mean of the respective algorithm; in [0, inf)")
  parser.add_argument("--to-threshold", type=parse_positive_float, metavar="TO-N-STD",
                      help="mask outlying ratings that lie below TO-N-STD standard deviations away from the mean of the respective algorithm (exclusive); in (0, inf)", default=math.inf)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_outlying_ratings(ns.project, ns.masks, ns.from_threshold, ns.to_threshold, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_workers_by_masked_ratings_count_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their percentual amount of masked ratings compared to all masked ratings."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("ref_masks", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="REF-MASK", help="masks on which the masked ratings should be counted", action=ConvertToSetAction)
  parser.add_argument("from_percent", type=parse_non_negative_float, metavar="FROM-PERCENT",
                      help="mask workers that have at least FROM-PERCENT of all masked ratings (inclusive); in [0, 100)")
  parser.add_argument("--to-percent", type=parse_positive_float, metavar="TO-PERCENT",
                      help="mask workers that have at maximum TO-PERCENT of all masked ratings (exclusive); in (0, 100]", default=100)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_ratings_by_masked_count(ns.project, ns.masks, ns.ref_masks,
                                 ns.from_percent / 100, ns.to_percent / 100, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_assignments_count_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their amount of assignments."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-count", type=parse_non_negative_integer, metavar="FROM-COUNT",
                      help="mask workers that have at least FROM-COUNT assignments (inclusive); in [0; inf)", default=0)
  parser.add_argument("to_count", type=parse_positive_integer, metavar="TO-COUNT",
                      help="mask workers that have fewer than TO-COUNT assignments (exclusive); in (0; inf)")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_assignment_count(
      ns.project, ns.masks, ns.from_count, ns.to_count, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-threshold", type=float, metavar="FROM-THRESHOLD",
                      help="mask workers that have a correlation with at least FROM-THRESHOLD (inclusive); in [-1; 1)", default=-1)
  parser.add_argument("to_threshold", type=float, metavar="TO-THRESHOLD",
                      help="mask workers that have a correlation smaller than TO-THRESHOLD (exclusive); in (-1; 1]")
  add_mode_argument(parser)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation(ns.project, ns.masks, ns.from_threshold,
                                ns.to_threshold, ns.mode, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_percent_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers (percentage-wise)."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-percent", type=parse_non_negative_float,
                      metavar="FROM-PERCENT", help="inclusive lower boundary; in [0; 100)", default=0)
  parser.add_argument("to_percent", type=parse_positive_float, metavar="TO-PERCENT",
                      help="exclusive top boundary; in (0; 100]")
  add_mode_argument(parser)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation_percent(ns.project, ns.masks, ns.from_percent / 100,
                                        ns.to_percent / 100, ns.mode, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def add_mode_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="mode to calculate the correlations: sentence -> the correlation of the ratings of each audio url from a worker in comparison to the mean of the ratings of the other workers; algorithm -> the correlation of the mean ratings from one worker compared to the mean of all other workers for each algorithm; both -> the mean of sentence and algorithm correlation")
