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
                                                    parse_non_negative_float, parse_positive_float,
                                                    parse_positive_integer)
from tts_mos_test_mturk_cli.default_args import (add_dry_argument, add_masks_argument,
                                                 add_output_mask_argument, add_project_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def get_mask_assignments_by_device_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their listening devices."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("devices", type=str, metavar="DEVICE", choices=["in-ear", "over-the-ear", "desktop", "laptop"], nargs="+",
                      help="mask all assignments that were done on DEVICE", action=ConvertToSetAction)
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_listening_device(ns.project, ns.masks, ns.devices, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_assignments_by_worktime_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their worktime."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_integer, metavar="THRESHOLD",
                      help="mask all assignments, which have a worktime smaller than THRESHOLD")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_worktime(ns.project, ns.masks, ns.threshold, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_rating_outliers_parser(parser: ArgumentParser):
  parser.description = "Mask outlying ratings of each algorithm."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="N-STD",
                      help="mask outlying ratings that lie N-STD standard deviations away from the mean of the respective algorithm.")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_outlying_ratings(ns.project, ns.masks, ns.threshold, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_workers_by_masked_ratings_count_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their percentual amount of masked ratings compared to all masked ratings."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("ref_masks", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="REF-MASK", help="masks on which the masked ratings should be counted", action=ConvertToSetAction)
  parser.add_argument("percent", type=parse_positive_float, metavar="PERCENT",
                      help=f"mask workers that have at least PERCENT % of all masked ratings")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_ratings_by_masked_count(ns.project, ns.masks, ns.ref_masks, ns.percent, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_assignments_count_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their amount of assignments."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("count", type=parse_positive_integer, metavar="COUNT",
                      help="mask workers which have fewer assignments than COUNT")
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_assignment_count(ns.project, ns.masks, ns.count, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("threshold", type=parse_positive_float, metavar="THRESHOLD",
                      help="mask workers that have a correlation smaller than THRESHOLD")
  add_mode_argument(parser)
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation(ns.project, ns.masks, ns.threshold, ns.mode, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_percent_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers (percentage-wise)."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("from_percent", type=parse_non_negative_float, metavar="FROM-PERCENT",
                      help="inclusive lower boundary")
  parser.add_argument("to_percent", type=parse_positive_float, metavar="TO-PERCENT",
                      help="exclusive top boundary")
  add_mode_argument(parser)
  add_output_mask_argument(parser)
  add_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_workers_by_correlation_percent(ns.project, ns.masks, ns.from_percent,
                                        ns.to_percent, ns.mode, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def add_mode_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="mode to calculate the correlations: sentence -> the correlation of the ratings of each audio url from a worker in comparison to the mean of the ratings of the other workers; algorithm -> the correlation of the mean ratings from one worker compared to the mean of all other workers for each algorithm; both -> the mean of sentence and algorithm correlation")
