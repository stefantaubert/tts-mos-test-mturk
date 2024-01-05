import math
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import cast

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.age_group_mask import mask_workers_by_age_group
from tts_mos_test_mturk.masking.assignment_count_mask import mask_workers_by_assignment_count
from tts_mos_test_mturk.masking.assignment_id_mask import mask_assignments_by_id
from tts_mos_test_mturk.masking.gender_mask import mask_workers_by_gender
from tts_mos_test_mturk.masking.listening_device_mask import mask_assignments_by_listening_device
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masked_count_mask import mask_ratings_by_masked_count
from tts_mos_test_mturk.masking.merge import merge_masks
from tts_mos_test_mturk.masking.outlier_mask import mask_outlying_ratings
from tts_mos_test_mturk.masking.reverse import reverse_mask
from tts_mos_test_mturk.masking.state_mask import mask_assignments_by_state
from tts_mos_test_mturk.masking.time_mask import mask_assignments_by_time
from tts_mos_test_mturk.masking.worker_correlation_mask import (mask_workers_by_correlation,
                                                                mask_workers_by_correlation_percent)
from tts_mos_test_mturk.masking.worker_id_mask import mask_workers_by_id
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction, parse_datetime,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float,
                                                    parse_non_negative_integer, parse_percent,
                                                    parse_positive_float, parse_positive_integer)
from tts_mos_test_mturk_cli.default_args import (add_opt_dry_argument, add_opt_masks_argument,
                                                 add_req_output_mask_argument,
                                                 add_req_project_argument, add_req_ratings_argument)
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.validation import (ensure_assignments_exists, ensure_mask_exists,
                                               ensure_masks_exist, ensure_ratings_exist,
                                               ensure_workers_exist)


def get_mask_assignments_by_device_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their listening devices."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("devices", type=parse_non_empty_or_whitespace, metavar="DEVICE", nargs="+",
                      help="mask all assignments that were done on DEVICE", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_listening_device(ns.project, ns.masks, ns.devices, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def get_mask_assignments_by_status_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their statuses."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("states", type=parse_non_empty_or_whitespace, metavar="STATUS", nargs="+", choices=["Submitted", "Approved", "Rejected"],
                      help="mask all assignments that have status STATUS", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_state(ns.project, ns.masks, ns.states, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_assignments_by_time_parser(parser: ArgumentParser):
  parser.description = "Mask assignments by their submit time."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-time", type=parse_datetime, metavar="FROM-TIME",
                      help="mask all assignments, which have a submit time after than or equal to FROM-TIME (inclusive); format: 1970-12-31 23:59:59", default=datetime(1970, 1, 1, 12, 0, 0, 0, None))
  parser.add_argument("to_time", type=parse_datetime, metavar="TO-TIME",
                      help="mask all assignments, which have a submit time prior to TO-TIME (exclusive); format: 1970-12-31 23:59:59")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    mask_assignments_by_time(ns.project, ns.masks, ns.from_time, ns.to_time, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_rating_outliers_parser(parser: ArgumentParser):
  parser.description = "Mask outlying ratings of each algorithm."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  add_req_ratings_argument(parser)
  parser.add_argument("from_threshold", type=parse_non_negative_float, metavar="FROM-N-STD",
                      help="mask outlying ratings that lie at least FROM-N-STD standard deviations away from the mean of the respective algorithm; in [0, inf)")
  parser.add_argument("--to-threshold", type=parse_positive_float, metavar="TO-N-STD",
                      help="mask outlying ratings that lie below TO-N-STD standard deviations away from the mean of the respective algorithm (exclusive); in (0, inf)", default=math.inf)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_ratings_exist(ns.project, ns.ratings)
    mask_outlying_ratings(ns.project, ns.masks, ns.from_threshold,
                          ns.to_threshold, ns.output_mask, ns.ratings)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_workers_by_masked_ratings_count_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their percentual amount of masked ratings compared to all masked ratings."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("ref_masks", type=parse_non_empty_or_whitespace, nargs="+",
                      metavar="REF-MASK", help="masks on which the masked ratings should be counted", action=ConvertToSetAction)
  parser.add_argument("from_percent", type=parse_percent, metavar="FROM-PERCENT",
                      help="mask workers that have at least FROM-PERCENT of all masked ratings (inclusive); in [0, 100)")
  parser.add_argument("--to-percent", type=parse_percent, metavar="TO-PERCENT",
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


def init_mask_workers_by_id_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their WorkerId."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("worker_ids", type=parse_non_empty_or_whitespace,
                      metavar="WORKER-ID", nargs="+", help="mask workers with these WorkerId's", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_workers_exist(ns.project, ns.worker_ids)
    mask_workers_by_id(
      ns.project, ns.masks, ns.worker_ids, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_assignments_by_id_parser(parser: ArgumentParser):
  parser.description = "Mask assignments based on their AssignmentId."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("assignment_ids", type=parse_non_empty_or_whitespace,
                      metavar="ASSIGNMENT-ID", nargs="+", help="mask assignments with these AssignmentId's", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_assignments_exists(ns.project, ns.assignment_ids)
    mask_assignments_by_id(
      ns.project, ns.masks, ns.assignment_ids, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_create_mask_parser(parser: ArgumentParser):
  parser.description = "Create empty mask."
  add_req_project_argument(parser)
  parser.add_argument("mask_type", type=str, choices=["a", "w", "r"], help="type of the mask (assignment/worker/rating)")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    factory = MaskFactory(ns.project)
    if ns.mask_type == "a":
      mask = factory.get_amask()
    elif ns.mask_type == "w":
      mask = factory.get_wmask()
    elif ns.mask_type == "r":
      mask = factory.get_rmask()
    else:
      raise NotImplementedError()
    cast(EvaluationData, ns.project).add_or_update_mask(ns.output_mask, mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_age_group_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their age group."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("age_groups", type=parse_non_empty_or_whitespace,
                      metavar="AGE-GROUP", nargs="+", help="mask workers with these age groups", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    # maybe not useful to check
    # ensure_age_groups_exist(ns.project, ns.age_groups)
    mask_workers_by_age_group(ns.project, ns.masks, ns.age_groups, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_gender_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their age gender."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("genders", type=parse_non_empty_or_whitespace,
                      metavar="GENDER", nargs="+", help="mask workers with these genders", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    # maybe not useful to check
    # ensure_genders_exist(ns.project, ns.age_groups)
    mask_workers_by_gender(ns.project, ns.masks, ns.genders, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_merge_masks_parser(parser: ArgumentParser):
  parser.description = "Merge masks together."
  add_req_project_argument(parser)
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="+", metavar="MASK", help="merge these masks", action=ConvertToSetAction)
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    # maybe not useful to check
    # ensure_genders_exist(ns.project, ns.age_groups)
    merge_masks(ns.project, ns.masks, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_reverse_masks_parser(parser: ArgumentParser):
  # can be removed bc !maskname does the same thing
  parser.description = "Reverse mask."
  add_req_project_argument(parser)
  parser.add_argument("mask", type=parse_non_empty_or_whitespace,
                      metavar="MASK", help="reverse this mask")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_mask_exists(ns.project, ns.mask)
    reverse_mask(ns.project, ns.mask, ns.output_mask)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers."
  add_req_project_argument(parser)
  add_req_ratings_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-threshold", type=float, metavar="FROM-THRESHOLD",
                      help="mask workers that have a correlation with at least FROM-THRESHOLD (inclusive); in [-1; 1)", default=-1)
  parser.add_argument("to_threshold", type=float, metavar="TO-THRESHOLD",
                      help="mask workers that have a correlation smaller than TO-THRESHOLD (exclusive); in (-1; 1]")
  add_mode_argument(parser)
  parser.add_argument("--nan", action="store_true",
                      help="mask workers that have a correlation of NaN")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_ratings_exist(ns.project, ns.ratings)
    mask_workers_by_correlation(ns.project, ns.masks, ns.from_threshold,
                                ns.to_threshold, ns.mode, ns.output_mask, ns.ratings, ns.nan)

    if not ns.dry:
      save_project(ns.project)
  return main


def init_mask_workers_by_correlation_percent_parser(parser: ArgumentParser):
  parser.description = "Mask workers based on their sentence or algorithm correlation compared to other workers (percentage-wise)."
  add_req_project_argument(parser)
  add_req_ratings_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("--from-percent", type=parse_percent,
                      metavar="FROM-PERCENT", help="inclusive lower boundary; in [0; 100)", default=0)
  parser.add_argument("to_percent", type=parse_percent, metavar="TO-PERCENT",
                      help="exclusive top boundary; in (0; 100]")
  add_mode_argument(parser)
  parser.add_argument("--consider-masked-workers", action="store_true",
                      help="consider masked workers in percent calculation")
  add_req_output_mask_argument(parser)
  add_opt_dry_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_ratings_exist(ns.project, ns.ratings)
    mask_workers_by_correlation_percent(ns.project, ns.masks, ns.from_percent / 100,
                                        ns.to_percent / 100, ns.mode, ns.consider_masked_workers, ns.output_mask, ns.ratings)

    if not ns.dry:
      save_project(ns.project)
  return main


def add_mode_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--mode", type=str, choices=["sentence", "algorithm",
                      "both"], default="both", help="mode to calculate the correlations: sentence -> the correlation of the ratings of each audio file from a worker in comparison to the mean of the ratings of the other workers; algorithm -> the correlation of the mean ratings from one worker compared to the mean of all other workers for each algorithm; both -> the mean of sentence and algorithm correlation")
