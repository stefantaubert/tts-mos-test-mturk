from argparse import ArgumentParser, Namespace

from tts_mos_test_mturk.df_generation import generate_ground_truth_table, get_mos_df
from tts_mos_test_mturk.statistics.algorithm_sentence_stats import get_algorithm_sentence_stats
from tts_mos_test_mturk.statistics.algorithm_worker_stats import get_worker_algorithm_stats
from tts_mos_test_mturk.statistics.update_stats import print_stats
from tts_mos_test_mturk.statistics.worker_assignment_stats import get_worker_assignment_stats
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.helper import print_full_df, save_output_csv
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def add_optional_output_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--output", type=get_optional(parse_path),
                      help="save results to this CSV-file", metavar="OUTPUT-CSV", default=None)


def add_silent_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-s", "--silent", action="store_true",
                      help="don't print results to console")


def get_export_wa_stats_parser(parser: ArgumentParser):
  parser.description = "Print assignment statistics for each worker."
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = get_worker_assignment_stats(ns.project, ns.masks)

    if not ns.silent:
      print_full_df(result_df)

    if ns.output:
      return save_output_csv(ns.output, result_df)
  return main


def get_stats_parser(parser: ArgumentParser):
  parser.description = "Print masks statistics regarding masked workers, assignments and opinion scores."
  add_project_argument(parser)
  add_masks_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    print_stats(ns.project, set(), ns.masks)
  return main


def get_calculation_parser(parser: ArgumentParser):
  parser.description = "Print MOS and CI95 statistics for each algorithm."
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = get_mos_df(ns.project, ns.masks)

    if not ns.silent:
      print_full_df(result_df)

    if ns.output:
      return save_output_csv(ns.output, result_df)
  return main


def get_export_as_stats_parser(parser: ArgumentParser):
  parser.description = "Print sentence statistics for each algorithm."
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = get_algorithm_sentence_stats(ns.project, ns.masks)

    if not ns.silent:
      print_full_df(result_df)

    if ns.output:
      return save_output_csv(ns.output, result_df)
  return main


def get_export_aw_stats_parser(parser: ArgumentParser):
  parser.description = "Print worker statistics for each algorithm."
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = get_worker_algorithm_stats(ns.project, ns.masks)

    if not ns.silent:
      print_full_df(result_df)

    if ns.output:
      return save_output_csv(ns.output, result_df)
  return main


def get_export_gt_parser(parser: ArgumentParser):
  parser.description = "Print all opinion scores including all metadata."
  add_project_argument(parser)
  add_masks_argument(parser)
  add_optional_output_argument(parser)
  add_silent_argument(parser)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_ground_truth_table(ns.project, ns.masks)

    if not ns.silent:
      print_full_df(result_df)

    if ns.output:
      return save_output_csv(ns.output, result_df)
  return main
