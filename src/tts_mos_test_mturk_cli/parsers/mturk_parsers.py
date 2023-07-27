from argparse import ArgumentParser, Namespace

from tts_mos_test_mturk.df_generation import (generate_approve_csv, generate_bonus_csv,
                                              generate_reject_csv)
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction, get_optional,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float, parse_path,
                                                    parse_percent)
from tts_mos_test_mturk_cli.default_args import add_opt_masks_argument, add_req_project_argument
from tts_mos_test_mturk_cli.helper import save_csv, save_json
from tts_mos_test_mturk_cli.types import CLIError
from tts_mos_test_mturk_cli.validation import ensure_masks_exist

DEFAULT_AMAZON_FEE = 20


def init_prepare_approval_parser(parser: ArgumentParser):
  parser.description = "Generate a CSV-file in which all unmasked assignments will be listed for them to be approved via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("output", type=parse_path,
                      help="write CSV/JSON to this path", metavar="OUTPUT")
  parser.add_argument("--costs", type=get_optional(parse_non_negative_float),
                      metavar="DOLLAR", help="costs for one approval (in $)", default=None)
  parser.add_argument("--fee", type=get_optional(parse_percent),
                      metavar="PERCENT", help="Mechanical Turk fee (in percent)", default=DEFAULT_AMAZON_FEE)
  parser.add_argument("--reason", type=get_optional(parse_non_empty_or_whitespace), metavar="REASON",
                      help="use custom reason instead of \"x\"", default=None)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df, result_dict = generate_approve_csv(
      ns.project, ns.masks, ns.reason, ns.costs, ns.fee / 100)
    if str(ns.output).lower().endswith(".csv"):
      save_csv(ns.output, result_df)
    elif str(ns.output).lower().endswith(".json"):
      save_json(ns.output, result_dict)
    else:
      raise CLIError("File needs to be CSV or JSON!")
  return main


def init_prepare_rejection_parser(parser: ArgumentParser):
  parser.description = "Generate a CSV-file in which all masked assignments will be listed for them to be rejected via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("reject_masks", type=parse_non_empty_or_whitespace,
                      nargs="+", metavar="REJECT-MASK", help="reject masked assignments from REJECT-MASK", action=ConvertToSetAction, default=set())
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="use this reason")
  parser.add_argument("output", type=parse_path,
                      help="write CSV/JSON to this path", metavar="OUTPUT")

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    ensure_masks_exist(ns.project, ns.reject_masks)
    result_df, result_dict = generate_reject_csv(ns.project, ns.masks, ns.reject_masks, ns.reason)
    if str(ns.output).lower().endswith(".csv"):
      save_csv(ns.output, result_df)
    elif str(ns.output).lower().endswith(".json"):
      save_json(ns.output, result_dict)
    else:
      raise CLIError("File needs to be CSV or JSON!")
  return main


def init_prepare_bonus_payment_parser(parser: ArgumentParser):
  parser.description = "Generate a file in which all unmasked assignments will be listed for them to be paid a bonus via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("amount", type=parse_non_negative_float,
                      metavar="BONUS", help="bonus amount in $")
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="reason for paying the bonus")
  parser.add_argument("--fee", type=parse_percent,
                      metavar="PERCENT", help="Mechanical Turk fee (in percent)", default=DEFAULT_AMAZON_FEE)
  parser.add_argument("output", type=parse_path,
                      help="write file to this path (.json/.csv)", metavar="OUTPUT")

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df, result_dict = generate_bonus_csv(
      ns.project, ns.masks, ns.amount, ns.reason, ns.fee / 100)
    # print(result_df)
    if str(ns.output).lower().endswith(".csv"):
      save_csv(ns.output, result_df)
    elif str(ns.output).lower().endswith(".json"):
      save_json(ns.output, result_dict)
    else:
      raise CLIError("File needs to be CSV or JSON!")
  return main
