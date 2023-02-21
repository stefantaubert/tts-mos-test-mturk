from argparse import ArgumentParser, Namespace

import boto3
from mypy_boto3_mturk import MTurkClient

from tts_mos_test_mturk.api import approve_from_df, grant_bonuses_from_df, reject_from_df
from tts_mos_test_mturk.df_generation import (generate_approve_csv, generate_bonus_csv,
                                              generate_reject_csv)
from tts_mos_test_mturk_cli.argparse_helper import (get_optional, parse_data_frame,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float, parse_path)
from tts_mos_test_mturk_cli.default_args import add_opt_masks_argument, add_req_project_argument
from tts_mos_test_mturk_cli.globals import MTURK_SANDBOX
from tts_mos_test_mturk_cli.helper import save_csv
from tts_mos_test_mturk_cli.types import CLIError
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def create_aws_session(aws_access_key_id: str, aws_secret_access_key: str) -> boto3.Session:
  try:
    session = boto3.Session(
      aws_access_key_id=aws_access_key_id,
      aws_secret_access_key=aws_secret_access_key,
      # region_name="us-east-1",
    )
  except Exception as ex:
    raise CLIError("AWS session couldn't be created!") from ex
  return session


def create_mturk_client_from_session(session: boto3.Session, endpoint_url: str) -> MTurkClient:
  try:
    mturk = session.client('mturk', endpoint_url=endpoint_url)
  except Exception as ex:
    raise CLIError("MTurk client couldn't be established!") from ex
  return mturk


def create_mturk_client(aws_access_key_id: str, aws_secret_access_key: str, endpoint_url: str) -> MTurkClient:
  session = create_aws_session(aws_access_key_id, aws_secret_access_key)
  return create_mturk_client_from_session(session, endpoint_url)


def init_prepare_approval_parser(parser: ArgumentParser):
  parser.description = "Generate a CSV-file in which all unmasked assignments will be listed for them to be approved via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")
  parser.add_argument("--costs", type=get_optional(parse_non_negative_float),
                      metavar="DOLLAR", help="costs for one approval (in $)", default=None)
  parser.add_argument("--reason", type=get_optional(parse_non_empty_or_whitespace), metavar="REASON",
                      help="use custom reason instead of \"x\"", default=None)

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_approve_csv(ns.project, ns.masks, ns.reason, ns.costs)
    # print(result_df)
    save_csv(ns.output, result_df)
  return main


def init_approve_parser(parser: ArgumentParser):
  parser.description = "Approve all assignments in a CSV-file via API."
  parser.add_argument("aws_access_key_id", type=parse_non_empty_or_whitespace,
                      help="AWS access key id", metavar="AWS_ACCESS_KEY_ID")
  parser.add_argument("aws_secret_access_key", type=parse_non_empty_or_whitespace,
                      help="AWS secret access key", metavar="AWS_SECRET_ACCESS_KEY")
  parser.add_argument("approve_csv", type=parse_data_frame,
                      metavar="APPROVE-CSV", help="path containing the approval CSV")
  # parser.add_argument("-p", "--productive", action="store_true",
  #                     help="use production API and not sandbox")
  parser.add_argument("--endpoint", type=parse_non_empty_or_whitespace,
                      default=MTURK_SANDBOX, help="API endpoint")
  # parser.add_argument("--region", type=parse_non_empty_or_whitespace,
  #                     default="us-east-1", help="region")

  def main(ns: Namespace) -> None:
    mturk = create_mturk_client(ns.aws_access_key_id, ns.aws_secret_access_key, ns.endpoint)
    if not approve_from_df(ns.approve_csv, mturk):
      raise CLIError("Not all assignments could've been approved!")
  return main


def init_prepare_rejection_parser(parser: ArgumentParser):
  parser.description = "Generate a CSV-file in which all masked assignments will be listed for them to be rejected via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="use this reason")
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_reject_csv(ns.project, ns.masks, ns.reason)
    save_csv(ns.output, result_df)
  return main


def init_reject_parser(parser: ArgumentParser):
  parser.description = "Reject all assignments in a CSV-file via API."
  parser.add_argument("aws_access_key_id", type=parse_non_empty_or_whitespace,
                      help="AWS access key id", metavar="AWS_ACCESS_KEY_ID")
  parser.add_argument("aws_secret_access_key", type=parse_non_empty_or_whitespace,
                      help="AWS secret access key", metavar="AWS_SECRET_ACCESS_KEY")
  parser.add_argument("reject_csv", type=parse_data_frame,
                      metavar="REJECT-CSV", help="path containing the rejection CSV")
  # parser.add_argument("-p", "--productive", action="store_true",
  #                     help="use production API and not sandbox")
  parser.add_argument("--endpoint", type=parse_non_empty_or_whitespace,
                      default=MTURK_SANDBOX, help="API endpoint")
  # parser.add_argument("--region", type=parse_non_empty_or_whitespace,
  #                     default="us-east-1", help="region")

  def main(ns: Namespace) -> None:
    mturk = create_mturk_client(ns.aws_access_key_id, ns.aws_secret_access_key, ns.endpoint)
    if not reject_from_df(ns.reject_csv, mturk):
      raise CLIError("Not all assignments could've been rejected!")
  return main


def init_prepare_bonus_payment_parser(parser: ArgumentParser):
  parser.description = "Generate a CSV-file in which all unmasked assignments will be listed for them to be paid a bonus via API or the MTurk website."
  add_req_project_argument(parser)
  add_opt_masks_argument(parser)
  parser.add_argument("amount", type=parse_non_negative_float,
                      metavar="BONUS", help="bonus amount in $")
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="reason for paying the bonus")
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")

  def main(ns: Namespace) -> None:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_bonus_csv(ns.project, ns.masks, ns.amount, ns.reason)
    # print(result_df)
    save_csv(ns.output, result_df)
  return main


def init_pay_bonus_parser(parser: ArgumentParser):
  parser.description = "Pay all assignments in a CSV-file a bonus via API."
  parser.add_argument("aws_access_key_id", type=parse_non_empty_or_whitespace,
                      help="AWS access key id", metavar="AWS_ACCESS_KEY_ID")
  parser.add_argument("aws_secret_access_key", type=parse_non_empty_or_whitespace,
                      help="AWS secret access key", metavar="AWS_SECRET_ACCESS_KEY")
  parser.add_argument("bonus_csv", type=parse_data_frame,
                      metavar="BONUS-CSV", help="path containing the bonuses CSV")
  # parser.add_argument("-p", "--productive", action="store_true",
  #                     help="use production API and not sandbox")
  parser.add_argument("--endpoint", type=parse_non_empty_or_whitespace,
                      default=MTURK_SANDBOX, help="API endpoint")
  # parser.add_argument("--region", type=parse_non_empty_or_whitespace,
  #                     default="us-east-1", help="region")

  def main(ns: Namespace) -> None:
    mturk = create_mturk_client(ns.aws_access_key_id, ns.aws_secret_access_key, ns.endpoint)
    if not grant_bonuses_from_df(ns.bonus_csv, mturk):
      raise CLIError("Not all assignments could've been bonused!")
  return main
