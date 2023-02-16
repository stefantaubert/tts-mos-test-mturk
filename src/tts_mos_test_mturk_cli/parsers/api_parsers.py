from argparse import ArgumentParser, Namespace
from logging import Logger

import boto3
import pandas as pd

from tts_mos_test_mturk.api import approve_from_df, grant_bonuses_from_df, reject_from_df
from tts_mos_test_mturk.df_generation import (generate_approve_csv, generate_bonus_csv,
                                              generate_reject_csv)
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats
from tts_mos_test_mturk.statistics.worker_assignment_stats import get_worker_assignment_stats
from tts_mos_test_mturk_cli.argparse_helper import (get_optional, parse_data_frame,
                                                    parse_non_empty_or_whitespace,
                                                    parse_non_negative_float, parse_path)
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.globals import MTURK_SANDBOX
from tts_mos_test_mturk_cli.types import CLIError, ExecutionResult
from tts_mos_test_mturk_cli.validation import ensure_masks_exist


def get_api_approve_parser(parser: ArgumentParser):
  parser.description = "Approve via API."
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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    try:
      session = boto3.Session(
        aws_access_key_id=ns.aws_access_key_id,
        aws_secret_access_key=ns.aws_secret_access_key,
        # region_name="us-east-1",
      )
    except Exception as ex:
      flogger.debug(ex)
      logger.error("Session couldn't be created!")
      return False

    try:
      mturk = session.client('mturk', endpoint_url=ns.endpoint)
    except Exception as ex:
      flogger.debug(ex)
      logger.error("MTurk client couldn't be established!")
      return False

    success = approve_from_df(ns.approve_csv, mturk)

    return success

  return main


def create_aws_session(aws_access_key_id: str, aws_secret_access_key: str):
  try:
    session = boto3.Session(
      aws_access_key_id=aws_access_key_id,
      aws_secret_access_key=aws_secret_access_key,
      # region_name="us-east-1",
    )
  except Exception as ex:
    raise CLIError("Session couldn't be created!") from ex
  return session


def get_api_bonus_parser(parser: ArgumentParser):
  parser.description = "Bonus via API."
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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    try:
      session = boto3.Session(
        aws_access_key_id=ns.aws_access_key_id,
        aws_secret_access_key=ns.aws_secret_access_key,
        # region_name="us-east-1",
      )
    except Exception as ex:
      flogger.debug(ex)
      logger.error("Session couldn't be created!")
      return False

    try:
      mturk = session.client('mturk', endpoint_url=ns.endpoint)
    except Exception as ex:
      flogger.debug(ex)
      logger.error("MTurk client couldn't be established!")
      return False

    success = grant_bonuses_from_df(ns.bonus_csv, mturk)

    return success
  return main


def get_api_reject_parser(parser: ArgumentParser):
  parser.description = "Reject via API."
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

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    try:
      session = boto3.Session(
        aws_access_key_id=ns.aws_access_key_id,
        aws_secret_access_key=ns.aws_secret_access_key,
        # region_name="us-east-1",
      )
    except Exception as ex:
      flogger.debug(ex)
      logger.error("Session couldn't be created!")
      return False

    try:
      mturk = session.client('mturk', endpoint_url=ns.endpoint)
    except Exception as ex:
      flogger.debug(ex)
      logger.error("MTurk client couldn't be established!")
      return False

    success = reject_from_df(ns.reject_csv, mturk)

    return success
  return main


def get_approve_parser(parser: ArgumentParser):
  parser.description = "Write approvement CSV of all unmasked assignments."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")
  parser.add_argument("--reason", type=get_optional(parse_non_empty_or_whitespace), metavar="REASON",
                      help="use custom reason instead of \"x\"", default=None)

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_approve_csv(ns.project, ns.masks, ns.reason)
    # print(result_df)

    try:
      result_df.to_csv(ns.output, index=False)
    except Exception as ex:
      flogger.debug(ex)
      logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
      return False
    logger.info(f"Written output to: \"{ns.output.absolute()}\"")

    return True
  return main


def get_bonus_parser(parser: ArgumentParser):
  parser.description = "Write bonus CSV of all unmasked assignments."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("amount", type=parse_non_negative_float,
                      metavar="AMOUNT", help="bonus amount in $")
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="reason for bonus")
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_bonus_csv(ns.project, ns.masks, ns.amount, ns.reason)
    # print(result_df)

    try:
      result_df.to_csv(ns.output, index=False)
    except Exception as ex:
      flogger.debug(ex)
      logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
      return False
    logger.info(f"Written output to: \"{ns.output.absolute()}\"")

    return True
  return main


def get_reject_parser(parser: ArgumentParser):
  parser.description = "Write rejectment CSV of all masked assignments."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="use this reason")
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")

  def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
    ensure_masks_exist(ns.project, ns.masks)
    result_df = generate_reject_csv(ns.project, ns.masks, ns.reason)

    try:
      result_df.to_csv(ns.output, index=False)
    except Exception as ex:
      flogger.debug(ex)
      logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
      return False
    logger.info(f"Written output to: \"{ns.output.absolute()}\"")

    return True
  return main
