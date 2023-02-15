from argparse import ArgumentParser, Namespace
from logging import Logger

import boto3
import pandas as pd

from tts_mos_test_mturk.api import approve_from_df, reject_from_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.algorithm_worker_stats import get_worker_algorithm_stats
from tts_mos_test_mturk_cli.argparse_helper import (parse_data_frame, parse_existing_file,
                                                    parse_non_empty_or_whitespace, parse_path)
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.globals import MTURK_SANDBOX
from tts_mos_test_mturk_cli.types import ExecutionResult


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

  return main


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
