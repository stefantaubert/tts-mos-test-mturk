from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.bad_worker_filtering import generate_approve_csv
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToOrderedSetAction, get_optional,
                                                    parse_existing_file,
                                                    parse_non_empty_or_whitespace, parse_path)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_approve_parser(parser: ArgumentParser):
  parser.description = "Write approvement CSV of all unmasked assignments."
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToOrderedSetAction)
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")
  parser.add_argument("--reason", type=get_optional(parse_non_empty_or_whitespace), metavar="REASON",
                      help="use custom reason instead of \"x\"", default=None)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  result_df = generate_approve_csv(project, ns.masks, ns.reason)
  # print(result_df)

  if result_df is None:
    logger.info("No assignments exist to approve!")
    return True

  try:
    result_df.to_csv(ns.output, index=False)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Written output to: \"{ns.output.absolute()}\"")

  return True
