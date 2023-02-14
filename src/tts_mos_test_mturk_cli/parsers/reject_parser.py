from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.df_generation import generate_reject_csv
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import parse_non_empty_or_whitespace, parse_path
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_reject_parser(parser: ArgumentParser):
  parser.description = "Write rejectment CSV of all masked assignments."
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("reason", type=parse_non_empty_or_whitespace, metavar="REASON",
                      help="use this reason")
  parser.add_argument("output", type=parse_path,
                      help="write CSV to this path", metavar="OUTPUT-CSV")
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  result_df = generate_reject_csv(project, ns.masks, ns.reason)

  if result_df is None:
    logger.info("No assignments exist to reject!")
    return True
  # print(result_df)

  try:
    result_df.to_csv(ns.output, index=False)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Written output to: \"{ns.output.absolute()}\"")

  return True
