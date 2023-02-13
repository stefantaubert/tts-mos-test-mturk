from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.statistics.algorithm_sentence_stats import get_algorithm_sentence_stats
from tts_mos_test_mturk_cli.argparse_helper import (ConvertToSetAction, parse_existing_file,
                                                    parse_non_empty_or_whitespace, parse_path)
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_export_as_stats_parser(parser: ArgumentParser):
  parser.description = "Write algorithm sentence statistics CSV."
  parser.add_argument("project", type=parse_existing_file, metavar="PROJECT-PATH",
                      help="project file (.pkl)")
  parser.add_argument("masks", type=parse_non_empty_or_whitespace,
                      nargs="*", metavar="MASK", help="apply these masks", action=ConvertToSetAction)
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

  result_df = get_algorithm_sentence_stats(project, ns.masks)

  if result_df is None:
    logger.info("No data exist to export!")
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
