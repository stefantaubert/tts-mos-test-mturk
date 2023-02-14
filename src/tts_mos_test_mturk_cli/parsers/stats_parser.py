from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.stats import print_stats
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_stats_parser(parser: ArgumentParser):
  parser.description = "Calculate MOS for each algorithm"
  add_project_argument(parser)
  add_masks_argument(parser)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  print_stats(project, set(), ns.masks)

  return True
