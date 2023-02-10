from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd

from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import parse_existing_file, parse_path
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_init_parser(parser: ArgumentParser):
  parser.description = "This command reads the lines of a text file and initializes a dataset from it."
  parser.add_argument("ground_truth_path", type=parse_existing_file, metavar="GROUND-TRUTH-CSV",
                      help="path containing the ground truths for each url, i.e. a CSV-file with columns \"audio_url, algorithm, file\"")
  parser.add_argument("results_path", type=parse_existing_file, metavar="RESULTS-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("output", type=parse_path, metavar="OUTPUT-PROJECT-PATH",
                      help="output project file (.pkl)")
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    results_df = pd.read_csv(ns.results_path)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"File \"{ns.results_path.absolute()}\" couldn't be parsed!")
    return False

  try:
    ground_truth_df = pd.read_csv(ns.ground_truth_path)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"File \"{ns.ground_truth_path.absolute()}\" couldn't be parsed!")
    return False

  data = EvaluationData(results_df, ground_truth_df)

  try:
    data.save(ns.output)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"File \"{ns.output.absolute()}\" couldn't be saved!")
    return False
  logger.info(f"Written project to: \"{ns.output.absolute()}\"")
  return True
