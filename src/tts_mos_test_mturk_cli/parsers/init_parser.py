from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import parse_data_frame, parse_existing_file, parse_path
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_init_parser(parser: ArgumentParser):
  parser.description = "This command reads the lines of a text file and initializes a dataset from it."
  parser.add_argument("ground_truth_path", type=parse_data_frame, metavar="GROUND-TRUTH-CSV",
                      help="path containing the ground truths for each url, i.e. a CSV-file with columns \"audio_url, algorithm, file\"")
  parser.add_argument("results_path", type=parse_data_frame, metavar="RESULTS-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("output", type=parse_path, metavar="OUTPUT-PROJECT-PATH",
                      help="output project file (.pkl)")
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  data = EvaluationData(ns.results_path, ns.ground_truth_path)
  data.file_path = ns.output

  success = save_project(data, logger, flogger)
  return success
