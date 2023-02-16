from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import parse_data_frame, parse_path
from tts_mos_test_mturk_cli.helper import save_project


def get_init_parser(parser: ArgumentParser):
  parser.description = "Initialize a project from the ground truth and batch results."
  parser.add_argument("ground_truth_path", type=parse_data_frame, metavar="GROUND-TRUTH-CSV",
                      help="path containing the ground truths for each url, i.e. a CSV-file with columns \"audio_url, algorithm, file\"")
  parser.add_argument("results_path", type=parse_data_frame, metavar="RESULTS-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("output", type=parse_path, metavar="OUTPUT-PROJECT-PATH",
                      help="output project file (.pkl)")

  def main(ns: Namespace) -> None:
    data = EvaluationData(ns.results_path, ns.ground_truth_path)
    data.file_path = ns.output
    save_project(data)
  return main
