import json
from argparse import ArgumentParser, Namespace

from mturk_template.convert_to_json import convert_to_json
from tts_mos_test_mturk_cli.argparse_helper import parse_data_frame, parse_path
from tts_mos_test_mturk_cli.logging_configuration import get_cli_logger


def init_convert_to_json_parser(parser: ArgumentParser):
  parser.description = "Initialize a project from the ground truth and batch results."
  parser.add_argument("ground_truth_df", type=parse_data_frame, metavar="GROUND-TRUTH-CSV",
                      help="path containing the ground truths for each url, i.e. a CSV-file with columns \"audio_url, algorithm, file\"")
  parser.add_argument("results_df", type=parse_data_frame, metavar="RESULTS-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("output", type=parse_path, metavar="OUTPUT-PATH",
                      help="output file (.json)")

  def main(ns: Namespace) -> None:
    data = convert_to_json(ns.results_df, ns.ground_truth_df)
    with open(ns.output, mode="w", encoding="utf-8") as f:
      json.dump(data, f, indent=2, sort_keys=False)
    logger = get_cli_logger()
    logger.info(f"Written output to: \"{ns.output.absolute()}\"")
  return main
