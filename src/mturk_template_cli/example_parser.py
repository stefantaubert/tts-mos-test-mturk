from argparse import ArgumentParser, Namespace

from mturk_template.example_df_creation import generate_example_data
from tts_mos_test_mturk_cli.argparse_helper import (get_optional, parse_non_negative_integer,
                                                    parse_path)
from tts_mos_test_mturk_cli.helper import save_csv


def init_generate_example_input_parser(parser: ArgumentParser):
  parser.description = "Initialize a project from the ground truth and batch results."
  parser.add_argument("ground_truth_path", type=parse_path, metavar="GROUND-TRUTH-CSV",
                      help="path containing the ground truths for each url, i.e. a CSV-file with columns \"audio_url, algorithm, file\"")
  parser.add_argument("results_path", type=parse_path, metavar="RESULTS-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("upload_path", type=parse_path, metavar="UPLOAD-CSV",
                      help="path to the batch results file (something like \"Batch_374625_batch_results.csv\")")
  parser.add_argument("--seed", type=get_optional(parse_non_negative_integer),
                      metavar="SEED", default=None, help="custom seed for creating the data")

  def main(ns: Namespace) -> None:
    gt_df, input_df, output_df = generate_example_data(ns.seed)
    save_csv(ns.ground_truth_path, gt_df, "ground truth file")
    save_csv(ns.results_path, output_df, "batch results file")
    save_csv(ns.upload_path, input_df, "upload file")
  return main
