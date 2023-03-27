from argparse import ArgumentParser, Namespace

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.result import parse_result_from_json
from tts_mos_test_mturk_cli.argparse_helper import parse_json, parse_path
from tts_mos_test_mturk_cli.helper import save_project
from tts_mos_test_mturk_cli.logging_configuration import get_cli_logger


def init_init_project_parser(parser: ArgumentParser):
  parser.description = "Initialize a project from the ground truth and batch results."
  parser.add_argument("result_json", type=parse_json, metavar="RESULT-JSON",
                      help="path containing the results (.json-file)")
  parser.add_argument("output", type=parse_path, metavar="OUTPUT-PROJECT-PATH",
                      help="output project file (.pkl)")

  def main(ns: Namespace) -> None:
    result = parse_result_from_json(ns.result_json)
    data = EvaluationData(result)
    data.file_path = ns.output
    logger = get_cli_logger()
    logger.info(f"Parsed {data.n_workers} workers, {data.n_assignments} assignments and {len(data.rating_names)} x {data.n_ratings} ratings for {data.n_algorithms} algorithms and {data.n_files} files.")

    save_project(data)
  return main
