from argparse import ArgumentParser, Namespace
from logging import Logger

from tts_mos_test_mturk.df_generation import get_mos_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path
from tts_mos_test_mturk_cli.default_args import add_masks_argument, add_project_argument
from tts_mos_test_mturk_cli.types import ExecutionResult


def get_calculation_parser(parser: ArgumentParser):
  parser.description = "Calculate MOS for each algorithm"
  add_project_argument(parser)
  add_masks_argument(parser)
  parser.add_argument("--output", type=get_optional(parse_path),
                      help="write results to this CSV", metavar="OUTPUT-CSV", default=None)
  return main


def main(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    project = EvaluationData.load(ns.project)
  except Exception as ex:
    flogger.debug(ex)
    logger.error(f"Project \"{ns.project.absolute()}\" couldn't be loaded!")
    return False

  result_df = get_mos_df(project, ns.masks)

  if ns.output:
    try:
      result_df.to_csv(ns.output, index=False)
    except Exception as ex:
      flogger.debug(ex)
      logger.error(f"Output CSV \"{ns.output.absolute()}\" couldn't be saved!")
      return False
    logger.info(f"Written output to: \"{ns.output.absolute()}\"")
  else:
    print(result_df)

  return True
