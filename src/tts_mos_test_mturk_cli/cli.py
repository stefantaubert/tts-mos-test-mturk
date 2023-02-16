import argparse
import logging
import platform
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, List

from tts_mos_test_mturk.logging import (attach_boto_to_detail_logger,
                                        attach_urllib3_to_detail_logger, get_detail_logger,
                                        get_logger)
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path, parse_positive_integer
from tts_mos_test_mturk_cli.globals import APP_NAME
from tts_mos_test_mturk_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                          init_and_return_loggers,
                                                          try_init_file_buffer_logger)
from tts_mos_test_mturk_cli.parsers import *
from tts_mos_test_mturk_cli.types import CLIError, ExecutionResult

__version__ = version(APP_NAME)

INVOKE_HANDLER_VAR = "invoke_handler"
DEFAULT_LOGGING_BUFFER_CAP = 1000000000


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_parsers():
  yield "init", "initialize project", get_init_parser  # initialize
  yield "export-ground-truth", "export ground truth", get_export_gt_parser
  yield "calc-mos", "calculate MOS with CI95", get_calculation_parser  # calculate-mos
  yield "stats", "print statistics", get_stats_parser  # print-statistics
  yield "stats-worker-assignments", "export worker assignment stats", get_export_wa_stats_parser
  yield "stats-algorithm-worker", "export algorithm worker stats", get_export_aw_stats_parser
  yield "stats-algorithm-sentences", "export algorithm sentence stats", get_export_as_stats_parser
  yield "approve", "approve", get_approve_parser  # assignments create-approve-csv
  yield "reject", "reject", get_reject_parser  # assignments create-reject-csv
  yield "bonus", "bonus assignments", get_bonus_parser  # assignments create-bonus-csv
  yield "ignore-too-fast", "ignore too fast assignments", get_mask_assignments_by_work_time_parser  # mask-by-work_time
  # assignments mask-by-count
  yield "ignore-too-few", "ignore workers with to few assignments", get_mask_workers_by_assignment_count_parser
  yield "ignore-by-listening-device", "ignore by device", get_mask_assignments_by_listening_device_parser
  # workers mask-by-correlation
  yield "ignore-bad-workers", "ignore too bad workers", get_mask_workers_by_correlation_parser
  yield "ignore-bad-workers-percent", "ignore bad workers by percentage", get_mask_workers_by_correlation_percent_parser
  yield "ignore-outliers", "ignore outliers", get_mask_outlying_ratings_parser  # opinions mask-by-std
  # opinions mask-by-masked-count
  yield "ignore-os-count", "ignore workers who overreach a specific percentage of all masked ratings", get_mask_ratings_by_masked_count_parser
  yield "approve-via-api", "approve via API", get_api_approve_parser
  yield "reject-via-api", "reject via API", get_api_reject_parser
  yield "bonus-via-api", "bonus via API", get_api_bonus_parser


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="CLI to evaluate MOS results from MTurk and approve/reject workers.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")
  default_log_path = Path(gettempdir()) / f"{APP_NAME}.log"

  methods = get_parsers()
  for command, description, method in methods:
    method_parser = subparsers.add_parser(
      command, help=description, formatter_class=formatter)
    method_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: method(method_parser),
    })
    logging_group = method_parser.add_argument_group("logging arguments")
    logging_group.add_argument("--log", type=get_optional(parse_path), metavar="FILE",
                               nargs="?", const=None, help="path to write the log", default=default_log_path)
    logging_group.add_argument("--buffer-capacity", type=parse_positive_integer, default=DEFAULT_LOGGING_BUFFER_CAP,
                               metavar="CAPACITY", help="amount of logging lines that should be buffered before they are written to the log-file")
    logging_group.add_argument("--debug", action="store_true",
                               help="include debugging information in log")

  return main_parser


def configure_logger(productive: bool) -> None:
  loglevel = logging.INFO if productive else logging.DEBUG
  main_logger = getLogger()
  main_logger.setLevel(loglevel)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  logging_formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
    '%Y/%m/%d %H:%M:%S',
  )
  console.setFormatter(logging_formatter)
  console.setLevel(loglevel)


def parse_args(args: List[str]) -> None:
  configure_root_logger()
  root_logger = getLogger()

  local_debugging = debug_file_exists()
  if local_debugging:
    root_logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit as exception:
    error_code = exception.args[0]
    # -v -> 0; invalid arg -> 2
    sys.exit(error_code)

  if local_debugging:
    root_logger.debug(f"Parsed arguments: {str(ns)}")

  if not hasattr(ns, INVOKE_HANDLER_VAR):
    parser.print_help()
    sys.exit(0)

  invoke_handler: Callable[..., ExecutionResult] = getattr(ns, INVOKE_HANDLER_VAR)
  delattr(ns, INVOKE_HANDLER_VAR)
  log_to_file = ns.log is not None
  if log_to_file:
    # log_to_file = try_init_file_logger(ns.log, local_debugging or ns.debug)
    log_to_file = try_init_file_buffer_logger(
      ns.log, local_debugging or ns.debug, ns.buffer_capacity)
    if not log_to_file:
      root_logger.warning("Logging to file is not possible.")

  flogger = get_file_logger()
  if not local_debugging:
    sys_version = sys.version.replace('\n', '')
    flogger.debug(f"CLI version: {__version__}")
    flogger.debug(f"Python version: {sys_version}")
    flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

    my_system = platform.uname()
    flogger.debug(f"System: {my_system.system}")
    flogger.debug(f"Node Name: {my_system.node}")
    flogger.debug(f"Release: {my_system.release}")
    flogger.debug(f"Version: {my_system.version}")
    flogger.debug(f"Machine: {my_system.machine}")
    flogger.debug(f"Processor: {my_system.processor}")

  flogger.debug(f"Received arguments: {str(args)}")
  flogger.debug(f"Parsed arguments: {str(ns)}")

  start = perf_counter()
  cmd_flogger, cmd_logger = init_and_return_loggers()

  # success, changed_anything = invoke_handler(ns, cmd_logger, cmd_flogger)
  core_main_logger = get_logger()
  core_main_logger.parent = cmd_logger
  core_detail_logger = get_detail_logger()
  core_detail_logger.parent = cmd_flogger
  attach_boto_to_detail_logger()
  attach_urllib3_to_detail_logger()

  success = True
  try:
    invoke_handler(ns)
  except CLIError as error:
    cmd_logger.error(error.args[0])
    cmd_flogger.debug(error, exc_info=True)
    success = False
  except Exception as exception:
    cmd_logger.error("Unexpected error occurred!")
    cmd_flogger.debug(exception, exc_info=True)
    success = False

  exit_code = 0
  if success:
    flogger.info("Everything was successful!")
  else:
    exit_code = 1
    # cmd_logger.error(f"Validation error: {success.default_message}")
    if log_to_file:
      root_logger.error("Not everything was successful! See log for details.")
    else:
      root_logger.error("Not everything was successful!")
    flogger.error("Not everything was successful!")

  duration = perf_counter() - start
  flogger.debug(f"Total duration (s): {duration}")
  if log_to_file:
    # path not encapsulated in "" because it is only console out
    root_logger.info(f"Log: \"{ns.log.absolute()}\"")
    root_logger.info("Writing remaining buffered log lines...")
  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / f"{APP_NAME}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{APP_NAME}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run_prod()
