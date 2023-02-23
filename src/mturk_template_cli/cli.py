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
from typing import Callable, Generator, List, Tuple

from mturk_template.logging import get_detail_logger, get_logger
from mturk_template_cli.example_parser import init_generate_example_input_parser
from mturk_template_cli.init_parser import init_convert_to_json_parser
from mturk_template_cli.types import CLIError
from tts_mos_test_mturk_cli.argparse_helper import get_optional, parse_path, parse_positive_integer
from tts_mos_test_mturk_cli.globals import APP_NAME
from tts_mos_test_mturk_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                          init_and_return_loggers,
                                                          try_init_file_buffer_logger)

__version__ = version(APP_NAME)

INVOKE_HANDLER_VAR = "invoke_handler"
DEFAULT_LOGGING_BUFFER_CAP = 1000000000

Parsers = Generator[Tuple[str, str, Callable], None, None]


Parsers = Generator[Tuple[str, str, Callable[[ArgumentParser],
                                             Callable[..., None]]], None, None]


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_parsers():
  yield "prepare-evaluation", "convert input data and results to .json-file", init_convert_to_json_parser
  yield "gen-example-input", "generate example input data", init_generate_example_input_parser


def print_features():
  parsers = get_parsers()
  for parser_name, help_str, methods in parsers:
    is_parent_parser = isinstance(methods, list)
    if is_parent_parser:
      print(f"- `{parser_name}`")
      for command, description, method in methods:
        print(f"  - `{command}`: {description}")
    else:
      print(f"- `{parser_name}`: {help_str}")


def add_logging_group(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "mturk-template.log"
  logging_group = parser.add_argument_group("logging arguments")
  logging_group.add_argument("--log", type=get_optional(parse_path), metavar="FILE",
                             nargs="?", const=None, help="path to write the log", default=default_log_path)
  logging_group.add_argument("--buffer-capacity", type=parse_positive_integer, default=DEFAULT_LOGGING_BUFFER_CAP,
                             metavar="CAPACITY", help="amount of logging lines that should be buffered before they are written to the log-file")
  logging_group.add_argument("--debug", action="store_true",
                             help="include debugging information in log")


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="CLI to prepare MOS evaluation with results from MTurk.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")

  parsers = get_parsers()
  for parser_name, help_str, methods in parsers:
    is_parent_parser = isinstance(methods, list)
    if is_parent_parser:
      sub_parser = subparsers.add_parser(parser_name, help=help_str, formatter_class=formatter)
      subparsers_of_subparser = sub_parser.add_subparsers()
      for command, description, method in methods:
        method_parser = subparsers_of_subparser.add_parser(
          command, help=description, formatter_class=formatter)
        invoke_method = method(method_parser)
        method_parser.set_defaults(**{
          INVOKE_HANDLER_VAR: invoke_method,
        })
        add_logging_group(method_parser)
    else:
      command = parser_name
      description = help_str
      method = methods
      method_parser = subparsers.add_parser(
        command, help=description, formatter_class=formatter)
      invoke_method = method(method_parser)
      method_parser.set_defaults(**{
        INVOKE_HANDLER_VAR: invoke_method,
      })
      add_logging_group(method_parser)
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

  invoke_handler: Callable[..., None] = getattr(ns, INVOKE_HANDLER_VAR)
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
    # Note: doesn't take so long on average so that it needs to be printed
    # root_logger.info("Writing remaining buffered log lines...")
  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / f"mturk-template-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"mturk-template-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run_prod()
