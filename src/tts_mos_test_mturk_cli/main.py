from argparse import ArgumentParser, Namespace
from logging import Logger
from typing import Callable

from tts_mos_test_mturk_cli.argparse_helper import parse_existing_file
from tts_mos_test_mturk_cli.types import ExecutionResult


def init_from_mel_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the Mel-Cepstral Distance and the penalty between two mel files (.npy). Both audio files need to have the same sampling rate."
  parser.add_argument("mel1", type=parse_existing_file, metavar="MEL1",
                      help="path to the first .npy-file")
  parser.add_argument("mel2", type=parse_existing_file, metavar="MEL2",
                      help="path to the second .npy-file")
  return calc_mcd_from_mel_ns


def calc_mcd_from_mel_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  return True

