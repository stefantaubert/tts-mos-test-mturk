
from logging import Logger, getLogger


def get_logger() -> Logger:
  return getLogger("tts_mos_test_mturk")


def get_detail_logger() -> Logger:
  return getLogger("tts_mos_test_mturk.details")
