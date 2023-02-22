
from logging import Logger, getLogger


def get_logger() -> Logger:
  return getLogger("mturk_template")


def get_detail_logger() -> Logger:
  return getLogger("mturk_template.details")
