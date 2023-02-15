
from logging import Logger, getLogger


def get_logger() -> Logger:
  return getLogger("tts_mos_test_mturk")


def get_detail_logger() -> Logger:
  return getLogger("tts_mos_test_mturk.details")


def attach_boto_to_detail_logger():
  dlogger = get_detail_logger()
  getLogger("botocore.hooks").parent = dlogger
  getLogger("botocore.loaders").parent = dlogger
  getLogger("botocore.endpoint").parent = dlogger
  getLogger("botocore.client").parent = dlogger
  getLogger("botocore.regions").parent = dlogger
  getLogger("botocore.auth").parent = dlogger
  getLogger("botocore.httpsession").parent = dlogger
  getLogger("botocore.connectionpool").parent = dlogger
  getLogger("botocore.parsers").parent = dlogger
  getLogger("botocore.retryhandler").parent = dlogger


def attach_urllib3_to_detail_logger():
  dlogger = get_detail_logger()
  getLogger("urllib3.connectionpool").parent = dlogger
