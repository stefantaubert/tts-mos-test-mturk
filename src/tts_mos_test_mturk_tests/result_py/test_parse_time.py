import datetime

from tts_mos_test_mturk.result import parse_time


def test_component_PST():
  result = parse_time("Mon Mar 06 08:59:21 PST 2023")
  assert result == datetime.datetime(2023, 3, 6, 8, 59, 21)


def test_component_PDT():
  result = parse_time("Mon Mar 06 08:59:21 PDT 2023")
  assert result == datetime.datetime(2023, 3, 6, 8, 59, 21)
