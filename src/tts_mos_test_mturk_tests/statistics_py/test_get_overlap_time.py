from datetime import datetime, timedelta

from tts_mos_test_mturk.statistics.worker_assignment_stats import get_overlap_time


def get_date(day: int) -> datetime:
  return datetime(1, 1, day)


def test_a1b2b1():
  time1, time2 = (
    (get_date(1), get_date(4)),
    (get_date(1), get_date(3))
  )
  result = get_overlap_time(time1, time2)
  assert result == timedelta(days=2)


def test_a1a2b1():
  time1, time2 = (
    (get_date(1), get_date(4)),
    (get_date(2), get_date(4))
  )
  result = get_overlap_time(time1, time2)
  assert result == timedelta(days=2)


def test_a1b2b1_a1a2b1():
  time1, time2 = (
    (get_date(1), get_date(5)),
    (get_date(2), get_date(4))
  )
  result = get_overlap_time(time1, time2)
  assert result == timedelta(days=2)


def test_no_overlap():
  time1, time2 = (
    (get_date(1), get_date(4)),
    (get_date(4), get_date(8))
  )
  result = get_overlap_time(time1, time2)
  assert result == timedelta(days=0)
