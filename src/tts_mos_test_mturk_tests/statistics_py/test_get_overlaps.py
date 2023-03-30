from datetime import datetime

from tts_mos_test_mturk.statistics.worker_assignment_stats import get_overlaps


def get_date(year: int) -> datetime:
  return datetime(year, 1, 1)


def test_component():
  timings = {
    (get_date(1), get_date(4)),  # a1, b1; overlap with #1 #2 #3 #5
    (get_date(1), get_date(2)),  # a1 < b2 < b1
    (get_date(3), get_date(4)),  # a1 < a2 < b1
    (get_date(2), get_date(3)),  # both
    (get_date(5), get_date(6)),  # None
    (get_date(2), get_date(4)),  # overlap with #0 #2 #3
  }
  result = list(get_overlaps(timings))
  assert result == [
    (  # 0 #1
      (get_date(1), get_date(4)),
      (get_date(1), get_date(2))
    ),
    (  # 0 #3x
      (get_date(1), get_date(4)),
      (get_date(2), get_date(3))
    ),
    (  # 0 #5x
      (get_date(1), get_date(4)),
      (get_date(2), get_date(4))
    ),
    (  # 0 #2x
      (get_date(1), get_date(4)),
      (get_date(3), get_date(4))
    ),
    (  # 5 #3
      (get_date(2), get_date(4)),
      (get_date(2), get_date(3))
    ),
    (  # 5 #2x
      (get_date(2), get_date(4)),
      (get_date(3), get_date(4))
    )
  ]


def test_one_overlap():
  fmt = "%Y-%m-%d %H:%M:%S"
  times = [
    ("2023-03-28 08:03:06", "2023-03-28 08:07:20"),
    ("2023-03-28 08:10:20", "2023-03-28 08:12:07"),
    ("2023-03-28 08:11:48", "2023-03-28 08:14:42"),
    ("2023-03-28 08:20:55", "2023-03-28 08:22:58"),
  ]
  times = [(datetime.strptime(x, fmt), datetime.strptime(y, fmt)) for x, y in times]

  result = list(get_overlaps(set(times)))
  print(result)
  assert result == [(
    (
        datetime(2023, 3, 28, 8, 10, 20),
        datetime(2023, 3, 28, 8, 12, 7)
    ),
    (
        datetime(2023, 3, 28, 8, 11, 48),
        datetime(2023, 3, 28, 8, 14, 42)
    )
  )]
