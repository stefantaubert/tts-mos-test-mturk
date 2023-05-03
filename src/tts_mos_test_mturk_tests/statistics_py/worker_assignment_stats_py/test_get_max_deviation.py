from tts_mos_test_mturk.statistics.worker_assignment_stats import get_max_deviation


def test_132_returns_2():
  result = get_max_deviation([1, 3, 2])
  assert result == 2


def test_192_returns_8():
  result = get_max_deviation([1, 9, 2])
  assert result == 8


def test_291_returns_8():
  result = get_max_deviation([2, 9, 1])
  assert result == 8
