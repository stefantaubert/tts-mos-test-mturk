
from tts_mos_test_mturk.masking.worker_correlation_mask import get_range_start_end_percent


def test_4_0_25__returns_0_0():
  start, end = get_range_start_end_percent(4, 0, 0.25)
  assert start == 0
  assert end == 1


def test_4_0_50__returns_0_1():
  start, end = get_range_start_end_percent(4, 0, 0.5)
  assert start == 0
  assert end == 1


def test_4_0_51__returns_0_2():
  start, end = get_range_start_end_percent(4, 0, 0.51)
  assert start == 0
  assert end == 2


def test_4_0_75__returns_0_2():
  start, end = get_range_start_end_percent(4, 0, 0.75)
  assert start == 0
  assert end == 2


def test_4_0_76__returns_0_3():
  start, end = get_range_start_end_percent(4, 0, 0.76)
  assert start == 0
  assert end == 3


def test_4_0_100__returns_0_4():
  start, end = get_range_start_end_percent(4, 0, 1.0)
  assert start == 0
  assert end == 4
