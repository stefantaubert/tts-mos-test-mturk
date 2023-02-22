from pytest import raises

from tts_mos_test_mturk.masking.worker_correlation_mask import get_range_start_incl_percent


def test_2_0__returns_0():
  start = get_range_start_incl_percent(2, 0)
  assert start == 0


def test_2_50__returns_0():
  start = get_range_start_incl_percent(2, 0.50)
  assert start == 0


def test_2_75__returns_1():
  start = get_range_start_incl_percent(2, 0.75)
  assert start == 1


def test_2_100__raises_error():
  with raises(ValueError) as error:
    get_range_start_incl_percent(2, 1.00)


def test_4_0__returns_0():
  start = get_range_start_incl_percent(4, 0)
  assert start == 0


def test_4_25__returns_0():
  start = get_range_start_incl_percent(4, 0.25)
  assert start == 0


def test_4_50__returns_1():
  start = get_range_start_incl_percent(4, 0.5)
  assert start == 1


def test_4_75__returns_2():
  start = get_range_start_incl_percent(4, 0.75)
  assert start == 2


def test_4_76__returns_3():
  start = get_range_start_incl_percent(4, 0.76)
  assert start == 3


def test_4_100__raises_error():
  with raises(ValueError) as error:
    get_range_start_incl_percent(4, 1.0)


def test_14_20__returns_2():
  start = get_range_start_incl_percent(14, 0.20)
  assert start == 2


def test_10_10__returns_0():
  start = get_range_start_incl_percent(10, 0.10)
  assert start == 0
