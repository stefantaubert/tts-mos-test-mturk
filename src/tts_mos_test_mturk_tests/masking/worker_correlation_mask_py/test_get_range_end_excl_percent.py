from pytest import raises

from tts_mos_test_mturk.masking.worker_correlation_mask import get_range_end_excl_percent


def test_2_0__raises_error():
  with raises(ValueError) as error:
    get_range_end_excl_percent(2, 0)


def test_2_1__returns_0():
  end = get_range_end_excl_percent(2, 0.01)
  assert end == 1


def test_2_50__returns_0():
  end = get_range_end_excl_percent(2, 0.50)
  assert end == 1


def test_2_75__returns_1():
  end = get_range_end_excl_percent(2, 0.75)
  assert end == 1


def test_2_76__returns_1():
  end = get_range_end_excl_percent(2, 0.76)
  assert end == 1


def test_2_100__returns_1():
  end = get_range_end_excl_percent(2, 1.00)
  assert end == 2


def test_4_0__raise_error():
  with raises(ValueError) as error:
    get_range_end_excl_percent(4, 0)


def test_4_1__returns_1():
  end = get_range_end_excl_percent(4, 0.01)
  assert end == 1


def test_4_25__returns_1():
  end = get_range_end_excl_percent(4, 0.25)
  assert end == 1


def test_4_26__returns_2():
  end = get_range_end_excl_percent(4, 0.26)
  assert end == 1


def test_4_50__returns_2():
  end = get_range_end_excl_percent(4, 0.5)
  assert end == 1


def test_4_51__returns_3():
  end = get_range_end_excl_percent(4, 0.51)
  assert end == 2


def test_4_75__returns_3():
  end = get_range_end_excl_percent(4, 0.75)
  assert end == 2


def test_4_76__returns_3():
  end = get_range_end_excl_percent(4, 0.76)
  assert end == 3


def test_4_100__returns_4():
  end = get_range_end_excl_percent(4, 1.0)
  assert end == 4
