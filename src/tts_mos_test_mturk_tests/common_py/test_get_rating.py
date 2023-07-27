
from tts_mos_test_mturk.common import get_mean_rating


def test_get_first_rating():
  d = {
    "a": 5,
    "b": 6,
  }
  result = get_mean_rating(d, {"a"})
  assert result == 5


def test_get_last_rating():
  d = {
    "a": 5,
    "b": 6,
  }
  result = get_mean_rating(d, {"b"})
  assert result == 6


def test_mean_rating():
  d = {
    "a": 5,
    "b": 6,
  }
  result = get_mean_rating(d, {"a", "b"})
  assert result == 5.5
