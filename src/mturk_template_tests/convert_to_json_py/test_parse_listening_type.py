
from mturk_template.convert_to_json import parse_listening_type


def test_component_4_options():
  result = parse_listening_type({
    "Answer.listening-type.laptop": False,
    "Answer.listening-type.desktop": False,
    "Answer.listening-type.in-ear": False,
    "Answer.listening-type.over-the-ear": True,
  })
  assert result == "over-the-ear"


def test_component_2_options():
  result = parse_listening_type({
    "Answer.listening-type.headphones": False,
    "Answer.listening-type.no-headphones": True,
  })
  assert result == "no-headphones"


def test_one_entry__returns_entry():
  result = parse_listening_type({
    "Answer.listening-type.headphone1": True
  })
  assert result == "headphone1"


def test_two_entries__returns_true_one():
  result = parse_listening_type({
    "Answer.listening-type.headphone1": False,
    "Answer.listening-type.headphone2": True,
  })
  assert result == "headphone2"


def test_three_entries__returns_true_one():
  result = parse_listening_type({
    "Answer.listening-type.headphone1": False,
    "Answer.listening-type.headphone2": True,
    "Answer.listening-type.headphone3": False,
  })
  assert result == "headphone2"
