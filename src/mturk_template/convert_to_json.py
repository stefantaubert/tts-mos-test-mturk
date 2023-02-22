import re
from collections import OrderedDict
from typing import Dict
from typing import OrderedDict as ODType
from typing import Set

from pandas import DataFrame


def convert_to_json(results_df: DataFrame, ground_truth_df: DataFrame) -> ODType:
  results_df.sort_values(by=["WorkerId", "AssignmentId"], inplace=True)
  results_dict = results_df.to_dict("index")
  alg_dict = get_alg_dict_from_df(ground_truth_df)
  file_dict = get_file_dict_from_df(ground_truth_df)

  result = OrderedDict()
  result["algorithms"] = sorted(set(alg_dict.values()))
  result["files"] = sorted(set(file_dict.values()))

  workers_data = OrderedDict()
  result["workers"] = workers_data

  taken_assignment_ids: Set[str] = set()

  for row in results_dict.values():
    audios = parse_audio_files(row)
    mos = parse_mos_answers(row)
    device = parse_listening_type(row)
    worktime = int(row["WorkTimeInSeconds"])
    worker_id = row["WorkerId"]
    assignment_id = row["AssignmentId"]
    state = row["AssignmentStatus"]

    if worker_id not in workers_data:
      workers_data[worker_id] = OrderedDict()
    assignments = workers_data[worker_id]
    if assignment_id in taken_assignment_ids:
      raise ValueError(f"AssignmentId \"{assignment_id}\" exists multiple times!")
    taken_assignment_ids.add(assignment_id)
    assert assignment_id not in assignments
    assignment_data = OrderedDict()
    assignments[assignment_id] = assignment_data

    assignment_data["device"] = device
    assignment_data["state"] = state
    assignment_data["worktime"] = worktime

    ratings = []
    for sample_nr, audio_url in audios.items():
      audio_alg = alg_dict[audio_url]
      audio_file = file_dict[audio_url]
      assert sample_nr in mos
      mos_rating = mos[sample_nr]
      rating_data = OrderedDict((
        ("algorithm", audio_alg),
        ("file", audio_file),
        ("rating", mos_rating),
      ))
      ratings.append(rating_data)

    assignment_data["ratings"] = ratings
  return result


def get_file_dict_from_df(ground_truth_df: DataFrame) -> Dict[str, str]:
  ground_truth_dict = ground_truth_df.to_dict("index")
  file_dict: Dict[str, str] = {}

  for row in ground_truth_dict.values():
    audio_url = row["audio_url"]
    file_dict[audio_url] = row["file"]
  return file_dict


def get_alg_dict_from_df(ground_truth_df: DataFrame) -> Dict[str, str]:
  ground_truth_dict = ground_truth_df.to_dict("index")
  alg_dict: Dict[str, str] = {}

  for row in ground_truth_dict.values():
    audio_url = row["audio_url"]
    alg_dict[audio_url] = row["algorithm"]
  return alg_dict


def parse_audio_files(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Input\.audio_url_(\d+)")
  result = {}
  for identifier, val in row.items():
    mos_match = re.match(pattern, identifier)
    if isinstance(mos_match, re.Match):
      sample_nr = mos_match.group(1)
      sample_nr = int(sample_nr)
      assert sample_nr not in result
      result[sample_nr] = val
  return result


def parse_mos_answers(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Answer\.(\d+)-mos-rating\.([1-5])")
  result = {}
  for identifier, val in row.items():
    if val:
      mos_match = re.match(pattern, identifier)
      if isinstance(mos_match, re.Match):
        sample_nr, mos_val = mos_match.groups()
        sample_nr = int(sample_nr)
        assert sample_nr not in result
        result[sample_nr] = int(mos_val)
  return result


def parse_listening_type(row: Dict[str, str]) -> str:
  pattern = re.compile(r"Answer\.listening-type\.(.+)")
  result = None
  for identifier, val in row.items():
    if val:
      mos_match = re.match(pattern, identifier)
      if isinstance(mos_match, re.Match):
        lt = mos_match.group(1)
        assert result is None
        result = lt
  return result


# def get_n_urls_per_assignment(data: List[DataPoint]) -> int:
#   tmp = {}
#   for data_point in data:
#     if data_point.assignment_id not in tmp:
#       tmp[data_point.assignment_id] = 0
#     tmp[data_point.assignment_id] += 1
#   result = max(tmp.values())
#   return result
