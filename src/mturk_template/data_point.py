# import re
# from dataclasses import dataclass
# from typing import Dict, Generator, List, Literal

# DEVICE_IN_EAR = "in-ear"
# DEVICE_ON_EAR = "over-the-ear"
# DEVICE_DESKTOP = "desktop"
# DEVICE_LAPTOP = "laptop"

# STATE_ACCEPTED = "Accepted"
# STATE_REJECTED = "Rejected"
# STATE_APPROVED = "Approved"


# @dataclass
# class DataPoint():
#   rating: Literal[1, 2, 3, 4, 5]
#   worker_id: str
#   assignment_id: str
#   algorithm: str
#   file: str
#   audio_url: str
#   listening_device: Literal["in-ear", "over-the-ear", "desktop", "laptop"]
#   state: Literal["Accepted", "Rejected", "Approved"]
#   worktime: int


# def parse_data_points(results_dict: Dict, alg_dict: Dict, file_dict: Dict) -> Generator[DataPoint, None, None]:
#   for row in results_dict.values():
#     audios = parse_audio_files(row)
#     mos = parse_mos_answers(row)
#     lt = parse_listening_type(row)
#     worktime = int(row["WorkTimeInSeconds"])

#     for sample_nr, audio_url in audios.items():
#       audio_alg = alg_dict[audio_url]
#       audio_file = file_dict[audio_url]
#       assert sample_nr in mos
#       mos_rating = mos[sample_nr]
#       data_point = DataPoint(
#         assignment_id=row["AssignmentId"],
#         worker_id=row["WorkerId"],
#         algorithm=audio_alg,
#         audio_url=audio_url,
#         file=audio_file,
#         listening_device=lt,
#         rating=mos_rating,
#         state=row["AssignmentStatus"],
#         worktime=worktime,
#       )
#       # raise ValueError(f"AssignmentId \"{assignment_id}\" exists multiple times!")
#       yield data_point


# def parse_audio_files(row: Dict[str, str]) -> Dict[str, int]:
#   pattern = re.compile(r"Input\.audio_url_(\d+)")
#   result = {}
#   for identifier, val in row.items():
#     mos_match = re.match(pattern, identifier)
#     if isinstance(mos_match, re.Match):
#       sample_nr = mos_match.group(1)
#       sample_nr = int(sample_nr)
#       assert sample_nr not in result
#       result[sample_nr] = val
#   return result


# def parse_mos_answers(row: Dict[str, str]) -> Dict[str, int]:
#   pattern = re.compile(r"Answer\.(\d+)-mos-rating\.([1-5])")
#   result = {}
#   for identifier, val in row.items():
#     if val:
#       mos_match = re.match(pattern, identifier)
#       if isinstance(mos_match, re.Match):
#         sample_nr, mos_val = mos_match.groups()
#         sample_nr = int(sample_nr)
#         assert sample_nr not in result
#         result[sample_nr] = int(mos_val)
#   return result


# def parse_listening_type(row: Dict[str, str]) -> str:
#   pattern = re.compile(r"Answer\.listening-type\.(.+)")
#   result = None
#   for identifier, val in row.items():
#     if val:
#       mos_match = re.match(pattern, identifier)
#       if isinstance(mos_match, re.Match):
#         lt = mos_match.group(1)
#         assert result is None
#         result = lt
#   return result


# def get_n_urls_per_assignment(data: List[DataPoint]) -> int:
#   tmp = {}
#   for data_point in data:
#     if data_point.assignment_id not in tmp:
#       tmp[data_point.assignment_id] = 0
#     tmp[data_point.assignment_id] += 1
#   result = max(tmp.values())
#   return result
