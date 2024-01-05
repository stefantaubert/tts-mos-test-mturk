import datetime
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List
from typing import OrderedDict as ODType
from typing import Set, Tuple, Union, cast

from ordered_set import OrderedSet

from tts_mos_test_mturk.typing import AlgorithmName, AssignmentId, FileName, Ratings, WorkerName


@dataclass()
class RatingData:
  votes: Ratings = field(default_factory=OrderedDict)


@dataclass()
class Assignment:
  # TODO make optional
  device: str
  # TODO make optional
  state: str
  # TODO make optional
  hit_id: str
  # TODO make optional
  time: datetime.datetime
  # ratings: List[Rating] = field(default_factory=list)
  ratings: ODType[Tuple[AlgorithmName, FileName], RatingData] = field(default_factory=OrderedDict)


@dataclass()
class Worker():
  age_group: str
  gender: str
  assignments: ODType[AssignmentId, Assignment] = field(default_factory=OrderedDict)


@dataclass()
class Result():
  # TODO make optional
  algorithms: OrderedSet[AlgorithmName] = field(default_factory=OrderedSet)
  # TODO make optional
  files: OrderedSet[FileName] = field(default_factory=OrderedSet)
  workers: ODType[WorkerName, Worker] = field(default_factory=OrderedDict)


def parse_int_then_float(val: str) -> Union[int, float]:
  if val.isdigit():
    return int(val)
  return float(val)


def parse_time(val: str) -> datetime.datetime:
  # e.g., Mon Mar 06 08:59:21 PST 2023
  # Pacific Standard Time
  val = val.replace(" PST", "")
  # Pacific Daylight Time
  val = val.replace(" PDT", "")
  result = datetime.datetime.strptime(val, "%a %b %d %H:%M:%S %Y")
  return result

def parse_time2(val: str) -> datetime.datetime:
  result = datetime.datetime.strptime(val, "%d.%m.%y %H:%M:%S")
  return result


def parse_result_from_json(data: Dict) -> Result:
  res_data: ODType[str, Worker] = OrderedDict()
  files = OrderedSet(str(x) for x in data["files"])
  algorithms = OrderedSet(str(x) for x in data["algorithms"])
  result = Result(algorithms, files, res_data)
  assignment_ids: Set[str] = set()

  workers_data = cast(Dict[str, Dict[str, Dict[str, Any]]], data["workers"])
  for worker_id, worker_data in workers_data.items():
    age_group = str(worker_data["age_group"])
    gender = str(worker_data["gender"])
    worker = Worker(age_group, gender)
    assert worker_id not in res_data
    res_data[worker_id] = worker
    for assignment_id, assignment_data in worker_data["assignments"].items():
      if assignment_id in assignment_ids:
        raise ValueError(f"Assignment \"{assignment_id}\" exist multiple times!")
      assignment_ids.add(assignment_id)
      assignment = Assignment(
        device=str(assignment_data["device"]),
        hit_id=str(assignment_data["hit"]),
        state=str(assignment_data["state"]),
        time=parse_time2(str(assignment_data["time"])),
      )

      assert assignment_id not in worker.assignments
      worker.assignments[assignment_id] = assignment
      ratings = cast(List[Dict[str, Any]], assignment_data["ratings"])
      for rating_data in ratings:
        algorithm = str(rating_data["algorithm"])
        file = str(rating_data["file"])
        if algorithm not in algorithms:
          raise ValueError(
            f"Referenced algorithm \"{algorithm}\" was not defined in \"algorithms\"!")

        if file not in files:
          raise ValueError(f"Referenced file \"{algorithm}\" was not defined in \"files\"!")

        alg_file_comb = (algorithm, file)
        if alg_file_comb in assignment.ratings:
          raise ValueError("Rating for algorithm and file combination exist multiple times!")

        votes: Dict[str, Union[int, float]] = rating_data["votes"]
        votes_parsed = OrderedDict()
        for vote_name, vote in votes.items():
          assert isinstance(vote, (int, float))
          votes_parsed[vote_name] = vote

        rating_data = RatingData(
          votes=votes_parsed,
        )

        assignment.ratings[alg_file_comb] = rating_data

  return result
