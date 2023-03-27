import datetime
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List
from typing import OrderedDict as ODType
from typing import Set, Union, cast

from ordered_set import OrderedSet


@dataclass()
class Rating:
  algorithm: str
  file: str
  ratings: ODType[str, Union[int, float]] = field(default_factory=OrderedDict)


@dataclass()
class Assignment:
  # TODO make optional
  device: str
  # TODO make optional
  state: str
  # TODO make optional
  worktime: Union[int, float]
  # TODO make optional
  hit_id: str
  time: datetime.datetime
  ratings: List[Rating] = field(default_factory=list)


@dataclass()
class Worker():
  assignments: ODType[str, Assignment] = field(default_factory=OrderedDict)


@dataclass()
class Result():
  # TODO make optional
  algorithms: OrderedSet[str] = field(default_factory=OrderedSet)
  # TODO make optional
  files: OrderedSet[str] = field(default_factory=OrderedSet)
  workers: ODType[str, Worker] = field(default_factory=OrderedDict)


def parse_int_then_float(val: str) -> Union[int, float]:
  if val.isdigit():
    return int(val)
  return float(val)


def parse_time(val: str) -> datetime.datetime:
  # Pacific Standard Time
  val = val.replace(" PST", "")
  # Pacific Daylight Time
  val = val.replace(" PDT", "")
  result = datetime.datetime.strptime(val, "%a %b %d %H:%M:%S %Y")
  return result


def parse_result_from_json(data: Dict) -> Result:
  res_data: ODType[str, Worker] = OrderedDict()
  files = OrderedSet(str(x) for x in data["files"])
  algorithms = OrderedSet(str(x) for x in data["algorithms"])
  result = Result(algorithms, files, res_data)
  assignment_ids: Set[str] = set()
  worker_data = cast(Dict[str, Dict[str, Dict[str, Any]]], data["workers"])
  for worker_id, assignments in worker_data.items():
    assert worker_id not in res_data
    worker = Worker()
    res_data[worker_id] = worker
    for assignment_id, assignment_data in assignments.items():
      device = str(assignment_data["device"])
      state = str(assignment_data["state"])
      hit = str(assignment_data["hit"])
      time = parse_time(str(assignment_data["time"]))
      # Mon Mar 06 08:59:21 PST 2023
      worktime = parse_int_then_float(str(assignment_data["worktime"]))
      if assignment_id in assignment_ids:
        raise ValueError(f"Assignment \"{assignment_id}\" exist multiple times!")
      assignment_ids.add(assignment_id)
      assignment = Assignment(device, state, worktime, hit, time)
      assert assignment_id not in worker.assignments
      worker.assignments[assignment_id] = assignment
      parsed_alg_file_combinations = set()
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
        if alg_file_comb in parsed_alg_file_combinations:
          raise ValueError("Rating for algorithm and file combination exist multiple times!")
        rating_names = rating_data.keys() - {"algorithm", "file"}
        parsed_alg_file_combinations.add(alg_file_comb)
        if len(rating_names) == 0:
          raise ValueError("No rating for algorithm and file combination was given!")
        r = Rating(algorithm, file)
        assignment.ratings.append(r)
        for rating_name in rating_names:
          rating = parse_int_then_float(str(rating_data[rating_name]))
          r.ratings[rating_name] = rating
  return result
