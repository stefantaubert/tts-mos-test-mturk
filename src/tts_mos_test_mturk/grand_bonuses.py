
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from mypy_boto3_mturk import MTurkClient
from ordered_set import OrderedSet


def generate_bonus_csv(workers: Set[str], assignments: Dict[str, Set[str]], bonus: float, reason: str) -> Optional[pd.DataFrame]:
  results: List[Dict[str, Any]] = []
  for worker in sorted(workers):
    for assignment in sorted(assignments[worker]):
      line = OrderedDict((
        ("WorkerId", worker),
        ("AssignmentId", assignment),
        ("BonusAmount", bonus),
        ("Reason", reason),
      ))
      results.append(line)
  if len(results) == 0:
    return None
  result = pd.DataFrame(
    data=[x.values() for x in results],
    columns=results[0].keys(),
  )
  return result


def generate_approve_csv(workers: Set[str], assignments: Dict[str, Set[str]]) -> Optional[pd.DataFrame]:
  results: List[Dict[str, Any]] = []
  for worker in sorted(workers):
    for assignment in sorted(assignments[worker]):
      line = OrderedDict((
        ("WorkerId", worker),
        ("AssignmentId", assignment),
        ("Approve", "x"),
        ("Reject", ""),
      ))
      results.append(line)
  if len(results) == 0:
    return None
  result = pd.DataFrame(
    data=[x.values() for x in results],
    columns=results[0].keys(),
  )
  return result


def generate_reject_csv(workers: Set[str], assignments: Dict[str, Set[str]], reason: str) -> Optional[pd.DataFrame]:
  results: List[Dict[str, Any]] = []
  for worker in sorted(workers):
    for assignment in sorted(assignments[worker]):
      line = OrderedDict((
        ("WorkerId", worker),
        ("AssignmentId", assignment),
        ("Approve", ""),
        ("Reject", reason),
      ))
      results.append(line)
  if len(results) == 0:
    return None
  result = pd.DataFrame(
    data=[x.values() for x in results],
    columns=results[0].keys(),
  )
  return result


def grant_bonuses(workers: OrderedSet[str], bonus: float, reason: str, mturk: MTurkClient):
  for worker in workers:
    mturk.send_bonus(
      WorkerId=worker,
      BonusAmount=bonus,
      Reason=reason,
    )


def approve_workers(workers: OrderedSet[str], mturk: MTurkClient):
  mturk
