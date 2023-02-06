
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


def generate_approve_csv(workers: Set[str], assignments: Dict[str, Set[str]], reason: Optional[str]) -> Optional[pd.DataFrame]:
  results: List[Dict[str, Any]] = []
  if reason is None:
    reason = "x"
  for worker in sorted(workers):
    for assignment in sorted(assignments[worker]):
      line = OrderedDict((
        ("WorkerId", worker),
        ("AssignmentId", assignment),
        ("Approve", reason),
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


def grant_bonuses(df: pd.DataFrame, mturk: MTurkClient):
  for i, row in df.iterrows():
    mturk.send_bonus(
      WorkerId=row["WorkerId"],
      BonusAmount=row["BonusAmount"],
      Reason=row["Reason"],
      AssignmentId=row["AssignmentId"],
    )


def accept_reject(df: pd.DataFrame, mturk: MTurkClient):
  for i, row in df.iterrows():
    app_str = row["Approve"]
    rej_str = row["Reject"]
    if app_str != "":
      mturk.approve_assignment(
        AssignmentId=row["AssignmentId"],
        RequesterFeedback=app_str,
      )
    else:
      assert rej_str != ""
      if rej_str == "x":
        rej_str = None
      mturk.reject_assignment(
        AssignmentId=row["AssignmentId"],
        RequesterFeedback=rej_str,
      )
