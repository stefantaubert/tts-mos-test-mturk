from logging import getLogger

import numpy as np

from tts_mos_test_mturk.globals import STATE_TYPES
from tts_mos_test_mturk.types import Evaluation


def ignore_rejected(ev: Evaluation):

  logger = getLogger(__name__)
  statuses = get_statuses(ev)
  mask = get_rejected_assignments_mask(statuses, ev.Z_ass)
  old_count = ev.opinion_scores_count

  ev.remove_opinion_scores(mask)

  new_count = ev.opinion_scores_count
  rem_count = old_count - new_count
  logger.info(
    f"Ignored {rem_count} / {old_count} opinion scores ({rem_count/ev.n_urls_per_hit:.0f} assignments), kept {new_count}!")


def ignore_fast_hits(ev: Evaluation, min_speed: float):
  logger = getLogger(__name__)
  worktimes = get_worktimes(ev)
  mask = get_too_fast_hits_mask(worktimes, ev.Z_ass, min_speed)
  old_count = ev.opinion_scores_count

  ev.remove_opinion_scores(mask)

  new_count = ev.opinion_scores_count
  rem_count = old_count - new_count
  logger.info(
    f"Ignored {rem_count} / {old_count} opinion scores ({rem_count/ev.n_urls_per_hit:.0f} assignments), kept {new_count}!")



def get_too_fast_hits_mask(worktimes: np.ndarray, Z_ass: np.ndarray, min_speed: float) -> np.ndarray:
  mask: np.ndarray = worktimes < min_speed
  assignment_indices = mask.nonzero()[0]
  Z_mask = np.isin(Z_ass, assignment_indices)
  return Z_mask


def get_worktimes(ev: Evaluation) -> np.ndarray:
  worktimes = np.full(ev.n_assignments, fill_value=np.nan, dtype=np.float32)
  for row in ev.results_dict.values():
    ass_i = ev.assignments.get_loc(row["AssignmentId"])
    worktime = int(row["WorkTimeInSeconds"])
    worktimes[ass_i] = worktime
  return worktimes


def get_rejected_assignments_mask(statuses: np.ndarray, Z_ass: np.ndarray) -> np.ndarray:
  mask: np.ndarray = statuses == STATE_TYPES["Rejected"]
  assignment_indices = mask.nonzero()[0]
  Z_mask = np.isin(Z_ass, assignment_indices)
  return Z_mask


def get_statuses(ev: Evaluation) -> np.ndarray:
  result = np.full(ev.n_assignments, fill_value=np.nan, dtype=np.float32)
  for row in ev.results_dict.values():
    ass_i = ev.assignments.get_loc(row["AssignmentId"])
    status = STATE_TYPES[row["AssignmentStatus"]]
    result[ass_i] = status
  return result


