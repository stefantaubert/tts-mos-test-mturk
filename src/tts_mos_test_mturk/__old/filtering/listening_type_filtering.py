import re
from logging import getLogger
from typing import Dict, Optional

import numpy as np

from tts_mos_test_mturk.globals import LISTENING_TYPES
from tts_mos_test_mturk.types import Evaluation


def ignore_non_headphones(ev: Evaluation, reject: bool, reject_reason: Optional[str], ignore_rejected: bool):
  logger = getLogger(__name__)
  listening_types = get_listening_types(ev)
  ass_mask = get_non_headphones_mask(listening_types)
  already_rej_mask = None
  ignored_ass = ass_mask.nonzero()[0]
  if reject:
    for ass_i in ignored_ass:
      ass = ev.assignments[ass_i]
      reason = reject_reason
      ev.rejection_history[ass] = reason
  old_count = ev.opinion_scores_count

  ev.remove_assignments_opinion_scores(ass_mask)

  new_count = ev.opinion_scores_count
  rem_count = old_count - new_count
  logger.info(
    f"Ignored {rem_count} / {old_count} opinion scores ({rem_count/ev.n_urls_per_hit:.0f} assignments), kept {new_count}!")


def get_listening_types(ev: Evaluation) -> np.ndarray:
  result = np.full(ev.n_assignments, fill_value=np.nan, dtype=np.float32)
  for row in ev.results_dict.values():
    ass_i = ev.assignments.get_loc(row["AssignmentId"])
    listening_type = parse_listening_type(row)
    lt_nr = LISTENING_TYPES.get(listening_type)
    result[ass_i] = lt_nr
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


def get_non_headphones_mask(listening_types: np.ndarray) -> np.ndarray:
  mask: np.ndarray = (listening_types == LISTENING_TYPES["desktop"]) | (
    listening_types == LISTENING_TYPES["laptop"])
  return mask
