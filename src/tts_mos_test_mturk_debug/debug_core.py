from pathlib import Path

import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import analyze, analyze_v2, compute_bonuses
from tts_mos_test_mturk.api_parser import get_mturk_sandbox
from tts_mos_test_mturk.core.bad_worker_filtering import (calc_mos, ignore_bad_workers,
                                                          ignore_masked_count_opinion_scores,
                                                          ignore_outlier_opinion_scores,
                                                          ignore_too_fast_assignments)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.csv_parser import parse_df
from tts_mos_test_mturk.filtering.listening_type_filtering import ignore_non_headphones
from tts_mos_test_mturk.filtering_old import ignore_fast_hits, ignore_rejected
from tts_mos_test_mturk.grand_bonuses import (accept_reject, generate_approve_csv,
                                              generate_bonus_csv, generate_reject_csv,
                                              grant_bonuses)
from tts_mos_test_mturk_cli.logging_configuration import configure_root_logger

configure_root_logger()


def parse_v3():
  result_path = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv")
  ground_truth = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-gt.csv")

  result_csv = pd.read_csv(result_path)
  ground_truth = pd.read_csv(ground_truth)

  data = EvaluationData(result_csv, ground_truth, "base")

  calc_mos(data, OrderedSet(("base",)))
  ignore_outlier_opinion_scores(data, OrderedSet(("base",)), 1, "outliers")
  ignore_masked_count_opinion_scores(data, OrderedSet(("base",)), "outliers", 0.05, "outliers_0.1")
  calc_mos(data, OrderedSet(("base", "outliers", "outliers_0.1")))

  ignore_too_fast_assignments(data, OrderedSet(("base",)), 30, "too_fast_1")
  ignore_bad_workers(data, OrderedSet(("base", "too_fast_1")), 0.25, "bad_workers")
  # ignore_bad_workers(data, OrderedSet(("base", "bad_workers")), 0.25, "bad_workers_2")
  ignore_too_fast_assignments(data, OrderedSet(("base", "bad_workers")), 30, "too_fast")
  ignore_too_fast_assignments(data, OrderedSet(
    ("base", "bad_workers", "too_fast")), 30, "too_fast_2")


parse_v3()


# parse_gen()
# parse_gen_v2()
