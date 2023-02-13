from pathlib import Path

import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.core.bad_worker_filtering import (calc_mos, generate_approve_csv,
                                                          ignore_bad_workers,
                                                          ignore_bad_workers_percent,
                                                          ignore_masked_count_opinion_scores,
                                                          ignore_outlier_opinion_scores,
                                                          ignore_too_fast_assignments,
                                                          mask_assignments_by_lt)
from tts_mos_test_mturk.core.evaluation_data import EvaluationData
from tts_mos_test_mturk.core.statistics.algorithm_sentence_stats import get_algorithm_sentence_stats
from tts_mos_test_mturk.core.statistics.algorithm_worker_stats import get_worker_algorithm_stats
from tts_mos_test_mturk.core.statistics.worker_assignment_stats import get_worker_assignment_stats
from tts_mos_test_mturk.core.stats import print_stats
from tts_mos_test_mturk_cli.logging_configuration import configure_root_logger

configure_root_logger()


def parse_v3():
  result_path = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv")
  ground_truth = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-gt.csv")

  result_csv = pd.read_csv(result_path)
  ground_truth = pd.read_csv(ground_truth)

  data = EvaluationData(result_csv, ground_truth)

  data.save(Path("/tmp/data-2.pkl"))
  data = EvaluationData.load(Path("/tmp/data-2.pkl"))

  mask_assignments_by_lt(data, OrderedSet(), {"desktop"}, "lt_bad")
  df = get_algorithm_sentence_stats(data, {"lt_bad"})
  print(df)
  df = get_worker_algorithm_stats(data, {"lt_bad"})
  print(df)
  df = get_worker_assignment_stats(data, masks)
  print(df)
  calc_mos(data, OrderedSet())
  ignore_bad_workers(data, OrderedSet(), 0.25, "bad_workers")
  ignore_bad_workers_percent(data, OrderedSet(("bad_workers",)), 0.0, 0.5, "bonus_50_1")
  print_stats(data, set(), {"bad_workers", "bonus_50_1"})
  ignore_bad_workers_percent(data, OrderedSet(
    ("bad_workers",)), 0.9, 1.1, "bonus_50_2")
  print_stats(data, set(), {"bad_workers", "bonus_50_1", "bonus_50_2"})
  ignore_bad_workers_percent(data, OrderedSet(
    ("bad_workers",)), 0.0, 0.9, "bonus_10")
  print_stats(data, set(), {"bad_workers", "bonus_10"})

  generate_approve_csv(data, OrderedSet(("too_fast_1",)), "good")

  ignore_too_fast_assignments(data, OrderedSet(), 30, "too_fast_1")

  ignore_outlier_opinion_scores(data, OrderedSet(), 1, "outliers")
  ignore_masked_count_opinion_scores(data, OrderedSet(), "outliers", 0.05, "outliers_0.1")
  calc_mos(data, OrderedSet(("outliers", "outliers_0.1")))

  ignore_bad_workers(data, OrderedSet(("too_fast_1",)), 0.25, "bad_workers")
  # ignore_bad_workers(data, OrderedSet(( "bad_workers")), 0.25, "bad_workers_2")
  ignore_too_fast_assignments(data, OrderedSet(("bad_workers",)), 30, "too_fast")
  ignore_too_fast_assignments(data, OrderedSet(("bad_workers", "too_fast")), 30, "too_fast_2")


parse_v3()


# parse_gen()
# parse_gen_v2()
