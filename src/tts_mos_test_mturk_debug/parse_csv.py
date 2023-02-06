from pathlib import Path

import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.analyze_assignmens import analyze, compute_bonuses
from tts_mos_test_mturk.api_parser import get_mturk_sandbox
from tts_mos_test_mturk.csv_parser import parse_df
from tts_mos_test_mturk.grand_bonuses import (accept_reject, generate_approve_csv,
                                              generate_bonus_csv, generate_reject_csv,
                                              grant_bonuses)
from tts_mos_test_mturk_cli.logging_configuration import configure_root_logger

configure_root_logger()


def parse_real():

  result_path = Path("/home/mi/code/tts-mos-test-mturk/examples/Batch_374625_batch_results.csv")
  input_csv = Path("/home/mi/code/tts-mos-test-mturk/examples/sel-audios-url-ljs.csv")

  result_csv = pd.read_csv(result_path)
  input_csv = pd.read_csv(input_csv)

  parse_df(
    input_csv,
    result_csv,
    consider_lt={"in-ear", "over-the-ear"},
    paths=OrderedSet((
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/orig/Linda Johnson;2;eng;North American/",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/wg-synthesized/Linda Johnson;2;eng;North American/",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/synthesized/Linda Johnson;2;eng;North American/",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/synthesized-dur/Linda Johnson;2;eng;North American/",
    )),
    min_worktime_s=35,
  )


def parse_gen():
  result_path = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv")
  input_csv = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-input.csv")

  result_csv = pd.read_csv(result_path)
  input_csv = pd.read_csv(input_csv)

  Z_all, work_times_all, listening_types_all, workers, all_audio_paths, worker_assignments = parse_df(
    input_csv,
    result_csv,
    consider_lt={"in-ear", "over-the-ear"},
    paths=OrderedSet((
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg0",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg1",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg2",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg3",
    )),
    min_worktime_s=8 * 4 + 2,
  )

  scores, ignored_workers = analyze(Z_all, work_times_all, listening_types_all, workers, all_audio_paths, paths=OrderedSet((
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg0",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg1",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg2",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg3",
    )),
    fast_worker_threshold=8 * 4,
    bad_worker_threshold=0.25,
    lt={"in-ear", "over-the-ear"},
    bad_worker_threshold_2=0.3,  # 0.7,
  )
  print(scores)
  Path("examples/reject-workers.txt").write_text("\n".join(sorted(ignored_workers)))
  fast_workers, bad_workers, no_bonus_workers, remaining_workers, top_50_workers, top_10_workers = compute_bonuses(Z_all, workers, all_audio_paths, paths=OrderedSet((
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg0",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg1",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg2",
      "https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/alg3",
    )),
    min_count_ass=20,
  )
  Path("examples/tier1-workers.txt").write_text("\n".join(sorted(no_bonus_workers)))
  Path("examples/tier2-workers.txt").write_text("\n".join(sorted(remaining_workers)))
  Path("examples/tier3-workers.txt").write_text("\n".join(sorted(top_50_workers)))
  Path("examples/tier4-workers.txt").write_text("\n".join(sorted(top_10_workers)))

  aws_access_key_id = "AKIAXZSPCXFHHW76X3FZ"
  aws_secret_access_key = "tVtCPeYp+O1+5fixLZWBTqKryS/eIZG2SRMmypCV"

  approve_workers = no_bonus_workers | remaining_workers | top_50_workers | top_10_workers
  # TODO only tests
  bad_workers |= no_bonus_workers
  approve_workers -= no_bonus_workers
  df1 = generate_approve_csv(approve_workers, worker_assignments,
                             "thank you for participating in our study")
  df2 = generate_reject_csv(bad_workers, worker_assignments,
                            "assignment is significantly inaccurate")
  df3 = generate_reject_csv(fast_workers, worker_assignments,
                            "assignment was submitted too quickly to be accurate")

  bonuses_df = pd.concat([df1, df2, df3])
  bonuses_df.to_csv(Path("examples/assignments.csv"), index=False)

  df1 = generate_bonus_csv(remaining_workers, worker_assignments,
                           "0.10", f"At least #{20} HITs completed")

  df2 = generate_bonus_csv(top_50_workers, worker_assignments,
                           "0.25", f"At least #{20} HITs completed; set in the top 50%")

  df3 = generate_bonus_csv(top_10_workers, worker_assignments,
                           "0.50", f"At least #{20} HITs completed; set in the top 10%")

  bonuses_df = pd.concat([df1, df2, df3])
  bonuses_df.to_csv(Path("examples/bonuses.csv"), index=False)

  mturk = get_mturk_sandbox(aws_access_key_id, aws_secret_access_key)
  assignments_df = pd.read_csv(Path("examples/assignments.csv"))
  try:
    accept_reject(assignments_df, mturk)
  except:
    pass
  bonuses_df = pd.read_csv(Path("examples/bonuses.csv"))
  try:
    grant_bonuses(bonuses_df, mturk)
  except:
    pass
  # for worker_id in remaining_workers:
  #   for assignment_id in worker_accepted_assignments[worker_id]:
  #     mturk.send_bonus(
  #       WorkerId=worker_id,
  #       BonusAmount="0.10",  # $
  #       Reason=f"At least #{20} HITs completed",
  #       AssignmentId=assignment_id,
  #     )


parse_gen()
