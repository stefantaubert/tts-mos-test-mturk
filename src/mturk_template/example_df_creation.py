import random
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from tts_mos_test_mturk.logging import get_logger

N_ASSIGNMENTS_PER_HIT = 9
N_AUDIOS = 120
N_AUDIOS_PER_HIT = 8
N_SPEAKERS = 1
N_WORKERS = 20
MIN_WORK_TIME = 35
TOO_FAST_RATE = 0.07
REJECT_RATE = 0.07
WRONG_DEVICE_RATE = 0.07
# WRONG_ANNOTATION_RATE = 0.07

N_ALG = 4
ALG_WEIGHTS = {
  # MOS: 1, 2, 3, 4, 5
  0: [0, 1, 1, 2, 2],
  1: [1, 1, 2, 2, 1],
  2: [1, 2, 2, 1, 0],
  3: [2, 2, 1, 1, 0],
}

# ALG_WEIGHTS = {
#   # MOS: 1, 2, 3, 4, 5
#   0: [1, 1, 1, 1, 1],
#   1: [1, 1, 1, 1, 1],
#   2: [1, 1, 1, 1, 1],
#   3: [1, 1, 1, 1, 1],
# }


def generate_example_data(seed: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  logger = get_logger()

  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)

  workers = [f"WORKER{str(x).zfill(len(str(N_WORKERS - 1)))}" for x in range(N_WORKERS)]

  random_nums = np.random.normal(loc=0, scale=0.5, size=N_WORKERS)
  random_nums += abs(np.min(random_nums.flatten()))
  random_nums /= abs(np.max(random_nums.flatten()))

  # worker_bad = {
  #   w: random.choices([True, False], weights=[
  #                     1 - WRONG_ANNOTATION_RATE, WRONG_ANNOTATION_RATE], k=1)[0]
  #   for w in workers
  # }

  worker_performance = dict(zip(workers, [1] * len(workers)))
  worker_performance = dict(zip(workers, reversed(sorted(random_nums))))

  output_rows: List[Dict[str, str]] = []
  input_rows: List[Dict[str, str]] = []
  assignment_statuses = {
    "Rejected": REJECT_RATE,
    "Approved": 1 - REJECT_RATE,
  }

  listening_types = {
    "over-the-ear": (1 - WRONG_DEVICE_RATE) / 3 * 2,
    "in-ear": (1 - WRONG_DEVICE_RATE) / 3 * 1,
    "desktop": WRONG_DEVICE_RATE / 3 * 2,
    "laptop": WRONG_DEVICE_RATE / 3 * 1,
  }

  audio_urls = []

  mos_ground_truth = {}

  gt_csv_data = []
  for i_speaker in range(N_SPEAKERS):
    speaker_name = f"speaker{i_speaker}"
    for i_alg in range(N_ALG):
      alg_name = f"alg{i_alg}"
      alg_weights = ALG_WEIGHTS[i_alg]
      for i_audio in range(N_AUDIOS):
        i = i_audio + N_AUDIOS * (i_alg + N_ALG * (i_speaker))
        file_path = f"https://example.com/download?path=/{i}.wav"
        audio_urls.append(file_path)
        mos = random.choices(range(1, 6), weights=alg_weights, k=1)[0]
        mos_ground_truth[file_path] = mos
        gt_csv_data.append((file_path, alg_name, f"audio{i_audio}"))

  gt_df = DataFrame(
    data=gt_csv_data,
    columns=["audio_url", "algorithm", "file"],
  )

  random.shuffle(audio_urls)

  n_unique_audios = N_SPEAKERS * N_AUDIOS * N_ALG
  assert n_unique_audios == len(audio_urls)
  n_hits = int(n_unique_audios / N_AUDIOS_PER_HIT)

  hits_workers: Dict[int, Set[str]] = {i: set() for i in range(n_hits)}

  assignment_counter = 0
  for i_assignment in range(N_ASSIGNMENTS_PER_HIT):
    assignment_audio_urls = audio_urls.copy()

    for i_hit in range(n_hits):
      open_workers = sorted(set(workers).difference(hits_workers[i_hit]))
      assert len(open_workers) > 0
      hit_worker = random.choice(open_workers)
      hit_worker_performance = worker_performance[hit_worker]
      hits_workers[i_hit].add(hit_worker)
      hit_status = random.choices(list(assignment_statuses.keys()),
                                  assignment_statuses.values(), k=1)[0]
      worktimes = [
        random.randint(MIN_WORK_TIME, MIN_WORK_TIME + 10),
        random.randint(0, MIN_WORK_TIME),
      ]
      worktime_in_s = random.choices(worktimes, [1 - TOO_FAST_RATE, TOO_FAST_RATE], k=1)[0]

      output_row = {
        "HITId": f"HIT{i_hit}",
        "AssignmentId": f"ASSIGNMENT{assignment_counter}",
        "WorkerId": hit_worker,
        "AssignmentStatus": hit_status,
        "WorkTimeInSeconds": worktime_in_s,
      }
      input_row = {}

      alg_weights = []
      for sample_nr in range(N_AUDIOS_PER_HIT):
        sample_audio_url = assignment_audio_urls.pop()
        output_row[f"Input.audio_url_{sample_nr}"] = sample_audio_url
        input_row[f"audio_url_{sample_nr}"] = sample_audio_url
        gt = gt_df[gt_df["audio_url"] == sample_audio_url]
        alg = list(gt["algorithm"])[0]
        alg_nr_reg = re.match(r"alg(\d+)", alg)
        alg_nr = int(alg_nr_reg.group(1))
        # alg_rate = ALG_WEIGHTS[alg_nr]
        alg_rate = [
          hit_worker_performance if mos == mos_ground_truth[sample_audio_url] else 0.25
          for mos in range(1, 6)
        ]
        alg_weights.append(alg_rate)

      input_rows.append(input_row)

      for sample_nr in range(N_AUDIOS_PER_HIT):
        sample_alg_weights = alg_weights[sample_nr]
        mos = random.choices(range(1, 6), weights=sample_alg_weights, k=1)[0]
        for mos_nr in range(1, 6):
          mos_val = mos_nr == mos
          output_row[f"Answer.{sample_nr}-mos-rating.{mos_nr}"] = str(mos_val).lower()

      output_row["Answer.comments"] = ""
      selected_lt = random.choices(list(listening_types.keys()),
                                   listening_types.values(), k=1)[0]
      for lt in listening_types.keys():
        lt_val = selected_lt == lt
        output_row[f"Answer.listening-type.{lt}"] = str(lt_val).lower()
      output_rows.append(output_row)
      assignment_counter += 1

  input_df = DataFrame(
    data=[row.values() for row in input_rows],
    columns=input_rows[0].keys(),
  )

  # gt_df.to_csv("/home/mi/code/tts-mos-test-mturk/examples/gen-gt.csv", index=False)

  # print(input_df)
  # input_df.to_csv("/home/mi/code/tts-mos-test-mturk/examples/gen-input.csv", index=False)

  output_df = DataFrame(
    data=[row.values() for row in output_rows],
    columns=output_rows[0].keys(),
  )
  # print(output_df)
  # output_df.to_csv("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv", index=False)

  alg_moses = {
    alg_i: [] for alg_i in range(N_ALG)
  }

  for audio_url in audio_urls:
    gt = gt_df[gt_df["audio_url"] == audio_url]
    alg = list(gt["algorithm"])[0]
    alg_nr_reg = re.match(r"alg(\d+)", alg)
    alg_nr = int(alg_nr_reg.group(1))
    alg_moses[alg_nr].append(mos_ground_truth[audio_url])

  print("MOS ground truth")
  for a, moses in alg_moses.items():
    ground_truth_mos = np.mean(moses)
    print(a, ground_truth_mos)

  return gt_df, input_df, output_df
