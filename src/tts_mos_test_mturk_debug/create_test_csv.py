
import random
from typing import Dict, List

from ordered_set import OrderedSet
from pandas import DataFrame

N_ASSIGNMENTS = 9
N_AUDIOS = 120
N_ALG = 4
N_AUDIOS_PER_HIT = 8
N_SPEAKERS = 1
N_WORKERS = 20
SEED = 1111
MIN_WORK_TIME = 35
TOO_FAST_RATE = 0.07
REJECT_RATE = 0.07
WRONG_DEVICE_RATE = 0.07

random.seed(SEED)

workers = [f"worker{x}" for x in range(N_WORKERS)]

alg_audios = []
for i_speaker in range(N_SPEAKERS):
  speaker_name = f"speaker{i_speaker}"
  for i_alg in range(N_ALG):
    alg_name = f"alg{i_alg}"
    for i_audio in range(N_AUDIOS):
      audio_name = f"audio{i_audio}.wav"
      file_path = f"https://tuc.cloud/index.php/s/Fn5FzWsQwAeqRG4/download?path=/{alg_name}/{speaker_name}/{audio_name}"
      alg_audios.append(file_path)

random.shuffle(alg_audios)

n_unique_audios = N_SPEAKERS * N_AUDIOS * N_ALG
assert n_unique_audios == len(alg_audios)
n_hits = int(n_unique_audios / N_AUDIOS_PER_HIT)

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

assignment_counter = 0
for i_hit in range(n_hits):
  hit_worker = random.choice(workers)
  hit_status = random.choices(list(assignment_statuses.keys()),
                              assignment_statuses.values(), k=1)[0]
  work_times = [
    random.randint(MIN_WORK_TIME, MIN_WORK_TIME + 10),
    random.randint(0, MIN_WORK_TIME),
  ]
  work_time_in_s = random.choices(work_times, [1 - TOO_FAST_RATE, TOO_FAST_RATE], k=1)[0]

  output_row = {
    "HITId": f"hit{i_hit}",
    "AssignmentId": f"assignment{assignment_counter}",
    "WorkerId": hit_worker,
    "AssignmentStatus": hit_status,
    "WorkTimeInSeconds": work_time_in_s,
  }
  input_row = {}

  for sample_nr in range(N_AUDIOS_PER_HIT):
    sample_audio = alg_audios.pop()
    output_row[f"Input.audio_url_{sample_nr}"] = sample_audio
    input_row[f"audio_url_{sample_nr}"] = sample_audio

  input_rows.append(input_row)

  for sample_nr in range(N_AUDIOS_PER_HIT):
    selected_mos = random.randint(1, 5)
    for mos_nr in range(1, 6):
      mos_val = mos_nr == selected_mos
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

print(input_df)
input_df.to_csv("/home/mi/code/tts-mos-test-mturk/examples/gen-input.csv", index=False)

output_df = DataFrame(
  data=[row.values() for row in output_rows],
  columns=output_rows[0].keys(),
)
print(output_df)
output_df.to_csv("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv", index=False)
