import re
from pathlib import Path
from typing import Any, Dict, List, OrderedDict

import pandas as pd

from tts_mos_test_mturk.types import Evaluation
from tts_mos_test_mturk_cli.logging_configuration import configure_root_logger

configure_root_logger()

result_path = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-output.csv")
ground_truth = Path("/home/mi/code/tts-mos-test-mturk/examples/gen-gt.csv")

result_csv = pd.read_csv(result_path)
ground_truth = pd.read_csv(ground_truth)

ev = Evaluation(result_csv, ground_truth)


ground_truth_dict = ground_truth.to_dict("index")
results_dict = result_csv.to_dict("index")


def parse_audio_files(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Input\.audio_url_(\d+)")
  result = {}
  for identifier, val in row.items():
    mos_match = re.match(pattern, identifier)
    if isinstance(mos_match, re.Match):
      sample_nr = mos_match.group(1)
      sample_nr = int(sample_nr)
      assert sample_nr not in result
      result[sample_nr] = val
  return result


def parse_mos_answers(row: Dict[str, str]) -> Dict[str, int]:
  pattern = re.compile(r"Answer\.(\d+)-mos-rating\.([1-5])")
  result = {}
  for identifier, val in row.items():
    if val:
      mos_match = re.match(pattern, identifier)
      if isinstance(mos_match, re.Match):
        sample_nr, mos_val = mos_match.groups()
        sample_nr = int(sample_nr)
        assert sample_nr not in result
        result[sample_nr] = int(mos_val)
  return result


alg_dict: Dict[str, str] = {}
file_dict: Dict[str, str] = {}

for row in ground_truth_dict.values():
  audio_url = row["audio_url"]
  alg_dict[audio_url] = row["algorithm"]
  file_dict[audio_url] = row["file"]


n_urls_per_hit = 0
out_assignments = []
out_worker_ids = []
out_moses = []
out_sample = []
out_alg = []
out_dur = []
out_lt = []
out_state = []

data: List[Dict[str, Any]] = []


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


for row in results_dict.values():
  row["AssignmentId"]
  row['WorkerId']
  audios = parse_audio_files(row)
  mos = parse_mos_answers(row)
  lt = parse_listening_type(row)
  worktime = int(row["WorkTimeInSeconds"])
  n_urls_per_hit = max(n_urls_per_hit, len(audios))

  for sample_nr, audio_url in audios.items():
    audio_alg = alg_dict[audio_url]
    audio_file = file_dict[audio_url]
    assert sample_nr in mos
    mos_rating = mos[sample_nr]
    line = OrderedDict((
      ("AssignmentId", row["AssignmentId"]),
      ("AssignmentStatus", row["AssignmentStatus"]),
      ("WorkerId", row["WorkerId"]),
      ("Algorithm", audio_alg),
      ("File", audio_file),
      ("MOS", mos_rating),
      ("ListeningType", lt),
      ("WorkTimeInSeconds", worktime),
    ))
    data.append(line)

result_df = pd.DataFrame(
  data=[x.values() for x in data],
  columns=data[0].keys()
)

result_df.to_csv(
  Path("/home/mi/code/tts-mos-test-mturk/examples/gen-results.csv"),
  index=True,
  index_label="Index",
)
result_df = pd.read_csv(
  Path("/home/mi/code/tts-mos-test-mturk/examples/gen-results.csv"),
)

output_folder = Path("/home/mi/code/tts-mos-test-mturk/examples/out")
output_folder.mkdir(parents=True, exist_ok=True)
for col in result_df.columns:
  vals = result_df[col].values.astype(str)
  lines = "\n".join(vals)
  path = output_folder / f"{col}.txt"
  path.write_text(lines)
