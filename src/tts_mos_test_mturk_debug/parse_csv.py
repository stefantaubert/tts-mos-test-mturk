from pathlib import Path

import pandas as pd
from ordered_set import OrderedSet

from tts_mos_test_mturk.csv_parser import parse_df
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

  parse_df(
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


parse_gen()