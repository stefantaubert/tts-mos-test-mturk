# tts-mos-test-mturk

[![PyPI](https://img.shields.io/pypi/v/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![PyPI](https://img.shields.io/pypi/pyversions/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![MIT](https://img.shields.io/github/license/stefantaubert/tts-mos-test-mturk.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk/#files)
![PyPI](https://img.shields.io/pypi/implementation/tts-mos-test-mturk.svg)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/tts-mos-test-mturk/latest/master.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/compare/v0.0.1...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7669641.svg)](https://doi.org/10.5281/zenodo.7669641)

Command-line interface (CLI) and Python library to evaluate text-to-speech (TTS) mean opinion score (MOS) studies done on Amazon Mechanical Turk (MTurk).
The calculation of the confidence intervals is done in the same manner as described in (Ribeiro et al., 2011).

## Features

The tool is split into two command line interfaces named `mturk-template-cli` and `mos-cli` to be able to evaluate studies performed with a template other than the one proposed.

- `mturk-template-cli`: CLI to prepare MOS evaluation with results from MTurk.
  - `prepare-evaluation`: create evaluation .json-file
  - `gen-example-input`: generate example input data
- `mos-cli`: CLI to evaluate text-to-speech MOS studies done on MTurk.
  - `init`: initialize project from .json-file
  - `masks`
    - `mask-workers-by-assignments-count`: mask workers by their count of assignments
    - `mask-workers-by-masked-ratings-count`: mask workers by their count of masked ratings
    - `mask-workers-by-correlation`: mask workers by their algorithm/sentence correlation
    - `mask-workers-by-correlation-percent`: mask workers by their algorithm/sentence correlation (percentage-wise)
    - `mask-assignments-by-device`: mask assignments by their listening device
    - `mask-assignments-by-worktime`: mask assignments by their worktime
    - `mask-rating-outliers`: mask outlying ratings
  - `stats`
    - `print-mos`: print MOS and CI95
    - `print-masking-stats`: print masking statistics
    - `print-worker-stats`: print worker statistics for each algorithm
    - `print-assignment-stats`: print assignment statistics for each worker
    - `print-sentence-stats`: print sentence statistics for each algorithm
    - `print-data`: print all data points
  - `mturk`
    - `prepare-approval`: generate approval CSV-file
    - `approve`: approve assignments from CSV-file
    - `prepare-rejection`: generate rejection CSV-file
    - `reject`: reject assignments from CSV-file
    - `prepare-bonus-payment`: generate bonus payment CSV-file
    - `pay-bonus`: pay bonus to assignments from CSV-file

## Installation

```sh
pip install tts-mos-test-mturk --user
```

## Usage as CLI

### mturk-template-cli

```txt
usage: mturk-template-cli [-h] [-v] {prepare-evaluation,gen-example-input} ...

CLI to evaluate MOS results from MTurk and approve/reject workers.

positional arguments:
  {prepare-evaluation,gen-example-input}
                                        description
    prepare-evaluation                  convert input data and results to .json-file
    gen-example-input                   generate example input data

options:
  -h, --help                            show this help message and exit
  -v, --version                         show program's version number and exit
```

### mos-cli

```txt
usage: mos-cli [-h] [-v] {init,masks,stats,mturk} ...

CLI to evaluate MOS results from MTurk and approve/reject workers.

positional arguments:
  {init,masks,stats,mturk}
                                        description
    init                                initialize project
    masks                               masks commands
    stats                               stats commands
    mturk                               mturk commands

options:
  -h, --help                            show this help message and exit
  -v, --version                         show program's version number and exit
```

## Usage as library

```py
import numpy as np

from tts_mos_test_mturk import compute_mos, compute_ci95

_ = np.nan

ratings = np.array([
    # columns represent sentences
    # algorithm 1
    [
      [4, 5, _, 4],  # rater 1
      [4, 4, 4, 5],  # rater 2
      [_, 3, 5, 4],  # rater 3
      [_, _, _, _],  # rater 4
    ],
    # algorithm 2
    [
      [1, 2, _, _],  # rater 1
      [1, 1, 1, _],  # rater 2
      [_, 2, 5, 1],  # rater 3
      [_, 1, _, 1],  # rater 4
    ]
])

alg1_mos = compute_mos(ratings[0])
alg1_ci95 = compute_ci95(ratings[0])

print(f"MOS algorithm 1: {alg1_mos:.2f} ± {alg1_ci95:.4f}")
# MOS algorithm 1: 4.20 ± 0.6997

alg2_mos = compute_mos(ratings[1])
alg2_ci95 = compute_ci95(ratings[1])

print(f"MOS algorithm 2: {alg2_mos:.2f} ± {alg2_ci95:.4f}")
# MOS algorithm 2: 1.60 ± 1.7912
```

## Pipeline

Note: The creation of the template and survey is not fully described yet. The evaluation can be done using a .json-file which interacts as a interface between the template and the evaluation (see "Project JSON example"); start at step 4 in this case.

### 1. Create MTurk Template

To create the MTurk Template the script at [mturk-template/create-template.sh](mturk-template/create-template.sh) can be used. To prepare the audio files for the template and to create the upload CSV for MTurk the script at [mturk-template/create-upload-csv.sh](mturk-template/create-upload-csv.sh) can be used.

### 2. Run survey on MTurk

The survey needs to be started at MTurk. Alternatively some example data can be generated with:

```py
mturk-template-cli gen-example-input \
  "/tmp/algorithms-and-files.csv" \
  "/tmp/Batch_374625_batch_results.csv" \
  "/tmp/upload.csv" --seed 1234
```

### 3. Prepare evaluation

To prepare the evaluation these two files are needed:

- a file containing the algorithm and sentence for each url
  - it needs to contain 3 columns `audio_url`, `algorithm` and `file`
    - `audio_url`: the link to an audio file that was evaluated
    - `algorithm`: the name of the algorithm for that file, e.g., `ground-truth`
    - `file`: the name of the file on each algorithm, e.g., `1.wav`
- the batch results file from MTurk
  - Visit site [MTurk -> Manage](https://requester.mturk.com/manage)
  - Then click on `Review Results` for the batch you want to evaluate
  - Then click on `Download CSV`
  - You get a file which is named something like `Batch_374625_batch_results.csv`

Then create the .json-file for the evaluation with:

```sh
mturk-template-cli prepare-evaluation \
  "/tmp/algorithms-and-files.csv" \
  "/tmp/Batch_374625_batch_results.csv" \
  "/tmp/project.json"
```

The .json-file contains all relevant information for the evaluation.

Note: If *another template* is used then a .json-file needs to be created for the evaluation containing all required fields.

### 4. Initialize project

From the previously created .json-file a new project can be initialized with:

```sh
mos-cli init \
  "/tmp/project.json"
  "/tmp/project.pkl"
```

### 5. Mask workers/assignments/ratings

Workers/assignments/ratings can be masked in order to ignore them later in the MOS calculation. For these operations the command `mos-cli masks [operation]` is used. For example: Mask assignments that were done too fast  (e.g., less than 30 seconds):

```sh
mos-cli masks mask-assignments-by-worktime \
  "/tmp/project.pkl" \
  30 "too-fast"
```

Example output:

```txt
--- Assignment Statistics ---
0 out of all 540 assignments (0.00%) were already masked (i.e., 540 unmasked).
Masked 34 out of the 540 unmasked assignments (6.30%), kept 506 unmasked!
Result: 34 out of all 540 assignments (6.30%) are masked now!
--- Ratings Statistics ---
0 out of all 4320 ratings (0.00%) were already masked (i.e., 4320 unmasked).
Masked 272 out of the 4320 unmasked ratings (6.30%), kept 4048 unmasked!
Result: 272 out of all 4320 ratings (6.30%) are masked now!
Updated project at: "/tmp/project.pkl"
Log: "/tmp/tts-mos-test-mturk.log"
```

This operation masked all 34 assignments (incl. their 272 contained ratings) that were done too fast.

Then, to mask from the remaining assignments the ones done without a headphone (i.e., laptop or desktop), the following command can be used:

```sh
mos-cli masks mask-assignments-by-device \
  "/tmp/project.pkl" \
  "laptop" "desktop" \
  "too-fast > no-headphone" \
  --masks "too-fast"
```

Example output:

```log
--- Assignment Statistics ---
34 out of all 540 assignments (6.30%) were already masked (i.e., 506 unmasked).
Masked 54 out of the 506 unmasked assignments (10.67%), kept 452 unmasked!
Result: 88 out of all 540 assignments (16.30%) are masked now!
--- Ratings Statistics ---
272 out of all 4320 ratings (6.30%) were already masked (i.e., 4048 unmasked).
Masked 432 out of the 4048 unmasked ratings (10.67%), kept 3616 unmasked!
Result: 704 out of all 4320 ratings (16.30%) are masked now!
Updated project at: "/tmp/project.pkl"
Log: "/tmp/tts-mos-test-mturk.log"
```

This operation masked 54 further assignments (incl. their 432 ratings) that were done without a headphone. All assignments that were done too fast were already masked.

### 6. Calculate MOS and CI95

To calculate the MOS for all ratings while ignoring ratings that were done without a headphone or were taken too fast, the masks `too-fast` and `too-fast > no-headphone` need to be applied:

```sh
mos-cli stats print-mos \
  "/tmp/project.pkl" \
  --masks \
    "too-fast" \
    "too-fast > no-headphone"
```

Example output:

```log
Count of ratings (unmasked/all): 3616/4320 -> on average 904/1080 per algorithm

  Algorithm       MOS      CI95
0      alg0  3.155134  0.178079
1      alg1  2.985620  0.161751
2      alg2  2.868565  0.175135
3      alg3  2.890365  0.183059
Log: "/tmp/tts-mos-test-mturk.log"
```

### 7. Approve/reject assignments

To approve all assignments that weren't done too fast, a CSV can be generated using:

```sh
mos-cli mturk prepare-approval \
  "/tmp/project.pkl" \
  "/tmp/approve.csv" \
  --costs 0.10 \
  --reason "good work" \
  --masks "too-fast"
```

Example output:

```log
Count of assignments that will be approved: 506
Estimated costs (506 assignments x 0.10$): 50.60$
Written output to: "/tmp/approve.csv"
Log: "/tmp/tts-mos-test-mturk.log"
```

To finally approve the assignments:

```sh
mos-cli mturk approve \
  "AWS_ACCESS_KEY_ID" \
  "AWS_SECRET_ACCESS_KEY" \
  "/tmp/approve.csv"
```

To reject all assignments that were done too fast, a CSV can be generated using:

```sh
mos-cli mturk prepare-rejection \
  "/tmp/project.pkl" \
  "assignment was done too fast" \
  "/tmp/reject.csv" \
  --masks "too-fast"
```

Example output:

```log
Count of assignments that will be rejected: 34
Written output to: "/tmp/reject.csv"
Log: "/tmp/tts-mos-test-mturk.log"
```

To finally reject the assignments:

```sh
mos-cli mturk reject \
  "AWS_ACCESS_KEY_ID" \
  "AWS_SECRET_ACCESS_KEY" \
  "/tmp/reject.csv"
```

## Project JSON example

```json
{
  "files": [
    "file1",
    "file2"
  ],
  "algorithms": [
    "alg1",
    "alg2"
  ],
  "workers": {
    "Worker1": {
      "Assignment1": {
        "device": "in-ear",
        "state": "Approved",
        "worktime": 30,
        "ratings": [
          {
            "rating": 5,
            "algorithm": "alg1",
            "file": "file1"
          },
          {
            "rating": 3,
            "algorithm": "alg2",
            "file": "file1"
          }
        ]
      }
    }
  }
}
```

## Roadmap

- add `masks mask-workers-by-id`
- add `masks mask-assignments-by-id`
- add `masks mask-assignments-by-status`
- add `masks mask-assignments-by-date`
- add `masks mask-assignments-not-of-last-month/week/day`
- add `masks reverse-mask`

## Dependencies

- `pandas`
- `tqdm`
- `boto3`
- `boto3-stubs`
- `ordered-set>=4.1.0`
- `scipy`

## Contributing

If you notice an error, please don't hesitate to open an issue.

### Development setup

```sh
# update
sudo apt update
# install Python 3.8, 3.9, 3.10 & 3.11 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/stefantaubert/tts-mos-test-mturk.git
cd tts-mos-test-mturk
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first install the tool like in "Development setup"
# then, navigate into the directory of the repo (if not already done)
cd tts-mos-test-mturk
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

Final lines of test result output:

```log
  py38: commands succeeded
  py39: commands succeeded
  py310: commands succeeded
  py311: commands succeeded
  congratulations :)
```

## License

MIT License

## Acknowledgments

Calculation and template are based on:

- Ribeiro, F., Florêncio, D., Zhang, C., & Seltzer, M. (2011). CrowdMOS: An approach for crowdsourcing mean opinion score studies. 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2416–2419. [https://doi.org/10.1109/ICASSP.2011.5946971](https://doi.org/10.1109/ICASSP.2011.5946971)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry generated by GitHub (see *About => Cite this repository*).

```txt
Taubert, S. (2023). tts-mos-test-mturk (Version 0.0.1) [Computer software]. https://doi.org/10.5281/zenodo.7669641
```

## Changelog

- v0.0.1 (2023-02-23)
  - Initial release
