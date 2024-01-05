# tts-mos-test-mturk

[![PyPI](https://img.shields.io/pypi/v/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![PyPI](https://img.shields.io/pypi/pyversions/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![MIT](https://img.shields.io/github/license/stefantaubert/tts-mos-test-mturk.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk/#files)
![PyPI](https://img.shields.io/pypi/implementation/tts-mos-test-mturk.svg)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/tts-mos-test-mturk/latest/master.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/compare/v0.0.2...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10461539.svg)](https://doi.org/10.5281/zenodo.10461539)

Command-line interface (CLI) to evaluate text-to-speech (TTS) mean opinion score (MOS) studies done on Amazon Mechanical Turk (MTurk).

## Features

- `init`: initialize project from .json-file
- `masks`
  - `create`: create empty mask
  - `mask-workers-by-id`: mask workers by their WorkerId
  - `mask-workers-by-age-group`: mask workers by their age group
  - `mask-workers-by-gender`: mask workers by their gender
  - `mask-workers-by-assignments-count`: mask workers by their count of assignments
  - `mask-workers-by-masked-ratings-count`: mask workers by their count of masked ratings
  - `mask-workers-by-correlation`: mask workers by their algorithm/sentence correlation
  - `mask-workers-by-correlation-percent`: mask workers by their algorithm/sentence correlation (percentage-wise)
  - `mask-assignments-by-id`: mask assignments by their AssignmentId
  - `mask-assignments-by-device`: mask assignments by their listening device
  - `mask-assignments-by-status`: mask assignments by their status
  - `mask-assignments-by-time`: mask assignments by their submit time
  - `mask-rating-outliers`: mask outlying ratings
  - `merge-masks`: merge masks together
  - `reverse-mask`: reverse mask
- `stats`
  - `print-mos`: print MOS and CI95
  - `print-masking-stats`: print masking statistics
  - `print-worker-stats`: print worker statistics for each algorithm
  - `print-assignment-stats`: print assignment statistics for each worker
  - `print-sentence-stats`: print sentence statistics for each algorithm
  - `print-data`: print all data points
- `mturk`
  - `prepare-approval`: generate approval file
  - `prepare-rejection`: generate rejection file
  - `prepare-bonus-payment`: generate bonus payment file

## Installation

```sh
pip install tts-mos-test-mturk --user
```

## Usage

```txt
usage: mos-cli [-h] [-v] {init,masks,stats,mturk} ...

CLI to evaluate text-to-speech MOS studies done on MTurk.

positional arguments:
  {init,masks,stats,mturk}
                                        description
    init                                initialize project from .json-file
    masks                               masks commands
    stats                               stats commands
    mturk                               mturk commands

options:
  -h, --help                            show this help message and exit
  -v, --version                         show program's version number and exit
```

## Project JSON example

```json
{
  "algorithms": [
    "alg1",
    "alg2",
    "alg3",
    "alg4"
  ],
  "files": [
    "file1",
    "file2",
    "file3"
  ],
  "workers": {
    "worker1": {
      "gender": "male",
      "age_group": "18-29",
      "assignments": {
        "assignment1": {
          "device": "headphone",
          "state": "Approved",
          "hit": "hit1",
          "time": "13.07.23 05:08:04",
          "ratings": [
            {
              "algorithm": "alg1",
              "file": "file1",
              "votes": {
                "naturalness": 3,
                "intelligibility": 5
              }
            },
            {
              "algorithm": "alg2",
              "file": "file3",
              "votes": {
                "naturalness": 2,
                "intelligibility": 4
              }
            }
          ]
        }
      }
    }
  }
}
```

For a longer example see [etc/example.json](./etc/example.json). It contains 4 algorithms and 120 files which were rated by 36 dummy workers in batches of 10 files per assignment. An example parsing of that file is under [etc/example.sh](./etc/example.sh).

## Roadmap

- add `masks mask-assignments-not-of-last-month/week/day`
- make device, state, hit and time optional
- make tax value optional

## Dependencies

- `numpy`
- `pandas`
- `tqdm`
- `ordered-set>=4.1.0`
- `mean-opinion-score==0.0.2`

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

- Ribeiro, F., Florêncio, D., Zhang, C., & Seltzer, M. (2011). CrowdMOS: An approach for crowdsourcing mean opinion score studies. 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2416–2419. [https://doi.org/10.1109/ICASSP.2011.5946971](https://doi.org/10.1109/ICASSP.2011.5946971)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry generated by GitHub (see *About => Cite this repository*).

```txt
Taubert, S. (2024). tts-mos-test-mturk (Version 0.0.2) [Computer software]. https://doi.org/10.5281/zenodo.10461539
```

## Changelog

- v0.0.2 (2024-01-05)
  - Bugfix:
    - `mask mask-workers-by-correlation-percent`: sorting was not always correct
  - Added:
    - `mask mask-workers-by-correlation-percent`: added option to include masked workers in percentage calculation
    - `mturk prepare-bonus-payment`: added logging of fees for Mechanical Turk
    - `mturk prepare-approval`: added logging of fees for Mechanical Turk
    - added `mask mask-assignments-by-status`
    - added `mask mask-assignments-by-time`
    - added parsing of `HITId`
    - added option to mask assignments before preparing rejection CSV
    - added `!` before mask name reverses mask on input
    - added `masks mask-assignments-by-id`
    - added `masks mask-workers-by-id`
  - Changed:
    - moved template creation and preparation to another repository
    - removed worktime parsing
- v0.0.1 (2023-02-23)
  - Initial release
