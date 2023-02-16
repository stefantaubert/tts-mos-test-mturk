# tts-mos-test-mturk

[![PyPI](https://img.shields.io/pypi/v/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![PyPI](https://img.shields.io/pypi/pyversions/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk)
[![MIT](https://img.shields.io/github/license/stefantaubert/tts-mos-test-mturk.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/tts-mos-test-mturk.svg)](https://pypi.python.org/pypi/tts-mos-test-mturk/#files)
![PyPI](https://img.shields.io/pypi/implementation/tts-mos-test-mturk.svg)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/tts-mos-test-mturk/latest/master.svg)](https://github.com/stefantaubert/tts-mos-test-mturk/compare/v0.0.1...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7534707.svg)](https://doi.org/10.5281/zenodo.7534707)

CLI to evaluate MOS results from MTurk and approve/reject workers.

## Features

- `init`: initialize project
- `masks`
  - `mask-workers-by-assignments-count`: mask workers by their count of assignments
  - `mask-workers-by-masked-ratings-count`: mask workers by their count of masked ratings
  - `mask-workers-by-correlation`: mask workers by their algorithm/sentence correlation
  - `mask-workers-by-correlation-percent`: mask workers by their algorithm/sentence correlation (percentage-wise)
  - `mask-assignments-by-device`: mask assignments by their listening device
  - `mask-assignments-by-work-time`: mask assignments by their work time
  - `mask-rating-outliers`: mask outlying ratings
- `stats`
  - `print-mos`: print MOS and CI95
  - `print-masking-stats`: print masking statistics
  - `print-worker-stats`: print worker statistics for each algorithm
  - `print-assignment-stats`: print assignment statistics for each worker
  - `print-sentence-stats`: print sentence statistics for each algorithm
  - `print-data`: export all data points
- `mturk`
  - `prepare-approval`: generate approval CSV-file
  - `approve`: approve assignments from CSV-file
  - `prepare-rejection`: generate rejection CSV-file
  - `reject`: reject assignments from CSV-file
  - `prepare-bonus-payment`: generate bonus payment CSV-file
  - `pay-bonus`: pay bonus to assignments from CSV-file

## Roadmap

## Installation

```sh
pip install tts-mos-test-mturk --user
```

## Usage

```txt
usage: cli.py [-h] [-v] {init,masks,stats,mturk} ...

CLI to evaluate MOS results from MTurk and approve/reject workers.

positional arguments:
  {init,masks,stats,mturk}  description
    init                    initialize project
    masks                   masks commands
    stats                   stats commands
    mturk                   mturk commands

options:
  -h, --help                show this help message and exit
  -v, --version             show program's version number and exit
```

## Evaluation

### Create MTurk Template

To create the MTurk Template the script at [mturk-template/create-template.sh](mturk-template/create-template.sh) could be used.

### Calculate MOS and CI95

First initialize a project. For this two files are needed:

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

Then initialize a new project:

```sh
mos-cli init \
  algorithms-and-files.csv \
  Batch_374625_batch_results.csv \
  /tmp/project.pkl
```

To calculate the MOS:

```sh
mos-cli calc-mos /tmp/project.pkl
```

Example Output:

```txt
  Algorithm       MOS      CI95
0      alg0  3.186111  0.279992
1      alg1  2.977778  0.090401
2      alg2  2.938889  0.090390
3      alg3  2.896296  0.193317
```

## Dependencies

- `pandas`
- `tqdm`
- `boto3`
- `boto3-stubs`
- `xmltodict`
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

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry generated by GitHub (see *About => Cite this repository*).

## Changelog

- v0.0.1 (unreleased)
  - Initial release
