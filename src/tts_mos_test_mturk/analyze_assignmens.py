import math
import re
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Set

import boto3
import botocore
import numpy as np
import pandas as pd
import xmltodict
from mypy_boto3_mturk.type_defs import HITTypeDef
from ordered_set import OrderedSet
from pandas import DataFrame

from tts_mos_test_mturk.calculation.compute_mos_ci95_3gaussian import (compute_ci95, compute_mos,
                                                                       compute_mos_ci95_3gaussian)
from tts_mos_test_mturk.calculation.corrcoef import corrcoef
from tts_mos_test_mturk.calculation.etc import (mask_algos, mask_lower_outliers, mask_outliers,
                                                mask_smaller_than_val, mask_workers)


def get_worker_correlation(worker: int, Z: np.ndarray) -> float:

  pass

# used for finding outliers and computing bonuses


def get_sentence_mos_correlation(worker: int, Zs: List[np.ndarray]) -> float:

  sentences_worker_moses = {}
  sentences_total_moses = {}

  for algorithm_Z in Zs:
    all_remaining_workers = list(range(len(algorithm_Z)))
    all_remaining_workers.remove(worker)

    for sentence_i in range(algorithm_Z.shape[1]):
      if math.isnan(algorithm_Z[worker, sentence_i]):
        continue
      worker_mos = algorithm_Z[worker, sentence_i]
      others_moses = []
      for w_index in all_remaining_workers:
        if math.isnan(algorithm_Z[w_index, sentence_i]):
          continue
        others_moses.append(algorithm_Z[w_index, sentence_i])
      if len(others_moses) == 0:
        continue

      others_mos = np.mean(others_moses)

      if sentence_i not in sentences_worker_moses:
        sentences_worker_moses[sentence_i] = []
        sentences_total_moses[sentence_i] = []

      sentences_worker_moses[sentence_i].append(worker_mos)
      sentences_total_moses[sentence_i].append(others_mos)

  worker_moses = [np.mean(x) for x in sentences_worker_moses.values()]
  total_moses = [np.mean(x) for x in sentences_total_moses.values()]

  result = corrcoef(np.array(worker_moses), np.array(total_moses))
  return result


def get_algorithm_mos_correlation(worker: int, Zs: List[np.ndarray]) -> float:
  worker_moses = []
  total_moses = []
  for algorithm_Z in Zs:
    all_remaining_workers = list(range(len(algorithm_Z)))
    all_remaining_workers.remove(worker)
    worker_mos = compute_mos(algorithm_Z[worker, :])
    total_mos = compute_mos(algorithm_Z[all_remaining_workers, :])
    worker_moses.append(worker_mos)
    total_moses.append(total_mos)
  result = corrcoef(np.array(worker_moses), np.array(total_moses))
  return result


def analyze(Z_all: np.ndarray, work_times_all: np.ndarray, workers: OrderedSet[str], all_audio_paths: OrderedSet[str], paths: OrderedSet[str], min_worktime_s: float,):
  logger = getLogger(__name__)

  # algo_results = list(split_algos(Z_all, all_audio_paths, paths))

  algo_indices = list(mask_algos(all_audio_paths, paths))
  Zs = []
  for algo_i, indices in enumerate(algo_indices):
    Z = Z_all[:, indices]
    Zs.append(Z)

  for worker_index, worker_name in enumerate(workers):
    print(worker_index, get_algorithm_mos_correlation(worker_index, Zs))
    print(worker_index, get_sentence_mos_correlation(worker_index, Zs))

  ignored_workers = []

  # Ignore too fast workers
  worker_mask: np.ndarray = np.nanmin(work_times_all, axis=1) < min_worktime_s
  logger.info(f"Ignored {np.sum(worker_mask)} / {Z_all.shape[0]} too fast workers!")
  old_count = np.nansum(Z_all.flatten() > 0)
  ignored_workers.extend(worker_mask.nonzero()[0])
  Z_all = Z_all[(~worker_mask).nonzero()]
  new_count = np.nansum(Z_all.flatten() > 0)
  logger.info(f"Ignored {old_count - new_count} / {new_count} assignments!")

  # # Ignore too short assignments
  # filter_mask = mask_smaller_than_val(work_times_all, min_worktime_s)
  # logger.info(
  #   f"Ignored {np.sum(filter_mask.flatten())} / {np.nansum(Z_all.flatten() > 0)} too short assignments!")
  # Z_all[filter_mask] = np.nan
  # work_times_all[filter_mask] = np.nan

  # # Ignore too deviate assignments from mean duration
  # filter_mask = mask_lower_outliers(work_times_all, 1.5)
  # logger.info(
  #   f"Ignored {np.sum(filter_mask.flatten())} / {np.nansum(Z_all.flatten() > 0)}  too different assignments based on time!")
  # Z_all[filter_mask] = np.nan

  # # Ignore too slow workers
  # worker_mask = mask_workers(filter_mask, 0.05)
  # logger.info(f"Ignored {np.sum(worker_mask)} / {Z_all.shape[0]} slow workers!")
  # old_count = np.nansum(Z_all.flatten() > 0)
  # Z_all = Z_all[(~worker_mask).nonzero()]
  # new_count = np.nansum(Z_all.flatten() > 0)
  # logger.info(
  #   f"Ignored {old_count - new_count} / {new_count} assignments!")

  for algo_i, indices in enumerate(algo_indices):
    Z = Z_all[:, indices]
    mos = compute_mos(Z)
    ci95 = compute_ci95(Z)

    logger.info(f"MOS for alg{algo_i}: {mos} +- {ci95}")

    Z_worktime = work_times_all[:, indices]

    # Ignore workers with < 20 assignments
    assignment_counts = np.nansum(Z > 0, axis=1)
    worker_mask = mask_smaller_than_val(assignment_counts, 20)
    logger.info(f"Ignored {np.sum(worker_mask)} / {Z.shape[0]} unproductive workers!")
    old_count = np.nansum(Z.flatten() > 0)
    Z = Z[(~worker_mask).nonzero()]
    new_count = np.nansum(Z.flatten() > 0)
    logger.info(
      f"Ignored {old_count - new_count} / {new_count} assignments!")

    mos = compute_mos(Z)
    ci95 = compute_ci95(Z)

    logger.info(f"MOS for alg{algo_i}: {mos} +- {ci95}")

    # std_ci95 = np.mean(calc_worker_std2(Z) * 1.95996)
    # logger.info(f"MOS for alg{algo_i}: {mos} +- {std_ci95}")
    # logger.info(f"MOS for alg{algo_i}: {mos} +- {std2}")

  return resulting_ratings
