from logging import getLogger

import numpy as np


def remove_outliers(Z: np.ndarray, max_std_dev_diff: float = 3.0, bad_score_percent: float = 0.05) -> np.ndarray:
  logger = getLogger(__name__)
  N, M = Z.shape
  logger.info(f"Started with {N} workers and {np.sum(~np.isnan(Z.flatten()))} scores.")

  mu = np.nanmean(Z)
  s = np.nanstd(Z)

  mu_norm = abs(Z - np.tile(mu, (N, 1))) / np.tile(s, (N, 1))
  # remove scores which are more than 'max_std_dev_diff' std devs away from the mean
  outlying_scores: np.ndarray = mu_norm > max_std_dev_diff
  Z[outlying_scores] = np.nan
  logger.info(f"Removed {np.sum(outlying_scores.flatten())} outlying scores.")

  # remove subjects which have more than 'bad_score_percent' * 100 % of outlying scores
  outlying_workers: np.ndarray = np.sum(
    outlying_scores, axis=1) > bad_score_percent * np.sum(~np.isnan(Z), axis=1)
  Z = Z[~outlying_workers, :]
  logger.info(f"Removed {np.sum(outlying_workers.flatten())} outlying workers.")

  logger.info(f"Finished with {Z.shape[0]} workers and {np.sum(~np.isnan(Z.flatten()))} scores.")
  return Z
