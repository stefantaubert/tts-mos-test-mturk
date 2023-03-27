from typing import Dict, Optional, Union

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData


def get_ratings(data: EvaluationData, rating_name: Optional[str]) -> np.ndarray:
  ratings = np.full(
    (data.n_algorithms, data.n_workers, data.n_files),
    fill_value=np.nan,
    dtype=np.float32
  )

  for worker, worker_data in data.worker_data.items():
    worker_i = data.workers.get_loc(worker)
    for assignment_data in worker_data.assignments.values():
      for rating_data in assignment_data.ratings:
        alg_i = data.algorithms.get_loc(rating_data.algorithm)
        file_i = data.files.get_loc(rating_data.file)
        ratings[alg_i, worker_i, file_i] = get_rating(rating_data.ratings, rating_name)
  return ratings


def get_rating(ratings: Dict[str, Union[float, int]], rating_name: Optional[str]) -> Union[float, int]:
  if rating_name is None:
    return np.mean(list(ratings.values()))
  return ratings[rating_name]
