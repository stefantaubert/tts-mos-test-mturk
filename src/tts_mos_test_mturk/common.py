from typing import List, Set

import numpy as np

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.typing import RatingName, Ratings, RatingValue


def get_ratings(data: EvaluationData, rating_names: Set[str]) -> np.ndarray:
  ratings: List[List[List[List[RatingValue]]]] = [
    [
      [
        []
        for _ in range(data.n_files)
      ]
      for _ in range(data.n_workers)
    ]
    for _ in range(data.n_algorithms)
  ]

  for worker, worker_data in data.worker_data.items():
    worker_i = data.workers.get_loc(worker)
    for assignment_data in worker_data.assignments.values():
      for (alg_name, file_name), ass_ratings in assignment_data.ratings.items():
        alg_i = data.algorithms.get_loc(alg_name)
        file_i = data.files.get_loc(file_name)
        rating = get_mean_rating(ass_ratings.votes, rating_names)
        ratings[alg_i][worker_i][file_i].append(rating)

  final_ratings = np.full(
    (data.n_algorithms, data.n_workers, data.n_files),
    fill_value=np.nan,
    dtype=np.float32
  )

  for alg_i in range(data.n_algorithms):
    for worker_i in range(data.n_workers):
      for file_i in range(data.n_files):
        rs = ratings[alg_i][worker_i][file_i]
        if len(rs) > 0:
          # Note: theoretically its better to take the mean for each rating-name separately and then the mean of both but in case the amount per rating is equal it makes no difference
          avg_rating = np.mean(rs)
          final_ratings[alg_i, worker_i, file_i] = avg_rating
  return final_ratings


def get_mean_rating(ratings: Ratings, rating_names: Set[RatingName]) -> RatingValue:
  selected_ratings = [
    ratings[rating_name]
    for rating_name in rating_names
  ]
  result = np.mean(selected_ratings)
  return result
