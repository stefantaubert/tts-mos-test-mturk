import datetime
from collections import OrderedDict
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional
from typing import OrderedDict as ODType
from typing import Set, Tuple

import numpy as np
import pandas as pd
from mean_opinion_score import get_ci95, get_ci95_default, get_mos
from ordered_set import OrderedSet
from scipy import stats

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.typing import MaskName


def get_comps(ratings: np.ndarray):
  n_algs = ratings.shape[0]
  p = list(permutations(range(n_algs), 2))
  res = np.zeros((n_algs, n_algs))
  for i, (alg_1, alg_2) in enumerate(p):
    a = ratings[alg_1].flatten()
    b = ratings[alg_2].flatten()
    # a = a[~np.isnan(a)]
    # b = b[~np.isnan(b)]
        
    tStat, pValue =stats.ttest_rel(b, a, alternative="less", nan_policy="omit")
    res[alg_1, alg_2] = pValue * 100
  return res
  

def get_t_test_df(data: EvaluationData, mask_names: Set[MaskName]) -> List[ODType[str, Any]]:
  
  masks = data.get_masks_from_names(mask_names)
  factory = MaskFactory(data)
  rmask = factory.merge_masks_into_rmask(masks)

  rows = []
  for rating_name in data.rating_names:
    row_template = OrderedDict()
    row_template["Rating"] = rating_name
    current_ratings = get_ratings(data, {rating_name})
    current_ratings_masked = current_ratings.copy()
    rmask.apply_by_nan(current_ratings_masked)
    p_values = get_comps(current_ratings_masked)
    n_algs = p_values.shape[0]
    for alg_i in range(n_algs):
      for alg_j in range(n_algs):
        if alg_i==alg_j:
          continue
        alg_1 = data.algorithms[alg_i]
        alg_2 = data.algorithms[alg_j]
        row = row_template.copy()
        row["Alg1"] = alg_1
        row["Alg2"] = alg_2
        row["P-Wert"] = p_values[alg_i, alg_j]
        rows.append(row)
  return rows