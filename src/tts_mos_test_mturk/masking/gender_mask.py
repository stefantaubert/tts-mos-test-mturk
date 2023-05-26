from typing import Set

from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def get_wmask_by_gender(data: EvaluationData, gender: str):
  factory = MaskFactory(data)
  res_wmask = factory.get_wmask()
  for i, worker_id in enumerate(data.workers):
    worker_data = data.worker_data[worker_id]
    if worker_data.gender == gender:
      res_wmask.mask[i] = True
  
  return res_wmask