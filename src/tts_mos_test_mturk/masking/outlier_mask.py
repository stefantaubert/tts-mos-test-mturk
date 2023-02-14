
from typing import Set

from tts_mos_test_mturk.calculation.etc import mask_outliers
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def mask_outlying_scores(data: EvaluationData, mask_names: Set[str], max_std_dev_diff: float, output_mask_name: str):
  masks = data.get_masks_from_names(mask_names)
  factory = data.get_mask_factory()

  os = data.get_os()
  omask = factory.merge_masks_into_omask(masks)
  omask.apply_by_nan(os)

  outlier_np_mask = mask_outliers(os, max_std_dev_diff)
  outlier_omask = factory.convert_ndarray_to_omask(outlier_np_mask)
  data.add_or_update_mask(output_mask_name, outlier_omask)

  print_stats_masks(data, masks, [outlier_omask])
