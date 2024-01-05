from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.statistics.update_stats import print_stats_masks


def reverse_mask(data: EvaluationData, mask_name: str, output_mask_name: str):
  mask = data.get_mask(mask_name)

  new_mask = mask.clone()
  new_mask.reverse()

  data.add_or_update_mask(output_mask_name, new_mask)

  print_stats_masks(data, [mask], [new_mask])
