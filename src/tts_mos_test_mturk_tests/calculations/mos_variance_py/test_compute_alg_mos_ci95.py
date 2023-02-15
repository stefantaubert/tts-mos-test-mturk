import json
from pathlib import Path

import numpy as np

from tts_mos_test_mturk.calculation.mos_variance import compute_alg_mos_ci95


def load_opinion_scores_from_json(path: Path) -> np.ndarray:
  with open(path, "r", encoding="utf-8") as f:
    opinion_scores_list = json.load(f)

  opinion_scores = np.array(opinion_scores_list, dtype=np.float32)
  opinion_scores[opinion_scores == 0] = np.nan
  return opinion_scores


def test_blizzard_crowdmos1_hp():
  path = Path("src/tts_mos_test_mturk_tests/calculations/mos_variance_py/blizzard_crowdmos1_hp.json")
  opinion_scores = load_opinion_scores_from_json(path)

  result = compute_alg_mos_ci95(opinion_scores)

  np.testing.assert_allclose(result, [
    [
      4.908, 2.930328, 2.9626555, 2.580247, 2.3690987, 2.9387755, 3.3471074, 2.7206478, 3.792683, 3.0365853, 2.02834, 2.979339, 2.877551, 2.282258, 2.322314, 3.904, 2.701613, 2.498008
    ],
    [
      0.16898364, 0.3038698, 0.16945656, 0.19913639, 0.23570034, 0.2458218, 0.20753676, 0.17340976, 0.2445845, 0.32016918, 0.14383565, 0.24260935, 0.15716285, 0.20709334, 0.19619721, 0.18108727, 0.23186237, 0.2152706
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_hp():
  path = Path("src/tts_mos_test_mturk_tests/calculations/mos_variance_py/blizzard_crowdmos2_hp.json")
  opinion_scores = load_opinion_scores_from_json(path)

  result = compute_alg_mos_ci95(opinion_scores)

  np.testing.assert_allclose(result, [
    [
      4.921941, 2.8311965, 3.0, 2.811159, 2.4888394, 3.0592105, 3.145055, 2.6876357, 3.665962, 3.178022, 2.0172787, 3.0334077, 2.8394794, 2.124731, 2.249453, 3.9308856, 2.74375, 2.6858406
    ],
    [
      0.02763858, 0.23015681, 0.12691799, 0.20296347, 0.13052502, 0.12673648, 0.10730608, 0.10987953, 0.1664351, 0.24069504, 0.09752844, 0.20278077, 0.16899829, 0.14027223, 0.18874303, 0.10880725, 0.2503712, 0.21257696
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_crowdmos2_ls():
  path = Path("src/tts_mos_test_mturk_tests/calculations/mos_variance_py/blizzard_crowdmos2_ls.json")
  opinion_scores = load_opinion_scores_from_json(path)

  result = compute_alg_mos_ci95(opinion_scores)

  np.testing.assert_allclose(result, [
    [
      4.8125, 2.8155339, 3.3615024, 3.1534884, 2.495238, 3.162162, 3.2102804, 2.6439025, 3.6517413, 3.55, 2.1857142, 3.3640554, 2.9581394, 2.088372, 2.2488263, 3.9812207, 3.5727699, 3.0536585
    ],
    [
      0.13847275, 0.34916577, 0.24193545, 0.30666336, 0.2887459, 0.19313672, 0.25734243, 0.30944833, 0.39222386, 0.29053533, 0.16449067, 0.30231607, 0.2865923, 0.31055424, 0.33643198, 0.31109768, 0.3060731, 0.33317095
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_paid_participants():
  path = Path("src/tts_mos_test_mturk_tests/calculations/mos_variance_py/blizzard_paid_participants.json")
  opinion_scores = load_opinion_scores_from_json(path)

  result = compute_alg_mos_ci95(opinion_scores)

  np.testing.assert_allclose(result, [
    [
      4.8875, 2.8625, 2.8375, 2.4375, 2.2625, 2.7125, 3.5625, 2.475, 3.9375, 3.0, 2.1375, 2.9875, 2.3875, 2.225, 2.5125, 4.175, 2.025, 2.1125
    ],
    [
      0.17842996, 0.485494, 0.3627819, 0.24821484, 0.35267782, 0.32353535, 0.29889312, 0.39418054, 0.27424592, 0.43140996, 0.21034686, 0.25633854, 0.32016677, 0.26781017, 0.43821582, 0.22357036, 0.25432304, 0.21712728
    ]
  ], rtol=1e-7, atol=1e-8)


def test_blizzard_online_volunteers():
  path = Path("src/tts_mos_test_mturk_tests/calculations/mos_variance_py/blizzard_online_volunteers.json")
  opinion_scores = load_opinion_scores_from_json(path)

  result = compute_alg_mos_ci95(opinion_scores)

  np.testing.assert_allclose(result, [
    [
      4.903226, 3.1935484, 2.935484, 2.8064516, 2.8064516, 3.096774, 3.1935484, 2.7096775, 4.0, 3.2258065, 2.3225806, 2.612903, 2.3225806, 2.6451614, 2.3548386, 4.096774, 2.3225806, 2.451613
    ],
    [
      0.14559552, 0.33556616, 0.4504665, 0.50650746, 0.50103223, 0.44746563, 0.39197063, 0.41166425, 0.3736052, 0.5040992, 0.3752356, 0.5318231, 0.43948394, 0.466249, 0.428413, 0.33752534, 0.46237212, 0.50864595
    ]
  ], rtol=1e-7, atol=1e-8)
