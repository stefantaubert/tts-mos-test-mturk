
import datetime
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import OrderedDict as ODType
from typing import Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator
from mean_opinion_score import get_ci95, get_mos
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.df_generation import ALL_CELL_CONTENT, get_mos_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.typing import MaskName


def plot_mcd(data: EvaluationData, mask_names: Set[MaskName], rating_name: str) -> Figure:
  factory = MaskFactory(data)
  rmask = factory.merge_masks_into_rmask(mask_names)

  ratings = get_ratings(data, {rating_name})
  rmask.apply_by_nan(ratings)

  alg_ratings = np.reshape(ratings, (ratings.shape[0], ratings.shape[1] * ratings.shape[2]))
  d = []
  for i in range(len(alg_ratings)):
    r = alg_ratings[i]
    d.append(r[~np.isnan(r)])
  labels = data.algorithms

  fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(3.5, 4),
    dpi=300,
  )

  dtw_y_min = 1
  dtw_y_max = 5
  dtw_y_step_maj = 1
  dtw_y_step_min = 1

  axis_fontsize = 10

  # ax.minorticks_on()
  ax.grid(axis="both", which="major", zorder=1)
  ax.grid(axis="y", which="minor", zorder=1, alpha=0.2)
  ax.minorticks_on()

  # ax.set_title('Mel-Cepstral Distance')

  parts = ax.violinplot(
    d,
    widths=0.9,
    showmeans=False,
    showmedians=False,
    showextrema=False,
  )

  pc: collections.PolyCollection
  for pc in parts['bodies']:
    pc.set_facecolor('lightgray')
    pc.set_edgecolor('gray')
    pc.set_alpha(1)
    pc.set_zorder(2)

  y_ticks_maj = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_maj, step=dtw_y_step_maj)
  y_ticks_min = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_min, step=dtw_y_step_min)

  ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=15, minor=False)
  x_ticks_min = np.arange(0, len(labels) + 1, 0.5)
  # ax.set_xticks(x_ticks_min, minor=True)
  ax.set_xlim(0.25, len(labels) + 0.75)
  ax.set_xlabel("Algorithm")

  for xmin in x_ticks_min:
    line = ax.axvline(x=xmin, ls='-')
    line.set_linewidth(0.5)
    line.set_color("gray")
    line.set_alpha(1)
    line.set_zorder(0)

  x_ticks_min2 = np.arange(0, len(labels) + 1, 0.25)
  for xmin in x_ticks_min2:
    line = ax.axvline(x=xmin, ls='-')
    line.set_linewidth(0.5)
    line.set_color("gray")
    line.set_alpha(0.5)
    line.set_zorder(0)

  ax.set_ylim(dtw_y_min, dtw_y_max)
  ax.set_yticks(y_ticks_maj)
  ax.set_yticklabels(ax.get_yticks(), fontsize=axis_fontsize)
  ax.yaxis.set_minor_locator(FixedLocator(y_ticks_min))
  # disable minor x ticks
  ax.tick_params(axis='x', which='minor', bottom=False)
  # ax.tick_params(axis='x', which='minor')
  ax.set_ylabel("Rating")

  q1, q2, q3 = np.percentile(alg_ratings, [25, 50, 75], axis=1)

  inds = np.arange(1, len(labels) + 1)
  ax.scatter(inds, q2, marker='o', color='white', s=15, zorder=3)
  ax.vlines(inds, q1, q3, color='k', linestyle='-', lw=5)

  fig.subplots_adjust(bottom=0.2, wspace=0.05, left=0.15, top=0.98, right=0.99)

  return fig


def plot_mean_group(ax: Axes, ratings: List[Dict], label: str):
  algorithms = [x["Algorithm"] for x in ratings]
  mos_ratings = [x["MOS"] for x in ratings]
  ci95_ratings = [x["CI95"] for x in ratings]
  # ci95_ratings = [x["CI95 (default)"] for x in ratings]

  ax.set_title(label)

  parts = ax.bar(
    algorithms,
    mos_ratings,
    yerr=ci95_ratings,
    width=0.8,
    capsize=10,
  )

  # todo for intell 3
  dtw_y_min = 2.5
  dtw_y_max = 5
  dtw_y_step_maj = 0.5
  dtw_y_step_min = 0.25

  axis_fontsize = 10

  # ax.minorticks_on()
  ax.grid(axis="both", which="major", zorder=1)
  ax.grid(axis="y", which="minor", zorder=1, alpha=0.2)
  ax.minorticks_on()

  pc: matplotlib.patches.Rectangle
  for pc in parts:
    pc.set_facecolor('lightgray')
    pc.set_edgecolor('gray')
    pc.set_alpha(1)
    pc.set_zorder(2)

  ax.set_xticks(np.arange(len(algorithms)),
                labels=algorithms, rotation=15, minor=False)
  # ax.set_xticks(x_ticks_min, minor=True)
  # ax.set_xlim(0.25, len(algorithms) + 0.75)
  ax.set_xlabel("Algorithm")

  y_ticks_maj = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_maj, step=dtw_y_step_maj)
  y_ticks_min = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_min, step=dtw_y_step_min)

  ax.set_ylim(dtw_y_min, dtw_y_max)
  ax.set_yticks(y_ticks_maj)
  ax.set_yticklabels(ax.get_yticks(), fontsize=axis_fontsize)
  ax.yaxis.set_minor_locator(FixedLocator(y_ticks_min))
  # disable minor x ticks
  ax.tick_params(axis='x', which='minor', bottom=False)
  # ax.tick_params(axis='x', which='minor')
  ax.set_ylabel("Rating")


def plot_means(data: EvaluationData, mask_names: Set[MaskName], rating_name: str) -> Figure:
  if False:
    res = get_mos_df(data, mask_names)
    with Path("/tmp/pickle-res.pkl").open("wb") as f:
      pickle.dump(res, f)
  else:
    with Path("/tmp/pickle-res.pkl").open("rb") as f:
      res = pickle.load(f)
      
  res = [x for x in res if x["Rating"] == rating_name]
  all_ratings = [
    x
    for x in res
    if x["Gender"] == ALL_CELL_CONTENT and
      x["AgeGroup"] == ALL_CELL_CONTENT and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  all_male = [
    x
    for x in res
    if x["Gender"] == "male" and
      x["AgeGroup"] == ALL_CELL_CONTENT and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  all_female = [
    x
    for x in res
    if x["Gender"] == "female" and
      x["AgeGroup"] == ALL_CELL_CONTENT and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  all_18_29 = [
    x
    for x in res
    if x["Gender"] == ALL_CELL_CONTENT and
      x["AgeGroup"] == "18-29" and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  all_30_49 = [
    x
    for x in res
    if x["Gender"] == ALL_CELL_CONTENT and
      x["AgeGroup"] == "30-49" and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  all_50 = [
    x
    for x in res
    if x["Gender"] == ALL_CELL_CONTENT and
      x["AgeGroup"] == "50+" and
      x["WorkerId"] == ALL_CELL_CONTENT
  ]

  fig, ax = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(3 * 3, 2 * 4),
    dpi=300,
  )

  plot_mean_group(ax[0, 0], all_ratings, "All")
  plot_mean_group(ax[0, 1], all_male, "Male")
  plot_mean_group(ax[0, 2], all_female, "Female")
  plot_mean_group(ax[1, 0], all_18_29, "18-29")
  plot_mean_group(ax[1, 1], all_30_49, "30-49")
  plot_mean_group(ax[1, 2], all_50, "50+")

  fig.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.5, left=0.15, top=0.95, right=0.99)

  return fig
