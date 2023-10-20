
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
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from mean_opinion_score import get_ci95, get_mos
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.df_generation import ALL_CELL_CONTENT, get_mos_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.globals import LATEX_FONT, LATEX_FONT_SIZE, LATEX_TEXT_WIDTH, LEGEND_FONT
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.statistics.worker_assignment_stats import WorkerEntry, get_wass_stat_data
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


def plot_correlations(data: EvaluationData, mask_names: Set[MaskName], rating_name: str) -> Figure:
  stats: Dict[str, WorkerEntry]
  if True:
    masks = data.get_masks_from_names(mask_names)
    stats = get_wass_stat_data(data, masks)
    with Path("/tmp/get_wass_stat_data.pkl").open("wb") as f:
      pickle.dump(stats, f)
  else:
    with Path("/tmp/get_wass_stat_data.pkl").open("rb") as f:
      stats = pickle.load(f)

  N_COLS = 3
  N_ROWS = 2

  fig, axes = plt.subplots(
    ncols=N_COLS,
    nrows=N_ROWS,
    figsize=(3 * 3, 2 * 4),
    dpi=300,
    edgecolor="black",
    linewidth=0,
  )

  group_names = {
    1: "M/18-29",
    2: "M/30-49",
    3: "M/50+",
    4: "W/18-29",
    5: "W/30-49",
    6: "W/50+",
  }

  group_params = {
    1: ("male", "18-29"),
    2: ("male", "30-49"),
    3: ("male", "50+"),
    4: ("female", "18-29"),
    5: ("female", "30-49"),
    6: ("female", "50+"),
  }

  dem_groups = np.arange(1, 7).reshape((N_ROWS, N_COLS))

  for row in range(N_ROWS):
    for col in range(N_COLS):
      ax: Axes = axes[row][col]
      dem_group = dem_groups[row][col]
      gender, age = group_params[dem_group]
      title = group_names[dem_group]
      ax.set_title(title, fontsize=LATEX_FONT_SIZE, font=LATEX_FONT)

      stat_entries = OrderedDict((
        (w, stat)
        for w, stat in stats.items()
        if stat.gender == gender and stat.age_group == age
      ))

      disp_keys = [x[-3:] for x in stat_entries]
      
      all_vals = []
      
      for rating_name in ["naturalness", "intelligibility"]:
        alg_correlations = [
          stat_entries[k].algorithm_correlations[rating_name]
          for k in stat_entries
        ]
        all_vals.append(alg_correlations)

        sent_correlations = [
          stat_entries[k].sentence_correlations[rating_name]
          for k in stat_entries
        ]
        all_vals.append(sent_correlations)
        
      means = [
        np.mean([
          x[i]
          for x in all_vals
        ])
        for i in range(len(all_vals[0]))
      ]
      all_vals.insert(0, means)
      all_vals = np.array(all_vals)
      sort_order = list(reversed(np.argsort(all_vals[0])))
      all_vals= all_vals[:,sort_order]
      disp_keys = [disp_keys[i] for i in sort_order]

        
      colors = [    0.1, 0.4,0.5, 0.65, 0.75]
      # legend_keys = ["Naturalness", "Intelligibility"]
      legend_keys = [
        "Durchschnitt",
        "Natuerlichkeit (Alg.)",
        "Natuerlichkeit (Satz)", 
        "Verstaendlichkeit (Alg.)",
        "Verstaendlichkeit (Satz)",
      ]
      keys_nr = np.arange(len(disp_keys))

      multiplier = 0
      width = (1-0.2)/len(all_vals)

      parts = []
      for val_i, vals in enumerate(all_vals):
            
        offset = width * multiplier
        color = tuple([colors[val_i]]*3+ [1.0])
        rects = ax.bar(
          keys_nr + offset, 
          vals, 
          width, 
          color=color, 
          fill=True,
        )
        multiplier += 1

        pc: Rectangle
        for pc in rects:
          # pc.set_facecolor('gray')
          # pc.set_edgecolor('gray')
          pc.set_alpha(1)
          pc.set_zorder(2)
        parts.append(rects)
      
      legend_recs = [
        [x for x in part if x._height > 0][0]
        for part in parts
      ]
      
      #if col == 2 and row==0:
      #ax.legend(legend_recs, legend_keys,loc='best', ncols=1, fontsize='x-small')
      
      # todo for intell 3
      dtw_y_min = -1
      dtw_y_max = 1
      dtw_y_step_maj = 0.5
      dtw_y_step_min = 0.1

      axis_fontsize = 10

      # ax.minorticks_on()
      ax.grid(axis="x", which="major", visible=True)
      ax.grid(axis="x", which="minor", visible=False)
      ax.grid(axis="y", which="major", zorder=1)
      ax.grid(axis="y", which="minor", zorder=1, alpha=0.2)
      ax.minorticks_on()

      ax.set_xlabel("Arbeiter", fontsize=LATEX_FONT_SIZE, font=LATEX_FONT)
      ax.set_xticks(keys_nr + width*((len(all_vals)-1)/2), disp_keys, rotation=0, minor=False)
      # ax.set_xticks(x_ticks_min, minor=True)
      # ax.set_xlim(0.25, len(algorithms) + 0.75)
      ax.set_xticklabels(disp_keys, font=LATEX_FONT)
      ax.xaxis.set_tick_params(labelsize=LATEX_FONT_SIZE)

      y_ticks_maj = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_maj, step=dtw_y_step_maj)
      y_ticks_min = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_min, step=dtw_y_step_min)

      ax.set_ylim(dtw_y_min, dtw_y_max)
      ax.set_yticks(y_ticks_maj)
      ax.set_yticklabels(ax.get_yticks(), font=LATEX_FONT)
      ax.yaxis.set_tick_params(labelsize=LATEX_FONT_SIZE)
      ax.yaxis.set_minor_locator(FixedLocator(y_ticks_min))
      # disable minor x ticks
      ax.tick_params(axis='x', which='minor', bottom=False)
      # ax.tick_params(axis='x', which='minor')
      ax.set_ylabel("Korrelationskoeff.", fontsize=LATEX_FONT_SIZE, font=LATEX_FONT)
      if row < N_ROWS - 1:
        ax.set_xlabel("")
        # ax.set_xticklabels([])
      if col > 0:
        ax.set_ylabel("")
        ax.set_yticklabels([])

      tick: Text
      for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
  # plt.subplots_adjust(
  #   top=0.78,
  #   bottom=0.1,
  #   hspace=0.47,
  #   wspace=0.05,
  #   right=0.99,
  #   left=0.09,
  # )
  
  fig.legend(
    legend_recs,
    legend_keys,
    loc="upper right",
    ncols=2,
    facecolor='white',
    framealpha=1,
    prop = LEGEND_FONT,
  )
  
  plt.gcf().set_size_inches(LATEX_TEXT_WIDTH, 4)
  
  plt.tight_layout(
    pad=0.2,
    h_pad=0.2,
    rect=(0,0,1.0,0.83),
  )

  return fig
