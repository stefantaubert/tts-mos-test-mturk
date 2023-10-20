
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


def plot_ratings_workerwise(data: EvaluationData, mask_names: Set[MaskName], rating_name: str, y_min: float) -> Figure:
  stat_rows: List[ODType[str, Any]]
  if False:
    stat_rows = get_mos_df(data, mask_names)
    with Path("/tmp/get_mos_df.pkl").open("wb") as f:
      pickle.dump(stat_rows, f)
  else:
    with Path("/tmp/get_mos_df.pkl").open("rb") as f:
      stat_rows = pickle.load(f)

  N_COLS = 3
  N_ROWS = 2

  fig, axes = plt.subplots(
    ncols=N_COLS,
    nrows=N_ROWS,
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
  ylabel = "Natürlichkeit"
  dem_groups = np.arange(1, 7).reshape((N_ROWS, N_COLS))

  for row in range(N_ROWS):
    for col in range(N_COLS):
      ax: Axes = axes[row][col]
      dem_group = dem_groups[row][col]
      gender, age = group_params[dem_group]
      title = group_names[dem_group]
      ax.set_title(title, fontsize=LATEX_FONT_SIZE, font=LATEX_FONT)

      stat_entries = OrderedDict((
        ((stat_row["WorkerId"], stat_row["Algorithm"]), stat_row)
        for stat_row in stat_rows
        if stat_row["Gender"] == gender and \
           stat_row["AgeGroup"] == age and \
           stat_row["Rating"] == rating_name and \
           stat_row["WorkerId"] != ALL_CELL_CONTENT
      ))
      
      all_workers = list(sorted({x for x, _ in stat_entries}))
      all_algos = list({x for _, x in stat_entries})
      all_algos = [
        "orig", 
        "wg-synthesized",
        "synthesized-dur",
        "synthesized",
      ]
      all_algos_disp = [
        "GT",
        "GT-WG",
        "EXPL",
        "IMPL",
      ]
      
      assert len(stat_entries) == len(all_workers) * len(all_algos)
      
      vals = {}
      
      for (worker_id, algo), stat_row in stat_entries.items():
        if worker_id not in vals:
          vals[worker_id] = {}
        assert algo not in vals[worker_id]
        vals[worker_id][algo] = stat_row
      
      all_vals: List[List[float]] = [[], [], [], []]
      all_ci95s: List[List[float]] = [[], [], [], []]
      
      for w_index, worker in enumerate(all_workers):
        for alg_index, alg in enumerate(all_algos):
          mos = vals[worker][alg]["MOS"]
          ci95 = vals[worker][alg]["CI95"]
          #ci95 = vals[worker][alg]["CI95 (default)"]
          all_vals[alg_index].append(mos)
          all_ci95s[alg_index].append(ci95)
      
      all_vals = np.array(all_vals)
      all_ci95s = np.array(all_ci95s)
      
      disp_keys = [x[-3:] for x in all_workers]
      
      sort_after_name = True
      if sort_after_name:
        disp_keys, sort_order = zip(*sorted(zip(disp_keys, range(len(disp_keys))),key=lambda x: x[0]))
        all_vals= all_vals[:,sort_order]
        all_ci95s= all_ci95s[:,sort_order]
      
      sort_after_mean = False
      if sort_after_mean:
        sort_order = list(reversed(np.argsort(np.mean(all_vals, axis=0))))
        all_vals= all_vals[:,sort_order]
        all_ci95s= all_ci95s[:,sort_order]
        disp_keys = [disp_keys[i] for i in sort_order]

      assert len(all_algos) == 4
      colors = [    0.4, 0.5, 0.65, 0.75]
      # legend_keys = ["Naturalness", "Intelligibility"]
      legend_keys = all_algos_disp
      keys_nr = np.arange(len(disp_keys))

      multiplier = 0
      width = (1-0.1)/len(all_vals)

      parts = []
      for val_i, (vals, ci95s) in enumerate(zip(all_vals, all_ci95s)):
            
        offset = width * multiplier
        color = tuple([colors[val_i]]*3+ [1.0])
        rects = ax.bar(
          keys_nr + offset, 
          vals, 
          width, 
          color=color, 
          fill=True,
          yerr=ci95s,
          capsize=12/len(vals),
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
      dtw_y_min = y_min
      dtw_y_max = 5
      dtw_y_step_maj = 1
      dtw_y_step_min = 0.5

      # ax.minorticks_on()
      ax.grid(axis="x", which="both", visible=False)
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
      ax.set_ylabel("Bewertung", fontsize=LATEX_FONT_SIZE, font=LATEX_FONT)
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
  #   top=0.83,
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
    ncols=4,
    facecolor='white',
    framealpha=1,
    prop = LEGEND_FONT,
  )
  
  plt.gcf().set_size_inches(LATEX_TEXT_WIDTH, 4)
  
  plt.tight_layout(
    pad=0.2,
    h_pad=0.2,
    rect=(0,0,1.0,0.92),
  )

  return fig
