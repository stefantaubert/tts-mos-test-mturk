
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
from matplotlib.ticker import FixedLocator
from mean_opinion_score import get_ci95, get_mos
from ordered_set import OrderedSet

from tts_mos_test_mturk.common import get_ratings
from tts_mos_test_mturk.df_generation import ALL_CELL_CONTENT, get_mos_df
from tts_mos_test_mturk.evaluation_data import EvaluationData
from tts_mos_test_mturk.logging import get_detail_logger, get_logger
from tts_mos_test_mturk.masking.mask_factory import MaskFactory
from tts_mos_test_mturk.masking.masks import MaskBase
from tts_mos_test_mturk.statistics.worker_assignment_stats import WorkerEntry, get_wass_stat_data
from tts_mos_test_mturk.typing import MaskName


def plot_ratings_groupwise(data: EvaluationData, mask_names: Set[MaskName]) -> Figure:
  stat_rows: List[ODType[str, Any]]
  if False:
    stat_rows = get_mos_df(data, mask_names)
    with Path("/tmp/get_mos_df.pkl").open("wb") as f:
      pickle.dump(stat_rows, f)
  else:
    with Path("/tmp/get_mos_df.pkl").open("rb") as f:
      stat_rows = pickle.load(f)


  fig, axes = plt.subplots(
    ncols=2,
    nrows=1,
    figsize=(3, 8),
    dpi=300,
    edgecolor="black",
    linewidth=0,
  )
  
  rating_names = OrderedDict()
  rating_names["intelligibility"]= "Verständlichkeit"
  rating_names["naturalness"]= "Natürlichkeit"

  for col, (rating_name, disp_rating_name) in enumerate(rating_names.items()):
    ax: Axes = axes[col]
    res = [x for x in stat_rows if x["Rating"] == rating_name]
    all_ratings = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == ALL_CELL_CONTENT and
        x["AgeGroup"] == ALL_CELL_CONTENT and
        x["WorkerId"] == ALL_CELL_CONTENT
    }

    all_male = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == "male" and
        x["AgeGroup"] == ALL_CELL_CONTENT and
        x["WorkerId"] == ALL_CELL_CONTENT
    }

    all_female = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == "female" and
        x["AgeGroup"] == ALL_CELL_CONTENT and
        x["WorkerId"] == ALL_CELL_CONTENT
    }

    all_18_29 = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == ALL_CELL_CONTENT and
        x["AgeGroup"] == "18-29" and
        x["WorkerId"] == ALL_CELL_CONTENT
    }

    all_30_49 = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == ALL_CELL_CONTENT and
        x["AgeGroup"] == "30-49" and
        x["WorkerId"] == ALL_CELL_CONTENT
    }

    all_50 = {
      x["Algorithm"]: x
      for x in res
      if x["Gender"] == ALL_CELL_CONTENT and
        x["AgeGroup"] == "50+" and
        x["WorkerId"] == ALL_CELL_CONTENT
    }
    
    ax.set_title(disp_rating_name)

    all_algos = OrderedDict()
    all_algos["orig"]="GT"
    all_algos["wg-synthesized"]="GT-WG"
    all_algos["synthesized-dur"]="EXPL"
    all_algos["synthesized"]="IMPL"
    
    all_vals: List[List[float]] = [[], [], [], []]
    all_ci95s: List[List[float]] = [[], [], [], []]
    
    vals = OrderedDict()
    vals["Alles"] = all_ratings
    vals["♂"] = all_male
    vals["♀"] = all_female
    vals["18-29"] = all_18_29
    vals["30-49"] = all_30_49
    vals["50+"] = all_50
    
    for alg_index, alg in enumerate(all_algos):
      for v in vals.values():
        all_vals[alg_index].append(v[alg]["MOS"])
        
      for v in vals.values():
        all_ci95s[alg_index].append(v[alg]["CI95"])
  
    all_vals = np.array(all_vals)
    all_ci95s = np.array(all_ci95s)
    
    disp_keys = list(vals.keys())
    
    assert len(all_algos) == 4
    colors = [    0.4, 0.5, 0.65, 0.75]
    # legend_keys = ["Naturalness", "Intelligibility"]
    legend_keys = list(all_algos.values())
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
        capsize=3,
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
    dtw_y_min = 2.5
    dtw_y_max = 5
    dtw_y_step_maj = 0.5
    dtw_y_step_min = 0.25

    axis_fontsize = 10

    # ax.minorticks_on()
    ax.grid(axis="both", which="major", zorder=1)
    ax.grid(axis="y", which="minor", zorder=1, alpha=0.2)
    ax.minorticks_on()

    ax.set_xlabel("Gruppe")
    ax.set_xticks(keys_nr + width*((len(all_vals)-1)/2), disp_keys, rotation=0, minor=False)
    # ax.set_xticks(x_ticks_min, minor=True)
    # ax.set_xlim(0.25, len(algorithms) + 0.75)
    ax.set_xticklabels(disp_keys, fontsize="medium")

    y_ticks_maj = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_maj, step=dtw_y_step_maj)
    y_ticks_min = np.arange(dtw_y_min, dtw_y_max + dtw_y_step_min, step=dtw_y_step_min)

    ax.set_ylim(dtw_y_min, dtw_y_max)
    ax.set_yticks(y_ticks_maj)
    ax.set_yticklabels(ax.get_yticks(), fontsize=axis_fontsize)
    ax.yaxis.set_minor_locator(FixedLocator(y_ticks_min))
    # disable minor x ticks
    ax.tick_params(axis='x', which='minor', bottom=False)
    # ax.tick_params(axis='x', which='minor')
    ax.set_ylabel("Bewertung")
    # if row < 1:
    #   ax.set_xlabel("")
    #   # ax.set_xticklabels([])

    if col > 0:
      ax.set_ylabel("")
      ax.set_yticklabels([])
      
  plt.subplots_adjust(
    top=0.75,
    bottom=0.15,
    hspace=0.47,
    wspace=0.05,
    right=0.99,
    left=0.075,
  )
  
  fig.legend(legend_recs, legend_keys,
        loc=(0.755,0.85), ncols=2, fontsize='small')

  plt.gcf().set_size_inches(7, 3)

  return fig
