import numpy as np
from pathlib import Path
from neural_analysis import utils
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
import yaml
import config

# ============================================================================
# PAPER FORMAT SETTINGS
# ============================================================================

SUBPLOT_W = 60 / 25.4   # inches per subplot
SUBPLOT_H = 45 / 25.4   # inches per subplot
H_PAD = 0.9       # horizontal padding between subplots (inches)
V_PAD = 0.8       # vertical padding between subplots (inches)
LEFT_MARGIN = 0.7  # left margin (inches)
RIGHT_MARGIN = 0.2 # right margin (inches)
TOP_MARGIN = 0.55  # top margin for suptitle (inches)
BOTTOM_MARGIN = 0.5 # bottom margin (inches)

FONT_DEFAULT = 8
FONT_SUPTITLE = 10
FONT_LEGEND = 6
FONT_COMP_TITLE = 8

mpl.rcParams.update({
    'font.size': FONT_DEFAULT,
    'axes.titlesize': FONT_COMP_TITLE,
    'axes.labelsize': FONT_DEFAULT,
    'xtick.labelsize': FONT_DEFAULT,
    'ytick.labelsize': FONT_DEFAULT,
    'legend.fontsize': FONT_LEGEND,
    'figure.titlesize': FONT_SUPTITLE,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def make_figure_with_fixed_subplots(n_rows, n_cols, projection=None):
    """
    Create a figure with exactly SUBPLOT_W × SUBPLOT_H inch subplots,
    manually positioned so they never get auto-scaled.

    Parameters
    ----------
    projection : str or None
        Passed to ``fig.add_axes``. Use ``None`` for ordinary 2D axes
        (default) or ``'3d'`` for 3D axes.
    """
    fig_w = LEFT_MARGIN + n_cols * SUBPLOT_W + (n_cols - 1) * H_PAD + RIGHT_MARGIN
    fig_h = TOP_MARGIN + n_rows * SUBPLOT_H + (n_rows - 1) * V_PAD + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            left = (LEFT_MARGIN + c * (SUBPLOT_W + H_PAD)) / fig_w
            bottom = (BOTTOM_MARGIN + (n_rows - 1 - r) * (SUBPLOT_H + V_PAD)) / fig_h
            width = SUBPLOT_W / fig_w
            height = SUBPLOT_H / fig_h
            axes[r, c] = fig.add_axes([left, bottom, width, height],
                                      projection=projection)

    return fig, axes


# ============================================================================
# CONFIGURATION
# ============================================================================

dpca_dir = config.resolve_pp("dPCA", "lALM", "subspace_results")  # dir containing dpca data (set area)

path_root = config.DATA_DIR_EXP
experiment = ["ALM_1", "ALM_2", "ALM_3"]
animals = "all"
sessions_exclude = ['008202304121458', '006202306141358', '021202402021327',
                    '022202403051220', '026202403191125', '026202403201103']
opto_lat = 'unilat'

# ============================================================================
# CONDITION DEFINITIONS
# ============================================================================
HITS = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
     'label': 'Visual Hit', 'color': 'blue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
     'label': 'Auditory Hit', 'color': 'red', 'linestyle': '--'},
]

ERRORS = [
    {'filter': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
     'label': 'Visual Error', 'color': 'blue', 'linestyle': '--'},
    {'filter': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
     'label': 'Auditory Error', 'color': 'red', 'linestyle': '-'},
]

MISSES = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Visual Miss', 'color': 'blue', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Auditory Miss', 'color': 'red', 'linestyle': ':'},
]

CORRECT_REJECTIONS = [
    {'filter': {'trialType': 'P', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Correct Rejection', 'color': 'green', 'linestyle': '-'},
]

FALSE_ALARMS = [
    {'filter': {'trialType': 'P', 'responseSide': 'L', 'hasphotostim': False},
     'label': 'False Alarm (left)', 'color': 'orange', 'linestyle': '-'},
    {'filter': {'trialType': 'P', 'responseSide': 'R', 'hasphotostim': False},
     'label': 'False Alarm (right)', 'color': 'goldenrod', 'linestyle': '--'},
]

HITS_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Hit Opto', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Hit Opto', 'color': 'pink', 'linestyle': '--'},
]

ERRORS_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Error Opto', 'color': 'lightskyblue', 'linestyle': '--'},
    {'filter': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Error Opto', 'color': 'lightsalmon', 'linestyle': '-'},
]

MISSES_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Miss Opto', 'color': 'slategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto', 'color': 'rosybrown', 'linestyle': ':'},
]

CORRECT_REJECTIONS_OPTO = [
    {'filter': {'trialType': 'P', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Correct Rejection Opto', 'color': 'lightgreen', 'linestyle': '-'},
]

HITS_THRESHOLD = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False, 'visualOriChange_norm': 1},
     'label': 'Visual Hit (threshold)', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False, 'audioOctChange_norm': 1},
     'label': 'Auditory Hit (threshold)', 'color': 'lightsalmon', 'linestyle': '--'},
]

HITS_MAX = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False, 'visualOriChange_norm': 2},
     'label': 'Visual Hit (max)', 'color': 'darkblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False, 'audioOctChange_norm': 2},
     'label': 'Auditory Hit (max)', 'color': 'darkred', 'linestyle': '--'},
]

MISSES_THRESHOLD = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False, 'visualOriChange_norm': 1},
     'label': 'Visual Miss (threshold)', 'color': 'steelblue', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False, 'audioOctChange_norm': 1},
     'label': 'Auditory Miss (threshold)', 'color': 'indianred', 'linestyle': ':'},
]

MISSES_MAX = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False, 'visualOriChange_norm': 2},
     'label': 'Visual Miss (max)', 'color': 'midnightblue', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False, 'audioOctChange_norm': 2},
     'label': 'Auditory Miss (max)', 'color': 'maroon', 'linestyle': ':'},
]

HITS_THRESHOLD_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 1, 'optoLat': opto_lat},
     'label': 'Visual Hit Opto (threshold)', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 1, 'optoLat': opto_lat},
     'label': 'Auditory Hit Opto (threshold)', 'color': 'lightsalmon', 'linestyle': '--'},
]

HITS_MAX_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 2, 'optoLat': opto_lat},
     'label': 'Visual Hit Opto (max)', 'color': 'royalblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 2, 'optoLat': opto_lat},
     'label': 'Auditory Hit Opto (max)', 'color': 'salmon', 'linestyle': '--'},
]

MISSES_THRESHOLD_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 1, 'optoLat': opto_lat},
     'label': 'Visual Miss Opto (threshold)', 'color': 'slategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 1, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto (threshold)', 'color': 'rosybrown', 'linestyle': ':'},
]

MISSES_MAX_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 2, 'optoLat': opto_lat},
     'label': 'Visual Miss Opto (max)', 'color': 'darkslategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 2, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto (max)', 'color': 'sienna', 'linestyle': ':'},
]

# --- Lick trials (any lick to either spout, regardless of correctness) ---
VISUAL_LICK = [
    {'filter': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
     'label': 'Visual Lick', 'color': 'blue', 'linestyle': '-'},
]

AUDITORY_LICK = [
    {'filter': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
     'label': 'Auditory Lick', 'color': 'red', 'linestyle': '-'},
]

VISUAL_LICK_OPTO = [
    {'filter': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Lick Opto', 'color': 'lightblue', 'linestyle': '-'},
]

AUDITORY_LICK_OPTO = [
    {'filter': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Lick Opto', 'color': 'pink', 'linestyle': '-'},
]

# Convenience groupings
LICK_TRIALS = VISUAL_LICK + AUDITORY_LICK
LICK_TRIALS_OPTO = VISUAL_LICK_OPTO + AUDITORY_LICK_OPTO

DEFAULT_COMPONENTS = [0, 1]

marg_names = {
    't': 'Time (condition-independent)',
    's': 'Stimulus',
    'd': 'Decision (lick direction)',
    'st': 'Stimulus × Time',
    'dt': 'Decision × Time',
    'sd': 'Stimulus × Decision',
    'sdt': 'Stimulus × Decision × Time'
}


def make_panel_stim(conditions, components=None, title=''):
    if components is None:
        components = DEFAULT_COMPONENTS
    return {
        'component': components,
        'align_to': 'stimChange',
        'title': title,
        'conditions': conditions,
    }


def make_panel_lick(conditions, components=None, title=''):
    if components is None:
        components = DEFAULT_COMPONENTS
    return {
        'component': components,
        'align_to': 'firstLick',
        'title': title,
        'conditions': conditions,
    }


# ============================================================================
# STATE-SPACE FIGURE DEFINITIONS (2D)
# ============================================================================
# Each state-space figure plots 2D trajectories where the x and y axes are
# dPCA component projections (potentially from different marginalizations).
#
# Structure:
#   'fig_name':   output filename stem
#   'fig_title':  suptitle
#   'align_to':   'stimChange' or 'firstLick'
#   'conditions':  list of condition dicts (same format as HITS, ERRORS, etc.)
#   'subplots':   list of subplot specs, each with:
#       'x_marg', 'x_comp':  marginalization and component index for x-axis
#       'y_marg', 'y_comp':  marginalization and component index for y-axis
#       'title':             (optional) subplot title
#
# The plotting function will project each condition through the relevant
# decoder matrices and plot the resulting 2D trajectories.

STATE_SPACE_FIGURES = [
    # --- Hits ± Opto in stimulus space ---
    {
        'fig_name': 'ss_hits_stim_aligned_opto',
        'fig_title': 'Stimulus aligned state space — Hits ± Opto',
        'align_to': 'stimChange',
        'conditions': HITS + HITS_OPTO,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Hits ± Opto in decision space ---
    {
        'fig_name': 'ss_hits_lick_aligned_opto',
        'fig_title': 'Lick aligned state space — Hits ± Opto',
        'align_to': 'firstLick',
        'conditions': HITS + HITS_OPTO,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    {
        'fig_name': 'ss_misses_stim_aligned_opto',
        'fig_title': 'Stimulus aligned state space — Hits ± Opto',
        'align_to': 'stimChange',
        'conditions': MISSES + MISSES_OPTO,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Hits ± Error in stimulus space ---
    {
        'fig_name': 'ss_hits_stim_aligned',
        'fig_title': 'Stimulus aligned state space — Hits + errors',
        'align_to': 'stimChange',
        'conditions': HITS + ERRORS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Hits + errors in decision space ---
    {
        'fig_name': 'ss_hits_lick_aligned',
        'fig_title': 'Lick aligned state space — Hits + errors',
        'align_to': 'firstLick',
        'conditions': HITS + ERRORS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    {
        'fig_name': 'ss_misses_stim_aligned',
        'fig_title': 'Stimulus aligned state space — Hits + errors',
        'align_to': 'stimChange',
        'conditions': MISSES + ERRORS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },

    {
        'fig_name': 'ss_non_ortho',
        'fig_title': 'Stimulus aligned state space - Non-orthogonal stimuls and decision components',
        'align_to': 'stimChange',
        'conditions': HITS + ERRORS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 0,
             'title': 's1 vs d0'},
            {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 2,
             'title': 's1 vs d2'},
        ],
    },

    {
        'fig_name': 'ss_non_ortho_licl_aligned',
        'fig_title': 'Stimulus aligned state space - Non-orthogonal stimuls and decision components',
        'align_to': 'firstLick',
        'conditions': HITS + ERRORS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 0,
             'title': 's1 vs d0'},
            {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 2,
             'title': 's1 vs d2'},
        ],
    },

    # --- Lick trials (visual vs auditory) — stimulus aligned ---
    {
        'fig_name': 'ss_lick_stim_aligned',
        'fig_title': 'Stimulus aligned state space — Lick trials',
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Lick trials (visual vs auditory) — lick aligned ---
    {
        'fig_name': 'ss_lick_lick_aligned',
        'fig_title': 'Lick aligned state space — Lick trials',
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Lick trials ± Opto — stimulus aligned ---
    {
        'fig_name': 'ss_lick_stim_aligned_opto',
        'fig_title': 'Stimulus aligned state space — Lick trials ± Opto',
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
    # --- Lick trials ± Opto — lick aligned ---
    {
        'fig_name': 'ss_lick_lick_aligned_opto',
        'fig_title': 'Lick aligned state space — Lick trials ± Opto',
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': [
            {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1,
             'title': 's0 vs s1'},
            {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1,
             'title': 'd0 vs d1'},
        ],
    },
]


# ============================================================================
# STATE-SPACE FIGURE DEFINITIONS (3D)
# ============================================================================
# 3D state-space figures plot trajectories in a space defined by THREE dPCA
# component projections. Subplot specs mirror the 2D version but add a third
# axis ('z_marg', 'z_comp'). By default each subplot uses the first three
# components (0, 1, 2) of a single marginalization.

def _ss3d_subplots():
    """Standard 3D subplots: first 3 components of 's' and of 'd'."""
    return [
        {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1, 'z_marg': 's', 'z_comp': 2,
         'title': 's0 / s1 / s2'},
        {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1, 'z_marg': 'd', 'z_comp': 2,
         'title': 'd0 / d1 / d2'},
    ]


STATE_SPACE_FIGURES_3D = [
    # --- Hits ± Opto ---
    {
        'fig_name': 'ss3d_hits_stim_aligned_opto',
        'fig_title': '3D stimulus aligned state space — Hits ± Opto',
        'align_to': 'stimChange',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_hits_lick_aligned_opto',
        'fig_title': '3D lick aligned state space — Hits ± Opto',
        'align_to': 'firstLick',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_misses_stim_aligned_opto',
        'fig_title': '3D stimulus aligned state space — Misses ± Opto',
        'align_to': 'stimChange',
        'conditions': MISSES + MISSES_OPTO,
        'subplots': _ss3d_subplots(),
    },
    # --- Hits + errors ---
    {
        'fig_name': 'ss3d_hits_stim_aligned',
        'fig_title': '3D stimulus aligned state space — Hits + errors',
        'align_to': 'stimChange',
        'conditions': HITS + ERRORS,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_hits_lick_aligned',
        'fig_title': '3D lick aligned state space — Hits + errors',
        'align_to': 'firstLick',
        'conditions': HITS + ERRORS,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_misses_stim_aligned',
        'fig_title': '3D stimulus aligned state space — Misses + errors',
        'align_to': 'stimChange',
        'conditions': MISSES + ERRORS,
        'subplots': _ss3d_subplots(),
    },
    # --- Lick trials (visual vs auditory) ---
    {
        'fig_name': 'ss3d_lick_stim_aligned',
        'fig_title': '3D stimulus aligned state space — Lick trials',
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_lick_lick_aligned',
        'fig_title': '3D lick aligned state space — Lick trials',
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS,
        'subplots': _ss3d_subplots(),
    },
    # --- Lick trials ± Opto ---
    {
        'fig_name': 'ss3d_lick_stim_aligned_opto',
        'fig_title': '3D stimulus aligned state space — Lick trials ± Opto',
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    {
        'fig_name': 'ss3d_lick_lick_aligned_opto',
        'fig_title': '3D lick aligned state space — Lick trials ± Opto',
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': _ss3d_subplots(),
    },
]


# ============================================================================
# TEMPORAL FIGURES (same as before)
# ============================================================================

MARGINALIZATIONS = ['s', 'd', 'sd']

def build_figures_for_marg(marg):
    marg_label = marg_names.get(marg, marg)
    return [
        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_hits_errors_misses',
            'fig_title': f'{marg_label} — Hits, Errors, Misses',
            'panels': [
                make_panel_stim(HITS + ERRORS + MISSES,
                                title='Hits, Errors & Misses — stimulus aligned'),
                make_panel_lick(HITS + ERRORS,
                                title='Hits & Errors — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_cr_fa',
            'fig_title': f'{marg_label} — Correct Rejections & False Alarms',
            'panels': [
                make_panel_stim(CORRECT_REJECTIONS + FALSE_ALARMS,
                                title='CR & FA — stimulus aligned'),
                make_panel_lick(CORRECT_REJECTIONS + FALSE_ALARMS,
                                title='CR & FA — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_hits',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(HITS + HITS_OPTO,
                                title='Hits ± Opto — stimulus aligned'),
                make_panel_lick(HITS + HITS_OPTO,
                                title='Hits ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_errors',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(ERRORS + ERRORS_OPTO,
                                title='Errors ± Opto — stimulus aligned'),
                make_panel_lick(ERRORS + ERRORS_OPTO,
                                title='Errors ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_misses_cr',
            'fig_title': f'{marg_label} — Misses & Correct Rej. ± Opto',
            'panels': [
                make_panel_stim(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                title='Misses & CR ± Opto — stimulus aligned'),
                make_panel_lick(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                title='Misses & CR ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_saliency_hits_misses',
            'fig_title': f'{marg_label} — Hits & Misses by Saliency',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + MISSES_THRESHOLD + MISSES_MAX,
                                title='Hits & Misses by saliency — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX,
                                title='Hits by saliency — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_saliency_hits_misses',
            'fig_title': f'{marg_label} — Hits & Misses by Saliency ± Opto',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO
                                + MISSES_THRESHOLD + MISSES_MAX + MISSES_THRESHOLD_OPTO + MISSES_MAX_OPTO,
                                title='Hits & Misses by saliency ± Opto — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO,
                                title='Hits by saliency ± Opto — lick aligned'),
            ],
        },
    ]


figures_joined = []
for m in MARGINALIZATIONS:
    figures_joined.extend(build_figures_for_marg(m))

MARGINALIZATIONS_NOT_JOINED = ['st', 'dt']

figures_not_joined = []
for m in MARGINALIZATIONS_NOT_JOINED:
    figures_not_joined.extend(build_figures_for_marg(m))


# ============================================================================
# LOAD dPCA RESULTS
# ============================================================================

print(f"Loading dPCA results from: {dpca_dir}")

with open(os.path.join(dpca_dir, 'metadata.yaml'), 'r') as f:
    metadata = yaml.safe_load(f)

area = metadata['area']
window_dpca = tuple(metadata['window'])
bin_size = metadata['bin_size']
step_size = metadata['step_size']

if 'join_marginalizations' in metadata:
    join_marginalizations = metadata['join_marginalizations']
else:
    join_marginalizations = False

if 'z_score' in metadata:
    z_score_data = metadata['z_score']
else:
    print("WARNING: No 'z_score' found in metadata. Defaulting to z_score_data == True.")
    z_score_data = True

z_score_mode = 'custom_baseline' if z_score_data else None

print(f"Area: {area}")
print(f"Original dPCA window: {window_dpca}, bin_size: {bin_size}, step_size: {step_size}")
print(f"z_score_data: {z_score_data}")

weights_npz = np.load(os.path.join(dpca_dir, 'dpca_weights.npz'), allow_pickle=True)
D = weights_npz['D'].item()
P = weights_npz['P'].item()

results_npz = np.load(os.path.join(dpca_dir, 'dpca_results.npz'), allow_pickle=True)
neuron_ids_dpca = results_npz['neuron_ids']
X_dpca = results_npz['X']  # (n_neurons, n_stimuli, n_decisions, n_timebins)

neuron_means = X_dpca.reshape(X_dpca.shape[0], -1).mean(axis=1)
n_neurons = len(neuron_ids_dpca)

try:
    demixing_index = results_npz['demixing_index'].item()
    has_demixing = True
except KeyError:
    demixing_index = {}
    has_demixing = False

try:
    var_explained = results_npz['component_var'].item()
    has_var_explained = True
except KeyError:
    var_explained = {}
    has_var_explained = False

print(f"Neurons in dPCA model: {n_neurons}")
print(f"Available marginalizations: {sorted(D.keys())}")
for key in sorted(D.keys()):
    print(f"  {key}: {D[key].shape[0]} neurons × {D[key].shape[1]} components")


# ============================================================================
# LOAD EXPERIMENTAL DATA
# ============================================================================

print("\nLoading experimental data...")
trialData, sessionData, spikeData = utils.load_data(path_root, experiment, animals)

(trialData, sessionData, spikeData, _) = utils.filter_sessions_hitrate(
    trialData, sessionData=sessionData, spikeData=spikeData, min_hitrate=0.3
)

trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
spikeData = spikeData[~spikeData['session_ID'].isin(sessions_exclude)]
sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]

spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=0.5, quality='good')
spikeData = spikeData[spikeData['area'].isin(area)]

trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)

trialData_full = trialData.copy()

print(f"Loaded {len(spikeData)} neurons, {len(trialData_full)} trials")


# ============================================================================
# SUBSET AND ORDER SPIKE DATA TO MATCH dPCA NEURON ORDER
# ============================================================================

spikeData_proj = spikeData[spikeData['cluster_ID'].isin(neuron_ids_dpca)].copy()

neuron_order_map = {nid: i for i, nid in enumerate(neuron_ids_dpca)}
spikeData_proj['_dpca_order'] = spikeData_proj['cluster_ID'].map(neuron_order_map)
spikeData_proj = spikeData_proj.sort_values('_dpca_order').drop('_dpca_order', axis=1)

assert np.array_equal(spikeData_proj['cluster_ID'].values, neuron_ids_dpca), \
    "Neuron order mismatch between dPCA model and spike data!"

print(f"Matched {len(spikeData_proj)} neurons to dPCA model")

neuron_sessions = spikeData_proj['session_ID'].values


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_trials(trialData, filter_dict):
    """Apply a dictionary of column filters to trialData."""
    df = trialData.copy()
    for col, val in filter_dict.items():
        if isinstance(val, list):
            df = df[df[col].isin(val)]
        elif isinstance(val, bool):
            df = df[df[col] == val]
        else:
            df = df[df[col] == val]
    return df


def make_hashable_filter(filter_dict):
    """Convert a filter dict to a hashable key for caching."""
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in filter_dict.items()))


def build_condition_average(spikeData_proj, trialData_subset, neuron_sessions,
                            bins, align_to, window, z_score_data, z_score_mode,
                            trialData_full):
    """Build a condition-averaged PSTH vector: (n_neurons, n_timebins)."""
    n_neurons = len(spikeData_proj)
    n_timebins = len(bins)

    X_cond = np.full((n_neurons, n_timebins), np.nan)

    if align_to == 'firstLick':
        trialData_subset = trialData_subset[trialData_subset['responseSide'].isin(['L', 'R'])]
        trialData_subset = trialData_subset[trialData_subset['reactionTime'] > (0.1 * 1e6)]

    n_trials = len(trialData_subset)

    if n_trials == 0:
        return X_cond, 0

    for ses in np.unique(neuron_sessions):
        neuron_mask = neuron_sessions == ses
        neuron_indices = np.where(neuron_mask)[0]

        if len(neuron_indices) == 0:
            continue

        trialData_ses = trialData_subset[trialData_subset['session_ID'] == ses]
        if len(trialData_ses) == 0:
            continue

        spikeData_ses = spikeData_proj.iloc[neuron_indices]

        psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
                                      z_score_data=z_score_data, align_to=align_to,
                                      window=window, trialData_full=trialData_full,
                                      z_score_mode=z_score_mode)

        for local_idx, global_idx in enumerate(neuron_indices):
            trial_data = psth_ses[local_idx, :, :]
            valid = ~np.isnan(trial_data[:, 0])
            if np.sum(valid) > 0:
                X_cond[global_idx, :] = np.nanmean(trial_data[valid, :], axis=0)

    return X_cond, n_trials


def project_condition_average(X_cond, decoder_matrix, neuron_means):
    """Project a condition-averaged PSTH through dPCA decoders."""
    n_neurons, n_timebins = X_cond.shape

    X_centered = X_cond - neuron_means[:, np.newaxis]

    nan_mask = np.isnan(X_centered[:, 0])
    X_centered[nan_mask, :] = 0.0

    if nan_mask.any():
        n_missing = np.sum(nan_mask)
        print(f"    Note: {n_missing}/{n_neurons} neurons have no trials "
              f"for this condition (set to grand mean).")

    projections = decoder_matrix.T @ X_centered

    return projections


def ensure_projection(cond, align_to, marg_key, cond_avg_cache, proj_cache,
                      spikeData_proj, neuron_sessions, bins, window_dpca,
                      z_score_data, z_score_mode, trialData_full, D, neuron_means):
    """
    Ensure condition average and projection are computed and cached.
    Returns (projections, n_trials) where projections is (n_components, n_timebins).
    """
    cond_filter = cond['filter']
    label = cond['label']
    filter_key = make_hashable_filter(cond_filter)
    avg_key = (filter_key, align_to)
    proj_key = (filter_key, align_to, marg_key)

    if avg_key not in cond_avg_cache:
        trialData_subset = filter_trials(trialData_full, cond_filter)
        print(f"    {label}: {len(trialData_subset)} trials (computing condition average...)")
        X_cond, n_trials = build_condition_average(
            spikeData_proj, trialData_subset, neuron_sessions,
            bins, align_to, window_dpca, z_score_data, z_score_mode,
            trialData_full
        )
        cond_avg_cache[avg_key] = (X_cond, n_trials)
    else:
        X_cond, n_trials = cond_avg_cache[avg_key]
        print(f"    {label}: cached")

    if proj_key not in proj_cache:
        proj_cache[proj_key] = project_condition_average(
            X_cond.copy(), D[marg_key], neuron_means
        )

    return proj_cache[proj_key], n_trials


# ============================================================================
# STATE-SPACE PLOTTING FUNCTIONS
# ============================================================================

def plot_state_space_projected(fig_spec, time_centers, cond_avg_cache, proj_cache,
                                spikeData_proj, neuron_sessions, bins, window_dpca,
                                z_score_data, z_score_mode, trialData_full, D,
                                neuron_means, var_explained, demixing_index,
                                has_var_explained, has_demixing, area):
    """
    Plot 2D state-space trajectories for projected conditions.

    Each subplot shows trajectories in a space defined by two dPCA components
    (possibly from different marginalizations). Conditions are plotted as
    colored trajectories with markers at t=0 (circle) and trial start (square).

    Uses the same visual style as plot_state_space_flex from dpca_plot.py.
    """
    fig_name = fig_spec['fig_name']
    fig_title = fig_spec.get('fig_title', fig_name)
    align_to = fig_spec.get('align_to', 'stimChange')
    conditions = fig_spec['conditions']
    subplot_specs = fig_spec['subplots']

    n_plots = len(subplot_specs)

    fig, axes = make_figure_with_fixed_subplots(1, n_plots)

    print(f"\n  State-space figure: {fig_name}")
    print(f"    {n_plots} subplots, {len(conditions)} conditions, aligned to {align_to}")

    # Collect all marginalizations needed
    all_margs = set()
    for spec in subplot_specs:
        all_margs.add(spec['x_marg'])
        all_margs.add(spec['y_marg'])

    # Check availability
    for m in all_margs:
        if m not in D:
            print(f"    WARNING: marginalization '{m}' not in decoder — skipping figure.")
            plt.close(fig)
            return None

    # Pre-compute all needed projections
    cond_projections = {}  # (cond_idx, marg_key) -> (projections, n_trials)
    for c_idx, cond in enumerate(conditions):
        for marg_key in all_margs:
            proj, n_trials = ensure_projection(
                cond, align_to, marg_key,
                cond_avg_cache, proj_cache,
                spikeData_proj, neuron_sessions, bins, window_dpca,
                z_score_data, z_score_mode, trialData_full, D, neuron_means
            )
            cond_projections[(c_idx, marg_key)] = (proj, n_trials)

    # Plot each subplot
    for plot_idx, spec in enumerate(subplot_specs):
        ax = axes[0, plot_idx]

        x_marg = spec['x_marg']
        x_comp = spec['x_comp']
        y_marg = spec['y_marg']
        y_comp = spec['y_comp']

        for c_idx, cond in enumerate(conditions):
            label = cond['label']
            color = cond['color']
            linestyle = cond.get('linestyle', '-')

            proj_x, n_trials_x = cond_projections[(c_idx, x_marg)]
            proj_y, n_trials_y = cond_projections[(c_idx, y_marg)]

            n_trials = min(n_trials_x, n_trials_y)
            if n_trials == 0:
                continue

            # Check component availability
            if x_comp >= proj_x.shape[0] or y_comp >= proj_y.shape[0]:
                continue

            traj_x = proj_x[x_comp, :]
            traj_y = proj_y[y_comp, :]

            # Trajectory line
            ax.plot(traj_x, traj_y,
                    color=color, linestyle=linestyle, linewidth=1.5,
                    label=f"{label} (n={n_trials})")

            # Marker at t=0 (stimulus onset or first lick)
            t0_idx = np.argmin(np.abs(time_centers))
            ax.plot(traj_x[t0_idx], traj_y[t0_idx], 'o',
                    markerfacecolor=color, markeredgecolor='black',
                    markersize=7, markeredgewidth=0.5, alpha=0.7, zorder=5)

            # Marker at trial start (first time bin)
            ax.plot(traj_x[0], traj_y[0], 's',
                    markerfacecolor=color, markeredgecolor='black',
                    markersize=7, markeredgewidth=0.5, alpha=0.7, zorder=5)

        # --- Axis labels with variance info ---
        def _axis_label(marg, comp):
            marg_display = marg_names.get(marg, marg)
            if has_var_explained and marg in var_explained and comp < len(var_explained[marg]):
                ve = var_explained[marg][comp]
                return f"{marg_display} comp {comp} ({ve:.1f}%)"
            else:
                return f"{marg_display} comp {comp}"

        ax.set_xlabel(_axis_label(x_marg, x_comp), fontsize=FONT_DEFAULT)
        ax.set_ylabel(_axis_label(y_marg, y_comp), fontsize=FONT_DEFAULT)

        title = spec.get('title', '')
        if title:
            ax.set_title(title, fontsize=FONT_COMP_TITLE, fontweight='bold')

        # Clean up ticks (state space plots don't need numeric ticks)
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       left=False, right=False, labelbottom=False, labelleft=False)

        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

        if plot_idx == 0:
            ax.legend(fontsize=FONT_LEGEND, loc='best', frameon=False)

    # Suptitle
    fig.text(0.5, 1.0 - (TOP_MARGIN * 0.35) / fig.get_size_inches()[1],
             f"{fig_title} — {', '.join(area)}",
             ha='center', va='top',
             fontsize=FONT_SUPTITLE, fontweight='bold')

    return fig


def plot_state_space_projected_3d(fig_spec, time_centers, cond_avg_cache, proj_cache,
                                  spikeData_proj, neuron_sessions, bins, window_dpca,
                                  z_score_data, z_score_mode, trialData_full, D,
                                  neuron_means, var_explained, demixing_index,
                                  has_var_explained, has_demixing, area):
    """
    Plot 3D state-space trajectories for projected conditions.

    Each subplot shows trajectories in a 3D space defined by three dPCA
    components (possibly from different marginalizations; by default the first
    three components of a single marginalization). Conditions are plotted as
    colored 3D trajectories with markers at t=0 (circle) and trial start
    (square), matching the 2D state-space style.

    Subplot specs require 'x_marg'/'x_comp', 'y_marg'/'y_comp' and
    'z_marg'/'z_comp'.
    """
    fig_name = fig_spec['fig_name']
    fig_title = fig_spec.get('fig_title', fig_name)
    align_to = fig_spec.get('align_to', 'stimChange')
    conditions = fig_spec['conditions']
    subplot_specs = fig_spec['subplots']

    n_plots = len(subplot_specs)

    fig, axes = make_figure_with_fixed_subplots(1, n_plots, projection='3d')

    print(f"\n  3D state-space figure: {fig_name}")
    print(f"    {n_plots} subplots, {len(conditions)} conditions, aligned to {align_to}")

    # Collect all marginalizations needed (x, y and z axes)
    all_margs = set()
    for spec in subplot_specs:
        all_margs.add(spec['x_marg'])
        all_margs.add(spec['y_marg'])
        all_margs.add(spec['z_marg'])

    # Check availability
    for m in all_margs:
        if m not in D:
            print(f"    WARNING: marginalization '{m}' not in decoder — skipping figure.")
            plt.close(fig)
            return None

    # Pre-compute all needed projections
    cond_projections = {}  # (cond_idx, marg_key) -> (projections, n_trials)
    for c_idx, cond in enumerate(conditions):
        for marg_key in all_margs:
            proj, n_trials = ensure_projection(
                cond, align_to, marg_key,
                cond_avg_cache, proj_cache,
                spikeData_proj, neuron_sessions, bins, window_dpca,
                z_score_data, z_score_mode, trialData_full, D, neuron_means
            )
            cond_projections[(c_idx, marg_key)] = (proj, n_trials)

    # Plot each subplot
    for plot_idx, spec in enumerate(subplot_specs):
        ax = axes[0, plot_idx]

        x_marg, x_comp = spec['x_marg'], spec['x_comp']
        y_marg, y_comp = spec['y_marg'], spec['y_comp']
        z_marg, z_comp = spec['z_marg'], spec['z_comp']

        for c_idx, cond in enumerate(conditions):
            label = cond['label']
            color = cond['color']
            linestyle = cond.get('linestyle', '-')

            proj_x, n_tx = cond_projections[(c_idx, x_marg)]
            proj_y, n_ty = cond_projections[(c_idx, y_marg)]
            proj_z, n_tz = cond_projections[(c_idx, z_marg)]

            n_trials = min(n_tx, n_ty, n_tz)
            if n_trials == 0:
                continue

            # Check component availability
            if (x_comp >= proj_x.shape[0] or y_comp >= proj_y.shape[0]
                    or z_comp >= proj_z.shape[0]):
                continue

            traj_x = proj_x[x_comp, :]
            traj_y = proj_y[y_comp, :]
            traj_z = proj_z[z_comp, :]

            # Trajectory line
            ax.plot(traj_x, traj_y, traj_z,
                    color=color, linestyle=linestyle, linewidth=1.5,
                    label=f"{label} (n={n_trials})")

            # Marker at t=0 (stimulus onset or first lick)
            t0_idx = np.argmin(np.abs(time_centers))
            ax.plot([traj_x[t0_idx]], [traj_y[t0_idx]], [traj_z[t0_idx]], 'o',
                    markerfacecolor=color, markeredgecolor='black',
                    markersize=7, markeredgewidth=0.5, alpha=0.7, zorder=5)

            # Marker at trial start (first time bin)
            ax.plot([traj_x[0]], [traj_y[0]], [traj_z[0]], 's',
                    markerfacecolor=color, markeredgecolor='black',
                    markersize=7, markeredgewidth=0.5, alpha=0.7, zorder=5)

        # --- Axis labels with variance info ---
        def _axis_label(marg, comp):
            marg_display = marg_names.get(marg, marg)
            if has_var_explained and marg in var_explained and comp < len(var_explained[marg]):
                ve = var_explained[marg][comp]
                return f"{marg_display} c{comp} ({ve:.1f}%)"
            else:
                return f"{marg_display} c{comp}"

        ax.set_xlabel(_axis_label(x_marg, x_comp), fontsize=FONT_LEGEND, labelpad=-6)
        ax.set_ylabel(_axis_label(y_marg, y_comp), fontsize=FONT_LEGEND, labelpad=-6)
        ax.set_zlabel(_axis_label(z_marg, z_comp), fontsize=FONT_LEGEND, labelpad=-6)

        title = spec.get('title', '')
        if title:
            ax.set_title(title, fontsize=FONT_COMP_TITLE, fontweight='bold')

        # Clean up ticks (state space plots don't need numeric ticks).
        # Keep the 3D box/panes for depth cues but drop the numeric labels.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis='both', which='both', length=0, pad=-1)

        # A fixed viewing angle keeps figures comparable across conditions.
        ax.view_init(elev=20, azim=-60)

        if plot_idx == 0:
            ax.legend(fontsize=FONT_LEGEND, loc='upper left', frameon=False)

    # Suptitle
    fig.text(0.5, 1.0 - (TOP_MARGIN * 0.35) / fig.get_size_inches()[1],
             f"{fig_title} — {', '.join(area)}",
             ha='center', va='top',
             fontsize=FONT_SUPTITLE, fontweight='bold')

    return fig


# ============================================================================
# COMPUTE PROJECTIONS AND PLOT
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING PROJECTIONS (condition-averaged)")
print("=" * 80)

bins, time_centers = utils.custom_bins(bin_size=bin_size, step_size=step_size,
                                        window=window_dpca, causal=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(config.FIGURE_DIR, "dPCA", "".join(area) if isinstance(area, (list, tuple)) else str(area), "projections", "1D", f"projections_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

if join_marginalizations:
    figure_list = figures_joined
else:
    figure_list = figures_not_joined

# ---- Shared caches across temporal, 2D and 3D state-space figures ----
cond_avg_cache = {}
proj_cache = {}


# ============================================================================
# TEMPORAL FIGURES (component × time)
# ============================================================================

print("\n" + "-" * 60)
print("TEMPORAL FIGURES")
print("-" * 60)

for fig_spec in figure_list:
    marg_key = fig_spec['marg']
    fig_name = fig_spec.get('fig_name', marg_key)
    panels = fig_spec['panels']

    if marg_key not in D:
        print(f"\nWARNING: marginalization '{marg_key}' not found. Skipping '{fig_name}'.")
        continue

    fig_title = fig_spec.get('fig_title', marg_names.get(marg_key, marg_key))
    decoder_matrix = D[marg_key]

    n_rows = len(panels)
    n_cols = len(panels[0]['component'])

    print(f"\n{'='*60}")
    print(f"Figure: {fig_name} — {fig_title}")
    print(f"  Marginalization: {marg_key}, {n_rows} rows × {n_cols} components")
    print(f"{'='*60}")

    fig, axes = make_figure_with_fixed_subplots(n_rows, n_cols)

    for row_idx, panel in enumerate(panels):
        components = panel['component']
        align_to = panel.get('align_to', 'stimChange')
        row_title = panel.get('title', f'Row {row_idx}')

        print(f"\n  Row {row_idx}: {row_title} (aligned to {align_to})")

        for cond in panel['conditions']:
            cond_filter = cond['filter']
            label = cond['label']
            color = cond['color']
            linestyle = cond.get('linestyle', '-')

            filter_key = make_hashable_filter(cond_filter)
            avg_key = (filter_key, align_to)
            proj_key = (filter_key, align_to, marg_key)

            if avg_key not in cond_avg_cache:
                trialData_subset = filter_trials(trialData_full, cond_filter)
                print(f"    {label}: {len(trialData_subset)} trials "
                      f"(computing condition average...)")
                X_cond, n_trials = build_condition_average(
                    spikeData_proj, trialData_subset, neuron_sessions,
                    bins, align_to, window_dpca, z_score_data, z_score_mode,
                    trialData_full
                )
                cond_avg_cache[avg_key] = (X_cond, n_trials)
            else:
                X_cond, n_trials = cond_avg_cache[avg_key]
                print(f"    {label}: cached")

            if proj_key not in proj_cache:
                proj_cache[proj_key] = project_condition_average(
                    X_cond.copy(), decoder_matrix, neuron_means
                )

            projections = proj_cache[proj_key]

            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]

                if comp >= decoder_matrix.shape[1]:
                    if cond is panel['conditions'][0]:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes, fontsize=FONT_DEFAULT)
                    continue

                if n_trials == 0:
                    continue

                trajectory = projections[comp, :]

                ax.plot(time_centers, trajectory,
                        color=color, linestyle=linestyle, linewidth=1.5,
                        label=f"{label} (n={n_trials})")

        for col_idx, comp in enumerate(components):
            ax = axes[row_idx, col_idx]

            ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.grid(False)
            ax.tick_params(axis='both', which='major',
                           labelsize=FONT_DEFAULT, length=3, width=0.6)

            if has_demixing and marg_key in demixing_index:
                if comp < len(demixing_index[marg_key]):
                    di_val = demixing_index[marg_key][comp]['demixing_index']
                    ve_val = var_explained[marg_key][comp]
                    comp_title = f"Component {comp} (DI={di_val:.2f}, VE={ve_val:.2f})"
                else:
                    comp_title = f"Component {comp}"
            else:
                comp_title = f"Component {comp}"

            if row_idx == 0:
                ax.set_title(comp_title, fontsize=FONT_COMP_TITLE, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n\nProjection (a.u.)", fontsize=FONT_DEFAULT)
            else:
                ax.set_ylabel('Projection (a.u.)', fontsize=FONT_DEFAULT)

            if row_idx == n_rows - 1:
                if align_to == 'firstLick':
                    ax.set_xlabel('Time from first lick (s)', fontsize=FONT_DEFAULT)
                else:
                    ax.set_xlabel('Time from stimulus (s)', fontsize=FONT_DEFAULT)

            if col_idx == 0:
                ax.legend(fontsize=FONT_LEGEND, loc='best', frameon=False)

    fig.text(0.5, 1.0 - (TOP_MARGIN * 0.35) / fig.get_size_inches()[1],
             f"{fig_title} — {', '.join(area)}",
             ha='center', va='top',
             fontsize=FONT_SUPTITLE, fontweight='bold')

    fig.savefig(os.path.join(save_dir, f"{fig_name}_projections.png"),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{fig_name}_projections.svg"),
                dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {fig_name}")
    plt.close(fig)


# ============================================================================
# STATE-SPACE FIGURES (2D)
# ============================================================================

print("\n" + "-" * 60)
print("STATE-SPACE FIGURES (2D)")
print("-" * 60)

for fig_spec in STATE_SPACE_FIGURES:
    fig = plot_state_space_projected(
        fig_spec, time_centers,
        cond_avg_cache, proj_cache,
        spikeData_proj, neuron_sessions, bins, window_dpca,
        z_score_data, z_score_mode, trialData_full, D, neuron_means,
        var_explained, demixing_index, has_var_explained, has_demixing, area
    )

    if fig is not None:
        fig_name = fig_spec['fig_name']
        fig.savefig(os.path.join(save_dir, f"{fig_name}.png"),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f"{fig_name}.svg"),
                    dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig_name}")
        plt.close(fig)


# ============================================================================
# STATE-SPACE FIGURES (3D)
# ============================================================================

print("\n" + "-" * 60)
print("STATE-SPACE FIGURES (3D)")
print("-" * 60)

for fig_spec in STATE_SPACE_FIGURES_3D:
    fig = plot_state_space_projected_3d(
        fig_spec, time_centers,
        cond_avg_cache, proj_cache,
        spikeData_proj, neuron_sessions, bins, window_dpca,
        z_score_data, z_score_mode, trialData_full, D, neuron_means,
        var_explained, demixing_index, has_var_explained, has_demixing, area
    )

    if fig is not None:
        fig_name = fig_spec['fig_name']
        fig.savefig(os.path.join(save_dir, f"{fig_name}.png"),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f"{fig_name}.svg"),
                    dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig_name}")
        plt.close(fig)


print(f"\nDone! All outputs saved to: {save_dir}")