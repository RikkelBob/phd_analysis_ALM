import numpy as np
from pathlib import Path
from neural_analysis import utils
import pandas as pd
import os
import config
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml

# ============================================================================
# PAPER FORMAT SETTINGS
# ============================================================================

SUBPLOT_W = 0.9  # inches per subplot
SUBPLOT_H = 0.75  # inches per subplot
H_PAD = 0.9  # horizontal padding between subplots (inches) — room for y-labels
V_PAD = 0.8  # vertical padding between subplots (inches) — room for x-labels/titles
LEFT_MARGIN = 0.7  # left margin (inches)
RIGHT_MARGIN = 0.2  # right margin (inches)
TOP_MARGIN = 0.55  # top margin for suptitle (inches)
BOTTOM_MARGIN = 0.5  # bottom margin (inches)
X_RANGE = [-1, 1]  # x_lim range

FONT_DEFAULT = 6  # regular text (axis labels, titles, tick labels)
FONT_SUPTITLE = 10  # figure suptitle
FONT_LEGEND = 6  # legend text (minimum size)
FONT_COMP_TITLE = 8  # component titles above columns

# Per-figure y-axis limits, keyed by fig_name. Values in PLOTTED units
# (i.e. after the trajectory / 1000 scaling below). Figures not listed
# here keep matplotlib autoscaling.
Y_RANGES = {
    's_opto_misses_cr_zoom': [-0.22, 0.4],
    's_opto_misses_cr': [-1.5, 1.5],
    's_opto_lick_trials': [-1.5, 1.5],
    'd_opto_hits_errors': [-1.5, 1.5]
}

# Set global matplotlib defaults
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
    'pdf.fonttype': 42,  # TrueType fonts in PDF (editable in Illustrator)
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


# ============================================================================
# SIGNIFICANCE LINE HELPER
# ============================================================================

def add_sig_lines_ax(ax, bool_arrs, colors=None, linewidth=5, alpha=0.5, timeline=None):
    """
    Draw horizontal significance bars at the bottom of an axis.

    Parameters
    ----------
    ax : matplotlib Axes
    bool_arrs : list of 1-D boolean arrays
        Each array marks significant timepoints for one condition/component.
    colors : list of colors, optional
        One per bool_arr. Falls back to colours of existing data lines.
    linewidth : float
    alpha : float
    timeline : array-like, optional
        x-values corresponding to the boolean arrays. Strongly recommended.
    """
    if colors is None:
        data_lines = ax.get_lines()
        colors = [data_line.get_color() for data_line in data_lines]

    if timeline is not None:
        x_data = timeline
    else:
        line = ax.get_lines()[0]
        x_data = line.get_xdata()

    y_range = np.ptp(ax.get_ylim())
    spacing = y_range * 0.05
    y_min = ax.get_ylim()[0]

    for idx, (bool_arr, color) in enumerate(zip(bool_arrs, colors)):
        clusters = []
        start = None

        for i, flag in enumerate(bool_arr):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                clusters.append((start, i - 1))
                start = None
        if start is not None:
            clusters.append((start, len(bool_arr) - 1))

        for start_idx, end_idx in clusters:
            x_start = x_data[start_idx]
            x_end = x_data[end_idx]
            ax.hlines(y=(spacing * (idx + 1)) + y_min, xmin=x_start, xmax=x_end,
                      color=color, linewidth=linewidth, alpha=alpha)

    return ax


def make_figure_with_fixed_subplots(n_rows, n_cols):
    """
    Create a figure with exactly SUBPLOT_W × SUBPLOT_H inch subplots,
    manually positioned so they never get auto-scaled.

    Returns (fig, axes) where axes is a 2D array [row, col].
    """
    fig_w = LEFT_MARGIN + n_cols * SUBPLOT_W + (n_cols - 1) * H_PAD + RIGHT_MARGIN
    fig_h = TOP_MARGIN + n_rows * SUBPLOT_H + (n_rows - 1) * V_PAD + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            # Position in figure coordinates (fraction of fig size)
            left = (LEFT_MARGIN + c * (SUBPLOT_W + H_PAD)) / fig_w
            # rows go top-to-bottom
            bottom = (BOTTOM_MARGIN + (n_rows - 1 - r) * (SUBPLOT_H + V_PAD)) / fig_h
            width = SUBPLOT_W / fig_w
            height = SUBPLOT_H / fig_h

            axes[r, c] = fig.add_axes([left, bottom, width, height])

    return fig, axes


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to dPCA output directory (contains dpca_results.npz, dpca_weights.npz, metadata.yaml)
dpca_dir = config.resolve_pp("dPCA", "lALM", "subspace_results")  # (set area)

# Data loading parameters (must match the dPCA script)
path_root = config.DATA_DIR_EXP
experiment = ["ALM_2", "ALM_3"]
animals = "all"
sessions_exclude = ['008202304121458', '006202306141358', '021202402021327',
                    '022202403051220', '026202403191125', '026202403201103']
opto_lat = 'unilat'

# ============================================================================
# FIGURE DEFINITIONS
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
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 1,
                'optoLat': opto_lat},
     'label': 'Visual Hit Opto (threshold)', 'color': 'lightblue', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 1,
                'optoLat': opto_lat},
     'label': 'Auditory Hit Opto (threshold)', 'color': 'lightsalmon', 'linestyle': ':'},
]

HITS_MAX_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 2,
                'optoLat': opto_lat},
     'label': 'Visual Hit Opto (max)', 'color': 'royalblue', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 2,
                'optoLat': opto_lat},
     'label': 'Auditory Hit Opto (max)', 'color': 'salmon', 'linestyle': ':'},
]

MISSES_THRESHOLD_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 1,
                'optoLat': opto_lat},
     'label': 'Visual Miss Opto (threshold)', 'color': 'slategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 1, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto (threshold)', 'color': 'rosybrown', 'linestyle': ':'},
]

MISSES_MAX_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'visualOriChange_norm': 2,
                'optoLat': opto_lat},
     'label': 'Visual Miss Opto (max)', 'color': 'darkslategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'audioOctChange_norm': 2, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto (max)', 'color': 'sienna', 'linestyle': ':'},
]

LICK_TRIALS = [
    {'filter': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
     'label': 'Visual Lick Trials', 'color': 'blue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
     'label': 'Auditory Lick Trials', 'color': 'red', 'linestyle': '-'},
]

LICK_TRIALS_OPTO = [
    {'filter': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True},
     'label': 'Visual Lick Trials', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True},
     'label': 'Auditory Lick Trials', 'color': 'pink', 'linestyle': '-'},
]

DEFAULT_COMPONENTS = [0, 1, 2, 3, 4]

marg_names = {
    't': 'Time (condition-independent)',
    's': 'Stimulus',
    'd': 'Decision (lick direction)',
    'st': 'Stimulus × Time',
    'dt': 'Decision × Time',
    'sd': 'Stimulus × Decision',
    'sdt': 'Stimulus × Decision × Time',
    'go': 'Go axis (go vs no-go)'
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
# FIGURES LIST
# ============================================================================

MARGINALIZATIONS = ['s', 'd', 'sd', 't', 'go']


def build_figures_for_marg_just_opto(marg):
    marg_label = marg_names.get(marg, marg)

    # Go axis is a single component
    comps = [0] if marg == 'go' else None  # None → uses DEFAULT_COMPONENTS

    return [
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_hits',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(HITS + HITS_OPTO,
                                components=comps, title='Hits ± Opto — stimulus aligned'),
                make_panel_lick(HITS + HITS_OPTO,
                                components=comps, title='Hits ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_errors',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(ERRORS + ERRORS_OPTO,
                                components=comps, title='Errors ± Opto — stimulus aligned'),
                make_panel_lick(ERRORS + ERRORS_OPTO,
                                components=comps, title='Errors ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_misses_cr',
            'fig_title': f'{marg_label} — Misses & Correct Rej. ± Opto',
            'panels': [
                make_panel_stim(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — stimulus aligned'),
                make_panel_lick(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_misses_cr_zoom',
            'fig_title': f'{marg_label} — Misses & Correct Rej. ± Opto',
            'panels': [
                make_panel_stim(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — stimulus aligned'),
                make_panel_lick(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_saliency_hits_misses',
            'fig_title': f'{marg_label} — Hits & Misses by Saliency',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + MISSES_THRESHOLD + MISSES_MAX,
                                components=comps, title='Hits & Misses by saliency — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX,
                                components=comps, title='Hits by saliency — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_saliency_hits',
            'fig_title': f'{marg_label} — Hits & Misses by Saliency ± Opto',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO,
                                components=comps, title='Hits by saliency ± Opto — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO,
                                components=comps, title='Hits by saliency ± Opto — lick aligned'),
            ],
        },
    ]


def build_figures_for_marg(marg):
    marg_label = marg_names.get(marg, marg)

    # Go axis is a single component
    comps = [0] if marg == 'go' else None  # None → uses DEFAULT_COMPONENTS

    return [
        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_hits_errors_misses',
            'fig_title': f'{marg_label} — Hits, Errors, Misses',
            'panels': [
                make_panel_stim(HITS + ERRORS + MISSES,
                                components=comps, title='Hits, Errors & Misses — stimulus aligned'),
                make_panel_lick(HITS + ERRORS,
                                components=comps, title='Hits & Errors — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_cr_fa',
            'fig_title': f'{marg_label} — Correct Rejections & False Alarms',
            'panels': [
                make_panel_stim(CORRECT_REJECTIONS + FALSE_ALARMS,
                                components=comps, title='CR & FA — stimulus aligned'),
                make_panel_lick(CORRECT_REJECTIONS + FALSE_ALARMS,
                                components=comps, title='CR & FA — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_hits',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(HITS + HITS_OPTO,
                                components=comps, title='Hits ± Opto — stimulus aligned'),
                make_panel_lick(HITS + HITS_OPTO,
                                components=comps, title='Hits ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_errors',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(ERRORS + ERRORS_OPTO,
                                components=comps, title='Errors ± Opto — stimulus aligned'),
                make_panel_lick(ERRORS + ERRORS_OPTO,
                                components=comps, title='Errors ± Opto — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_misses_cr',
            'fig_title': f'{marg_label} — Misses & Correct Rej. ± Opto',
            'panels': [
                make_panel_stim(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — stimulus aligned'),
                make_panel_lick(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_misses_cr_zoom',
            'fig_title': f'{marg_label} — Misses & Correct Rej. ± Opto',
            'panels': [
                make_panel_stim(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — stimulus aligned'),
                make_panel_lick(MISSES + MISSES_OPTO + CORRECT_REJECTIONS + CORRECT_REJECTIONS_OPTO,
                                components=comps, title='Misses & CR ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_ctrl_saliency_hits_misses',
            'fig_title': f'{marg_label} — Hits & Misses by Saliency',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + MISSES_THRESHOLD + MISSES_MAX,
                                components=comps, title='Hits & Misses by saliency — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX,
                                components=comps, title='Hits by saliency — lick aligned'),
            ],
        },
        {
            'marg': marg,
            'fig_name': f'{marg}_opto_saliency_hits',
            'fig_title': f'{marg_label} — Hits by Saliency ± Opto',
            'panels': [
                make_panel_stim(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO,
                                components=comps, title='Hits by saliency ± Opto — stimulus aligned'),
                make_panel_lick(HITS_THRESHOLD + HITS_MAX + HITS_THRESHOLD_OPTO + HITS_MAX_OPTO,
                                components=comps, title='Hits by saliency ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_hits_errors',
            'fig_title': f'{marg_label} — Hits & Errors ± Opto',
            'panels': [
                make_panel_stim(HITS + HITS_OPTO + ERRORS + ERRORS_OPTO,
                                components=comps, title='Hits ± Opto — stimulus aligned'),
                make_panel_lick(HITS + HITS_OPTO + ERRORS + ERRORS_OPTO,
                                components=comps, title='Hits ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_lick_trials',
            'fig_title': f'{marg_label} — Visual & Auditory ± Opto',
            'panels': [
                make_panel_stim(LICK_TRIALS + LICK_TRIALS_OPTO,
                                components=comps, title='Visual & Auditory ± Opto — stimulus aligned'),
                make_panel_lick(LICK_TRIALS + LICK_TRIALS_OPTO,
                                components=comps, title='Visual & Auditory ± Opto — lick aligned'),
            ],
        },

        {
            'marg': marg,
            'fig_name': f'{marg}_opto_L_R',
            'fig_title': f'{marg_label} — Left/right lick (vis/hit) ± Opto',
            'panels': [
                make_panel_stim(LICK_TRIALS + LICK_TRIALS_OPTO,
                                components=comps, title='Visual & Auditory ± Opto — stimulus aligned'),
                make_panel_lick(LICK_TRIALS + LICK_TRIALS_OPTO,
                                components=comps, title='Visual & Auditory ± Opto — lick aligned'),
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

# Grand mean per neuron from the original dPCA fit.
neuron_means = X_dpca.reshape(X_dpca.shape[0], -1).mean(axis=1)  # (n_neurons,)
n_neurons = len(neuron_ids_dpca)

go_axis_path = os.path.join(dpca_dir, 'go_axis_results.npz')
if os.path.isfile(go_axis_path):
    print(f"Loading Go axis from: {go_axis_path}")
    go_data = np.load(go_axis_path, allow_pickle=True)
    go_neuron_ids = go_data['neuron_ids']
    go_axis_raw = go_data['go_axis']

    # Find shared neurons and reindex the Go axis to match dPCA neuron order
    go_id_to_idx = {nid: i for i, nid in enumerate(go_neuron_ids)}
    shared_mask = np.array([nid in go_id_to_idx for nid in neuron_ids_dpca])

    if shared_mask.sum() < len(neuron_ids_dpca):
        n_missing = len(neuron_ids_dpca) - shared_mask.sum()
        print(f"  Warning: {n_missing} dPCA neurons not in Go axis — "
              f"using {shared_mask.sum()} shared neurons")

    # Build aligned Go axis vector (same length/order as dPCA axes)
    go_axis_vector = np.zeros(len(neuron_ids_dpca))
    for i, nid in enumerate(neuron_ids_dpca):
        if nid in go_id_to_idx:
            go_axis_vector[i] = go_axis_raw[go_id_to_idx[nid]]

    # Normalize
    norm = np.linalg.norm(go_axis_vector)
    if norm > 0:
        go_axis_vector /= norm
    print(f"  Go axis aligned ({shared_mask.sum()} neurons)")

    # Inject go axis into D as a single-component marginalization
    D['go'] = go_axis_vector.reshape(-1, 1)  # (n_neurons, 1)
    P['go'] = go_axis_vector.reshape(1, -1)  # (1, n_neurons) — encoder
    print(f"  Injected 'go' into D and P as single-component marginalization")
else:
    print("No go_axis_results.npz found — skipping 'go' marginalization.")
    # Remove 'go' from MARGINALIZATIONS so figures aren't attempted
    if 'go' in MARGINALIZATIONS:
        MARGINALIZATIONS.remove('go')

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

# ============================================================================
# LOAD SIGNIFICANCE MASKS (if cross-validation was run)
# ============================================================================

try:
    significance_masks = results_npz['significance_masks'].item()
    has_significance = True
    print(f"Significance masks loaded. Available marginalizations: {sorted(significance_masks.keys())}")
    for key in sorted(significance_masks.keys()):
        mask = significance_masks[key]
        print(f"  {key}: shape {mask.shape} "
              f"(components × timebins)")
except KeyError:
    significance_masks = {}
    has_significance = False
    print("No significance masks found — cross-validation was not run or not saved.")

# Also check metadata for run_validation flag
run_validation_flag = metadata.get('run_validation', False)
if has_significance:
    print(f"Cross-validation significance data available (run_validation={run_validation_flag}).")
else:
    print(f"No cross-validation significance data (run_validation={run_validation_flag}).")

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

# Build session mapping
neuron_sessions = spikeData_proj['session_ID'].values  # (n_neurons,)


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
    """
    Build a condition-averaged PSTH vector: (n_neurons, n_timebins).
    """
    n_neurons = len(spikeData_proj)
    n_timebins = len(bins)

    X_cond = np.full((n_neurons, n_timebins), np.nan)

    # For lick-aligned, filter out no-response and very fast RT trials
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
    """
    Project a condition-averaged PSTH through dPCA decoders.
    """
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

# ---- Caching ----
cond_avg_cache = {}
proj_cache = {}

# Significance line colour per marginalization
SIG_COLORS = {
    't': 'gray',
    's': 'black',
    'd': 'black',
    'sd': 'black',
    'st': 'black',
    'dt': 'black',
    'sdt': 'black',
}

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

    # Determine whether to overlay significance on this figure
    show_significance = (
            has_significance
            and fig_name.endswith('_ctrl_hits_errors_misses')
            and marg_key in significance_masks
    )

    print(f"\n{'=' * 60}")
    print(f"Figure: {fig_name} — {fig_title}")
    print(f"  Marginalization: {marg_key}, {n_rows} rows × {n_cols} components")

    if show_significance:
        print(f"  → Will overlay significance lines (from cross-validated classification)")
    print(f"{'=' * 60}")

    # --- Create figure with fixed 2×2 inch subplots ---
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

            # --- Build condition average once per (condition, alignment) ---
            if avg_key not in cond_avg_cache:
                trialData_subset = filter_trials(trialData_full, cond_filter)
                print(f"    {label}: {len(trialData_subset)} trials "
                      f"(computing condition average...)")
                trialData_full_for_zscore = trialData_full[trialData_full['hasphotostim'] == False].copy()
                X_cond, n_trials = build_condition_average(
                    spikeData_proj, trialData_subset, neuron_sessions,
                    bins, align_to, window_dpca, z_score_data, z_score_mode,
                    trialData_full_for_zscore
                )
                cond_avg_cache[avg_key] = (X_cond, n_trials)

            else:
                X_cond, n_trials = cond_avg_cache[avg_key]
                print(f"    {label}: cached")

            # --- Project once per (condition, alignment, marg) ---
            if proj_key not in proj_cache:
                proj_cache[proj_key] = project_condition_average(
                    X_cond.copy(), decoder_matrix, neuron_means
                )

            projections = proj_cache[proj_key]

            # --- Plot ---
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

                ax.plot(time_centers, trajectory / 1000,
                        color=color, linestyle=linestyle, linewidth=1.5,
                        label=f"{label} (n={n_trials})")

        # --- Apply custom y-range (before sig lines so bars are placed correctly) ---
        if fig_name in Y_RANGES:
            for col_idx, comp in enumerate(components):
                axes[row_idx, col_idx].set_ylim(Y_RANGES[fig_name])

        # --- Add significance lines (only for ctrl_hits_errors_misses, stimulus-aligned panels) ---
        if show_significance and align_to == 'stimChange':
            sig_mask_marg = significance_masks[marg_key]  # (n_components, n_timebins)

            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]

                if comp >= decoder_matrix.shape[1]:
                    continue
                if comp >= sig_mask_marg.shape[0]:
                    continue

                sig_bool = sig_mask_marg[comp, :]  # (n_timebins,)

                # The significance mask timebins should match the dPCA
                # time_centers. If lengths differ (e.g. different binning),
                # truncate to the shorter of the two.
                n_sig = len(sig_bool)
                n_tc = len(time_centers)
                if n_sig != n_tc:
                    n_use = min(n_sig, n_tc)
                    print(f"    WARNING: sig mask length ({n_sig}) != "
                          f"time_centers length ({n_tc}); using first {n_use}")
                    sig_bool = sig_bool[:n_use]
                    tc_use = time_centers[:n_use]
                else:
                    tc_use = time_centers

                sig_color = SIG_COLORS.get(marg_key, 'black')

                add_sig_lines_ax(
                    ax,
                    bool_arrs=[sig_bool],
                    colors=[sig_color],
                    linewidth=3,
                    alpha=0.4,
                    timeline=tc_use,
                )

        # --- Formatting ---
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

            if X_RANGE is not None:
                ax.set_xlim(X_RANGE)

    # Place suptitle manually at the top of the figure
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

print(f"\nDone! All outputs saved to: {save_dir}")