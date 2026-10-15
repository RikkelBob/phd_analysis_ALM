"""
Interactive 3D state-space plotter.

For each requested figure (FIGURES_TO_PLOT) and each of its subplots:
  1. a normal-size rotation window pops up so you can orient the sub-space;
  2. close the window to confirm the angle;
  3. a separate 1x1 inch export figure is rendered at that angle and saved
     as PNG + SVG (so the SVG imports at exactly 1x1 inch).

Confirmed angles accumulate in a named library you can reuse. Four preset
angles from a previous run are pre-loaded (named by the subplot they were
picked for) and the matching preset is suggested for each subplot.

Output formatting: no legend, no text, light 0.5pt background grid,
1.5pt trajectories, 4pt event markers with 0.5pt black outline, and a single
4pt combined start marker (average of all conditions' start points).
"""

import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import config
import matplotlib

# --- Make sure an interactive backend is active so the pop-up works. ---
# Delete this block if your IDE already shows plots in a window. If no window
# appears, set the backend by hand, e.g. matplotlib.use('TkAgg').
if matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps', 'template'):
    for _b in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(_b, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from neural_analysis import utils


# ============================================================================
# USER SETTINGS
# ============================================================================

FIGURES_TO_PLOT = [
    'ss3d_lick_stim_aligned_opto',
    'ss3d_lick_lick_aligned_opto',
    'ss3d_hits_lick_aligned_opto',
    'ss3d_errors_lick_aligned_opto'
]

# Starting angle for a fresh ("new") pick.
DEFAULT_ANGLE = {'elev': 20.0, 'azim': -60.0, 'roll': 0.0}

# Preset angles carried over from a previous run, named by the subplot they
# were picked for. Always selectable from the menu.
PRESET_ANGLES = [
    {'name': 'stim_s', 'elev': 37.9, 'azim': -122.5, 'roll': 0.0},  # stim-aligned, s0/s1/s2
    {'name': 'stim_d', 'elev': 27.6, 'azim': -44.3,  'roll': 0.0},  # stim-aligned, d0/d1/d2
    {'name': 'lick_s', 'elev': 38.7, 'azim': -102.6, 'roll': 0.0},  # lick-aligned, s0/s1/s2
    {'name': 'lick_d', 'elev': 24.7, 'azim': 44.0,   'roll': 0.0},  # lick-aligned, d0/d1/d2
]

# Which preset to suggest for each (figure, subplot title).
SUGGESTED = {
    ('ss3d_lick_stim_aligned_opto', 's0 s1 s2'): 'stim_s',
    ('ss3d_lick_stim_aligned_opto', 'd0 d1 d2'): 'stim_d',
    ('ss3d_lick_lick_aligned_opto', 's0 s1 s2'): 'lick_s',
    ('ss3d_lick_lick_aligned_opto', 'd0 d1 d2'): 'lick_d',
}

# --- Output geometry ---
EXPORT_SIZE_IN = 1.0    # saved figure size in inches (SVG imports at this size)
DISPLAY_SIZE_IN = 5.5   # comfortable size for the interactive rotation window
DPI = 300               # raster resolution for the PNG

# --- Output formatting (all line/marker sizes are in points) ---
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)   # light grey background lines
GRID_WIDTH = 0.5                       # background grid / pane / spine width (pt)
PANE_COLOR = (1.0, 1.0, 1.0, 0.0)      # transparent panes
TRAJ_WIDTH = 1.5                       # trajectory line width (pt)
MARKER_SIZE = 4.0                      # event + start marker size (pt)
MARKER_EDGE_WIDTH = 0.5                # marker outline width (pt)
MARKER_EDGE_COLOR = 'black'            # fully black outline
START_MARKER_FACE = 'black'           # fill of the single combined start marker
AXIS_LABEL_SIZE = 6.0    # point size of the component axis labels
AXIS_LABEL_PAD  = -6.0   # gap between label and axis (negative pulls inward; tune to taste)

# ============================================================================
# CONFIGURATION
# ============================================================================

dpca_dir = config.resolve_pp("dPCA", "lALM", "subspace_results")  # dir containing dpca results (set area)

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

HITS_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Hit Opto', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Hit Opto', 'color': 'pink', 'linestyle': '--'},
]

MISSES_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Miss Opto', 'color': 'slategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Miss Opto', 'color': 'rosybrown', 'linestyle': ':'},
]

ERRORS_OPTO = [
    {'filter': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Visual Error', 'color': 'lightblue', 'linestyle': '--'},
    {'filter': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
     'label': 'Auditory Error', 'color': 'pink', 'linestyle': '-'},
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

LICK_TRIALS = VISUAL_LICK + AUDITORY_LICK
LICK_TRIALS_OPTO = VISUAL_LICK_OPTO + AUDITORY_LICK_OPTO

marg_names = {
    't': 'Time (condition-independent)',
    's': 'Stimulus',
    'd': 'Decision (lick direction)',
    'st': 'Stimulus × Time',
    'dt': 'Decision × Time',
    'sd': 'Stimulus × Decision',
    'sdt': 'Stimulus × Decision × Time',
}


# ============================================================================
# 3D FIGURE LIBRARY
# ============================================================================

def _ss3d_subplots():
    return [
        {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1, 'z_marg': 's', 'z_comp': 2,
         'title': 's0 s1 s2'},
        {'x_marg': 'd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 1, 'z_marg': 'd', 'z_comp': 2,
         'title': 'd0 d1 d2'},
    ]


AVAILABLE_3D = {
    'ss3d_lick_stim_aligned_opto': {
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_lick_lick_aligned_opto': {
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS + LICK_TRIALS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_lick_stim_aligned': {
        'align_to': 'stimChange',
        'conditions': LICK_TRIALS,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_lick_lick_aligned': {
        'align_to': 'firstLick',
        'conditions': LICK_TRIALS,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_hits_stim_aligned_opto': {
        'align_to': 'stimChange',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_hits_lick_aligned_opto': {
        'align_to': 'firstLick',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_misses_stim_aligned_opto': {
        'align_to': 'stimChange',
        'conditions': MISSES + MISSES_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_hits_stim_aligned': {
        'align_to': 'stimChange',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_hits_lick_aligned': {
        'align_to': 'firstLick',
        'conditions': HITS + HITS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_errors_stim_aligned_opto': {
        'align_to': 'stimChange',
        'conditions': ERRORS + ERRORS_OPTO,
        'subplots': _ss3d_subplots(),
    },
    'ss3d_errors_lick_aligned_opto': {
        'align_to': 'firstLick',
        'conditions': ERRORS + ERRORS_OPTO,
        'subplots': _ss3d_subplots(),
    },
}


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

z_score_data = metadata.get('z_score', True)
if 'z_score' not in metadata:
    print("WARNING: No 'z_score' found in metadata. Defaulting to z_score_data == True.")
z_score_mode = 'custom_baseline' if z_score_data else None

print(f"Area: {area}")
print(f"dPCA window: {window_dpca}, bin_size: {bin_size}, step_size: {step_size}")
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
    var_explained = results_npz['component_var'].item()
    has_var_explained = True
except KeyError:
    var_explained = {}
    has_var_explained = False

print(f"Neurons in dPCA model: {n_neurons}")
print(f"Available marginalizations: {sorted(D.keys())}")


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
# HELPER FUNCTIONS (projection machinery)
# ============================================================================

def filter_trials(trialData, filter_dict):
    df = trialData.copy()
    for col, val in filter_dict.items():
        if isinstance(val, list):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]
    return df


def make_hashable_filter(filter_dict):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in filter_dict.items()))


def build_condition_average(spikeData_proj, trialData_subset, neuron_sessions,
                            bins, align_to, window, z_score_data, z_score_mode,
                            trialData_full):
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
        neuron_indices = np.where(neuron_sessions == ses)[0]
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
    X_centered = X_cond - neuron_means[:, np.newaxis]
    nan_mask = np.isnan(X_centered[:, 0])
    X_centered[nan_mask, :] = 0.0
    return decoder_matrix.T @ X_centered


def ensure_projection(cond, align_to, marg_key, cond_avg_cache, proj_cache,
                      spikeData_proj, neuron_sessions, bins, window_dpca,
                      z_score_data, z_score_mode, trialData_full, D, neuron_means):
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
            bins, align_to, window_dpca, z_score_data, z_score_mode, trialData_full)
        cond_avg_cache[avg_key] = (X_cond, n_trials)
    else:
        X_cond, n_trials = cond_avg_cache[avg_key]

    if proj_key not in proj_cache:
        proj_cache[proj_key] = project_condition_average(
            X_cond.copy(), D[marg_key], neuron_means)

    return proj_cache[proj_key], n_trials


# ============================================================================
# STYLING + PLOTTING
# ============================================================================

def make_3d_fig(size_inches, full_bleed):
    fig = plt.figure(figsize=(size_inches, size_inches))
    if full_bleed:
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def component_label(marg, comp):
    """Axis label for a (marginalization, component) pair.
    Matches the subplot-title convention: 's', comp 0 -> 's0'.
    For the verbose form instead, use e.g.
        f"{marg_names[marg]} {comp}"   ->  'Stimulus 0'
    """
    return f"{marg}{comp}"


def style_clean_3d(ax, spec=None):
    """Strip ticks/title, lighten the grid, and (if a spec is given) label each
    axis with its component name."""
    if spec is not None:
        ax.set_xlabel(component_label(spec['x_marg'], spec['x_comp']),
                      fontsize=AXIS_LABEL_SIZE, labelpad=AXIS_LABEL_PAD)
        ax.set_ylabel(component_label(spec['y_marg'], spec['y_comp']),
                      fontsize=AXIS_LABEL_SIZE, labelpad=AXIS_LABEL_PAD)
        ax.set_zlabel(component_label(spec['z_marg'], spec['z_comp']),
                      fontsize=AXIS_LABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    ax.set_title('')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        # Nice, evenly-spaced gridlines anchored to round values; prune the
        # outermost ticks so no gridline sits right next to the box edge.
        axis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        axis.set_major_formatter(NullFormatter())  # hide numbers, keep gridlines
        try:
            axis._axinfo['grid'].update(color=GRID_COLOR, linewidth=GRID_WIDTH)
        except Exception:
            pass
        try:
            axis.set_pane_color(PANE_COLOR)
        except Exception:
            try:
                axis.pane.set_facecolor(PANE_COLOR)
            except Exception:
                pass
        try:
            axis.pane.set_edgecolor(GRID_COLOR)
            axis.pane.set_linewidth(GRID_WIDTH)
        except Exception:
            pass
        try:
            axis.line.set_color(GRID_COLOR)
            axis.line.set_linewidth(GRID_WIDTH)
        except Exception:
            pass
        try:
            axis.set_pane_color(PANE_COLOR)
        except Exception:
            try:
                axis.pane.set_facecolor(PANE_COLOR)
            except Exception:
                pass
        try:
            axis.pane.set_edgecolor(GRID_COLOR)
            axis.pane.set_linewidth(GRID_WIDTH)
        except Exception:
            pass
        try:
            axis.line.set_color(GRID_COLOR)
            axis.line.set_linewidth(GRID_WIDTH)
        except Exception:
            pass


def plot_subplot_3d(ax, spec, conditions, cond_projections, time_centers):
    """Draw condition trajectories + per-condition event markers + a single
    combined start marker (average of all conditions' start points)."""
    x_marg, x_comp = spec['x_marg'], spec['x_comp']
    y_marg, y_comp = spec['y_marg'], spec['y_comp']
    z_marg, z_comp = spec['z_marg'], spec['z_comp']
    t0_idx = int(np.argmin(np.abs(time_centers)))

    start_points = []
    for c_idx, cond in enumerate(conditions):
        color = cond['color']
        linestyle = cond.get('linestyle', '-')

        proj_x, n_tx = cond_projections[(c_idx, x_marg)]
        proj_y, n_ty = cond_projections[(c_idx, y_marg)]
        proj_z, n_tz = cond_projections[(c_idx, z_marg)]
        n_trials = min(n_tx, n_ty, n_tz)
        if n_trials == 0:
            continue
        if (x_comp >= proj_x.shape[0] or y_comp >= proj_y.shape[0]
                or z_comp >= proj_z.shape[0]):
            continue

        tx, ty, tz = proj_x[x_comp], proj_y[y_comp], proj_z[z_comp]

        # trajectory
        ax.plot(tx, ty, tz, color=color, linestyle=linestyle, linewidth=TRAJ_WIDTH)

        # event marker at t=0 (stimChange or firstLick)
        ax.plot([tx[t0_idx]], [ty[t0_idx]], [tz[t0_idx]], 'o',
                markerfacecolor=color, markeredgecolor=MARKER_EDGE_COLOR,
                markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
                alpha=1.0, zorder=6)

        start_points.append((tx[0], ty[0], tz[0]))

    # single combined start marker = mean start position across conditions
    if start_points:
        s = np.mean(np.array(start_points), axis=0)
        ax.plot([s[0]], [s[1]], [s[2]], 'o',
                markerfacecolor=START_MARKER_FACE, markeredgecolor=MARKER_EDGE_COLOR,
                markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
                alpha=1.0, zorder=7)

    style_clean_3d(ax, spec)


def precompute_projections(fig_spec, cond_avg_cache, proj_cache):
    all_margs = set()
    for spec in fig_spec['subplots']:
        all_margs.update([spec['x_marg'], spec['y_marg'], spec['z_marg']])

    for m in all_margs:
        if m not in D:
            print(f"  WARNING: marginalization '{m}' not in decoder — skipping figure.")
            return None

    cp = {}
    for c_idx, cond in enumerate(fig_spec['conditions']):
        for m in all_margs:
            proj, n_trials = ensure_projection(
                cond, fig_spec['align_to'], m, cond_avg_cache, proj_cache,
                spikeData_proj, neuron_sessions, bins, window_dpca,
                z_score_data, z_score_mode, trialData_full, D, neuron_means)
            cp[(c_idx, m)] = (proj, n_trials)
    return cp


# ============================================================================
# ANGLE PICKING
# ============================================================================

def get_view_angle(ax):
    return {'elev': float(ax.elev),
            'azim': float(ax.azim),
            'roll': float(getattr(ax, 'roll', 0.0) or 0.0)}


def set_view_angle(ax, a):
    try:
        ax.view_init(elev=a['elev'], azim=a['azim'], roll=a.get('roll', 0.0))
    except TypeError:  # matplotlib < 3.6 has no 'roll'
        ax.view_init(elev=a['elev'], azim=a['azim'])


def _angles_close(a, b, tol=0.5):
    return (abs(a['elev'] - b['elev']) < tol
            and abs(a['azim'] - b['azim']) < tol
            and abs(a.get('roll', 0.0) - b.get('roll', 0.0)) < tol)


def choose_angle(disp_fig, disp_ax, angle_lib, context, suggested_name):
    """
    Console menu + (optional) interactive rotation window.
    Returns the chosen angle dict; appends newly confirmed angles to angle_lib.
    """
    print(f"\n=== Viewing angle for: {context} ===")
    sugg_idx = None
    for i, a in enumerate(angle_lib):
        if suggested_name and a.get('name') == suggested_name:
            sugg_idx = i

    print("  Saved angles:")
    for i, a in enumerate(angle_lib):
        tag = "  <-- suggested" if i == sugg_idx else ""
        print(f"    [{i}] {a.get('name', ''):<28} "
              f"elev={a['elev']:6.1f}  azim={a['azim']:7.1f}  roll={a['roll']:5.1f}{tag}")
    print("  Enter a NUMBER  -> reuse that angle exactly (no window, aligns figures)")
    print("  Enter e<NUMBER> -> open window starting from that angle (e.g. e0)")
    print("  Enter / 'n'     -> open window from the default angle")
    if sugg_idx is not None:
        print(f"  (press Enter-less shortcut: type {sugg_idx} to reuse the suggested angle)")

    choice = input("  Choice: ").strip().lower()

    start = None
    reuse_exact = False
    if choice in ('', 'n'):
        start = DEFAULT_ANGLE
    elif choice.startswith('e'):
        try:
            start = angle_lib[int(choice[1:])]
        except (ValueError, IndexError):
            print("  Invalid index — opening at default angle.")
            start = DEFAULT_ANGLE
    else:
        try:
            start = angle_lib[int(choice)]
            reuse_exact = True
        except (ValueError, IndexError):
            print("  Invalid input — opening at default angle.")
            start = DEFAULT_ANGLE

    if reuse_exact:
        print(f"  Reusing '{start.get('name', '')}' exactly: "
              f"elev={start['elev']:.1f}, azim={start['azim']:.1f}, roll={start['roll']:.1f}")
        return {'elev': start['elev'], 'azim': start['azim'], 'roll': start.get('roll', 0.0)}

    set_view_angle(disp_ax, start)
    print("  >>> Rotate the figure, then CLOSE the window to confirm the angle.")
    plt.show()  # blocks until the window is closed

    final = get_view_angle(disp_ax)
    if not any(_angles_close(final, a) for a in angle_lib):
        entry = dict(final)
        entry['name'] = context
        angle_lib.append(entry)
        print(f"  Confirmed and saved as angle [{len(angle_lib) - 1}] ('{context}'): "
              f"elev={final['elev']:.1f}, azim={final['azim']:.1f}, roll={final['roll']:.1f}")
    else:
        print(f"  Confirmed (matches an existing saved angle): "
              f"elev={final['elev']:.1f}, azim={final['azim']:.1f}, roll={final['roll']:.1f}")
    return final


def _slug(s):
    s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
    return s.strip('_').lower()


# ============================================================================
# MAIN
# ============================================================================

bins, time_centers = utils.custom_bins(bin_size=bin_size, step_size=step_size,
                                        window=window_dpca, causal=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(config.FIGURE_DIR, "dPCA", "".join(area) if isinstance(area, (list, tuple)) else str(area), "projections", "3D", f"state_space_3d_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
print(f"\nSaving to: {save_dir}")

cond_avg_cache = {}
proj_cache = {}
angle_lib = [dict(a) for a in PRESET_ANGLES]   # named presets, always selectable

for fig_name in FIGURES_TO_PLOT:
    if fig_name not in AVAILABLE_3D:
        print(f"\nWARNING: '{fig_name}' is not defined in AVAILABLE_3D — skipping.")
        continue

    fig_spec = AVAILABLE_3D[fig_name]
    conditions = fig_spec['conditions']

    print(f"\n{'='*70}\nFigure: {fig_name}  (aligned to {fig_spec['align_to']})\n{'='*70}")

    cond_projections = precompute_projections(fig_spec, cond_avg_cache, proj_cache)
    if cond_projections is None:
        continue

    for spec in fig_spec['subplots']:
        sub_title = spec.get('title', '')
        context = f"{fig_name} | {sub_title}"
        suggested_name = SUGGESTED.get((fig_name, sub_title))

        # 1) interactive display figure (comfortable size) for picking the angle
        disp_fig, disp_ax = make_3d_fig(DISPLAY_SIZE_IN, full_bleed=False)
        plot_subplot_3d(disp_ax, spec, conditions, cond_projections, time_centers)
        angle = choose_angle(disp_fig, disp_ax, angle_lib, context, suggested_name)
        plt.close(disp_fig)

        # 2) export figure rendered at exactly EXPORT_SIZE_IN inches
        exp_fig, exp_ax = make_3d_fig(EXPORT_SIZE_IN, full_bleed=True)
        plot_subplot_3d(exp_ax, spec, conditions, cond_projections, time_centers)
        set_view_angle(exp_ax, angle)

        out_base = os.path.join(save_dir, f"{fig_name}__{_slug(sub_title)}")
        # bbox_inches=None preserves the exact figure size in the output.
        exp_fig.savefig(out_base + ".svg", bbox_inches=None, pad_inches=0, transparent=True)
        exp_fig.savefig(out_base + ".png", dpi=DPI, bbox_inches=None, pad_inches=0, transparent=True)
        print(f"  Saved (1x1 in): {os.path.basename(out_base)}.svg / .png")
        plt.close(exp_fig)

print(f"\nDone! All outputs saved to: {save_dir}")