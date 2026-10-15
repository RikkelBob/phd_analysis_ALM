# Grab motion energy data computed using Facemap and generate datasets/figures for relevant conditions
# Set load_preprocessed_data to True to only redo plotting, False to redo data loading from Facemap files

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from neural_analysis import utils
import pandas as pd
from neural_analysis.motion.tools.utils_motion import extract_frames, sum_motion_3D, causal_gaussian_kernel, bad_motion_sessions
from neural_analysis.plot_utils import paper_colors
import matplotlib
import config

load_preprocessed_data = True

# ── Font & style configuration ──────────────────────────────────────
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})

MM_TO_INCH = 1 / 25.4


def plot_session_heatmap(motion_dict_per_ses, condition_name, save_path, timeline,
                         label_to_plot=None, legend=None, z_score=True,
                         baseline_window=(-1, -0.1), data_window=(-1, 1.5),
                         cmap='viridis', vmin=None, vmax=None, symmetric_clim=False,
                         fig_width_mm=120, fig_height_mm=60,
                         margin_left_mm=25, margin_bottom_mm=12,
                         margin_right_mm=12, margin_top_mm=10,
                         colorbar_width_mm=3, colorbar_gap_mm=2, y_ticks=False):
    """
    Plot mean motion per session as rows in a heatmap.

    Parameters
    ----------
    motion_dict_per_ses : dict
        {condition_name -> {label -> {session_ID -> trials (n_trials, n_timepoints)}}}
    condition_name : str
        Which condition to plot.
    save_path : str
        File path without extension.
    timeline : np.ndarray
        Time axis.
    label_to_plot : int or None
        Which label to plot. If None, averages across all labels.
    legend : list of str or None
        Label names for the title.
    z_score : bool
        Whether to z-score each session's mean using the baseline window.
    baseline_window : tuple
        (start, end) in seconds for z-scoring baseline.
    data_window : tuple
        (start, end) in seconds for the full data window.
    cmap : str or matplotlib.colors.Colormap
        Colormap for the heatmap (e.g. 'viridis', 'RdBu_r', 'inferno').
    vmin, vmax : float or None
        Color limits. If None, derived from data.
    symmetric_clim : bool
        If True and vmin/vmax are None, set limits symmetric around zero.
    fig_width_mm, fig_height_mm : float
        Exact figure dimensions in millimeters.
    margin_left_mm, margin_bottom_mm, margin_right_mm, margin_top_mm : float
        Margins in mm.
    colorbar_width_mm : float
        Width of the colorbar in mm.
    colorbar_gap_mm : float
        Gap between heatmap and colorbar in mm.
    """
    condition_data = motion_dict_per_ses.get(condition_name, {})
    if not condition_data:
        print(f"[Warning] No data for condition '{condition_name}', skipping heatmap.")
        return

    # Collect session means
    session_ids = []
    session_means = []

    labels_to_use = [label_to_plot] if label_to_plot is not None else sorted(condition_data.keys())

    for label in labels_to_use:
        if label not in condition_data:
            continue
        for ses_id, trials in condition_data[label].items():
            trials = np.array(trials)
            if trials.ndim == 1:
                trials = trials[np.newaxis, :]

            if z_score:
                trials = z_score_motion(trials, baseline_window=baseline_window, data_window=data_window)

            ses_mean = np.mean(trials, axis=0)

            if ses_id not in session_ids:
                session_ids.append(ses_id)
                session_means.append(ses_mean)

    if not session_means:
        print(f"[Warning] No session data for condition '{condition_name}', skipping heatmap.")
        return

    heatmap_data = np.stack(session_means, axis=0)

    # Sort by peak time for visual clarity
    peak_indices = np.argmax(heatmap_data, axis=1)
    sort_order = np.argsort(peak_indices)
    heatmap_data = heatmap_data[sort_order]
    session_ids = [session_ids[i] for i in sort_order]

    # Color limits
    if vmin is None or vmax is None:
        if symmetric_clim:
            abs_max = np.max(np.abs(heatmap_data))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = vmin if vmin is not None else np.min(heatmap_data)
            vmax = vmax if vmax is not None else np.max(heatmap_data)

    # Build figure
    fig = make_figure_mm(fig_width_mm, fig_height_mm)

    heatmap_w_mm = fig_width_mm - margin_left_mm - margin_right_mm - colorbar_width_mm - colorbar_gap_mm
    heatmap_h_mm = fig_height_mm - margin_bottom_mm - margin_top_mm

    ax = make_axes_mm(fig, margin_left_mm, margin_bottom_mm, heatmap_w_mm, heatmap_h_mm)
    cbar_ax = make_axes_mm(fig, margin_left_mm + heatmap_w_mm + colorbar_gap_mm,
                           margin_bottom_mm, colorbar_width_mm, heatmap_h_mm)

    # Plot heatmap
    extent = [timeline[0], timeline[-1], heatmap_data.shape[0] - 0.5, -0.5]
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation='nearest')

    ax.axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=0.8)

    ax.set_xlabel('Time from stimulus (s)')
    ax.set_ylabel('Session')
    if y_ticks:
        ax.set_yticks(range(len(session_ids)))
        ax.set_yticklabels(session_ids, fontsize=4)

    label_str = ''
    if label_to_plot is not None and legend is not None:
        idx = int(label_to_plot) - 1
        if 0 <= idx < len(legend):
            label_str = f' — {legend[idx]}'
    ax.set_title(f'Session heatmap — {condition_name}{label_str}', fontweight='bold')
    ax.grid(False)

    # Colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-scored motion' if z_score else 'Motion energy (a.u.)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.savefig(save_path + ".png", dpi=1000)
    fig.savefig(save_path + ".svg", dpi=1000)
    plt.close(fig)
    print(f"  Saved: {save_path}.png and .svg")


def mm_to_inch(mm):
    """Convert millimeters to inches."""
    return mm * MM_TO_INCH


def make_figure_mm(width_mm, height_mm, **kwargs):
    """
    Create a matplotlib figure with exact dimensions in millimeters.

    Parameters
    ----------
    width_mm : float
        Figure width in millimeters.
    height_mm : float
        Figure height in millimeters.
    **kwargs
        Additional keyword arguments passed to plt.figure().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    return plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)), **kwargs)


def make_axes_mm(fig, left_mm, bottom_mm, width_mm, height_mm):
    """
    Add an axes to a figure at an exact position and size in millimeters.
    Origin is bottom-left of the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    left_mm, bottom_mm : float
        Position of the axes bottom-left corner in mm from figure bottom-left.
    width_mm, height_mm : float
        Axes width and height in mm.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    fig_w_mm, fig_h_mm = fig_size_mm(fig)
    left   = left_mm   / fig_w_mm
    bottom = bottom_mm / fig_h_mm
    width  = width_mm  / fig_w_mm
    height = height_mm / fig_h_mm
    return fig.add_axes([left, bottom, width, height])


def make_gridspec_mm(fig, nrows, ncols,
                     left_mm, bottom_mm, width_mm, height_mm,
                     hspace_mm=2, wspace_mm=2,
                     row_heights=None, col_widths=None):
    """
    Create a GridSpec anchored at an exact position and size in millimeters.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    nrows, ncols : int
    left_mm, bottom_mm : float
        Position of the grid bottom-left corner in mm.
    width_mm, height_mm : float
        Total grid width and height in mm.
    hspace_mm : float
        Vertical space between rows in mm.
    wspace_mm : float
        Horizontal space between columns in mm.
    row_heights : list of float or None
        Relative row heights (e.g. [2, 1]). Defaults to equal.
    col_widths : list of float or None
        Relative column widths. Defaults to equal.

    Returns
    -------
    gs : matplotlib.gridspec.GridSpec
    """
    fig_w_mm, fig_h_mm = fig_size_mm(fig)

    left   = left_mm   / fig_w_mm
    bottom = bottom_mm / fig_h_mm
    right  = (left_mm + width_mm)  / fig_w_mm
    top    = (bottom_mm + height_mm) / fig_h_mm

    # Convert absolute mm spacing to GridSpec's fractional wspace/hspace.
    # GridSpec hspace is relative to the average subplot height.
    avg_subplot_h_mm = (height_mm - hspace_mm * (nrows - 1)) / nrows
    avg_subplot_w_mm = (width_mm  - wspace_mm * (ncols - 1)) / ncols
    hspace = hspace_mm / avg_subplot_h_mm if nrows > 1 else 0
    wspace = wspace_mm / avg_subplot_w_mm if ncols > 1 else 0

    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        left=left, right=right,
        bottom=bottom, top=top,
        hspace=hspace, wspace=wspace,
        height_ratios=row_heights,
        width_ratios=col_widths,
    )
    return gs


def fig_size_mm(fig):
    """Return (width_mm, height_mm) of a figure."""
    w_in, h_in = fig.get_size_inches()
    return w_in / MM_TO_INCH, h_in / MM_TO_INCH


# ── Layout configuration ─────────────────────────────────────────────
class FigureLayoutConfig:
    """
    Central place to define exact figure and subplot dimensions in mm.
    Edit these values to change the layout.
    """
    # ── Grand-average figure ──────────────────────────────────────────
    GA_FIG_W_MM        = 70   # total figure width
    GA_FIG_H_MM        = 50    # total figure height

    # Margins (mm from figure edge to the plot area)
    GA_MARGIN_LEFT_MM   = 18
    GA_MARGIN_BOTTOM_MM = 12
    GA_MARGIN_RIGHT_MM  = 6
    GA_MARGIN_TOP_MM    = 10

    GA_HSPACE_MM        = 6    # vertical gap between rows (mm)
    GA_WSPACE_MM        = 8    # horizontal gap between columns (mm)

    # ── Per-session figure ────────────────────────────────────────────
    SES_FIG_W_MM        = 70   # match GA width
    SES_FIG_H_MM        = 80   # per condition; scaled in code

    SES_MARGIN_LEFT_MM   = 18  # match GA margins
    SES_MARGIN_BOTTOM_MM = 12
    SES_MARGIN_RIGHT_MM  = 6
    SES_MARGIN_TOP_MM    = 10

    SES_HSPACE_MM        = 8
    SES_ROW_RATIO        = [2, 1]  # mean-trace row : snake-plot row

    # ── Heatmap / snake plot settings ─────────────────────────────────
    HEATMAP_CMAP         = 'inferno'
    HEATMAP_SYMMETRIC    = True
    HEATMAP_VMIN         = -1
    HEATMAP_VMAX         = 5
    COLORBAR_WIDTH_MM    = 3
    COLORBAR_GAP_MM      = 2


def get_condition_colors(condition_name):
    """
    Return consistent per-label colors for a condition,
    matching the colors used in grand average and spaghetti plots.
    """
    if condition_name == 'au':
        return [paper_colors('aud_hit'), 'grey']
    else:
        return [paper_colors('vis_hit'), 'grey']


def plot_spaghetti(motion_dict_per_ses, condition_name, save_path, timeline,
                   legend=None, label_ignore=None,
                   layout: FigureLayoutConfig = None, z_score=True, colors=None, plot_mean=False):
    """
    Plot one semi-transparent line per session per label (spaghetti plot).

    Parameters
    ----------
    motion_dict_per_ses : dict
        {condition_name -> {label -> {session_ID -> trials (n_trials, n_timepoints)}}}
    condition_name : str
        Which condition to plot (e.g. 'vis', 'au').
    save_path : str
        File path without extension.
    timeline : np.ndarray
        Time axis.
    legend : list of str or None
        Label names in label order.
    label_ignore : list or None
        Labels to skip.
    layout : FigureLayoutConfig or None
    """
    if label_ignore is None:
        label_ignore = []
    if layout is None:
        layout = FigureLayoutConfig()

    lyt = layout
    condition_data = motion_dict_per_ses.get(condition_name, {})
    if not condition_data:
        print(f"[Warning] No data for condition '{condition_name}', skipping spaghetti plot.")
        return

    fig = make_figure_mm(lyt.GA_FIG_W_MM, lyt.GA_FIG_H_MM)

    plot_w_mm = lyt.GA_FIG_W_MM - lyt.GA_MARGIN_LEFT_MM - lyt.GA_MARGIN_RIGHT_MM
    plot_h_mm = lyt.GA_FIG_H_MM - lyt.GA_MARGIN_BOTTOM_MM - lyt.GA_MARGIN_TOP_MM

    gs = make_gridspec_mm(
        fig, nrows=1, ncols=1,
        left_mm=lyt.GA_MARGIN_LEFT_MM,
        bottom_mm=lyt.GA_MARGIN_BOTTOM_MM,
        width_mm=plot_w_mm,
        height_mm=plot_h_mm,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(False)

    if colors is None:
        colors = get_condition_colors(condition_name)

    color_idx = 0
    for i, (label, sessions) in enumerate(condition_data.items()):

        if label in label_ignore:
            continue

        label_name = legend[i] if legend and i < len(legend) else f'Label {label}'
        color = colors[color_idx]
        color_idx += 1
        session_means = []
        first = True
        for ses_id, trials in sessions.items():
            trials = np.array(trials)
            if trials.ndim == 1:
                trials = trials[np.newaxis, :]
            ses_mean = np.mean(trials, axis=0)

            if z_score:
                ses_mean = np.mean(z_score_motion(trials, baseline_window=(-1, -0.1), data_window=(-1, 1.5)), axis=0)

            session_means.append(ses_mean)

            # One line per session; only label the first for a clean legend
            ax.plot(timeline, ses_mean,
                    color=color, alpha=0.3, linewidth=0.8,
                    label=label_name if first else '_nolegend_')
            first = False

        # Grand mean on top as a bold line
        if session_means and plot_mean:
            grand_mean = np.mean(np.stack(session_means, axis=0), axis=0)
            ax.plot(timeline, grand_mean,
                    color=color, linewidth=1.5,
                    label=f'{label_name} mean (n={len(sessions)} ses)')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray',  linestyle='-',  alpha=0.3)
    ax.set_xlabel('Time from stimulus (s)')
    ax.set_ylabel('Motion Energy (a.u.)')
    ax.set_title(f'Per-session motion — {condition_name}', fontweight='bold')
    ax.legend(loc='best')

    fig.savefig(save_path + ".png", dpi=300)
    fig.savefig(save_path + ".svg")
    plt.close(fig)
    print(f"  Saved: {save_path}.png and .svg")



def z_score_motion(motion_data, baseline_window, data_window):
    """Z-score motion data using baseline window."""
    num_timepoints = motion_data.shape[1]
    time_axis = np.linspace(data_window[0], data_window[1], num_timepoints)
    baseline_indices = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    baseline_data = motion_data[:, baseline_indices]
    return (motion_data - np.mean(baseline_data)) / np.std(baseline_data)


def plot_grand_average(motion_data_dict, title, save_path, timeline, sigma=1,
                       per_label=True, z_score=False, label_ignore=None, legend=None,
                       window=(-1, 1.5), baseline_window=None,
                       layout: FigureLayoutConfig = None, error_type='std',
                       colors=None):
    """
    Plot the grand-average motion data using exact mm dimensions.

    Parameters
    ----------
    layout : FigureLayoutConfig or None
        If None, a default FigureLayoutConfig() is used.
    colors : list or None
        Per-label colors. If None, uses get_condition_colors.
    (all other parameters unchanged from original)
    """
    if baseline_window is None:
        baseline_window = [-1, -0.2]
    if label_ignore is None:
        label_ignore = []
    if layout is None:
        layout = FigureLayoutConfig()

    lyt = layout  # shorthand

    # ── Build figure & axes with mm precision ────────────────────────
    fig = make_figure_mm(lyt.GA_FIG_W_MM, lyt.GA_FIG_H_MM)

    plot_w_mm = lyt.GA_FIG_W_MM - lyt.GA_MARGIN_LEFT_MM - lyt.GA_MARGIN_RIGHT_MM
    plot_h_mm = lyt.GA_FIG_H_MM - lyt.GA_MARGIN_BOTTOM_MM - lyt.GA_MARGIN_TOP_MM

    gs = make_gridspec_mm(
        fig, nrows=1, ncols=1,
        left_mm=lyt.GA_MARGIN_LEFT_MM,
        bottom_mm=lyt.GA_MARGIN_BOTTOM_MM,
        width_mm=plot_w_mm,
        height_mm=plot_h_mm,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(False)

    # ── Plot ─────────────────────────────────────────────────────────
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.5, 10))

    if per_label:
        for i, (label, traces) in enumerate(motion_data_dict.items()):
            if label in label_ignore:
                continue
            traces = np.array(traces)
            mean_data = np.mean(traces, axis=0)

            if error_type == 'sem':
                error_data  = np.std(traces, axis=0) / np.sqrt(traces.shape[0])
            elif error_type == 'std':
                error_data = np.std(traces, axis=0)
            else:
                print(f"ERROR: Incorrect errorbar type {error_type}")

            label_name = legend[i] if legend and i < len(legend) else f'Label {label}'
            ax.plot(timeline, mean_data,
                    label=f'{label_name} (n={traces.shape[0]})',
                    color=colors[i], linewidth=1)
            ax.fill_between(timeline,
                            mean_data - error_data,
                            mean_data + error_data,
                            alpha=0.3, color=colors[i], linewidth=0)
    else:
        all_traces = []
        for label, traces in motion_data_dict.items():
            if label not in label_ignore:
                all_traces.extend(traces)
        if not all_traces:
            print(f"[Warning] No data to plot for: {title}")
            plt.close(fig)
            return
        all_traces = np.array(all_traces)
        mean_data = np.mean(all_traces, axis=0)

        if error_type == 'sem':
            error_data = np.std(traces, axis=0) / np.sqrt(traces.shape[0])
        elif error_type == 'std':
            error_data = np.std(traces, axis=0)
        else:
            print(f"ERROR: Incorrect errorbar type {error_type}")

        ax.plot(timeline, mean_data,
                label=f'Grand Average (n={all_traces.shape[0]})', linewidth=1)
        ax.fill_between(timeline,
                        mean_data - error_data,
                        mean_data + error_data,
                        alpha=0.3, label=f'±{error_type}', linewidth=0)

    ax.set_xlabel('Time from stimulus (s)')
    ax.set_ylabel('Motion Energy (a.u.)' + (' (z-scored)' if z_score else ''))
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(False)

    fig.savefig(save_path + ".png", dpi=300)
    fig.savefig(save_path + ".svg")
    plt.close(fig)
    print(f"  Saved: {save_path}.png and .svg")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class MotionAnalysisConfig:
    """Configuration class for motion analysis parameters."""

    def __init__(self):
        self.root_path   = config.DATA_DIR_PP / "motion/facemap"
        self.data_path   = config.DATA_DIR_EXP
        self.output_dir_pp  = config.DATA_DIR_PP_NEW / "motion/"
        self.figure_dir = config.FIGURE_DIR / "motion_energy"
        
        self.experiments     = ['ALM_2', 'ALM_3']
        self.window          = (-1, 1.5)
        self.baseline_window = [-1, -0.2]
        self.smooth_sigma    = 0
        self.n_components    = 50
        self.n_min_trials    = 10
        self.plot_per_session = True
        self.num_timepoints  = 200

        self.layout = FigureLayoutConfig()   # ← mm layout lives here
        self.conditions = self.define_conditions()

    def define_conditions(self):
        return {
            'vis': {
                'trial_type': ['X'],
                'labels': {
                    1: lambda trial_data: trial_data['correctResponse'] == 1,
                    2: lambda trial_data: trial_data['noResponse'] == 1,
                },
                'legend': ['Visual hit', 'Visual miss'],
                'label_ignore': []
            },
            'au': {
                'trial_type': ['Y'],
                'labels': {
                    1: lambda trial_data: trial_data['correctResponse'] == 1,
                    2: lambda trial_data: trial_data['noResponse'] == 1,
                },
                'legend': ['Auditory hit', 'Auditory miss'],
                'label_ignore': []
            },

            'probe': {
                'trial_type': ['P'],
                'labels': {
                    1: lambda trial_data: trial_data['noResponse'] == 0,
                    2: lambda trial_data: trial_data['noResponse'] == 1,
                },
                'legend': ['False alarm', 'Correct rejection'],
                'label_ignore': []
            }
        }


def process_session_data(session, sessionData, trialData, motion_config, motion_dict_per_ses,
                         motion_type='motion', smooth_motion=True):
    session_folder   = os.path.basename(session['session'])
    sessionData_ses  = sessionData[sessionData['folderName'] == session_folder]
    if sessionData_ses.empty:
        return None

    sampling_interval = 1/25 if (sessionData_ses['experiment'] == 'ALM_1').iloc[0] else 1/80
    session_ID        = sessionData_ses['session_ID'].item()
    trialData_ses     = trialData[trialData['session_ID'] == session_ID]

    session_plot_data = {}

    for condition_name, condition_params in motion_config.conditions.items():
        trial_type  = condition_params['trial_type']

        trialData_cond = trialData_ses[
            trialData_ses['trialType'].isin(trial_type)
        ]
        if trialData_cond.empty:
            continue

        motion_data = session[motion_type][1]
        if smooth_motion:
            smooth_kernel = causal_gaussian_kernel(0.1, 1/sampling_interval)
            motion_data = np.convolve(motion_data, smooth_kernel, mode='full')[:len(motion_data)]

        trial_aligned_motion, good_trials = extract_frames(
            session['timeline'], motion_data,
            trialData_cond['stimChange'],
            motion_config.window, motion_config.n_components, sampling_interval
        )
        trialData_cond = trialData_cond[good_trials]

        labels = np.zeros(len(trialData_cond))
        for label_value, label_func in condition_params['labels'].items():
            labels[label_func(trialData_cond)] = label_value

        valid_trials = labels > 0
        if not np.any(valid_trials):
            print(f"Warning: No valid trials for {condition_name} in session {session_ID}")
            continue

        # Check minimum trial count per label
        skip_condition = False
        for label_value in condition_params['labels'].keys():
            n_label = np.sum(labels[valid_trials] == label_value)
            if n_label < motion_config.n_min_trials:
                print(f"Skipping {condition_name} in session {session_ID}: "
                      f"label {label_value} has {n_label} trials (min={motion_config.n_min_trials})")
                skip_condition = True
                break
        if skip_condition:
            continue

        if motion_type == 'motSVD':
            trial_aligned_motion = sum_motion_3D(trial_aligned_motion)[valid_trials]
        else:
            trial_aligned_motion = trial_aligned_motion[valid_trials]
        labels = labels[valid_trials]

        session_plot_data[condition_name] = {
            'trial_aligned_motion': trial_aligned_motion,
            'labels': labels,
            'condition_params': condition_params,
        }

        for label in np.unique(labels):
            if label not in motion_dict_per_ses[condition_name]:
                motion_dict_per_ses[condition_name][label] = {}

            motion_dict_per_ses[condition_name][label][session_ID] = trial_aligned_motion[labels == label]

    if motion_config.plot_per_session and session_plot_data:
        plot_session_data_combined(session_plot_data, session_ID, motion_config)

    return session_plot_data


def plot_session_data_combined(session_plot_data, session_ID, motion_config):
    """
    Plot per-session data using exact mm dimensions, matching the style
    of grand average (trace) and heatmap (snake) plots.
    All x-axes are shared across conditions and plot types.
    Trace plot heights match the grand-average plot height.
    Y-limits are shared across all trace (mean±error) plots.
    """
    lyt          = motion_config.layout
    time_axis    = np.linspace(motion_config.window[0], motion_config.window[1], motion_config.num_timepoints)
    n_conditions = len(session_plot_data)

    # ── Compute trace row height to match grand-average plot height ───
    # GA plot area height in mm (what the GA axes actually occupy)
    ga_plot_h_mm = lyt.GA_FIG_H_MM - lyt.GA_MARGIN_BOTTOM_MM - lyt.GA_MARGIN_TOP_MM
    # Snake row height derived from the GA trace height and the row ratio
    trace_ratio, snake_ratio = lyt.SES_ROW_RATIO
    snake_h_mm = ga_plot_h_mm * snake_ratio / trace_ratio

    # Total plot area: n_conditions * (trace + snake) + gaps between them
    n_rows = n_conditions * 2
    total_hspace_mm = lyt.SES_HSPACE_MM * (n_rows - 1)
    plot_h_mm = n_conditions * (ga_plot_h_mm + snake_h_mm) + total_hspace_mm

    # Total figure height
    fig_h_mm = plot_h_mm + lyt.SES_MARGIN_BOTTOM_MM + lyt.SES_MARGIN_TOP_MM
    fig = make_figure_mm(lyt.SES_FIG_W_MM, fig_h_mm)

    # Account for colorbar in the horizontal layout (matching heatmap style)
    plot_w_mm = (lyt.SES_FIG_W_MM - lyt.SES_MARGIN_LEFT_MM - lyt.SES_MARGIN_RIGHT_MM
                 - lyt.COLORBAR_WIDTH_MM - lyt.COLORBAR_GAP_MM)

    # Row heights in mm: [ga_plot_h, snake_h, ga_plot_h, snake_h, ...]
    row_heights_mm = [ga_plot_h_mm, snake_h_mm] * n_conditions

    gs = make_gridspec_mm(
        fig,
        nrows=n_rows, ncols=1,
        left_mm=lyt.SES_MARGIN_LEFT_MM,
        bottom_mm=lyt.SES_MARGIN_BOTTOM_MM,
        width_mm=plot_w_mm,
        height_mm=plot_h_mm,
        hspace_mm=lyt.SES_HSPACE_MM,
        row_heights=row_heights_mm,
    )

    # ── Compute shared snake-plot color limits across all conditions ──
    all_snake_data = []
    for data in session_plot_data.values():
        tam = data['trial_aligned_motion']
        sd = np.mean(tam, axis=2) if np.ndim(tam) == 3 else tam
        all_snake_data.append(sd)
    all_snake_data = np.concatenate(all_snake_data, axis=0)
    vmin = np.min(all_snake_data)
    vmax = np.max(all_snake_data)

    # ── Pre-compute shared y-limits for all trace (mean±std) plots ────
    trace_ymin = np.inf
    trace_ymax = -np.inf
    for data in session_plot_data.values():
        tam = data['trial_aligned_motion']
        labels = data['labels']
        for label in np.unique(labels):
            label_mask = labels == label
            mean_data = np.mean(tam[label_mask], axis=0)
            std_data  = np.std(tam[label_mask], axis=0)
            trace_ymin = min(trace_ymin, np.min(mean_data - std_data))
            trace_ymax = max(trace_ymax, np.max(mean_data + std_data))
    # Add a small margin (5%) for visual breathing room
    trace_y_range = trace_ymax - trace_ymin
    trace_ymin -= 0.05 * trace_y_range
    trace_ymax += 0.05 * trace_y_range

    # ── Track axes for shared x-axis ─────────────────────────────────
    all_axes = []
    trace_axes = []
    ax_first = None

    for cond_idx, (condition_name, data) in enumerate(session_plot_data.items()):
        # Create axes with shared x-axis
        if ax_first is None:
            ax_trace = fig.add_subplot(gs[cond_idx * 2, 0])
            ax_first = ax_trace
        else:
            ax_trace = fig.add_subplot(gs[cond_idx * 2, 0], sharex=ax_first)
        ax_snake = fig.add_subplot(gs[cond_idx * 2 + 1, 0], sharex=ax_first)
        all_axes.extend([ax_trace, ax_snake])
        trace_axes.append(ax_trace)

        trial_aligned_motion = data['trial_aligned_motion']
        labels = data['labels']
        condition_params = data['condition_params']

        # ── Use consistent condition colors ──────────────────────────
        colors = get_condition_colors(condition_name)

        for i, label in enumerate(np.unique(labels)):
            label_mask   = labels == label
            mean_data    = np.mean(trial_aligned_motion[label_mask], axis=0)
            std_data     = np.std(trial_aligned_motion[label_mask], axis=0)
            legend_labels = condition_params.get('legend', [])
            legend_label  = legend_labels[int(label) - 1] if len(legend_labels) >= label else f'Label {label}'
            color = colors[i] if i < len(colors) else 'grey'

            ax_trace.plot(time_axis, mean_data, color=color,
                          label=f'{legend_label} (n={np.sum(label_mask)})',
                          linewidth=1)
            ax_trace.fill_between(time_axis,
                                  mean_data - std_data,
                                  mean_data + std_data,
                                  alpha=0.3, color=color, linewidth=0)

        # ── Match grand-average trace styling ────────────────────────
        ax_trace.set_title(f'{session_ID} — {condition_name}', fontweight='bold')
        ax_trace.set_ylabel('Motion Energy (a.u.)')
        ax_trace.legend(loc='best')
        ax_trace.grid(False)
        ax_trace.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax_trace.axhline(y=0, color='gray',  linestyle='-',  alpha=0.3)

        # ── Snake plot matching heatmap style ────────────────────────
        plot_snake_motion(trial_aligned_motion, labels=labels, ax=ax_snake,
                          time_axis=time_axis, motion_config=motion_config, vmin=vmin, vmax=vmax)
        ax_snake.set_title(f'Trial-by-trial motion — {condition_name}', fontweight='bold')

        # ── Add colorbar for the snake plot (matching heatmap layout) ─
        snake_pos = ax_snake.get_position()
        fig_w_in, fig_h_in = fig.get_size_inches()
        fig_w_mm_actual, fig_h_mm_actual = fig_w_in / MM_TO_INCH, fig_h_in / MM_TO_INCH

        cbar_left = (lyt.SES_MARGIN_LEFT_MM + plot_w_mm + lyt.COLORBAR_GAP_MM) / fig_w_mm_actual
        cbar_width = lyt.COLORBAR_WIDTH_MM / fig_w_mm_actual
        cbar_ax = fig.add_axes([cbar_left, snake_pos.y0, cbar_width, snake_pos.height])

        # Re-create the imshow to grab the mappable for the colorbar
        im = ax_snake.images[0] if ax_snake.images else None
        if im is not None:
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Motion energy (a.u.)', fontsize=6)
            cbar.ax.tick_params(labelsize=5)

    # ── Apply shared y-limits to all trace axes ──────────────────────
    for ax in trace_axes:
        ax.set_ylim(trace_ymin, trace_ymax)

    # ── Hide x-tick labels on all but the bottom axis ────────────────
    for ax in all_axes[:-1]:
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=False)
    # Set x-label only on the bottom-most axis
    all_axes[-1].set_xlabel('Time from stimulus (s)')

    motion_dir = os.path.join(motion_config.output_dir, "session_plots")
    ensure_dir(os.path.join(motion_dir, 'png'))
    ensure_dir(os.path.join(motion_dir, 'svg'))
    fig.savefig(os.path.join(motion_dir, 'png', f'{session_ID}.png'), dpi=1000)
    fig.savefig(os.path.join(motion_dir, 'svg', f'{session_ID}.svg'), dpi=1000)
    plt.close(fig)


def plot_snake_motion(trial_aligned_motion, labels=None, ax=None,
                      time_axis=None, motion_config=None, vmin=None, vmax=None):
    """
    Snake plot of trial-by-trial motion, styled to match session heatmaps.

    Parameters
    ----------
    trial_aligned_motion : np.ndarray
        (n_trials, n_timepoints) or (n_trials, n_timepoints, n_components).
    labels : array-like or None
        Trial labels for sorting and boundary lines.
    ax : matplotlib.axes.Axes or None
    time_axis : np.ndarray or None
        Time axis for x-extent. If None, uses timepoint indices.
    motion_config : MotionAnalysisConfig or None
        Used to pull heatmap colormap/clim settings from layout.
    """
    if ax is None:
        ax = plt.gca()

    data = np.mean(trial_aligned_motion, axis=2) if np.ndim(trial_aligned_motion) == 3 else trial_aligned_motion

    if labels is not None:
        if len(labels) != trial_aligned_motion.shape[0]:
            print("Number of labels is not equal to number of trials")
            return
        if hasattr(labels, 'iloc'):
            sorted_indices = np.argsort(labels.values)
            sorted_labels  = labels.iloc[sorted_indices]
        else:
            sorted_indices = np.argsort(labels)
            sorted_labels  = labels[sorted_indices]
        data = data[sorted_indices]
        _, label_change_indices = np.unique(sorted_labels, return_index=True)

    # ── Use heatmap-consistent colormap and extent ───────────────────
    lyt = motion_config.layout if motion_config is not None else FigureLayoutConfig()
    cmap = lyt.HEATMAP_CMAP

    if time_axis is not None:
        extent = [time_axis[0], time_axis[-1], data.shape[0] - 0.5, -0.5]
    else:
        extent = None

    ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest',
              extent=extent, vmin=vmin, vmax=vmax)

    if labels is not None:
        for idx in label_change_indices[1:]:
            ax.axhline(y=idx - 0.5, color='white', linestyle='--', linewidth=0.8)

    # ── Match heatmap axis styling ───────────────────────────────────
    if time_axis is not None:
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=0.8)
        ax.set_xlabel('Time from stimulus (s)')
    else:
        ax.set_xlabel('Timepoints')
    ax.set_ylabel('Trials (sorted by label)')
    ax.grid(False)

    return ax


def compute_grand_averages(motion_data_per_ses, motion_config):
    grand_average   = {}
    grand_average_z = {}

    for condition_name in motion_data_per_ses:
        grand_average[condition_name]   = {}
        grand_average_z[condition_name] = {}

        for label in motion_data_per_ses[condition_name]:
            all_trials, all_trials_z = [], []
            print(f"Processing {condition_name} - Label: {label}")
            for ses_id, ses_data in motion_data_per_ses[condition_name][label].items():
                all_trials.append(ses_data)
                all_trials_z.append(z_score_motion(ses_data, motion_config.baseline_window, motion_config.window))
            if all_trials:
                grand_average[condition_name][label]   = np.concatenate(all_trials,   axis=0)
                grand_average_z[condition_name][label] = np.concatenate(all_trials_z, axis=0)

    return grand_average, grand_average_z


def main():
    motion_config = MotionAnalysisConfig()

    ses_exclude = bad_motion_sessions()

    if load_preprocessed_data:
        data_folder = config.resolve_pp('motion', 'pp_for_plotting')
        motion_dict_per_ses = np.load(os.path.join(data_folder, 'motion_dict_per_ses.npy'), allow_pickle=True).item()
        grand_average = np.load(os.path.join(data_folder, 'grand_average.npy'), allow_pickle=True).item()
        grand_average_z = np.load(os.path.join(data_folder, 'grand_average_z.npy'), allow_pickle=True).item()
    else:
        print("Loading data...")
        trialData, sessionData, _ = utils.load_data(motion_config.data_path, motion_config.experiments)
        session_paths = [s for s in utils.get_session_paths(motion_config.root_path)
                         if any(sub in s for sub in motion_config.experiments)]

        # Only include sessions with kfolds trials for every condition
        valid_session_IDs = sessionData[~sessionData['session_ID'].isin(ses_exclude)]['session_ID']
        valid_session_folders = sessionData[sessionData['session_ID'].isin(valid_session_IDs)]['folderName']

        session_paths = [full_path for full_path in session_paths
                               if any(valid_session_IDs in full_path for valid_session_IDs in valid_session_folders)]

        #session_paths = session_paths[0:2]

        trialData = trialData[trialData['hasphotostim'] == 0]

        motion_type = 'motion'
        print("Loading motion data...")

        if motion_type == 'motSVD':
            motion_data = utils.get_motSVD(session_paths, timeline_file="realigned_timeline_nlx_ts.npy")
        else:
            motion_data = utils.get_motion(session_paths, timeline_file="realigned_timeline_nlx_ts.npy")
            motion_config.n_components = 1

        motion_data = pd.DataFrame(motion_data)
        motion_dict_per_ses = {key: {} for key in motion_config.conditions}

        print("Processing sessions...")
        for idx_ses, session in motion_data.iterrows():
            print(f"Processing session {idx_ses + 1}/{len(motion_data)}")
            process_session_data(session, sessionData, trialData, motion_config, motion_dict_per_ses)

        print("Computing grand averages...")
        grand_average, grand_average_z = compute_grand_averages(motion_dict_per_ses, motion_config)

    print("Plotting grand averages...")
    time_axis = np.linspace(motion_config.window[0], motion_config.window[1], motion_config.num_timepoints)
    ga_dir    = os.path.join(motion_config.figure_dir, "grand_averages")
    ensure_dir(ga_dir)

    # Grand average plots
    for condition_name, data in grand_average_z.items():
        if not data:
            continue
        condition_params = motion_config.conditions[condition_name]
        colors = get_condition_colors(condition_name)
        plot_grand_average(
            data,
            f"Grand Average (Z-scored) - {condition_name.replace('_', ' ').title()}",
            os.path.join(ga_dir, f"grand_avg_z_{condition_name}"),
            time_axis, motion_config.smooth_sigma,
            per_label=True, z_score=False,
            label_ignore=condition_params.get('label_ignore', []),
            legend=condition_params.get('legend'),
            window=motion_config.window,
            baseline_window=motion_config.baseline_window,
            layout=motion_config.layout,
            colors=colors,
        )

    for condition_name, data in grand_average.items():
        if not data:
            continue
        condition_params = motion_config.conditions[condition_name]
        colors = get_condition_colors(condition_name)
        plot_grand_average(
            data,
            f"Grand Average - {condition_name.replace('_', ' ').title()}",
            os.path.join(ga_dir, f"grand_avg_{condition_name}"),
            time_axis, motion_config.smooth_sigma,
            per_label=True, z_score=False,
            label_ignore=condition_params.get('label_ignore', []),
            legend=condition_params.get('legend'),
            window=motion_config.window,
            baseline_window=motion_config.baseline_window,
            layout=motion_config.layout,
            colors=colors,
        )

    spag_dir = os.path.join(motion_config.figure_dir, "spaghetti")
    ensure_dir(spag_dir)

    for condition_name, condition_params in motion_config.conditions.items():
        colors = get_condition_colors(condition_name)
        plot_spaghetti(
            motion_dict_per_ses,
            condition_name,
            save_path=os.path.join(spag_dir, f"spaghetti_{condition_name}"),
            timeline=time_axis,
            legend=condition_params.get('legend'),
            label_ignore=condition_params.get('label_ignore', []),
            layout=motion_config.layout,
            colors=colors,
        )

    # Heatmaps (single-ses)
    heatmap_dir = os.path.join(motion_config.figure_dir, "heatmaps")
    ensure_dir(heatmap_dir)

    for condition_name, condition_params in motion_config.conditions.items():
        for label in motion_dict_per_ses.get(condition_name, {}).keys():
            plot_session_heatmap(
                motion_dict_per_ses,
                condition_name,
                save_path=os.path.join(heatmap_dir, f"heatmap_{condition_name}_label{int(label)}"),
                timeline=time_axis,
                label_to_plot=label,
                legend=None,#condition_params.get('legend'),
                z_score=True,
                cmap=motion_config.layout.HEATMAP_CMAP,
                symmetric_clim=motion_config.layout.HEATMAP_SYMMETRIC,
                fig_width_mm=motion_config.layout.GA_FIG_W_MM,
                fig_height_mm=motion_config.layout.GA_FIG_H_MM,
                vmin=motion_config.layout.HEATMAP_VMIN,
                vmax=motion_config.layout.HEATMAP_VMAX,
            )

    print("Saving data...")
    data_save_path = os.path.join(motion_config.output_dir_pp, 'pp_for_plotting')
    ensure_dir(data_save_path)
    np.save(os.path.join(data_save_path, 'motion_dict_per_ses.npy'),  motion_dict_per_ses, allow_pickle=True)
    np.save(os.path.join(data_save_path, 'grand_average.npy'),        grand_average,       allow_pickle=True)
    np.save(os.path.join(data_save_path, 'grand_average_z.npy'),      grand_average_z,     allow_pickle=True)
    print("Analysis complete!")


if __name__ == "__main__":
    main()