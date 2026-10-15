"""
dPCA Plotting Script
====================
Generates all dPCA figures (except novel condition projections) from saved results. Can be called standalone to
replot without recalculating.

Usage:
    python dpca_plot.py --input_dir <path_to_results_directory>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import argparse
import os
import yaml
from pathlib import Path

mpl.rcParams['svg.fonttype'] = 'none'

MM_TO_INCH = 1 / 25.4


def plot_cumvar_per_marginalization(var_explained, area, n_neurons, n_comp_plot=5,
                                    subplot_size_mm=(12.7, 19.05)):
    """Cumulative variance per marginalization, normalized within each
    marginalization (each curve reaches 100% at its last component).

    Each subplot's DATA area (the axes box, excluding ticks/labels/title) is
    exactly subplot_size_mm — default (25.4, 19.05) mm = 1 x 0.75 inch.
    Components 1-10; one subplot per marginalization.
    """
    marg_keys = sorted(var_explained.keys())
    n_marg = len(marg_keys)
    n_cols = min(4, n_marg)
    n_rows = int(np.ceil(n_marg / n_cols))

    fig, axes = _make_mm_figure(
        n_rows=n_rows, n_cols=n_cols,
        subplot_w_mm=subplot_size_mm[0], subplot_h_mm=subplot_size_mm[1],
        spacing_h_mm=16, spacing_v_mm=18,
        margin_mm={'left': 12, 'right': 4, 'top': 14, 'bottom': 12},
    )

    for idx, marg_key in enumerate(marg_keys):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.patch.set_visible(False)   # no background fill behind the data

        vals = np.asarray(var_explained[marg_key], dtype=float)
        total = np.sum(vals)
        n_plot = min(n_comp_plot, len(vals))
        cumvar = np.cumsum(vals[:n_plot]) / total * 100 if total > 0 else np.zeros(n_plot)
        comps = np.arange(1, n_plot + 1)

        ax.plot(comps, cumvar, 'o-', color='steelblue',
                linewidth=1, markersize=3)

        marg_display = marg_names.get(marg_key, marg_key)
        ax.set_title(f"{marg_display}\n(total {total:.1f}% of overall)", fontsize=FONT_BIG)
        ax.set_xlabel('Component', fontsize=FONT_SMALL)
        ax.set_ylabel('Cumulative var (% of marg.)', fontsize=FONT_SMALL)
        ax.set_xticks(comps)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylim(45, 105)
        ax.axhline(95, color='grey', linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=FONT_SMALL)

    # Hide any unused grid cells
    for idx in range(n_marg, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    fig.suptitle(f"Cumulative Variance per Marginalization (normalized within marg.) — "
                 f"{', '.join(area)}, n={n_neurons} neurons",
                 fontsize=FONT_BIG, fontweight='bold')
    # NOTE: do NOT call plt.tight_layout() here — it would override the exact
    # axes sizes set by _make_mm_figure.
    return fig
def _find_significance_windows(mask_1d, time_centers, half_bin):
    """Return contiguous True runs in a 1-D boolean mask as time windows."""
    windows = []
    in_window = False
    start_idx = None
    for i, val in enumerate(mask_1d):
        if val and not in_window:
            in_window, start_idx = True, i
        elif not val and in_window:
            in_window = False
            windows.append((start_idx, i - 1))
    if in_window:
        windows.append((start_idx, len(mask_1d) - 1))

    return [
        {
            'start_time': round(float(time_centers[s] - half_bin), 4),
            'end_time': round(float(time_centers[e] + half_bin), 4),
            'start_bin': int(s),
            'end_bin': int(e),
            'n_bins': int(e - s + 1),
        }
        for s, e in windows
    ]


def save_crossval_yaml(masks, var_explained, time_centers, step_size,
                       n_shuffles, n_splits, output_dir):
    """Save significance windows per component per marginalization to YAML."""
    if step_size is not None:
        half_bin = step_size / 2
    else:
        half_bin = (time_centers[1] - time_centers[0]) / 2 if len(time_centers) > 1 else 0.005

    crossval = {
        'parameters': {
            'n_shuffles': int(n_shuffles),
            'n_splits': int(n_splits),
            'significance_threshold': round(float(1 / (n_shuffles + 1)), 4),
            'half_bin_s': round(float(half_bin), 4),
            'time_start': round(float(time_centers[0]), 4),
            'time_end': round(float(time_centers[-1]), 4),
        },
        'marginalizations': {}
    }

    for marg_key in sorted(masks.keys()):
        mask = masks[marg_key]  # (n_comp, n_time)
        comp_dict = {}
        for comp_idx in range(mask.shape[0]):
            mask_1d = mask[comp_idx]
            ve = var_explained[marg_key]
            comp_dict[f'comp_{comp_idx}'] = {
                'variance_explained': round(float(ve[comp_idx]), 4) if comp_idx < len(ve) else None,
                'fraction_significant': round(float(np.mean(mask_1d)), 4),
                'n_significant_bins': int(np.sum(mask_1d)),
                'n_windows': len(_find_significance_windows(mask_1d, time_centers, half_bin)),
                'significant_windows': _find_significance_windows(mask_1d, time_centers, half_bin),
            }
        crossval['marginalizations'][marg_key] = {
            'display_name': marg_names.get(marg_key, marg_key),
            'components': comp_dict,
        }

    out_path = os.path.join(output_dir, 'crossval_significance.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(crossval, f, default_flow_style=False, sort_keys=False)
    return out_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot dPCA results from saved data')
    parser.add_argument('--input_dir', type=str,
                        help='Path to directory containing dpca_results.npz and metadata.yaml')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')
    parser.add_argument('--svg', action='store_true', default=True,
                        help='Also save SVG versions (default: True)')
    parser.add_argument('--no_svg', action='store_true', default=False,
                        help='Disable SVG output')
    parser.add_argument('--go_axis', type=str, default=None,
                        help='Path to go_axis_results.npz to include Go axis in orthogonality plot')
    return parser.parse_args()


def load_results(input_dir):
    """Load all saved dPCA results and metadata."""
    # Load metadata
    with open(os.path.join(input_dir, 'metadata.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)

    # Load main results (allow_pickle needed for dict objects)
    results_path = os.path.join(input_dir, 'dpca_results.npz')
    data = np.load(results_path, allow_pickle=True)

    # Extract arrays and dicts (np.load wraps dicts in 0-d arrays)
    def unwrap(val):
        """Unwrap 0-d numpy arrays that contain dicts."""
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return val.item()
        return val

    results = {}
    for key in data.files:
        results[key] = unwrap(data[key])

    return results, metadata


# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

# Font sizes
FONT_SMALL = 6
FONT_BIG = 8

# Color scheme
stim_colors = {'X': 'blue', 'Y': 'red', 'P': 'green'}
stim_labels_plot = {'X': 'Visual', 'Y': 'Auditory', 'P': 'Probe'}
decision_colors = {'L': 'solid', 'R': 'dashed', 'N': 'dotted'}
decision_labels_plot = {'L': 'Left lick', 'R': 'Right lick', 'N': 'No lick'}

# Marginalization display names
marg_names = {
    't': 'Time (condition-independent)',
    's': 'Stimulus',
    'd': 'Decision (lick direction)',
    'st': 'Stimulus × Time',
    'dt': 'Decision × Time',
    'sd': 'Stimulus × Decision',
    'sdt': 'Stimulus × Decision × Time'
}


# ============================================================================
# HELPER: CREATE FIGURE WITH EXACT MM SUBPLOT SIZES
# ============================================================================

def _make_mm_figure(n_rows, n_cols, subplot_w_mm, subplot_h_mm,
                    spacing_h_mm=8, spacing_v_mm=8, margin_mm=None):
    """
    Create a figure with subplots at exact mm dimensions using manual axes placement.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    subplot_w_mm, subplot_h_mm : float
        Exact width and height of each subplot in mm.
    spacing_h_mm : float
        Horizontal gap between subplots in mm.
    spacing_v_mm : float
        Vertical gap between subplots in mm.
    margin_mm : dict, optional
        Figure margins in mm. Keys: 'left', 'right', 'top', 'bottom'.

    Returns
    -------
    fig : Figure
    axes : 2-D list of Axes, shape (n_rows, n_cols)
    """
    if margin_mm is None:
        margin_mm = {}
    ml = margin_mm.get('left', 10)
    mr = margin_mm.get('right', 2)
    mt = margin_mm.get('top', 8)
    mb = margin_mm.get('bottom', 10)

    fig_w_mm = ml + n_cols * subplot_w_mm + (n_cols - 1) * spacing_h_mm + mr
    fig_h_mm = mt + n_rows * subplot_h_mm + (n_rows - 1) * spacing_v_mm + mb

    fig = plt.figure(figsize=(fig_w_mm * MM_TO_INCH, fig_h_mm * MM_TO_INCH))

    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            left_mm = ml + c * (subplot_w_mm + spacing_h_mm)
            # rows go top-to-bottom, but figure coords go bottom-to-top
            bottom_mm = mb + (n_rows - 1 - r) * (subplot_h_mm + spacing_v_mm)
            ax = fig.add_axes([
                left_mm / fig_w_mm,
                bottom_mm / fig_h_mm,
                subplot_w_mm / fig_w_mm,
                subplot_h_mm / fig_h_mm,
            ])
            row_axes.append(ax)
        axes.append(row_axes)

    return fig, axes


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_components(Z, var_explained, time_centers, stimulus_types, decision_types,
                    n_components, area, n_neurons, bin_size, step_size):
    """Figure 1: Top components per marginalization."""
    n_comp_plot = min(3, n_components)
    marginalizations_to_plot = [k for k in sorted(Z.keys()) if k != 't']
    priority_order = ['st', 'dt', 'sd', 's', 'd', 'sdt']
    marginalizations_to_plot = [m for m in priority_order if m in Z] + \
                               [m for m in marginalizations_to_plot if m not in priority_order]

    n_marg = len(marginalizations_to_plot)
    fig, axes = plt.subplots(n_marg, n_comp_plot, figsize=(5 * n_comp_plot, 3.5 * n_marg),
                              squeeze=False)

    for row_idx, marg_key in enumerate(marginalizations_to_plot):
        for comp_idx in range(n_comp_plot):
            ax = axes[row_idx, comp_idx]

            for s_idx, s_type in enumerate(stimulus_types):
                for d_idx, d_type in enumerate(decision_types):
                    trajectory = Z[marg_key][comp_idx, s_idx, d_idx, :]
                    linestyle = decision_colors[d_type]

                    label = f"{stim_labels_plot[s_type]} - {decision_labels_plot[d_type]}" \
                        if comp_idx == 0 and row_idx == 0 else None

                    ax.plot(time_centers, trajectory,
                            color=stim_colors[s_type],
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label)

            ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.grid(True, alpha=0.2)

            marg_display = marg_names.get(marg_key, marg_key)
            var_str = f"{var_explained[marg_key][comp_idx]:.1f}%"
            ax.set_title(f"{marg_display}\nComp {comp_idx+1} ({var_str})", fontsize=FONT_BIG)

            if row_idx == n_marg - 1:
                ax.set_xlabel('Time from stimulus (s)', fontsize=FONT_SMALL)
            if comp_idx == 0:
                ax.set_ylabel('dPC (a.u.)', fontsize=FONT_SMALL)

    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', ncol=3, fontsize=FONT_SMALL,
               bbox_to_anchor=(0.5, 1.02))

    step_label = f"{step_size*1000:.0f}ms" if step_size is not None else "None"
    fig.suptitle(f"dPCA Components - {', '.join(area)}\n"
                 f"n={n_neurons} neurons, bin={bin_size*1000:.0f}ms, step={step_label}",
                 fontsize=FONT_BIG, fontweight='bold', y=1.06)

    plt.tight_layout()
    return fig


def plot_variance_pie(var_explained, area, n_neurons):
    """Figure 2: Variance explained pie chart."""
    fig_pie, ax_pie = plt.subplots(1, 1, figsize=(2, 2))

    marg_var = {}
    for key in sorted(var_explained.keys()):
        marg_var[marg_names.get(key, key)] = np.sum(var_explained[key])

    labels_pie = list(marg_var.keys())
    sizes = list(marg_var.values())
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))

    wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels_pie, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90, pctdistance=0.85)
    for text in autotexts:
        text.set_fontsize(FONT_SMALL)
    ax_pie.set_title(f"Variance Explained by Marginalization\n{', '.join(area)}, n={n_neurons} neurons",
                      fontsize=FONT_BIG, fontweight='bold')

    plt.tight_layout()
    return fig_pie


def plot_state_space(Z, var_explained, time_centers, stimulus_types, decision_types,
                     area, n_neurons, subplot_size_mm=(50, 40), spacing_mm=8,
                     margin_mm=None):
    """Figure 3: Top 2 components of stimulus and decision in 2D state space.

    Parameters
    ----------
    subplot_size_mm : tuple of (width_mm, height_mm)
        Exact size of each subplot in millimeters. Default: (65, 40).
    spacing_mm : float
        Horizontal spacing between subplots in mm.
    margin_mm : dict, optional
        Figure margins in mm. Keys: 'left', 'right', 'top', 'bottom'.
    """
    state_space_margs = [m for m in ['s', 'd', 'st', 'dt'] if m in Z][:2]
    n_plots = len(state_space_margs)

    fig, axes_grid = _make_mm_figure(
        n_rows=1, n_cols=n_plots,
        subplot_w_mm=subplot_size_mm[0], subplot_h_mm=subplot_size_mm[1],
        spacing_h_mm=spacing_mm,
        margin_mm=margin_mm,
    )
    axes_state = axes_grid[0]  # single row

    marker_edge = {'L': 'red', 'R': 'blue', 'N': 'black'}

    for plot_idx, marg_key in enumerate(state_space_margs):
        ax = axes_state[plot_idx]

        for s_idx, s_type in enumerate(stimulus_types):
            for d_idx, d_type in enumerate(decision_types):
                traj_x = Z[marg_key][0, s_idx, d_idx, :]
                traj_y = Z[marg_key][1, s_idx, d_idx, :]

                linestyle = decision_colors[d_type]
                label = f"{stim_labels_plot[s_type]} - {decision_labels_plot[d_type]}"

                face = stim_colors[s_type]
                edge = marker_edge[d_type]
                # Only show outline if it differs from fill
                show_edge = face == edge

                ax.plot(traj_x, traj_y,
                        color=stim_colors[s_type],
                        linestyle=linestyle,
                        linewidth=1.5,
                        label=label)

                t0_idx = np.argmin(np.abs(time_centers))
                # t0 marker (circle)
                ax.plot(traj_x[t0_idx], traj_y[t0_idx], 'o',
                        markerfacecolor=face,
                        markeredgecolor='none',
                        markersize=8, zorder=5)
                if show_edge:
                    ax.plot(traj_x[t0_idx], traj_y[t0_idx], 'o',
                            markerfacecolor='none',
                            markeredgecolor=edge,
                            markersize=8, markeredgewidth=1.5, zorder=6)

                # start marker (square)
                ax.plot(traj_x[0], traj_y[0], 's',
                        markerfacecolor=face,
                        markeredgecolor='none',
                        markersize=8, zorder=5)
                if show_edge:
                    ax.plot(traj_x[0], traj_y[0], 's',
                            markerfacecolor='none',
                            markeredgecolor=edge,
                            markersize=8, markeredgewidth=1.5, zorder=6)

        marg_display = marg_names.get(marg_key, marg_key)
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       left=False, right=False, labelbottom=False, labelleft=False)
        ax.set_xlabel(f'Component 1', fontsize=FONT_SMALL)
        ax.set_ylabel(f'Component 2', fontsize=FONT_SMALL)
        ax.set_title(f'{marg_display} State Space', fontsize=FONT_BIG, fontweight='bold')
        ax.legend(fontsize=FONT_SMALL, loc='best')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

    return fig


def plot_state_space_flex(Z, var_explained, time_centers, stimulus_types, decision_types,
                          area, n_neurons, axes_spec, subplot_size_mm=(65, 40),
                          spacing_mm=8, margin_mm=None):
    """
    Flexible state space plot: each subplot uses a user-specified marginalization
    and component pair, with exact subplot sizing in millimeters.

    Parameters
    ----------
    axes_spec : list of dicts
        Each dict defines one subplot with keys:
            'x_marg': str — marginalization for x-axis (e.g. 's', 'dt')
            'x_comp': int — component index for x-axis (0-based)
            'y_marg': str — marginalization for y-axis
            'y_comp': int — component index for y-axis
            'title':  str — (optional) subplot title

        Example:
            axes_spec = [
                {'x_marg': 's', 'x_comp': 0, 'y_marg': 's', 'y_comp': 1},
                {'x_marg': 's', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 0, 'title': 'Stim vs Dec'},
                {'x_marg': 'dt', 'x_comp': 0, 'y_marg': 'dt', 'y_comp': 1},
            ]
    subplot_size_mm : tuple of (width_mm, height_mm)
        Exact size of each subplot in millimeters. Default: (65, 40).
    spacing_mm : float
        Horizontal spacing between subplots in mm.
    margin_mm : dict, optional
        Figure margins in mm. Keys: 'left', 'right', 'top', 'bottom'.
    """
    n_plots = len(axes_spec)

    fig, axes_grid = _make_mm_figure(
        n_rows=1, n_cols=n_plots,
        subplot_w_mm=subplot_size_mm[0], subplot_h_mm=subplot_size_mm[1],
        spacing_h_mm=spacing_mm,
        margin_mm=margin_mm,
    )
    axes = axes_grid[0]  # single row

    marker_edge = {'L': 'blue', 'R': 'red', 'N': 'black'}

    for plot_idx, spec in enumerate(axes_spec):
        ax = axes[plot_idx]

        x_marg = spec['x_marg']
        x_comp = spec['x_comp']
        y_marg = spec['y_marg']
        y_comp = spec['y_comp']

        for s_idx, s_type in enumerate(stimulus_types):
            for d_idx, d_type in enumerate(decision_types):
                traj_x = Z[x_marg][x_comp, s_idx, d_idx, :]
                traj_y = Z[y_marg][y_comp, s_idx, d_idx, :]

                linestyle = decision_colors[d_type]
                label = f"{stim_labels_plot[s_type]} - {decision_labels_plot[d_type]}"

                face = stim_colors[s_type]
                edge = marker_edge[d_type]
                show_edge = face != edge

                ax.plot(traj_x, traj_y,
                        color=stim_colors[s_type],
                        linestyle=linestyle,
                        linewidth=1.5,
                        label=label)

                t0_idx = np.argmin(np.abs(time_centers))
                ax.plot(traj_x[t0_idx], traj_y[t0_idx], 'o',
                        markerfacecolor=face,
                        markeredgecolor=edge if show_edge else 'none',
                        markersize=8, markeredgewidth=0.7, alpha=0.5, zorder=5)
                ax.plot(traj_x[0], traj_y[0], 's',
                        markerfacecolor=face,
                        markeredgecolor=edge if show_edge else 'none',
                        markersize=8, markeredgewidth=0.7, alpha=0.5, zorder=5)

        # Labels
        x_label = f"{x_marg} comp {x_comp + 1} ({var_explained[x_marg][x_comp]:.1f}%)"
        y_label = f"{y_marg} comp {y_comp + 1} ({var_explained[y_marg][y_comp]:.1f}%)"
        ax.set_xlabel(x_label, fontsize=FONT_SMALL)
        ax.set_ylabel(y_label, fontsize=FONT_SMALL)

        title = spec.get('title', None)
        if title:
            ax.set_title(title, fontsize=FONT_BIG, fontweight='bold')

        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       left=False, right=False, labelbottom=False, labelleft=False)
        if plot_idx == 0:
            ax.legend(fontsize=FONT_SMALL, loc='best')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

    return fig


def plot_demixing_index(demixing_index, var_explained, n_components, area):
    """Figure 4: Demixing index heatmap."""
    n_comp_demix = min(5, n_components)
    marg_order = [k for k in ['st', 'dt', 't', 's', 'd', 'sd', 'sdt'] if k in demixing_index]

    fig_demix, axes_demix = plt.subplots(1, len(marg_order),
                                          figsize=(3.5 * len(marg_order), 4),
                                          squeeze=False)

    for col_idx, marg_key in enumerate(marg_order):
        ax = axes_demix[0, col_idx]

        matrix = np.zeros((n_comp_demix, len(marg_order)))
        for comp_idx in range(n_comp_demix):
            fracs = demixing_index[marg_key][comp_idx]['var_fractions']
            for src_idx, src_marg in enumerate(marg_order):
                matrix[comp_idx, src_idx] = fracs.get(src_marg, 0)

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                        interpolation='nearest')

        ax.set_xticks(range(len(marg_order)))
        ax.set_xticklabels(marg_order, fontsize=FONT_SMALL, rotation=45)
        ax.set_yticks(range(n_comp_demix))
        ax.set_yticklabels([f"C{i}" for i in range(n_comp_demix)], fontsize=FONT_SMALL)
        ax.set_xlabel('Source', fontsize=FONT_SMALL)
        if col_idx == 0:
            ax.set_ylabel('Component', fontsize=FONT_SMALL)

        marg_display = marg_names.get(marg_key, marg_key)
        demix_vals = [demixing_index[marg_key][c]['demixing_index'] for c in range(n_comp_demix)]
        ax.set_title(f"{marg_display}\nDI: {np.mean(demix_vals):.2f}", fontsize=FONT_BIG, fontweight='bold')

        for i in range(n_comp_demix):
            for j in range(len(marg_order)):
                val = matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=FONT_SMALL, color=color)

    fig_demix.suptitle(f"Demixing Index — {', '.join(area)}\n"
                        f"Fraction of component variance from each marginalization",
                        fontsize=FONT_BIG, fontweight='bold')
    plt.tight_layout()
    return fig_demix


def plot_axis_orthogonality(dot_products, correlations, significant_pairs,
                            axis_labels_orth, axis_marg, area):
    """Figure 5: Axis orthogonality & component correlations (Kobak et al. Fig 3e)."""
    n_total_axes = len(axis_labels_orth)

    fig_orth, ax_orth = plt.subplots(1, 1, figsize=(2, 2))

    combined_matrix = np.zeros((n_total_axes, n_total_axes))
    for i in range(n_total_axes):
        for j in range(n_total_axes):
            if i == j:
                combined_matrix[i, j] = 0
            elif j > i:
                combined_matrix[i, j] = dot_products[i, j]
            else:
                combined_matrix[i, j] = correlations[i, j]

    im = ax_orth.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                         interpolation='nearest')

    for i in range(n_total_axes):
        for j in range(i + 1, n_total_axes):
            if significant_pairs[i, j]:
                ax_orth.plot(j, i, marker='*', color='black', markersize=3, zorder=5, markeredgewidth=0.2)

    ax_orth.plot([-0.5, n_total_axes - 0.5], [-0.5, n_total_axes - 0.5],
                  'k-', linewidth=0.5)

    prev_marg = axis_marg[0]
    for idx in range(1, n_total_axes):
        if axis_marg[idx] != prev_marg:
            ax_orth.axhline(idx - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax_orth.axvline(idx - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            prev_marg = axis_marg[idx]

    ax_orth.set_xticks(range(n_total_axes))
    ax_orth.set_xticklabels(axis_labels_orth, fontsize=FONT_SMALL, rotation=45, ha='right')
    ax_orth.set_yticks(range(n_total_axes))
    ax_orth.set_yticklabels(axis_labels_orth, fontsize=FONT_SMALL)

    cbar = plt.colorbar(im, ax=ax_orth, shrink=0.8)
    cbar.ax.tick_params(labelsize=FONT_SMALL)
    cbar.set_label('Value', fontsize=FONT_SMALL)

    ax_orth.text(n_total_axes * 0.75, n_total_axes * 0.15, 'Dot products',
                  ha='center', va='center', fontsize=FONT_SMALL, fontstyle='italic', alpha=0.6)
    ax_orth.text(n_total_axes * 0.25, n_total_axes * 0.85, 'Correlations',
                  ha='center', va='center', fontsize=FONT_SMALL, fontstyle='italic', alpha=0.6)

    plt.tight_layout()
    return fig_orth


def plot_significance(Z, masks, var_explained, time_centers, step_size,
                      stimulus_types, decision_types, n_components, area,
                      n_shuffles, n_splits):
    """Figure 6: Significance overlaid on components."""
    sig_margs = [k for k in ['s', 'd', 'sd', 'st', 'dt'] if k in masks]
    n_sig_comp = min(3, n_components)

    fig_sig, axes_sig = plt.subplots(len(sig_margs), n_sig_comp,
                                      figsize=(5 * n_sig_comp, 3.5 * len(sig_margs)),
                                      squeeze=False)

    for row_idx, marg_key in enumerate(sig_margs):
        for comp_idx in range(n_sig_comp):
            ax = axes_sig[row_idx, comp_idx]

            for s_idx, s_type in enumerate(stimulus_types):
                for d_idx, d_type in enumerate(decision_types):
                    trajectory = Z[marg_key][comp_idx, s_idx, d_idx, :]
                    linestyle = decision_colors[d_type]
                    label = f"{stim_labels_plot[s_type]} - {decision_labels_plot[d_type]}" \
                        if comp_idx == 0 and row_idx == 0 else None
                    ax.plot(time_centers, trajectory,
                            color=stim_colors[s_type], linestyle=linestyle,
                            linewidth=1.5, label=label)

            if marg_key in masks and comp_idx < masks[marg_key].shape[0]:
                sig_mask = masks[marg_key][comp_idx]
                # Determine half-bin width for shading
                if step_size is not None:
                    half_bin = step_size / 2
                else:
                    # Infer from time_centers spacing
                    half_bin = (time_centers[1] - time_centers[0]) / 2 if len(time_centers) > 1 else 0.005
                for t_idx in range(len(time_centers)):
                    if sig_mask[t_idx]:
                        ax.axvspan(
                            time_centers[t_idx] - half_bin,
                            time_centers[t_idx] + half_bin,
                            color='yellow', alpha=0.3, zorder=0
                        )

            ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.grid(True, alpha=0.2)

            marg_display = marg_names.get(marg_key, marg_key)
            var_str = f"{var_explained[marg_key][comp_idx]:.1f}%"
            frac = np.mean(masks[marg_key][comp_idx]) if comp_idx < masks[marg_key].shape[0] else 0
            ax.set_title(f"{marg_display}\nComp {comp_idx+1} ({var_str}, sig={frac:.0%})", fontsize=FONT_BIG)

            if row_idx == len(sig_margs) - 1:
                ax.set_xlabel('Time from stimulus (s)', fontsize=FONT_SMALL)
            if comp_idx == 0:
                ax.set_ylabel('dPC (a.u.)', fontsize=FONT_SMALL)

    handles, labels_leg = axes_sig[0, 0].get_legend_handles_labels()
    fig_sig.legend(handles, labels_leg, loc='upper center', ncol=4, fontsize=FONT_SMALL,
                    bbox_to_anchor=(0.5, 1.02))

    fig_sig.suptitle(f"Significance Analysis — {', '.join(area)}\n"
                      f"Yellow = significant (p<{1/(n_shuffles+1):.2f}), "
                      f"n_shuffles={n_shuffles}, n_splits={n_splits}",
                      fontsize=FONT_BIG, fontweight='bold', y=1.06)
    plt.tight_layout()
    return fig_sig


def plot_cumvar(dpca_cumvar, pca_cumvar, area, n_neurons):
    """Figure 7: Cumulative variance comparison dPCA vs PCA."""
    fig_cumvar, ax_cumvar = plt.subplots(1, 1, figsize=(8, 5))

    n_plot = min(len(dpca_cumvar), len(pca_cumvar), 30)
    ax_cumvar.plot(range(1, n_plot + 1), dpca_cumvar[:n_plot], 'o-', color='blue',
                    linewidth=2, markersize=5, label='dPCA')
    ax_cumvar.plot(range(1, n_plot + 1), pca_cumvar[:n_plot], 's-', color='gray',
                    linewidth=2, markersize=5, label='PCA')

    ax_cumvar.set_xlabel('Number of components', fontsize=FONT_SMALL)
    ax_cumvar.set_ylabel('Cumulative variance explained (%)', fontsize=FONT_SMALL)
    ax_cumvar.set_title(f"dPCA vs PCA — {', '.join(area)}, n={n_neurons} neurons",
                         fontsize=FONT_BIG, fontweight='bold')
    ax_cumvar.legend(fontsize=FONT_SMALL)
    ax_cumvar.grid(True, alpha=0.3)
    ax_cumvar.set_xlim(0.5, n_plot + 0.5)
    plt.tight_layout()
    return fig_cumvar


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_arguments()

    # uncomment and specify directory to run plotting script without terminal
    #args.input_dir = ""
    
    args.go_axis = os.path.join(args.input_dir, 'go_axis_results.npz')

    if not os.path.isfile(args.go_axis):
        args.go_axis = None

    input_dir = args.input_dir
    dpi = args.dpi
    save_svg = args.svg and not args.no_svg

    single_ses = True if os.path.isfile(os.path.join(input_dir, 'session_summary.yaml')) else False
    if single_ses:
        dirs = list(Path(input_dir).rglob("*/"))
    else:
        dirs = [input_dir]
    for output_dir in dirs:
        print(f"Loading results from: {output_dir}")
        results, metadata = load_results(output_dir)
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Extract data from results
        Z = results['Z']
        var_explained = results['var_explained']
        time_centers = results['time_centers']
        stimulus_types = list(results['stimulus_types'])
        decision_types = list(results['decision_types'])
        demixing_index = results['demixing_index']
        dot_products = results['dot_products']
        correlations = results['correlations_between_components']
        significant_pairs = results['significant_pairs']
        axis_labels_orth = list(results['axis_labels_orth'])
        axis_marg = list(results['axis_marg_orth'])

        # Extract metadata
        area = metadata['area']
        n_neurons = metadata['n_neurons']
        n_components = metadata['n_components']
        bin_size = metadata['bin_size']
        step_size = metadata['step_size']
        run_validation = metadata.get('run_validation', False)
        n_shuffles = metadata.get('n_shuffles', 100)
        n_splits = metadata.get('n_splits', 100)

        # Check if validation data is present
        has_validation = ('significance_masks' in results and
                          'dpca_cumvar' in results and
                          'pca_cumvar' in results)

        print(f"Area: {area}")
        print(f"n_neurons: {n_neurons}, n_components: {n_components}")
        print(f"Validation data present: {has_validation}")

        # Save per-component variance explained as readable YAML
        var_explained_readable = {}
        for key in sorted(var_explained.keys()):
            marg_display = marg_names.get(key, key)
            var_explained_readable[marg_display] = {
                'total': round(float(np.sum(var_explained[key])), 4),
                'components': {
                    f'comp_{i}': round(float(v), 4)
                    for i, v in enumerate(var_explained[key])
                }
            }

        with open(os.path.join(output_dir, 'variance_explained.yaml'), 'w') as f:
            yaml.dump(var_explained_readable, f, default_flow_style=False, sort_keys=False)

        print(f"Saved per-component variance to: {os.path.join(output_dir, 'variance_explained.yaml')}")

        # --- Optionally load Go axis ---
        go_axis_vector = None
        if args.go_axis is not None:
            print(f"Loading Go axis from: {args.go_axis}")
            go_data = np.load(args.go_axis, allow_pickle=True)
            go_neuron_ids = go_data['neuron_ids']
            go_axis_raw = go_data['go_axis']

            dpca_neuron_ids = results['neuron_ids']

            # Find shared neurons and reindex the Go axis to match dPCA neuron order
            go_id_to_idx = {nid: i for i, nid in enumerate(go_neuron_ids)}
            shared_mask = np.array([nid in go_id_to_idx for nid in dpca_neuron_ids])

            if shared_mask.sum() < len(dpca_neuron_ids):
                n_missing = len(dpca_neuron_ids) - shared_mask.sum()
                print(f"  Warning: {n_missing} dPCA neurons not in Go axis — "
                      f"using {shared_mask.sum()} shared neurons")

            # Build aligned Go axis vector (same length/order as dPCA axes)
            go_axis_vector = np.zeros(len(dpca_neuron_ids))
            for i, nid in enumerate(dpca_neuron_ids):
                if nid in go_id_to_idx:
                    go_axis_vector[i] = go_axis_raw[go_id_to_idx[nid]]

            # Normalize
            norm = np.linalg.norm(go_axis_vector)
            if norm > 0:
                go_axis_vector /= norm
            print(f"  Go axis aligned ({shared_mask.sum()} neurons)")

        # --- Generate all plots ---
        print("\nCreating plots...")

        # Figure 1: Components
        fig_comp = plot_components(Z, var_explained, time_centers, stimulus_types,
                                   decision_types, n_components, area, n_neurons,
                                   bin_size, step_size)

        # Figure 2: Variance pie
        fig_pie = plot_variance_pie(var_explained, area, n_neurons)

        # Figure 3: State space
        fig_state = plot_state_space(Z, var_explained, time_centers, stimulus_types,
                                     decision_types, area, n_neurons)

        # Figure 4: Demixing index
        fig_demix = plot_demixing_index(demixing_index, var_explained, n_components, area)

        # Figure 5: Axis orthogonality
        if go_axis_vector is not None:
            encoder_axes = results['encoder_axes']  # (n_axes, n_neurons)

            n_existing = len(axis_labels_orth)

            # Compute dot products and correlations between Go axis and each dPCA axis
            new_dots = np.array([np.dot(go_axis_vector, encoder_axes[i])
                                 for i in range(n_existing)])
            new_corrs = np.array([np.corrcoef(go_axis_vector, encoder_axes[i])[0, 1]
                                  for i in range(n_existing)])

            # Extend the matrices by one row and column
            dp_new = np.zeros((n_existing + 1, n_existing + 1))
            dp_new[:n_existing, :n_existing] = dot_products
            dp_new[:n_existing, n_existing] = new_dots
            dp_new[n_existing, :n_existing] = new_dots
            dot_products = dp_new

            co_new = np.zeros((n_existing + 1, n_existing + 1))
            co_new[:n_existing, :n_existing] = correlations
            co_new[:n_existing, n_existing] = new_corrs
            co_new[n_existing, :n_existing] = new_corrs
            correlations = co_new

            sp_new = np.zeros((n_existing + 1, n_existing + 1), dtype=bool)
            sp_new[:n_existing, :n_existing] = significant_pairs
            significant_pairs = sp_new

            axis_labels_orth = list(axis_labels_orth) + ['Go']
            axis_marg = list(axis_marg) + ['go']

            print("  Go axis appended to orthogonality matrices")

        fig_orth = plot_axis_orthogonality(dot_products, correlations, significant_pairs,
                                           axis_labels_orth, axis_marg, area)

        # Figure 6: Mixed state space (stimulus x decision)
        fig_state_mixed_s_d = plot_state_space_flex(Z, var_explained, time_centers, stimulus_types, decision_types,
                                    area, n_neurons, axes_spec=[
                {'x_marg': 's', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 0, 'title': ''},
                {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 1, 'title': ''},
            ])

        # Figure 7:
        fig_state_mixed_sd = plot_state_space_flex(Z, var_explained, time_centers, stimulus_types, decision_types,
                                    area, n_neurons, axes_spec=[
                {'x_marg': 'sd', 'x_comp': 0, 'y_marg': 'sd', 'y_comp': 1, 'title': ''},
                {'x_marg': 'sd', 'x_comp': 0, 'y_marg': 'd', 'y_comp': 0, 'title': ''},
                {'x_marg': 'sd', 'x_comp': 0, 'y_marg': 's', 'y_comp': 0, 'title': ''}
            ])

        # Figure 8:
        fig_state_mixed_s_d_non_orthogonal = plot_state_space_flex(Z, var_explained, time_centers, stimulus_types, decision_types,
                                    area, n_neurons, axes_spec=[
                {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 0, 'title': ''},
                {'x_marg': 's', 'x_comp': 1, 'y_marg': 'd', 'y_comp': 2, 'title': ''},
            ])

        # Figure: Cumulative variance per marginalization
        fig_cumvar_marg = plot_cumvar_per_marginalization(var_explained, area, n_neurons)

        # --- Save figures ---
        print("\nSaving figures...")

        fig_comp.savefig(os.path.join(fig_dir, 'dpca_components.png'), dpi=dpi, bbox_inches='tight')
        fig_pie.savefig(os.path.join(fig_dir, 'dpca_variance_explained.png'), dpi=dpi, bbox_inches='tight')
        fig_state.savefig(os.path.join(fig_dir, 'dpca_state_space.png'), dpi=dpi, bbox_inches='tight')
        fig_demix.savefig(os.path.join(fig_dir, 'dpca_demixing_index.png'), dpi=dpi, bbox_inches='tight')
        fig_orth.savefig(os.path.join(fig_dir, 'dpca_axis_orthogonality.png'), dpi=dpi, bbox_inches='tight')
        fig_state_mixed_s_d.savefig(os.path.join(fig_dir, 'dpca_state_space_mixed.png'), dpi=dpi, bbox_inches='tight')
        fig_state_mixed_sd.savefig(os.path.join(fig_dir, 'dpca_state_space_mixed_sd.png'), dpi=dpi, bbox_inches='tight')
        fig_state_mixed_s_d_non_orthogonal.savefig(os.path.join(fig_dir, 'dpca_state_space_mixed_non_orthogonal.png'), dpi=dpi, bbox_inches='tight')
        fig_cumvar_marg.savefig(os.path.join(fig_dir, 'dpca_cumvar_per_marginalization.png'), dpi=dpi, bbox_inches='tight')

        if save_svg:
            fig_comp.savefig(os.path.join(fig_dir, 'dpca_components.svg'), dpi=dpi, bbox_inches='tight')
            fig_pie.savefig(os.path.join(fig_dir, 'dpca_variance_explained.svg'), dpi=dpi, bbox_inches='tight')
            fig_state.savefig(os.path.join(fig_dir, 'dpca_state_space.svg'), dpi=dpi, bbox_inches='tight')
            fig_demix.savefig(os.path.join(fig_dir, 'dpca_demixing_index.svg'), dpi=dpi, bbox_inches='tight')
            fig_orth.savefig(os.path.join(fig_dir, 'dpca_axis_orthogonality.svg'), dpi=dpi, bbox_inches='tight')
            fig_state_mixed_s_d.savefig(os.path.join(fig_dir, 'dpca_state_space_mixed.svg'), dpi=dpi, bbox_inches='tight')
            fig_state_mixed_sd.savefig(os.path.join(fig_dir, 'dpca_state_space_mixed_sd.svg'), dpi=dpi, bbox_inches='tight')
            fig_cumvar_marg.savefig(os.path.join(fig_dir, 'dpca_cumvar_per_marginalization.svg'), dpi=dpi, bbox_inches='tight')

        # --- Validation plots (only if data is present) ---
        if has_validation:
            print("Generating validation plots...")

            masks = results['significance_masks']
            dpca_cumvar = results['dpca_cumvar']
            pca_cumvar = results['pca_cumvar']

            crossval_path = save_crossval_yaml(masks, var_explained, time_centers, step_size,
                                               n_shuffles, n_splits, output_dir)
            print(f"Saved cross-validation results to: {crossval_path}")

            dpca_cumvar = dpca_cumvar * 100

            # Figure 6: Significance
            fig_sig = plot_significance(Z, masks, var_explained, time_centers, step_size,
                                        stimulus_types, decision_types, n_components, area,
                                        n_shuffles, n_splits)
            fig_sig.savefig(os.path.join(fig_dir, 'dpca_significance.png'), dpi=dpi, bbox_inches='tight')
            if save_svg:
                fig_sig.savefig(os.path.join(fig_dir, 'dpca_significance.svg'), dpi=dpi, bbox_inches='tight')

            # Figure 7: Cumulative variance
            fig_cumvar = plot_cumvar(dpca_cumvar, pca_cumvar, area, n_neurons)
            fig_cumvar.savefig(os.path.join(fig_dir, 'dpca_cumvar_vs_pca.png'), dpi=dpi, bbox_inches='tight')
            if save_svg:
                fig_cumvar.savefig(os.path.join(fig_dir, 'dpca_cumvar_vs_pca.svg'), dpi=dpi, bbox_inches='tight')
        else:
            print("No validation data found — skipping significance and cumvar plots.")

        plt.close('all')

        print(f"\nPlotting complete! Figures saved to: {fig_dir}")
        print(f"Figures: dpca_components.png, dpca_variance_explained.png, dpca_state_space.png, "
              f"dpca_demixing_index.png, dpca_axis_orthogonality.png")
        if has_validation:
            print(f"Validation: dpca_significance.png, dpca_cumvar_vs_pca.png")


if __name__ == '__main__':
    main()