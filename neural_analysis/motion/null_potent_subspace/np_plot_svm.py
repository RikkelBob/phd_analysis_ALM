# Plot SVM decoding results from null vs potent subspace reconstructions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

from scipy.stats import mannwhitneyu
from scipy.ndimage import label as ndi_label
from statsmodels.stats.multitest import multipletests

import config

# ============================================================
# Settings
# ============================================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'   # keep text as text in SVG (editable)
plt.rcParams['pdf.fonttype'] = 42       # editable text in PDF too, just in case
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
})

# Path to the SVM output folder produced by the decoding script
subdir_subspace = "lALMrALM"
subspace_dir = config.resolve_pp("null_potent_subspace", "decode_from_subspace", subdir_subspace)

# Use the newest run (subfolder with output.npy), whatever it is named
runs = sorted((p for p in subspace_dir.iterdir() if (p / 'output.npy').exists()),
              key=lambda p: p.stat().st_mtime)
if not runs:
    raise FileNotFoundError(f"No subfolder with output.npy found in {subspace_dir}")
results_dir = runs[-1]
print(f"[data] using decoding run: {results_dir.name}")

# Derived results are always saved to the NEW data dir, never next to the shipped data
results_out_dir = config.DATA_DIR_PP_NEW / "SVM" / "decode_from_subspace" / subdir_subspace
results_out_dir.mkdir(parents=True, exist_ok=True)

fig_dir = config.FIGURE_DIR / "SVM" / "decode_from_subspace"
fig_dir.mkdir(exist_ok=True)

# Colors
COL_NULL = '#2a9d8f'      # teal — movement-null
COL_POTENT = '#e76f51'    # orange — movement-potent
COL_SHUF = '#888888'      # gray — shuffled

# Modality colors (used in the by-subspace figure)
COL_VIS = 'blue'       # blue — visual
COL_AUD = 'red'       # purple — auditory

MODALITY_COLORS = {
    'vis_presence_miss_cr': COL_VIS,
    'aud_presence_miss_cr': COL_AUD,
}
MODALITY_LABELS = {
    'vis_presence_miss_cr': 'visual',
    'aud_presence_miss_cr': 'auditory',
}

SUBSPACE_TITLES = {
    'null': 'movement-null subspace',
    'potent': 'movement-potent subspace',
}

ANALYSIS_TITLES = {
    'vis_presence_miss_cr': 'Visual miss vs. correct rejection',
    'aud_presence_miss_cr': 'Auditory miss vs. correct rejection',
}

# CI percentile (e.g. 95 → 2.5 / 97.5)
CI_PCT = 95
LO, HI = (100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2

# Cluster-based permutation test settings (Maris & Oostenveld 2007)
CLUSTER_WINDOW = None              # None → search all bins; otherwise (t_start, t_end)
CLUSTER_THRESHOLD_PCT = 95.0       # cluster-forming threshold (percentile of null per bin)
CLUSTER_P_ALPHA = 0.01             # cluster-level significance threshold

# ---- Panel size control --------------------------------------------------
# Each data panel is exactly PANEL_W x PANEL_H inches.
PANEL_W = 1.5   # inches
PANEL_H = 20/25.4   # inches

# Margins around the data panel(s), in inches. These are physical, fixed.
# Tweak if labels get clipped — the panel size itself stays 1x1.
MARGIN_LEFT   = 0.55
MARGIN_RIGHT  = 0.15
MARGIN_BOTTOM = 0.45
MARGIN_TOP    = 0.30

# Spacing between panels in the summary grid (inches)
WSPACE_IN = 0.35
HSPACE_IN = 0.35


def make_fixed_panel_fig(nrows=1, ncols=1):
    """Create a figure where each subplot's data area is exactly
    PANEL_W x PANEL_H inches, with fixed physical margins.
    """
    fig_w = MARGIN_LEFT + ncols * PANEL_W + (ncols - 1) * WSPACE_IN + MARGIN_RIGHT
    fig_h = MARGIN_BOTTOM + nrows * PANEL_H + (nrows - 1) * HSPACE_IN + MARGIN_TOP

    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Convert physical margins to figure-fraction for subplots_adjust
    left   = MARGIN_LEFT / fig_w
    right  = 1 - MARGIN_RIGHT / fig_w
    bottom = MARGIN_BOTTOM / fig_h
    top    = 1 - MARGIN_TOP / fig_h

    # wspace/hspace are fractions of the *average axes size*
    wspace = WSPACE_IN / PANEL_W if ncols > 1 else 0
    hspace = HSPACE_IN / PANEL_H if nrows > 1 else 0

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)
    return fig, axs


# ============================================================
# Load
# ============================================================

output = np.load(results_dir / 'output.npy', allow_pickle=True).item()
with open(results_dir / 'metadata.yaml', 'r') as f:
    metadata = yaml.unsafe_load(f)

analyses_present = sorted({
    key.replace('results_', '').rsplit('_', 1)[0]
    for key in output.keys()
    if key.startswith('results_') and not key.startswith('results_shuffled_')
})


# ============================================================
# Helpers
# ============================================================

def _get_bin_size():
    """Pull bin_size (s) from metadata. Falls back to inferring from spacing."""
    for k in ('bin_size', 'binsize', 'bin_width'):
        if k in metadata:
            return float(metadata[k])
    for sub in metadata.values():
        if isinstance(sub, dict):
            for k in ('bin_size', 'binsize', 'bin_width'):
                if k in sub:
                    return float(sub[k])
    return None


BIN_SIZE = _get_bin_size()


def to_iter_df(records):
    """Turn list of {'bin', 'iteration', 'scores': {...}} into a DataFrame
    with one row per (iteration, bin) — the score is the mean across CV folds.

    Stored bin values are left edges; we shift by bin_size/2 to plot at centers.
    """
    rows = []
    for r in records:
        bin_val = r['bin']
        if hasattr(bin_val, '__len__'):
            bin_val = float(bin_val[0])
        score = float(np.mean(r['scores']['test_score']))
        rows.append({
            'bin': bin_val,
            'iteration': r.get('iteration', np.nan),
            'score': score,
        })
    df = pd.DataFrame(rows)

    bin_size = BIN_SIZE
    if bin_size is None:
        unique_bins = np.sort(df['bin'].unique())
        if len(unique_bins) >= 2:
            bin_size = float(np.median(np.diff(unique_bins)))
    if bin_size is not None:
        df['bin'] = (df['bin'] / 1e6) + bin_size

    return df


def aggregate_by_bin(df, ci_pct=CI_PCT):
    lo_pct = (100 - ci_pct) / 2
    hi_pct = 100 - lo_pct
    grouped = df.groupby('bin')['score']
    mean = grouped.mean()
    lo = grouped.quantile(lo_pct / 100)
    hi = grouped.quantile(hi_pct / 100)
    return mean, lo, hi


def plot_trace(ax, df, color, label, linestyle='-'):
    median, lo, hi = aggregate_by_bin(df)
    bin_centers = median.index.values
    ax.plot(bin_centers, median.values, color=color, linewidth=1.0,
            label=label, linestyle=linestyle)
    ax.fill_between(bin_centers, lo.values, hi.values,
                    color=color, alpha=0.2, linewidth=0)


# ============================================================
# CLUSTER-BASED PERMUTATION TEST (Maris & Oostenveld 2007)
# ============================================================

def _build_cluster_time_mask(time_centers, cluster_window):
    if cluster_window is None:
        return np.ones(len(time_centers), dtype=bool)
    start, end = cluster_window
    mask = np.ones(len(time_centers), dtype=bool)
    if start is not None:
        mask &= (time_centers >= start)
    if end is not None:
        mask &= (time_centers <= end)
    return mask


def _find_clusters_1d(trace_windowed, threshold_windowed):
    above = trace_windowed > threshold_windowed
    if not np.any(above):
        return []
    labeled, n_clusters = ndi_label(above)
    clusters = []
    for k in range(1, n_clusters + 1):
        mask = labeled == k
        idx = np.where(mask)[0]
        mass = float(np.sum(trace_windowed[mask] - threshold_windowed[mask]))
        clusters.append((int(idx[0]), int(idx[-1]), mass))
    return clusters


def cluster_permutation_test(observed, null_distribution, time_centers,
                             cluster_window=None,
                             threshold_percentile=95.0,
                             p_alpha=0.05):
    """One-sided cluster-based permutation test (observed > null).

    Parameters
    ----------
    observed : (n_timebins,) array
        Observed decoding trace (e.g. mean across real iterations).
    null_distribution : (n_perms, n_timebins) array
        Null/shuffle distribution; one row per shuffle iteration.
    time_centers : (n_timebins,) array
        Time at center of each bin (used only for the optional window
        and for reporting cluster times).
    cluster_window : (t_start, t_end) or None
        Restrict cluster search to this time window. None → all bins.
    threshold_percentile : float
        Cluster-forming threshold (per-bin percentile of null).
    p_alpha : float
        Cluster-level significance.

    Returns
    -------
    dict with keys:
        observed_clusters : list of dicts (start_idx, end_idx, t_start, t_end, mass, p_value)
        max_null_masses   : (n_perms,) max cluster mass under each shuffle
        threshold_full    : (n_timebins,) per-bin threshold (NaN outside window)
        cluster_mask      : (n_timebins,) bool, True where part of a significant cluster
        p_map             : (n_timebins,) cluster p-value at each bin (NaN if not in sig cluster)
        cluster_window, threshold_percentile : echoes of inputs
    """
    n_perms, n_timebins = null_distribution.shape
    win_mask = _build_cluster_time_mask(time_centers, cluster_window)
    win_idx = np.where(win_mask)[0]

    threshold_full = np.full(n_timebins, np.nan)
    threshold_windowed = np.nanpercentile(
        null_distribution[:, win_mask], threshold_percentile, axis=0
    )
    threshold_full[win_mask] = threshold_windowed

    obs_clusters_win = _find_clusters_1d(observed[win_mask], threshold_windowed)

    max_null_masses = np.zeros(n_perms)
    for p in range(n_perms):
        clusters_p = _find_clusters_1d(null_distribution[p, win_mask], threshold_windowed)
        if clusters_p:
            max_null_masses[p] = max(c[2] for c in clusters_p)

    observed_clusters = []
    cluster_mask = np.zeros(n_timebins, dtype=bool)
    p_map = np.full(n_timebins, np.nan)
    for (s_win, e_win, mass) in obs_clusters_win:
        p = (np.sum(max_null_masses >= mass) + 1) / (n_perms + 1)
        s_full = int(win_idx[s_win])
        e_full = int(win_idx[e_win])
        observed_clusters.append({
            'start_idx': s_full,
            'end_idx': e_full,
            't_start': float(time_centers[s_full]),
            't_end': float(time_centers[e_full]),
            'mass': float(mass),
            'p_value': float(p),
        })
        if p < p_alpha:
            cluster_mask[s_full:e_full + 1] = True
            p_map[s_full:e_full + 1] = p

    return {
        'observed_clusters': observed_clusters,
        'max_null_masses': max_null_masses,
        'threshold_full': threshold_full,
        'cluster_mask': cluster_mask,
        'p_map': p_map,
        'cluster_window': cluster_window if cluster_window is not None else (None, None),
        'threshold_percentile': float(threshold_percentile),
    }


def _print_cluster_summary(cluster_result, prefix="    ", p_alpha=CLUSTER_P_ALPHA):
    clusters = cluster_result['observed_clusters']
    if not clusters:
        print(f"{prefix}Cluster test: no clusters above threshold.")
        return
    n_sig = sum(1 for c in clusters if c['p_value'] < p_alpha)
    print(f"{prefix}Cluster test: {len(clusters)} cluster(s), {n_sig} significant (p<{p_alpha})")
    for i, c in enumerate(clusters):
        sig_flag = " *" if c['p_value'] < p_alpha else ""
        print(f"{prefix}  Cluster {i}: t=[{c['t_start']:+.3f}, {c['t_end']:+.3f}]s, "
              f"mass={c['mass']:.3f}, p={c['p_value']:.4f}{sig_flag}")


def run_cluster_test(df_real, df_shuf,
                     cluster_window=CLUSTER_WINDOW,
                     threshold_percentile=CLUSTER_THRESHOLD_PCT,
                     p_alpha=CLUSTER_P_ALPHA):
    """Wrap cluster_permutation_test to take the same DataFrames we use
    elsewhere. Observed = mean across real iterations; null = shuffle
    iterations × bins.

    Returns the cluster_result dict, plus the bin centers used. The
    returned bin_centers are the ONLY axis whose indices line up with
    cluster_mask / observed_clusters — always draw bars against these.
    """
    pivot_real = df_real.pivot_table(index='iteration', columns='bin', values='score')
    pivot_shuf = df_shuf.pivot_table(index='iteration', columns='bin', values='score')
    pivot_shuf = pivot_shuf[pivot_real.columns]

    bin_centers = pivot_real.columns.values.astype(float)
    observed = pivot_real.values.mean(axis=0)
    null_distribution = pivot_shuf.values

    result = cluster_permutation_test(
        observed=observed,
        null_distribution=null_distribution,
        time_centers=bin_centers,
        cluster_window=cluster_window,
        threshold_percentile=threshold_percentile,
        p_alpha=p_alpha,
    )
    return result, bin_centers


def _draw_cluster_bar(ax, cluster_res, bin_centers, color, y_axes_frac,
                      height_frac=0.012, p_alpha=CLUSTER_P_ALPHA):
    """Draw a horizontal bar spanning each SIGNIFICANT cluster.

    bin_centers MUST be the axis returned alongside cluster_res by
    run_cluster_test, so the cluster start/end indices line up with the
    plotted times. x is in data coords, y in axes-fraction.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    # half-bin so the bar covers the full extent of edge bins, not just centers
    if len(bin_centers) >= 2:
        half = float(np.median(np.diff(bin_centers))) / 2.0
    else:
        half = 0.0
    trans = ax.get_xaxis_transform()  # x = data, y = axes fraction
    for c in cluster_res['observed_clusters']:
        if c['p_value'] >= p_alpha:
            continue
        x0 = bin_centers[c['start_idx']] - half
        x1 = bin_centers[c['end_idx']] + half
        ax.add_patch(Rectangle(
            (x0, y_axes_frac), x1 - x0, height_frac,
            transform=trans, facecolor=color, edgecolor='none',
            clip_on=False, zorder=5))


# ============================================================
# Per-analysis plots: null vs potent (shuffle + cluster significance bars)
# ============================================================

# We collect cluster results per analysis/recon so they can also go into
# results.yaml. Store (cluster_result, bin_centers) so the axis travels with it.
cluster_results = {}  # cluster_results[analysis][recon_key] = (cluster_res, bin_centers)

for analysis in analyses_present:
    cluster_results[analysis] = {}
    fig, axs = make_fixed_panel_fig(nrows=1, ncols=1)
    ax = axs[0, 0]

    cluster_bars = {}  # recon_key -> (cluster_result, bin_centers, color)

    for recon_key, color in [('null', COL_NULL), ('potent', COL_POTENT)]:
        key = f'results_{analysis}_{recon_key}'
        shuf_key = f'results_shuffled_{analysis}_{recon_key}'
        if key not in output:
            continue

        df = to_iter_df(output[key])
        df_shuf = to_iter_df(output[shuf_key])

        plot_trace(ax, df, color=color, label=f'{recon_key} subspace')
        plot_trace(ax, df_shuf, color=color, label=None, linestyle='--')

        cluster_res, bin_centers = run_cluster_test(df, df_shuf)
        cluster_bars[recon_key] = (cluster_res, bin_centers, color)
        cluster_results[analysis][recon_key] = (cluster_res, bin_centers)
        print(f"[{analysis} / {recon_key}]")
        _print_cluster_summary(cluster_res)

    ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time from stim change (s)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title(ANALYSIS_TITLES.get(analysis, analysis))
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc='upper left', frameon=False, handlelength=1.2,
              borderpad=0.2, labelspacing=0.2)

    # Cluster significance bars — one row per recon, only significant clusters
    for i, (recon_key, (cluster_res, bin_centers, color)) in enumerate(cluster_bars.items()):
        y_bar = 1.02 + 0.025 * i
        _draw_cluster_bar(ax, cluster_res, bin_centers, color, y_bar)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # NOTE: do NOT call tight_layout — it would override the fixed margins
    fig.savefig(fig_dir / f'{analysis}_null_vs_potent.png', dpi=300)
    fig.savefig(fig_dir / f'{analysis}_null_vs_potent.svg')
    plt.close(fig)


# ============================================================
# Combined summary: all analyses on one figure, two columns (null | potent)
# ============================================================

if analyses_present:
    fig, axs = make_fixed_panel_fig(nrows=len(analyses_present), ncols=2)

    for row, analysis in enumerate(analyses_present):
        for col, (recon_key, color) in enumerate([('null', COL_NULL), ('potent', COL_POTENT)]):
            ax = axs[row, col]
            key = f'results_{analysis}_{recon_key}'
            shuf_key = f'results_shuffled_{analysis}_{recon_key}'

            if key in output:
                df = to_iter_df(output[key])
                df_shuf = to_iter_df(output[shuf_key])
                plot_trace(ax, df, color=color, label=recon_key)
                plot_trace(ax, df_shuf, color=COL_SHUF, label='shuffled')

                # Recompute so cluster_res and its axis are guaranteed paired.
                cluster_res, bin_centers = run_cluster_test(df, df_shuf)
                _draw_cluster_bar(ax, cluster_res, bin_centers, color, 0.96)

            ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':')
            ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
            ax.set_ylim(0.35, 1.0)

            if row == 0:
                ax.set_title(f'{recon_key} subspace')
            if row == len(analyses_present) - 1:
                ax.set_xlabel('Time from stim change (s)')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f'{ANALYSIS_TITLES.get(analysis, analysis)}\nbalanced accuracy')
            else:
                ax.set_yticklabels([])

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

    axs[0, 0].legend(loc='upper left', frameon=False, handlelength=1.2,
                     borderpad=0.2, labelspacing=0.2)

    fig.savefig(fig_dir / 'summary_null_vs_potent.png', dpi=300)
    fig.savefig(fig_dir / 'summary_null_vs_potent.svg')
    plt.close(fig)

    # ============================================================
    # By-subspace summary: one panel per subspace, modalities overlaid
    # ============================================================

    subspaces = ['null', 'potent']

    # Only include modalities we actually have keys for
    modalities_present = [
        a for a in analyses_present if a in MODALITY_COLORS
    ]

    if modalities_present:
        fig, axs = make_fixed_panel_fig(nrows=1, ncols=len(subspaces))

        for col, recon_key in enumerate(subspaces):
            ax = axs[0, col]
            clu_bars = []      # (cluster_result, bin_centers, color)

            for analysis in modalities_present:
                color = MODALITY_COLORS[analysis]
                label = MODALITY_LABELS[analysis]

                key = f'results_{analysis}_{recon_key}'
                shuf_key = f'results_shuffled_{analysis}_{recon_key}'
                if key not in output:
                    continue

                df = to_iter_df(output[key])
                df_shuf = to_iter_df(output[shuf_key])

                plot_trace(ax, df, color=color, label=label)
                plot_trace(ax, df_shuf, color=color, label=None, linestyle='--')

                # Always recompute cluster result with its own axis.
                cluster_res, bin_centers = run_cluster_test(df, df_shuf)
                clu_bars.append((cluster_res, bin_centers, color))

            ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':')
            ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
            ax.set_ylim(0.35, 1.0)
            ax.set_title(SUBSPACE_TITLES.get(recon_key, recon_key))
            ax.set_xlabel('Time from stim change (s)')
            if col == 0:
                ax.set_ylabel('Balanced accuracy')
            else:
                ax.set_yticklabels([])

            # One cluster bar row per modality
            for i, (cluster_res, bin_centers, color) in enumerate(clu_bars):
                y_bar = 1.02 + 0.025 * i
                _draw_cluster_bar(ax, cluster_res, bin_centers, color, y_bar)

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        axs[0, 0].legend(loc='upper left', frameon=False, handlelength=1.2,
                         borderpad=0.2, labelspacing=0.2)

        fig.savefig(fig_dir / 'summary_by_subspace.png', dpi=300)
        fig.savefig(fig_dir / 'summary_by_subspace.svg')
        plt.close(fig)

# ============================================================
# Write significant clusters + p-values to results.yaml (cluster only)
# ============================================================

results_summary = {
    'cluster': {
        'test': 'cluster-based permutation test (Maris & Oostenveld 2007), '
                'one-sided observed > null',
        'threshold_percentile': float(CLUSTER_THRESHOLD_PCT),
        'p_alpha': float(CLUSTER_P_ALPHA),
        'cluster_window': (
            list(CLUSTER_WINDOW) if CLUSTER_WINDOW is not None else None
        ),
    },
    'analyses': {},
}

for analysis in analyses_present:
    results_summary['analyses'][analysis] = {}
    for recon_key in ['null', 'potent']:
        key = f'results_{analysis}_{recon_key}'
        shuf_key = f'results_shuffled_{analysis}_{recon_key}'
        if key not in output or shuf_key not in output:
            continue

        df = to_iter_df(output[key])
        df_shuf = to_iter_df(output[shuf_key])

        # Reuse computed result if available, else compute now.
        stored = cluster_results.get(analysis, {}).get(recon_key)
        if stored is not None:
            cluster_res, _ = stored
        else:
            cluster_res, _ = run_cluster_test(df, df_shuf)

        clusters_yaml = [
            {
                't_start': float(c['t_start']),
                't_end': float(c['t_end']),
                'n_bins': int(c['end_idx'] - c['start_idx'] + 1),
                'mass': float(c['mass']),
                'p_value': float(c['p_value']),
                'significant': bool(c['p_value'] < CLUSTER_P_ALPHA),
            }
            for c in cluster_res['observed_clusters']
        ]
        n_sig_clusters = sum(1 for c in clusters_yaml if c['significant'])

        results_summary['analyses'][analysis][recon_key] = {
            'cluster': {
                'n_clusters': len(clusters_yaml),
                'n_significant_clusters': int(n_sig_clusters),
                'clusters': clusters_yaml,
            },
        }

with open(results_out_dir / 'results.yaml', 'w') as f:
    yaml.safe_dump(results_summary, f, sort_keys=False, default_flow_style=False)

print(f"Saved results to {results_out_dir / 'results.yaml'}")
print(f"Saved figures to {fig_dir}")