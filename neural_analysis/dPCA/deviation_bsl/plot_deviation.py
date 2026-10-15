# ============================================================================
# Plotting for dPCA-space deviation analyses (Test 1 and Test 2)
# ============================================================================
# Loads results saved by dpca_deviation.py and produces ONE SVG PER PANEL
#
# Output structure:
#   FIGURE_DIR/dPCA/{area}/bsl_deviation/{run}/
#       test1_<marg>_<condition_slug>.svg                     (per condition)
#       test1_<marg>_overlay_<base_condition_slug>.svg        (ctrl/opto overlay)
#       test2_<marg>_<pair_slug>__deviations.svg              (E_ctrl & E_opto)
#       test2_<marg>_<pair_slug>__delta.svg                   (delta + null)
#       test1_summary_<marg>.svg                              (bar chart)
#       test2_summary_<marg>.svg                              (bar chart)
#
# Exact-size axes:
#   Pass --axes-width / --axes-height (inches) or use --axes-width-cm /
#   --axes-height-cm to set the DATA PANEL size. "Data panel" = the area
#   inside the spines (where the traces live); tick marks, tick labels,
#   axis labels, titles and legends are drawn OUTSIDE this box. The figure
#   size is grown automatically to accommodate them via fixed inch-based
#   margins (see AXES_MARGINS).
#
# Manual x-limits:
#   Pass --xlim-min / --xlim-max (in seconds) to override the time axis range
#   on all time-series panels (Test 1 conditions/overlays, Test 2
#   deviations/delta). Either bound may be omitted, in which case that side
#   falls back to the first/last time center. Summary bar charts are
#   categorical and ignore xlim.
# ============================================================================

import os
import re
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml
import config
from datetime import datetime

# Re-use the loaders from the analysis script (assumed in same directory or PYTHONPATH).
try:
    from neural_analysis.dPCA.deviation_bsl.dPCA_deviation import (
        load_existing_deviation_by_label,
        load_existing_deviation_change_by_label,
    )
except ImportError:
    print("ERROR: cannot import dpca_deviation. Place this script next to "
          "dpca_deviation.py or add it to PYTHONPATH.")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 6, 'axes.titlesize': 8, 'axes.labelsize': 6,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'figure.titlesize': 10, 'axes.linewidth': 1, 'lines.linewidth': 1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLOR_CTRL      = '#2b6cb0'   # blue (fallback for unlabelled modality)
COLOR_VISUAL    = '#2b6cb0'   # blue
COLOR_AUDITORY  = '#c53030'   # red
COLOR_OPTO      = '#805ad5'   # purple
COLOR_NULL      = '#666666'   # gray
COLOR_DELTA     = '#2d3748'   # dark slate
COLOR_SIGCLUSTER = '#fbd38d'  # orange tint
COLOR_CI_ALPHA      = 0.25
COLOR_NULL_ALPHA    = 0.20
COLOR_CLUSTER_ALPHA = 0.30

AXES_W_DEFAULT = 0.9
AXES_H_DEFAULT = 0.6
X_LIM_DEFAULT = [-0.5, 1.5]
LICK_XLIM = [-1.0, 1.0]

MARG_FULL_NAMES = {
    'd': 'decision (d)',
    's': 'stimulus (s)',
    'sd': 'stim x decision (sd)',
    't': 'time-only (t)',
}

# Fixed inch margins around the data axes. The figure grows to
# (axes_w + left + right, axes_h + bottom + top).
AXES_MARGINS = {
    'left':   0.85,   # for ylabel + tick labels
    'right':  0.30,
    'bottom': 0.60,   # for xlabel + tick labels
    'top':    0.65,   # for title (multi-line title-friendly)
}
# A taller top margin for plots whose titles include the legend line.
AXES_MARGINS_LEGEND_INSIDE = dict(AXES_MARGINS, top=0.95)


# ----------------------------------------------------------------------------
# Y-axis linking: panels listed together share a y-limit = union of their
# natural data ranges (largest-amplitude panel sets the scale, nothing clips).
# Identify each panel by (test, marg, slug) where `slug` is EXACTLY the string
# in the output filename:
#   Test 1 overlay : _slug(base)   -> test1_<marg>_overlay_<slug>.svg
#   Test 2 delta   : _slug(label)  -> test2_<marg>_<slug>__delta.svg
# A wrong slug prints a warning at run time listing the available slugs.
# ----------------------------------------------------------------------------
Y_LINK_PAD_FRAC = 0.05   # margin added to each shared range (mimics autoscale)
Y_LINK_GROUPS = [
    # test2_s : auditory vs visual lick_trials delta
   # [
   #     ('test2', 's', 'Auditory_lick_trials_Ctrl_vs_Opto'),
   #     ('test2', 's', 'Visual_lick_trials_Ctrl_vs_Opto'),
   # ],
   # # test2_d : all left vs right licks delta
   # [
   #     ('test2', 'd', 'All_Left_licks_Ctrl_vs_Opto'),
   #     ('test2', 'd', 'All_Right_licks_Ctrl_vs_Opto'),
   # ],
   # # test1_d : all lick left vs right overlay
   # [
   #     ('test1', 'd', 'All_lick_left'),
   #     ('test1', 'd', 'All_lick_right'),
   # ],
   # # test1_s : visual vs auditory lick_trials overlay
   # [
   #     ('test1', 's', 'Visual_lick_trials'),
   #     ('test1', 's', 'Auditory_lick_trials'),
   # ],
]


# ----------------------------------------------------------------------------
# Exact-size axes plumbing
# ----------------------------------------------------------------------------
def make_exact_axes(axes_w_in, axes_h_in, margins=None):
    """Create a figure with a single Axes whose DATA PANEL is exactly
    (axes_w_in x axes_h_in) inches.

    The Axes position rectangle is the region bounded by the spines, i.e.
    the data area. Tick marks, tick labels, axis labels, the title and any
    legend are rendered in the surrounding margins and do NOT shrink this
    box. As long as savefig uses bbox_inches=None (see _save_svg), the panel
    keeps its exact physical size.

    Returns (fig, ax). The figure size is automatically set to
    (axes_w + left + right, axes_h + bottom + top).
    """
    m = dict(AXES_MARGINS if margins is None else margins)
    fig_w = axes_w_in + m['left'] + m['right']
    fig_h = axes_h_in + m['bottom'] + m['top']
    fig = plt.figure(figsize=(fig_w, fig_h))
    left   = m['left']   / fig_w
    bottom = m['bottom'] / fig_h
    width  = axes_w_in   / fig_w
    height = axes_h_in   / fig_h
    ax = fig.add_axes([left, bottom, width, height])
    return fig, ax


# ----------------------------------------------------------------------------
# Filename / label helpers
# ----------------------------------------------------------------------------
def _slug(s):
    """Make a label filesystem-safe."""
    s = s.replace('vs', 'vs').strip()
    s = re.sub(r'[^\w\-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


# ----------------------------------------------------------------------------
# Stats export (YAML)
# ----------------------------------------------------------------------------
def _py(x):
    """Coerce numpy scalars/arrays to native Python for clean YAML.
    NaN/Inf -> None so the file stays valid/parseable."""
    if isinstance(x, np.generic):
        x = x.item()
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:                 # 0-d array (e.g. scalar from np.savez)
            return _py(x.item())
        return [_py(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

def _cluster_list(res, time_centers, p_alpha):
    """Per-cluster stats for one panel. Includes ALL clusters that crossed the
    cluster-forming threshold (not just significant ones); 'significant' flags
    p < p_alpha. 'mass' is the cluster weight = sum over the cluster of
    (statistic - threshold)."""
    if 'cluster_mass' not in res:
        return []
    masses = np.asarray(res.get('cluster_mass', []),  float)
    pvals  = np.asarray(res.get('cluster_p_value', []), float)
    t0s    = np.asarray(res.get('cluster_t_start', []), float)
    t1s    = np.asarray(res.get('cluster_t_end', []),   float)
    i0s    = np.asarray(res.get('cluster_start_idx', []), int)
    i1s    = np.asarray(res.get('cluster_end_idx', []),   int)
    sig = res.get('observed', res.get('deviation', None))   # signal for in-cluster peak
    sig = None if sig is None else np.asarray(sig, float)
    out = []
    for k in range(len(masses)):
        i0 = int(i0s[k]) if k < len(i0s) else None
        i1 = int(i1s[k]) if k < len(i1s) else None
        entry = {
            'index':     int(k),
            't_start':   _py(t0s[k]) if k < len(t0s) else None,
            't_end':     _py(t1s[k]) if k < len(t1s) else None,
            'idx_start': i0,
            'idx_end':   i1,
            'n_bins':    (i1 - i0 + 1) if (i0 is not None and i1 is not None) else None,
            'mass':      _py(masses[k]),                 # cluster weight
            'p_value':   _py(pvals[k]) if k < len(pvals) else None,
            'significant': bool(pvals[k] < p_alpha) if k < len(pvals) else None,
        }
        if sig is not None and i0 is not None:
            seg = sig[i0:i1 + 1]
            if seg.size and not np.all(np.isnan(seg)):
                j = int(np.nanargmax(np.abs(seg)))
                entry['peak_value'] = _py(seg[j])
                entry['peak_time']  = _py(time_centers[i0 + j])
        out.append(entry)
    return out


def _test1_entry(label, res, time_centers, marg, p_alpha):
    e = {'svg': f"test1_{marg}_{_slug(label)}.svg"}
    if res.get('skipped', False):
        e['skipped'] = True
        e['n_trials'] = int(res.get('n_trials', 0))
        return e
    exc = np.asarray(res['deviation'], float)
    e['n_trials'] = int(res['n_trials'])
    if exc.size and not np.all(np.isnan(exc)):
        j = int(np.nanargmax(exc))
        e['peak_deviation'] = _py(exc[j])
        e['peak_time']      = _py(time_centers[j])
    pv = np.asarray(res.get('p_values', []), float)
    if pv.size:
        e['n_timebins']       = int(pv.size)
        e['n_sig_timepoints'] = int(np.nansum(pv < p_alpha))   # pointwise, uncorrected
    e['clusters']       = _cluster_list(res, time_centers, p_alpha)
    e['n_sig_clusters'] = int(sum(1 for c in e['clusters'] if c.get('significant')))
    return e


def _test2_entry(label, res, time_centers, marg, p_alpha):
    slug = _slug(label)
    e = {'svg_deviations': f"test2_{marg}_{slug}__deviations.svg",
         'svg_delta':      f"test2_{marg}_{slug}__delta.svg"}
    if res.get('skipped', False):
        e['skipped'] = True
        e['n_trials_ctrl'] = int(res.get('n_trials_ctrl', 0))
        e['n_trials_opto'] = int(res.get('n_trials_opto', 0))
        return e
    obs = np.asarray(res['observed'], float)
    e['n_trials_ctrl'] = int(res['n_trials_ctrl'])
    e['n_trials_opto'] = int(res['n_trials_opto'])
    if obs.size and not np.all(np.isnan(obs)):
        j = int(np.nanargmax(np.abs(obs)))
        e['peak_abs_delta']    = _py(abs(obs[j]))
        e['peak_delta_signed'] = _py(obs[j])
        e['peak_time']         = _py(time_centers[j])
    for k_in, k_out in [('deviation_ctrl', 'peak_deviation_ctrl'),
                        ('deviation_opto', 'peak_deviation_opto')]:
        a = np.asarray(res.get(k_in, []), float)
        if a.size and not np.all(np.isnan(a)):
            e[k_out] = _py(np.nanmax(a))
    pv = np.asarray(res.get('p_values', []), float)
    if pv.size:
        e['n_timebins']       = int(pv.size)
        e['n_sig_timepoints'] = int(np.nansum(pv < p_alpha))
    e['clusters']       = _cluster_list(res, time_centers, p_alpha)
    e['n_sig_clusters'] = int(sum(1 for c in e['clusters'] if c.get('significant')))
    return e


def write_stats_yaml(results_dir, t1, t2, t1_overlays, p_alpha,
                     out_name='deviation_stats.yaml'):
    """Write one YAML of all relevant stats for every panel in this results dir.

    t1 / t2     : {marg: (by_label, file_meta)}
    t1_overlays : {marg: {base: {kind: (label, res)}}}
    """
    doc = {
        'results_dir': results_dir,
        'generated':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'p_alpha':     float(p_alpha),
        'notes': ("Cluster 'mass' = cluster weight = sum over the cluster of "
                  "(statistic - cluster-forming threshold) (Maris & Oostenveld). "
                  "'significant' = cluster p < p_alpha. 'p_value' per cluster is "
                  "the permutation cluster-mass p. n_sig_timepoints counts bins "
                  "with pointwise permutation p < p_alpha (uncorrected). Test 1 "
                  "statistic is E(t); Test 2 is delta = E_ctrl - E_opto."),
        'meta': {},
        'test1': {},
        'test2': {},
    }

    # File-level meta (grab from any available file_meta).
    fm_any = next((fm for d in (t1, t2) for _, (_, fm) in d.items() if fm), None)
    if fm_any is not None:
        tc = np.asarray(fm_any.get('time_centers', []), float)
        if tc.size:
            doc['meta']['time_range'] = [_py(tc[0]), _py(tc[-1])]
            doc['meta']['n_timebins'] = int(tc.size)
        for k in ('n_bootstrap', 'ci', 'cluster_threshold_percentile'):
            if k in fm_any:
                doc['meta'][k] = _py(fm_any[k])
        for k in ('components', 'pre_window'):
            if k in fm_any:
                doc['meta'][k] = _py(fm_any[k])

    # Test 1
    for marg, (by_label, fm) in sorted(t1.items()):
        tc = np.asarray(fm['time_centers'], float)
        conds = {lbl: _test1_entry(lbl, res, tc, marg, p_alpha)
                 for lbl, res in by_label.items()}
        block = {'conditions': conds}
        overlays = {}
        for base, d in t1_overlays.get(marg, {}).items():
            if 'ctrl' in d and 'opto' in d:
                overlays[base] = {
                    'svg':        f"test1_{marg}_overlay_{_slug(base)}.svg",
                    'ctrl_label': d['ctrl'][0],
                    'opto_label': d['opto'][0],
                }
        if overlays:
            block['overlays'] = overlays
        doc['test1'][marg] = block

    # Test 2
    for marg, (by_label, fm) in sorted(t2.items()):
        tc = np.asarray(fm['time_centers'], float)
        pairs = {lbl: _test2_entry(lbl, res, tc, marg, p_alpha)
                 for lbl, res in by_label.items()}
        doc['test2'][marg] = {'pairs': pairs}

    out_path = os.path.join(results_dir, out_name)
    with open(out_path, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)
    print(f"  Saved stats: {os.path.basename(out_path)}")
    return out_path


def _is_lick_aligned(results_dir):
    return 'firstlick' in os.path.basename(results_dir.rstrip(os.sep)).lower()


def _is_ctrl(label):
    return '(Control)' in label or '(Ctrl)' in label


def _is_opto(label):
    return '(Opto)' in label


def _modality_color(label, default=COLOR_CTRL):
    """Opto -> purple, visual -> blue, auditory -> red, else `default`."""
    low = label.lower()
    if _is_opto(label):
        return COLOR_OPTO
    if 'visual' in low:
        return COLOR_VISUAL
    if 'auditory' in low:
        return COLOR_AUDITORY
    return default


def _base_condition_name(label):
    """Strip '(Control)' / '(Opto)' suffix to find matching ctrl/opto pairs."""
    for tag in [' (Control)', ' (Ctrl)', ' (Opto)']:
        if label.endswith(tag):
            return label[:-len(tag)]
    return label


def _shade_clusters(ax, time_centers, result,
                    color=COLOR_SIGCLUSTER, alpha=COLOR_CLUSTER_ALPHA):
    """Shade significant cluster windows (boolean cluster_mask)."""
    if 'cluster_mask' not in result:
        return
    mask = np.asarray(result['cluster_mask']).astype(bool)
    if not np.any(mask):
        return
    diffs = np.diff(mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, len(mask)]
    for s, e in zip(starts, ends):
        ax.axvspan(time_centers[s],
                   time_centers[min(e, len(time_centers) - 1)],
                   color=color, alpha=alpha, zorder=0, lw=0)


def _null_band(null_dist, lo=2.5, hi=97.5):
    return (np.nanpercentile(null_dist, lo, axis=0),
            np.nanpercentile(null_dist, hi, axis=0))


def _overlay_yrange(d):
    """Natural y data-range of a Test-1 ctrl/opto overlay panel."""
    los, his = [0.0], [0.0]          # include y=0 (axhline)
    for kind in ('ctrl', 'opto'):
        if kind not in d:
            continue
        _, res = d[kind]
        for key in ('ci_low', 'ci_high', 'deviation'):
            a = np.asarray(res[key], dtype=float)
            if a.size and not np.all(np.isnan(a)):
                los.append(float(np.nanmin(a)))
                his.append(float(np.nanmax(a)))
        if 'cluster_mask' in res and np.any(np.asarray(res['cluster_mask'])):
            exc = np.asarray(res['deviation'], dtype=float)
            if exc.size and not np.all(np.isnan(exc)):
                his.append(float(np.nanmax(exc)) * (1.10 if kind == 'opto' else 1.05))
    return (min(los), max(his))


def _delta_yrange(res):
    """Natural y data-range of a Test-2 delta panel."""
    los, his = [0.0], [0.0]          # include y=0 (axhline)
    arrays = []
    if 'null_distribution' in res:
        lo, hi = _null_band(res['null_distribution'])
        arrays += [lo, hi]
    for key in ('ci_low', 'ci_high', 'observed'):
        if key in res:
            arrays.append(res[key])
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if a.size and not np.all(np.isnan(a)):
            los.append(float(np.nanmin(a)))
            his.append(float(np.nanmax(a)))
    return (min(los), max(his))


def _compute_group_ylims(groups, t1_overlays, t2_by_marg, pad_frac=Y_LINK_PAD_FRAC):
    """Map (test, marg, slug) -> shared (lo, hi) for every y-link group.

    t1_overlays : {marg: {base: {kind: (label, res)}}}
    t2_by_marg  : {marg: {label: res}}
    """
    out = {}
    for group in groups:
        lo_all, hi_all, members = np.inf, -np.inf, []
        for (test, marg, slug) in group:
            yr = None
            if test == 'test1':
                by_base = t1_overlays.get(marg, {})
                for base, d in by_base.items():
                    if _slug(base) == slug and 'ctrl' in d and 'opto' in d:
                        yr = _overlay_yrange(d); break
                if yr is None:
                    avail = [_slug(b) for b, d in by_base.items()
                             if 'ctrl' in d and 'opto' in d]
                    print(f"  [y-link] WARNING: no test1 overlay for "
                          f"marg='{marg}' slug='{slug}'. Available: {avail}")
            elif test == 'test2':
                by_label = t2_by_marg.get(marg, {})
                for label, res in by_label.items():
                    if res.get('skipped', False):
                        continue
                    if _slug(label) == slug:
                        yr = _delta_yrange(res); break
                if yr is None:
                    avail = [_slug(l) for l, r in by_label.items()
                             if not r.get('skipped', False)]
                    print(f"  [y-link] WARNING: no test2 delta for "
                          f"marg='{marg}' slug='{slug}'. Available: {avail}")
            if yr is None:
                continue
            members.append((test, marg, slug))
            lo_all = min(lo_all, yr[0]); hi_all = max(hi_all, yr[1])
        if not members or not np.isfinite(lo_all):
            continue
        pad = pad_frac * (hi_all - lo_all) if hi_all > lo_all else 0.0
        ylim = (lo_all - pad, hi_all + pad)
        for key in members:
            out[key] = ylim
        print(f"  [y-link] {members} -> ylim=({ylim[0]:.4g}, {ylim[1]:.4g})")
    return out


def _decorate_time_axis(ax, time_centers, pre_window=None, show_xlabel=True,
                        xlim=None):
    """Add reference lines / pre-window shading and set the x-limits.

    xlim : None or (lo, hi). Either entry may be None to fall back to the
           first / last time center. When xlim is None, the full data range
           is used.
    """
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6, zorder=0)
    ax.axhline(0, color='k', lw=0.4, ls='-',  alpha=0.4, zorder=0)
    if pre_window is not None:
        ax.axvspan(pre_window[0], pre_window[1], color='#cbd5e0',
                   alpha=0.25, zorder=-1, lw=0)
    if xlim is not None:
        lo = xlim[0] if xlim[0] is not None else time_centers[0]
        hi = xlim[1] if xlim[1] is not None else time_centers[-1]
        ax.set_xlim(lo, hi)
    else:
        ax.set_xlim(time_centers[0], time_centers[-1])
    if show_xlabel:
        ax.set_xlabel('Time (s)')


def _save_svg(fig, out_path):
    # bbox_inches=None is essential: 'tight' would crop the figure and break
    # the exact data-panel sizing guaranteed by make_exact_axes.
    fig.savefig(out_path, format='svg', bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ============================================================================
# TEST 1 — one SVG per condition
# ============================================================================
def plot_test1_condition(label, res, time_centers, pre_window, marg_key,
                         out_path, axes_w, axes_h, p_alpha=0.01, xlim=None):
    color = _modality_color(label)
    fig, ax = make_exact_axes(axes_w, axes_h)

    _shade_clusters(ax, time_centers, res)
    null_lo, null_hi = _null_band(res['null_distribution'])
    ax.fill_between(time_centers, null_lo, null_hi,
                    color=COLOR_NULL, alpha=COLOR_NULL_ALPHA, lw=0)
    ax.fill_between(time_centers, res['ci_low'], res['ci_high'],
                    color=color, alpha=COLOR_CI_ALPHA, lw=0)
    ax.plot(time_centers, res['deviation'], color=color, lw=1.3)

    _decorate_time_axis(ax, time_centers, pre_window=pre_window, xlim=xlim)
    ax.set_ylabel('E(t)')
    ax.set_title(f"[{marg_key}] {label}\nn={res['n_trials']}", fontsize=8)
    _save_svg(fig, out_path)


def plot_test1_overlay(base, d, time_centers, pre_window, marg_key,
                       out_path, axes_w, axes_h, p_alpha=0.01, xlim=None, force_ylim=None):
    fig, ax = make_exact_axes(axes_w, axes_h)
    for kind in ('ctrl', 'opto'):
        if kind not in d: continue
        label, res = d[kind]
        color = COLOR_OPTO if kind == 'opto' else _modality_color(base)
        ax.fill_between(time_centers, res['ci_low'], res['ci_high'],
                        color=color, alpha=COLOR_CI_ALPHA, lw=0)
        ax.plot(time_centers, res['deviation'], color=color, lw=1.3, label=kind)
        # mark sig clusters with thin segment above the trace
        if 'cluster_mask' in res and np.any(np.asarray(res['cluster_mask'])):
            mask = np.asarray(res['cluster_mask']).astype(bool)
            yval = np.nanmax(res['deviation']) * (1.05 if kind == 'ctrl' else 1.10)
            y = np.full(len(time_centers), np.nan)
            y[mask] = yval
            ax.plot(time_centers, y, color=color, lw=2.0, solid_capstyle='butt')
    _decorate_time_axis(ax, time_centers, pre_window=pre_window, xlim=xlim)
    if force_ylim is not None:
        ax.set_ylim(force_ylim)
    ax.set_ylabel('E(t)')
    ax.set_title(f"[{marg_key}] {base}\nctrl vs opto", fontsize=8)
    ax.legend(loc='best', frameon=False)
    _save_svg(fig, out_path)


# ============================================================================
# TEST 2 — two SVGs per pair (deviations + delta)
# ============================================================================
def plot_test2_deviations(label, res, time_centers, pre_window, marg_key,
                          out_path, axes_w, axes_h, xlim=None):
    fig, ax = make_exact_axes(axes_w, axes_h)
    ax.plot(time_centers, res['deviation_ctrl'], color=_modality_color(label), lw=1.3, label='ctrl')
    ax.plot(time_centers, res['deviation_opto'], color=COLOR_OPTO, lw=1.3, label='opto')
    _decorate_time_axis(ax, time_centers, pre_window=pre_window, xlim=xlim)
    ax.set_ylabel('E(t)')
    ax.set_title(f"[{marg_key}] {label}\n"
                 f"ctrl n={res['n_trials_ctrl']}, opto n={res['n_trials_opto']}",
                 fontsize=8)
    ax.legend(loc='best', frameon=False)
    _save_svg(fig, out_path)


def plot_test2_delta(label, res, time_centers, pre_window, marg_key,
                     out_path, axes_w, axes_h, p_alpha=0.01, xlim=None, force_ylim=None):
    color = _modality_color(label, default=COLOR_DELTA)
    fig, ax = make_exact_axes(axes_w, axes_h)
    _shade_clusters(ax, time_centers, res)
    null_lo, null_hi = _null_band(res['null_distribution'])
    ax.fill_between(time_centers, null_lo, null_hi,
                    color=COLOR_NULL, alpha=COLOR_NULL_ALPHA, lw=0)
    ax.fill_between(time_centers, res['ci_low'], res['ci_high'],
                    color=color, alpha=COLOR_CI_ALPHA, lw=0)
    ax.plot(time_centers, res['observed'], color=color, lw=1.3)
    _decorate_time_axis(ax, time_centers, pre_window=pre_window, xlim=xlim)
    if force_ylim is not None:
        ax.set_ylim(force_ylim)
    ax.set_ylabel('E(t)')
    ax.set_title(f"[{marg_key}] {label}\n"
                 r'$\Delta$ = ctrl - opto', fontsize=8)
    _save_svg(fig, out_path)


# ============================================================================
# SUMMARY BAR CHARTS — one SVG per (test, marg)
# ============================================================================
def _peak_metric(res, key_obs):
    """Use peak deviation (or peak |delta|) as the summary metric, since the
    window-averaged stats are no longer saved."""
    arr = np.asarray(res[key_obs])
    if np.all(np.isnan(arr)):
        return np.nan
    if key_obs == 'observed':       # signed delta: take signed peak by |.|
        i = np.nanargmax(np.abs(arr))
        return float(arr[i])
    return float(np.nanmax(arr))


def _has_sig_cluster(res, p_alpha):
    if 'cluster_p_value' not in res:
        return False
    pvals = np.asarray(res['cluster_p_value'])
    if pvals.size == 0:
        return False
    return bool(np.any(pvals < p_alpha))


def plot_test1_summary(marg, by_label, out_path, axes_w, axes_h, p_alpha=0.01):
    rows = [(lbl, res) for lbl, res in by_label.items()
            if not res.get('skipped', False)]
    if not rows: return
    fig, ax = make_exact_axes(axes_w, axes_h,
                              margins=dict(AXES_MARGINS, bottom=2.0))  # extra room for rotated x labels
    xs = np.arange(len(rows))
    vals   = [_peak_metric(r, 'deviation') for _, r in rows]
    colors = [_modality_color(l, default=COLOR_CTRL if _is_ctrl(l) else '#718096')
              for l, _ in rows]
    ax.bar(xs, vals, color=colors, edgecolor='k', lw=0.5)
    for x, (lbl, res), v in zip(xs, rows, vals):
        if _has_sig_cluster(res, p_alpha):
            ax.text(x, v, '*', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _ in rows], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('peak E(t)')
    ax.set_title(f"[{marg}] Test 1 summary — peak deviation", fontsize=8)
    _save_svg(fig, out_path)


def plot_test2_summary(marg, by_label, out_path, axes_w, axes_h, p_alpha=0.01):
    rows = [(lbl, res) for lbl, res in by_label.items()
            if not res.get('skipped', False)]
    if not rows: return
    fig, ax = make_exact_axes(axes_w, axes_h,
                              margins=dict(AXES_MARGINS, bottom=2.0))
    xs = np.arange(len(rows))
    vals = [_peak_metric(r, 'observed') for _, r in rows]
    colors = [_modality_color(l, default=COLOR_DELTA) for l, _ in rows]
    ax.bar(xs, vals, color=colors, edgecolor='k', lw=0.5)
    for x, (lbl, res), v in zip(xs, rows, vals):
        if _has_sig_cluster(res, p_alpha):
            y_off = max(0.01, 0.02 * abs(v))
            ax.text(x, v + y_off * np.sign(v if v != 0 else 1.0),
                    '*', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _ in rows], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel(r'peak $\Delta$E(t) (signed)')
    ax.set_title(f"[{marg}] Test 2 summary — peak flattening", fontsize=8)
    _save_svg(fig, out_path)


# ============================================================================
# MAIN
# ============================================================================
def _figure_dir(results_dir, subdir=''):
    """FIGURE_DIR/dPCA/{area}/bsl_deviation/{run}[/subdir] for a results dir."""
    run = os.path.basename(os.path.normpath(results_dir))
    area = None
    meta_path = os.path.join(results_dir, 'metadata_deviation.yaml')
    if not os.path.isfile(meta_path):  # pre-rename runs
        meta_path = os.path.join(results_dir, 'metadata_excursion.yaml')
    if os.path.isfile(meta_path):
        with open(meta_path, 'r') as f:
            area = (yaml.safe_load(f) or {}).get('area')
    if area is None:  # fall back to .../dPCA/{area}/bsl_deviation/{run}
        area = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(results_dir))))
    area = ''.join(area) if isinstance(area, (list, tuple)) else str(area)
    return os.path.join(config.FIGURE_DIR, 'dPCA', area, 'bsl_deviation', run, subdir)


def plot_all(results_dir, axes_w, axes_h, p_alpha=0.01, svg_subdir='',
             xlim=None, y_link_groups=None):
    svg_dir = _figure_dir(results_dir, svg_subdir)
    os.makedirs(svg_dir, exist_ok=True)

    xlim_eff = LICK_XLIM if _is_lick_aligned(results_dir) else xlim
    if _is_lick_aligned(results_dir):
        print(f"  Lick-aligned run detected -> xlim = {LICK_XLIM}")

    # Files written before the excursion -> deviation rename use the old prefix.
    pre_t1, pre_t2 = {}, {}  # marg -> filename prefix present on disk
    for f in os.listdir(results_dir):
        if not f.endswith('.npz'): continue
        for word in ('deviation', 'excursion'):
            if f.startswith(f'{word}_vs_baseline_'):
                pre_t1.setdefault(f[len(f'{word}_vs_baseline_'):-4], f'{word}_vs_baseline_')
            elif f.startswith(f'{word}_change_'):
                pre_t2.setdefault(f[len(f'{word}_change_'):-4], f'{word}_change_')
    margs_t1 = sorted(pre_t1)
    margs_t2 = sorted(pre_t2)

    # ---- Load all results once ----
    t1 = {}
    for marg in margs_t1:
        path = os.path.join(results_dir, f'{pre_t1[marg]}{marg}.npz')
        by_label, file_meta = load_existing_deviation_by_label(path)
        if by_label:
            t1[marg] = (by_label, file_meta)

    t2 = {}
    for marg in margs_t2:
        path = os.path.join(results_dir, f'{pre_t2[marg]}{marg}.npz')
        by_label, file_meta = load_existing_deviation_change_by_label(path)
        if by_label:
            t2[marg] = (by_label, file_meta)

    # ---- Pre-build Test-1 ctrl/opto overlay groupings ----
    t1_overlays = {}
    for marg, (by_label, _) in t1.items():
        by_base = {}
        for label, res in by_label.items():
            if res.get('skipped', False):
                continue
            base = _base_condition_name(label)
            kind = 'opto' if _is_opto(label) else ('ctrl' if _is_ctrl(label) else 'other')
            by_base.setdefault(base, {})[kind] = (label, res)
        t1_overlays[marg] = by_base

    # ---- Resolve shared y-limits for linked panels ----
    t2_by_marg = {marg: bl for marg, (bl, _) in t2.items()}
    group_ylims = _compute_group_ylims(y_link_groups or [], t1_overlays, t2_by_marg)

    # ----- Test 1 -----
    for marg, (by_label, file_meta) in t1.items():
        time_centers = np.asarray(file_meta['time_centers'])
        pre_window   = tuple(file_meta['pre_window'])
        print(f"\n[Test 1] Plotting {marg}: {len(by_label)} conditions...")

        for label, res in by_label.items():
            if res.get('skipped', False): continue
            out = os.path.join(svg_dir, f"test1_{marg}_{_slug(label)}.svg")
            plot_test1_condition(label, res, time_centers, pre_window, marg,
                                  out, axes_w, axes_h, p_alpha=p_alpha, xlim=xlim_eff)

        for base, d in t1_overlays[marg].items():
            if 'ctrl' in d and 'opto' in d:
                out = os.path.join(svg_dir, f"test1_{marg}_overlay_{_slug(base)}.svg")
                fy = group_ylims.get(('test1', marg, _slug(base)))
                plot_test1_overlay(base, d, time_centers, pre_window, marg,
                                    out, axes_w, axes_h, p_alpha=p_alpha,
                                    xlim=xlim_eff, force_ylim=fy)

        plot_test1_summary(marg, by_label,
                           os.path.join(svg_dir, f"test1_summary_{marg}.svg"),
                           axes_w, axes_h, p_alpha=p_alpha)


    # ----- Test 2 -----
    for marg, (by_label, file_meta) in t2.items():
        time_centers = np.asarray(file_meta['time_centers'])
        pre_window   = tuple(file_meta['pre_window'])
        print(f"\n[Test 2] Plotting {marg}: {len(by_label)} pairs...")
        for label, res in by_label.items():
            if res.get('skipped', False): continue
            slug = _slug(label)
            plot_test2_deviations(label, res, time_centers, pre_window, marg,
                                   os.path.join(svg_dir, f"test2_{marg}_{slug}__deviations.svg"),
                                   axes_w, axes_h, xlim=xlim_eff)
            fy = group_ylims.get(('test2', marg, slug))
            plot_test2_delta(label, res, time_centers, pre_window, marg,
                              os.path.join(svg_dir, f"test2_{marg}_{slug}__delta.svg"),
                              axes_w, axes_h, p_alpha=p_alpha, xlim=xlim_eff,
                              force_ylim=fy)

        plot_test2_summary(marg, by_label,
                           os.path.join(svg_dir, f"test2_summary_{marg}.svg"),
                           axes_w, axes_h, p_alpha=p_alpha)

    print(f"\nAll SVGs in: {svg_dir}")

    # ---- Write consolidated stats YAML ----
    try:
        write_stats_yaml(results_dir, t1, t2, t1_overlays, p_alpha)
    except Exception as e:
        print(f"  WARNING: failed to write stats YAML: {e}")

def _find_latest_results_dir(parent_dir):
    if not os.path.isdir(parent_dir):
        return None
    candidates = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full): continue
        try: entries = os.listdir(full)
        except OSError: continue
        if any(f.startswith(('deviation_', 'excursion_')) and f.endswith('.npz') for f in entries):
            candidates.append((os.path.getmtime(full), full))
    if not candidates: return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _main_cli():
    main_dir = str(config.resolve_pp("dPCA", "lALM", "bsl_deviation"))

    parser = argparse.ArgumentParser(
        description='Plot deviation analysis results as one SVG per panel, '
                    'each with the data axes drawn at an exact physical size.')
    parser.add_argument('--main-dir', type=str, default=main_dir,
                        help='Parent directory whose subdirectories are each '
                             'treated as an deviation results dir.')
    parser.add_argument('--svg-subdir', type=str, default='',
                        help="Optional subdirectory (inside the run's figure folder) for SVG output.")
    parser.add_argument('--p-alpha', type=float, default=0.01)

    # Exact axes size — accept either inches or cm. Inches take precedence
    # if both are given.
    parser.add_argument('--axes-width',  type=float, default=None,
                        help='Data panel width in inches.')
    parser.add_argument('--axes-height', type=float, default=None,
                        help='Data panel height in inches.')
    parser.add_argument('--axes-width-cm',  type=float, default=None,
                        help='Data panel width in cm.')
    parser.add_argument('--axes-height-cm', type=float, default=None,
                        help='Data panel height in cm.')

    # Manual x-limits (seconds) for the time-series panels. Either may be
    # omitted to fall back to the data edge on that side.
    parser.add_argument('--xlim-min', type=float, default=None,
                        help='Lower x-limit in seconds for time-series panels. '
                             'Omit to use the first time center.')
    parser.add_argument('--xlim-max', type=float, default=None,
                        help='Upper x-limit in seconds for time-series panels. '
                             'Omit to use the last time center.')
    args = parser.parse_args()

    # Resolve axes size, default 2.5 x 1.6 inches.
    CM_PER_IN = 2.54
    if args.axes_width is not None:
        axes_w = args.axes_width
    elif args.axes_width_cm is not None:
        axes_w = args.axes_width_cm / CM_PER_IN
    else:
        axes_w = AXES_W_DEFAULT
    if args.axes_height is not None:
        axes_h = args.axes_height
    elif args.axes_height_cm is not None:
        axes_h = args.axes_height_cm / CM_PER_IN
    else:
        axes_h = AXES_H_DEFAULT
    print(f"Data panel size: {axes_w:.3f} x {axes_h:.3f} inches "
          f"({axes_w*CM_PER_IN:.2f} x {axes_h*CM_PER_IN:.2f} cm)")

    # Resolve manual x-limits.
    if args.xlim_min is not None or args.xlim_max is not None:
        xlim = (args.xlim_min, args.xlim_max)
        print(f"Manual xlim: ({args.xlim_min}, {args.xlim_max}) s "
              "(None = data edge)")
    else:
        xlim = X_LIM_DEFAULT

    main_dir = args.main_dir
    if not os.path.isdir(main_dir):
        print(f'ERROR: main-dir not found: {main_dir}')
        sys.exit(1)

    # Collect every immediate subdirectory; each is treated as deviation_DIR.
    subdirs = sorted(
        os.path.join(main_dir, name)
        for name in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, name))
    )
    if not subdirs:
        print(f'ERROR: no subdirectories found under {main_dir}')
        sys.exit(1)

    print(f'Found {len(subdirs)} subdirectories to process.')

    n_ok, n_failed = 0, 0
    for i, deviation_dir in enumerate(subdirs, 1):
        print(f'\n[{i}/{len(subdirs)}] Processing: {deviation_dir}')
        try:
            plot_all(deviation_dir, axes_w=axes_w, axes_h=axes_h,
                     p_alpha=args.p_alpha, svg_subdir=args.svg_subdir,
                     xlim=xlim, y_link_groups=Y_LINK_GROUPS)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            print(f'  ERROR processing {deviation_dir}: {e}')

    print(f'\nDone. {n_ok} succeeded, {n_failed} failed.')

if __name__ == '__main__':
    _main_cli()