# ============================================================================
# dPCA-space deviation-from-Baseline Analysis
# ============================================================================
# Two tests:
#
#   Test 1 (deviation_vs_baseline)
#       For a single condition, ask: does the trial-averaged trajectory leave
#       its pre-stim location?
#         E(t) = || x_bar(t) - x_bar_pre ||      (dPCA-space Euclidean)
#       Null: circular shift of each trial's time axis (breaks stim-locking,
#       preserves within-trial autocorrelation).
#       Bootstrap: resample trials with replacement.
#       Optionally one-sided (E > 0 expected).
#
#   Test 2 (deviation_change)
#       For a (ctrl, opto) pair within the same task condition, ask: is the
#       opto deviation smaller than the ctrl deviation ("flattening")?
#         dE(t) = E_ctrl(t) - E_opto(t)
#       Baseline x_bar_pre is computed from POOLED ctrl+opto trials, per
#       session, so the reference is identical for both arms.
#       Null: pool ctrl+opto trials within session, shuffle into ctrl-sized
#       and opto-sized groups (optionally balanced), recompute dE(t).
#
# Both tests use:
#   - the same dPCA decoder/marginalization structure as the parent script,
#   - per-session trial structure (sessions shuffled independently),
#   - cluster-based permutation correction (Maris & Oostenveld 2007).
#
# On-disk format mirrors the parent script: incremental re-runs are supported
# via --output-dir, matching by label (Test 1) or label (Test 2).
# ============================================================================

# ----------------------------------------------------------------------------
# BLAS thread control — set before importing numpy
# ----------------------------------------------------------------------------
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from pathlib import Path
from neural_analysis import utils
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import yaml
import argparse
import sys
from scipy.ndimage import label as ndi_label
from joblib import Parallel, delayed
import config

mpl.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 6,
    'figure.titlesize': 10, 'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
DEFAULT_DPCA_AREA = "lALM"
DEFAULT_DPCA_DIR = None  # None -> config.resolve_pp("dPCA", DEFAULT_DPCA_AREA, "subspace_results")
DEFAULT_PATH_ROOT = config.DATA_DIR_EXP
DEFAULT_EXPERIMENT = ["ALM_1", "ALM_2", "ALM_3"]
DEFAULT_ANIMALS = "all"
DEFAULT_SESSIONS_EXCLUDE = ['008202304121458', '006202306141358', '021202402021327',
                            '022202403051220', '026202403191125', '026202403201103']
DEFAULT_OPTO_LAT = 'unilat'
DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_CI = 95
DEFAULT_ALIGN_TO = 'stimChange'
DEFAULT_SUBSAMPLE_TRIALS = False
DEFAULT_PRE_WINDOW = (-1, 0)       # baseline window relative to align event
DEFAULT_CLUSTER_TIME_WINDOW = None      # None = use whole trace for cluster test
DEFAULT_CLUSTER_THRESHOLD_PCTILE = 95.0
DEFAULT_P_ALPHA = 0.01
DEFAULT_N_JOBS = -1
DEFAULT_PARALLEL_BACKEND = 'loky'

# Default dimensions to include from each marginalization (applied uniformly
# across all marginalizations). Override via the `marg_components` argument
# of `run_deviation_analysis` to loop over different component sets from a
# calling script.
DEFAULT_MARG_COMPONENTS = [0]

# ============================================================================
# CONDITION DEFINITIONS
# ----------------------------------------------------------------------------
# Two structures:
#   1. SINGLE_CONDITIONS[marg]  -> list of (filter_dict, label) for Test 1.
#   2. PAIRED_CONDITIONS[marg]  -> list of (filter_ctrl, filter_opto, label)
#                                  for Test 2.
# Labels must be unique within each list (used as the on-disk key for
# incremental matching).
# ============================================================================
def get_single_conditions(opto_lat):
    """Conditions for Test 1 (deviation vs baseline within a single condition).

    Includes ctrl conditions (sanity check: deviation should be large), opto
    conditions (the real interesting test: is deviation small?), and a couple
    of pre-stim-only controls for negative-control purposes.
    """
    SC = {
        's': [
            ({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'Visual lick trials (Control)'),
            ({'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'Auditory lick trials (Control)'),

            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'Visual lick left (Control)'),
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'Visual lick right (Control)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Auditory lick left (Control)'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Auditory lick right (Control)'),

            ({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual lick trials (Opto)'),
            ({'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory lick trials (Opto)'),

            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual lick left (Opto)'),
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual lick right (Opto)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory lick left (Opto)'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory lick right (Opto)'),
        ],

        'd': [
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'Visual hits (Control)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Auditory hits (Control)'),
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'Visual lick left (Control)'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Auditory lick right (Control)'),

            ({'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': False},
             'All lick right (Control)'),
            ({'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': False},
             'All lick left (Control)'),

            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual hits (Opto)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory hits (Opto)'),
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual lick left (Opto)'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory lick right (Opto)'),
            ({'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'All lick right (Opto)'),
            ({'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'All lick left (Opto)'),
        ]
    }
    return SC


def get_paired_conditions(opto_lat):
    """Paired (ctrl, opto) conditions for Test 2 (deviation change / flattening)."""
    PC = {
        'd': [
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual L: Ctrl vs Opto'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory R: Ctrl vs Opto'),
            ({'responseSide': 'L', 'hasphotostim': False},
             {'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'All Left licks: Ctrl vs Opto'),
            ({'responseSide': 'R', 'hasphotostim': False},
             {'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'All Right licks: Ctrl vs Opto'),
        ],
        's': [
            ({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual lick trials: Ctrl vs Opto'),
            ({'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory lick trials: Ctrl vs Opto'),
            ({'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
             {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'Visual Misses: Ctrl vs Opto'),
            ({'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'Auditory Misses: Ctrl vs Opto'),
        ]
    }
    return PC


# ============================================================================
# CLUSTER-BASED PERMUTATION TEST (Maris & Oostenveld 2007)
# Identical to the parent script. One-sided positive: observed > threshold.
# ============================================================================
def _build_cluster_time_mask(time_centers, cluster_window):
    if cluster_window is None:
        return np.ones(len(time_centers), dtype=bool)
    start, end = cluster_window
    mask = np.ones(len(time_centers), dtype=bool)
    if start is not None: mask &= (time_centers >= start)
    if end is not None:   mask &= (time_centers <= end)
    return mask


def _find_clusters_1d(trace_windowed, threshold_windowed):
    above = trace_windowed > threshold_windowed
    if not np.any(above): return []
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
                              p_alpha=0.01):
    n_perms, n_timebins = null_distribution.shape
    win_mask = _build_cluster_time_mask(time_centers, cluster_window)
    win_idx = np.where(win_mask)[0]
    threshold_full = np.full(n_timebins, np.nan)
    threshold_windowed = np.nanpercentile(null_distribution[:, win_mask],
                                          threshold_percentile, axis=0)
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
        s_full = int(win_idx[s_win]); e_full = int(win_idx[e_win])
        observed_clusters.append({
            'start_idx': s_full, 'end_idx': e_full,
            't_start': float(time_centers[s_full]),
            't_end':   float(time_centers[e_full]),
            'mass': float(mass), 'p_value': float(p),
        })
        if p < p_alpha:
            cluster_mask[s_full:e_full+1] = True
            p_map[s_full:e_full+1] = p
    return {
        'observed_clusters': observed_clusters,
        'max_null_masses': max_null_masses,
        'threshold_full': threshold_full,
        'cluster_mask': cluster_mask,
        'p_map': p_map,
        'cluster_window': cluster_window if cluster_window is not None else (None, None),
        'threshold_percentile': float(threshold_percentile),
    }


def _print_cluster_summary(cluster_result, prefix="    ", p_alpha=0.01):
    clusters = cluster_result['observed_clusters']
    if not clusters:
        print(f"{prefix}Cluster test: no clusters above threshold."); return
    n_sig = sum(1 for c in clusters if c['p_value'] < p_alpha)
    print(f"{prefix}Cluster test: {len(clusters)} cluster(s), {n_sig} significant (p<{p_alpha})")
    for i, c in enumerate(clusters):
        sig_flag = " *" if c['p_value'] < p_alpha else ""
        print(f"{prefix}  Cluster {i}: t=[{c['t_start']:+.3f}, {c['t_end']:+.3f}]s, "
              f"mass={c['mass']:.3f}, p={c['p_value']:.4f}{sig_flag}")


def _cluster_fields_to_save_dict(cluster_result, prefix):
    out = {}
    out[f"{prefix}_cluster_max_null_masses"] = cluster_result['max_null_masses']
    out[f"{prefix}_cluster_threshold_full"]  = cluster_result['threshold_full']
    out[f"{prefix}_cluster_mask"]            = cluster_result['cluster_mask']
    out[f"{prefix}_cluster_p_map"]           = cluster_result['p_map']
    out[f"{prefix}_cluster_threshold_percentile"] = cluster_result['threshold_percentile']
    cw = cluster_result['cluster_window']
    out[f"{prefix}_cluster_window"] = np.array(
        [np.nan if cw[0] is None else cw[0],
         np.nan if cw[1] is None else cw[1]], dtype=float)
    for field, dtype in [('start_idx',int),('end_idx',int),('t_start',float),
                         ('t_end',float),('mass',float),('p_value',float)]:
        out[f"{prefix}_cluster_{field}"] = np.array(
            [c[field] for c in cluster_result['observed_clusters']], dtype=dtype)
    return out


# ============================================================================
# HELPER FUNCTIONS (lifted from parent, unchanged)
# ============================================================================
def _make_filter_key(f):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in f.items()))


def filter_trials(trialData, filter_dict):
    df = trialData.copy()
    for col, val in filter_dict.items():
        if isinstance(val, list):  df = df[df[col].isin(val)]
        elif isinstance(val, bool): df = df[df[col] == val]
        else:                       df = df[df[col] == val]
    return df


def precompute_single_trial_psths(spikeData_proj, trialData_subset, neuron_sessions,
                                   bins, align_to, window, z_score_data, z_score_mode,
                                   trialData_full):
    if align_to == 'firstLick':
        trialData_subset = trialData_subset[trialData_subset['responseSide'].isin(['L', 'R'])]
        trialData_subset = trialData_subset[trialData_subset['reactionTime'] > (0.1 * 1e6)]
    n_trials = len(trialData_subset)
    if n_trials == 0: return {}, 0
    session_psths = {}
    for ses in np.unique(neuron_sessions):
        neuron_mask = neuron_sessions == ses
        neuron_indices = np.where(neuron_mask)[0]
        if len(neuron_indices) == 0: continue
        trialData_ses = trialData_subset[trialData_subset['session_ID'] == ses]
        if len(trialData_ses) == 0: continue
        spikeData_ses = spikeData_proj.iloc[neuron_indices]
        psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
            z_score_data=z_score_data, align_to=align_to, window=window,
            trialData_full=trialData_full, z_score_mode=z_score_mode)
        session_psths[ses] = (psth_ses, neuron_indices)
    return session_psths, n_trials


def ensure_in_cache(psth_cache, filter_dict, spikeData_proj, trialData_full,
                    neuron_sessions, bins, align_to, window, z_score_data, z_score_mode):
    key = _make_filter_key(filter_dict)
    if key not in psth_cache:
        trials = filter_trials(trialData_full, filter_dict)
        sp, nt = precompute_single_trial_psths(
            spikeData_proj, trials, neuron_sessions, bins, align_to, window,
            z_score_data, z_score_mode, trialData_full)
        psth_cache[key] = (sp, nt)
        print(f"      Precomputed {nt} trials for new condition")
    return psth_cache


def build_psth_cache(filter_dicts, spikeData_proj, trialData_full, neuron_sessions,
                     bins, align_to, window, z_score_data, z_score_mode):
    psth_cache = {}; unique_keys = {}
    for f in filter_dicts:
        key = _make_filter_key(f)
        if key not in unique_keys: unique_keys[key] = f
    print(f"  Precomputing single-trial PSTHs for {len(unique_keys)} unique conditions...")
    for key, fd in unique_keys.items():
        trials = filter_trials(trialData_full, fd)
        sp, nt = precompute_single_trial_psths(
            spikeData_proj, trials, neuron_sessions, bins, align_to, window,
            z_score_data, z_score_mode, trialData_full)
        psth_cache[key] = (sp, nt)
        print(f"    {nt} trials precomputed")
    print(f"  Precomputation done.\n")
    return psth_cache


# ============================================================================
# CONDITION AVERAGING (parent-script style)
# ============================================================================
def condition_average_from_precomputed(session_psths, n_neurons, n_timebins):
    X_cond = np.full((n_neurons, n_timebins), np.nan)
    for ses, (psth_ses, neuron_indices) in session_psths.items():
        if psth_ses.ndim != 3 or psth_ses.shape[1] == 0: continue
        ses_mean = np.nanmean(psth_ses, axis=1)
        X_cond[neuron_indices, :] = ses_mean
    return X_cond


def bootstrap_condition_average_from_precomputed(session_psths, n_neurons, n_timebins, rng):
    X_cond = np.full((n_neurons, n_timebins), np.nan)
    for ses, (psth_ses, neuron_indices) in session_psths.items():
        if psth_ses.ndim != 3: continue
        n_trials_ses = psth_ses.shape[1]
        if n_trials_ses == 0: continue
        boot_idx = rng.choice(n_trials_ses, size=n_trials_ses, replace=True)
        resampled = psth_ses[:, boot_idx, :]
        ses_mean = np.nanmean(resampled, axis=1)
        X_cond[neuron_indices, :] = ses_mean
    return X_cond


def project_condition_average(X_cond, decoder_matrix, neuron_means, components):
    D_use = decoder_matrix[:, components]
    X_centered = X_cond - neuron_means[:, np.newaxis]
    nan_mask = np.isnan(X_centered[:, 0])
    X_centered[nan_mask, :] = 0.0
    return D_use.T @ X_centered


# ============================================================================
# deviation COMPUTATION
# ----------------------------------------------------------------------------
# E(t) = || proj(t) - mean_t∈pre_window proj(t) ||_2
# Computed in the projected dPCA subspace (so the "distance" is in the
# components specified in the marg_components argument).
# ============================================================================
def _pre_window_mask(time_centers, pre_window):
    return (time_centers >= pre_window[0]) & (time_centers <= pre_window[1])


def compute_deviation(session_psths, n_neurons, n_timebins,
                      decoder_matrix, neuron_means, components,
                      pre_mask, baseline_override=None):
    """E(t) = || proj_cond(t) - baseline ||_2.

    If baseline_override is provided (a (n_components,) vector), uses it.
    Otherwise computes the baseline as the mean over pre-stim bins of this
    condition's own projection.
    """
    X = condition_average_from_precomputed(session_psths, n_neurons, n_timebins)
    proj = project_condition_average(X, decoder_matrix, neuron_means, components)
    if baseline_override is not None:
        baseline = baseline_override
    else:
        baseline = np.nanmean(proj[:, pre_mask], axis=1)
    diff = proj - baseline[:, np.newaxis]
    return np.sqrt(np.sum(diff ** 2, axis=0))


def compute_pooled_baseline(session_psths_list, n_neurons, n_timebins,
                            decoder_matrix, neuron_means, components, pre_mask):
    """Compute a single baseline vector from POOLED trials across multiple
    session_psths dicts (used for Test 2: ctrl+opto pooled baseline).

    The baseline is the trial-mean projection averaged over pre-stim bins.
    Implemented as a weighted average across input dicts (weighted by trial
    count per session, since condition_average_from_precomputed already takes
    nanmean per session).
    """
    pooled = {}
    for ses in set().union(*(sp.keys() for sp in session_psths_list)):
        trial_blocks = []
        neuron_indices = None
        for sp in session_psths_list:
            if ses not in sp: continue
            psth, nidx = sp[ses]
            if psth.ndim != 3 or psth.shape[1] == 0: continue
            trial_blocks.append(psth)
            neuron_indices = nidx
        if not trial_blocks: continue
        pooled_trials = np.concatenate(trial_blocks, axis=1)
        pooled[ses] = (pooled_trials, neuron_indices)
    X = condition_average_from_precomputed(pooled, n_neurons, n_timebins)
    proj = project_condition_average(X, decoder_matrix, neuron_means, components)
    return np.nanmean(proj[:, pre_mask], axis=1)


# ============================================================================
# TEST 1: NULL VIA CIRCULAR TRIAL SHIFTS
# ----------------------------------------------------------------------------
# Each trial's time axis is rolled by an independent random offset. This
# preserves single-trial autocorrelation and overall firing-rate structure
# but destroys stimulus-locking. After shifting, the condition-averaged
# trajectory should have no structured deviation at t=0 (or anywhere).
# ============================================================================
def _circular_shift_session_psths(session_psths, rng):
    """Return a new session_psths dict where each trial's time axis is rolled
    by an independent random offset."""
    out = {}
    for ses, (psth, neuron_indices) in session_psths.items():
        if psth.ndim != 3 or psth.shape[1] == 0:
            out[ses] = (psth, neuron_indices); continue
        n_trials_ses = psth.shape[1]
        n_t = psth.shape[2]
        offsets = rng.integers(0, n_t, size=n_trials_ses)
        # Build shifted array trial-by-trial. np.roll along axis=-1 with an
        # array of shifts requires manual indexing.
        shifted = np.empty_like(psth)
        for ti in range(n_trials_ses):
            shifted[:, ti, :] = np.roll(psth[:, ti, :], offsets[ti], axis=-1)
        out[ses] = (shifted, neuron_indices)
    return out


def _perm_deviation_worker(seed, session_psths, n_neurons, n_timebins,
                           decoder_matrix, neuron_means, components, pre_mask):
    rng = np.random.default_rng(seed)
    shifted = _circular_shift_session_psths(session_psths, rng)
    # After circular shifting, baseline drawn from shifted trace itself
    # (so the null genuinely says "no stimulus-locked deviation"). Using a
    # fixed real-data baseline would conflate null with bias.
    return compute_deviation(shifted, n_neurons, n_timebins,
                             decoder_matrix, neuron_means, components,
                             pre_mask, baseline_override=None)


def _boot_deviation_worker(seed, session_psths, n_neurons, n_timebins,
                           decoder_matrix, neuron_means, components, pre_mask):
    rng = np.random.default_rng(seed)
    X = bootstrap_condition_average_from_precomputed(session_psths, n_neurons, n_timebins, rng)
    proj = project_condition_average(X, decoder_matrix, neuron_means, components)
    baseline = np.nanmean(proj[:, pre_mask], axis=1)
    diff = proj - baseline[:, np.newaxis]
    return np.sqrt(np.sum(diff ** 2, axis=0))


# ============================================================================
# TEST 2: NULL VIA CTRL/OPTO TRIAL POOLING & SHUFFLE
# ----------------------------------------------------------------------------
# Identical scheme to parent script's _pool_ctrl_opto_sessions + shuffle.
# In each permutation:
#   - pool ctrl+opto trials per session,
#   - reshuffle into ctrl-sized and opto-sized groups,
#   - recompute baseline from the POOLED trials (invariant under shuffle but
#     conceptually recomputed each iter — actually constant, so we cache it),
#   - compute E_ctrl_perm(t) and E_opto_perm(t) wrt the pooled baseline,
#   - return delta = E_ctrl_perm(t) - E_opto_perm(t).
# ============================================================================
def _pool_ctrl_opto_sessions(session_psths_ctrl, session_psths_opto):
    pooled = {}
    for ses in set(session_psths_ctrl.keys()) | set(session_psths_opto.keys()):
        psth_ctrl, idx_ctrl = session_psths_ctrl.get(ses, (np.empty((0,0,0)), np.array([])))
        psth_opto, idx_opto = session_psths_opto.get(ses, (np.empty((0,0,0)), np.array([])))
        neuron_indices = idx_ctrl if len(idx_ctrl) > 0 else idx_opto
        n_ctrl = psth_ctrl.shape[1] if psth_ctrl.ndim == 3 else 0
        n_opto = psth_opto.shape[1] if psth_opto.ndim == 3 else 0
        if n_ctrl + n_opto == 0: continue
        trials = np.concatenate(
            [x for x in [psth_ctrl, psth_opto] if x.ndim == 3 and x.shape[1] > 0], axis=1)
        pooled[ses] = (trials, neuron_indices, n_ctrl)
    return pooled


def _shuffle_pooled(pooled, rng):
    g1, g2 = {}, {}
    for ses, (trials, neuron_indices, n_first) in pooled.items():
        perm = rng.permutation(trials.shape[1])
        g1[ses] = (trials[:, perm[:n_first], :], neuron_indices)
        g2[ses] = (trials[:, perm[n_first:], :], neuron_indices)
    return g1, g2


def _shuffle_pooled_balanced(pooled, rng):
    g1, g2 = {}, {}
    for ses, (trials, neuron_indices, n_first) in pooled.items():
        n_second = trials.shape[1] - n_first
        if n_first == 0 or n_second == 0:
            g1[ses] = (trials, neuron_indices)
            g2[ses] = (trials[:, :0, :], neuron_indices)
            continue
        n_bal = min(n_first, n_second)
        idx_first  = rng.choice(n_first, size=n_bal, replace=False)
        idx_second = rng.choice(n_second, size=n_bal, replace=False) + n_first
        balanced = trials[:, np.concatenate([idx_first, idx_second]), :]
        perm = rng.permutation(2 * n_bal)
        g1[ses] = (balanced[:, perm[:n_bal], :], neuron_indices)
        g2[ses] = (balanced[:, perm[n_bal:], :], neuron_indices)
    return g1, g2


def _perm_deviation_diff_worker(seed, pooled, baseline_vec, n_neurons, n_timebins,
                                decoder_matrix, neuron_means, components,
                                pre_mask, subsample):
    rng = np.random.default_rng(seed)
    sfn = _shuffle_pooled_balanced if subsample else _shuffle_pooled
    sa, sb = sfn(pooled, rng)
    Ea = compute_deviation(sa, n_neurons, n_timebins, decoder_matrix, neuron_means,
                            components, pre_mask, baseline_override=baseline_vec)
    Eb = compute_deviation(sb, n_neurons, n_timebins, decoder_matrix, neuron_means,
                            components, pre_mask, baseline_override=baseline_vec)
    return Ea - Eb


def _boot_deviation_diff_worker(seed, session_psths_ctrl, session_psths_opto,
                                baseline_vec, n_neurons, n_timebins,
                                decoder_matrix, neuron_means, components, pre_mask):
    rng = np.random.default_rng(seed)
    Xc = bootstrap_condition_average_from_precomputed(session_psths_ctrl, n_neurons, n_timebins, rng)
    Xo = bootstrap_condition_average_from_precomputed(session_psths_opto, n_neurons, n_timebins, rng)
    pc = project_condition_average(Xc, decoder_matrix, neuron_means, components)
    po = project_condition_average(Xo, decoder_matrix, neuron_means, components)
    Ec = np.sqrt(np.sum((pc - baseline_vec[:, np.newaxis]) ** 2, axis=0))
    Eo = np.sqrt(np.sum((po - baseline_vec[:, np.newaxis]) ** 2, axis=0))
    return Ec - Eo


def _spawn_seeds(n, base_seed=42):
    return np.random.SeedSequence(base_seed).spawn(n)


# ============================================================================
# TEST 1 DRIVER (deviation vs baseline)
# ============================================================================
def run_deviation_vs_baseline(spikeData_proj, trialData_full, neuron_sessions,
                              decoder_matrix, neuron_means, single_conditions,
                              bins, time_centers,
                              align_to, window_dpca, z_score_data, z_score_mode,
                              components, n_bootstrap, ci, pre_window,
                              cluster_window, cluster_threshold_pctile,
                              p_alpha, n_jobs, backend,
                              psth_cache=None, condition_seed_offsets=None):
    n_timebins = len(time_centers)
    n_neurons = len(spikeData_proj)
    pre_mask = _pre_window_mask(time_centers, pre_window)
    if not np.any(pre_mask):
        raise ValueError(f"pre_window {pre_window} contains no bins in time_centers "
                         f"[{time_centers.min():.3f}, {time_centers.max():.3f}]")

    if psth_cache is None:
        psth_cache = build_psth_cache(
            [fd for fd, _ in single_conditions],
            spikeData_proj, trialData_full, neuron_sessions,
            bins, align_to, window_dpca, z_score_data, z_score_mode)
    if condition_seed_offsets is None:
        condition_seed_offsets = list(range(len(single_conditions)))

    results = []
    for (fd, label), seed_offset in zip(single_conditions, condition_seed_offsets):
        print(f"  {label}")
        psth_cache = ensure_in_cache(psth_cache, fd, spikeData_proj, trialData_full,
                                      neuron_sessions, bins, align_to, window_dpca,
                                      z_score_data, z_score_mode)
        sp, n_trials = psth_cache[_make_filter_key(fd)]
        if n_trials < 2:
            print(f"    WARNING: Not enough trials ({n_trials}), skipping.")
            results.append({
                'label': label,
                'deviation': np.full(n_timebins, np.nan),
                'ci_low':    np.full(n_timebins, np.nan),
                'ci_high':   np.full(n_timebins, np.nan),
                'boot_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'null_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'p_values':  np.full(n_timebins, np.nan),
                'pre_window': pre_window,
                'n_trials':  n_trials,
                'skipped':   True,
            })
            continue

        # --- observed deviation ---
        deviation = compute_deviation(sp, n_neurons, n_timebins,
                                       decoder_matrix, neuron_means, components,
                                       pre_mask, baseline_override=None)

        base_seed = 1000 + seed_offset * 1000

        # --- bootstrap CI ---
        print(f"    Bootstrapping ({n_bootstrap} iter, n_jobs={n_jobs})...")
        boot_seeds = _spawn_seeds(n_bootstrap, base_seed)
        boot_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_boot_deviation_worker)(s, sp, n_neurons, n_timebins,
                                             decoder_matrix, neuron_means,
                                             components, pre_mask)
            for s in boot_seeds)
        boot = np.array(boot_rows)
        alpha = (100 - ci) / 2
        ci_low  = np.nanpercentile(boot, alpha,       axis=0)
        ci_high = np.nanpercentile(boot, 100 - alpha, axis=0)

        # --- null via circular shift ---
        print(f"    Circular-shift null ({n_bootstrap} iter, n_jobs={n_jobs})...")
        null_seeds = _spawn_seeds(n_bootstrap, base_seed + 500)
        null_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_perm_deviation_worker)(s, sp, n_neurons, n_timebins,
                                             decoder_matrix, neuron_means,
                                             components, pre_mask)
            for s in null_seeds)
        null_dist = np.array(null_rows)

        # --- p-values: one-sided (observed > null), since deviation is non-negative ---
        n_perms = null_dist.shape[0]
        pvals = (np.sum(null_dist >= deviation[np.newaxis, :], axis=0) + 1) / (n_perms + 1)

        # --- cluster test ---
        cr = cluster_permutation_test(deviation, null_dist, time_centers,
                                       cluster_window=cluster_window,
                                       threshold_percentile=cluster_threshold_pctile,
                                       p_alpha=p_alpha)

        print(f"    {n_trials} trials, peak E = {np.nanmax(deviation):.3f} "
              f"at t={time_centers[np.nanargmax(deviation)]:.3f}s")
        print(f"    Sig. timepoints (p<{p_alpha}): {np.sum(pvals < p_alpha)}/{n_timebins}")
        _print_cluster_summary(cr, p_alpha=p_alpha)

        results.append({
            'label': label,
            'deviation': deviation,
            'ci_low': ci_low, 'ci_high': ci_high,
            'boot_distribution': boot,
            'null_distribution': null_dist,
            'p_values': pvals,
            'pre_window': pre_window,
            'n_trials': n_trials,
            'cluster_result': cr,
        })
    return results, psth_cache


# ============================================================================
# TEST 2 DRIVER (deviation change ctrl vs opto)
# ============================================================================
def run_deviation_change(spikeData_proj, trialData_full, neuron_sessions,
                         decoder_matrix, neuron_means, paired_conditions,
                         bins, time_centers,
                         align_to, window_dpca, z_score_data, z_score_mode,
                         components, n_bootstrap, ci, pre_window,
                         cluster_window, cluster_threshold_pctile,
                         subsample, p_alpha, n_jobs, backend,
                         psth_cache=None, pair_seed_offsets=None):
    n_timebins = len(time_centers)
    n_neurons = len(spikeData_proj)
    pre_mask = _pre_window_mask(time_centers, pre_window)
    if not np.any(pre_mask):
        raise ValueError(f"pre_window {pre_window} contains no bins in time_centers")

    if psth_cache is None:
        all_fds = []
        for fc, fo, _ in paired_conditions: all_fds.extend([fc, fo])
        psth_cache = build_psth_cache(all_fds, spikeData_proj, trialData_full,
                                       neuron_sessions, bins, align_to, window_dpca,
                                       z_score_data, z_score_mode)
    if pair_seed_offsets is None:
        pair_seed_offsets = list(range(len(paired_conditions)))

    results = []
    for (fc, fo, label), seed_offset in zip(paired_conditions, pair_seed_offsets):
        print(f"  {label}")
        psth_cache = ensure_in_cache(psth_cache, fc, spikeData_proj, trialData_full,
                                      neuron_sessions, bins, align_to, window_dpca,
                                      z_score_data, z_score_mode)
        psth_cache = ensure_in_cache(psth_cache, fo, spikeData_proj, trialData_full,
                                      neuron_sessions, bins, align_to, window_dpca,
                                      z_score_data, z_score_mode)
        spc, nc = psth_cache[_make_filter_key(fc)]
        spo, no = psth_cache[_make_filter_key(fo)]
        if nc < 2 or no < 2:
            print(f"    WARNING: Not enough trials (ctrl={nc}, opto={no}), skipping.")
            results.append({
                'label': label,
                'observed': np.full(n_timebins, np.nan),
                'deviation_ctrl': np.full(n_timebins, np.nan),
                'deviation_opto': np.full(n_timebins, np.nan),
                'ci_low':  np.full(n_timebins, np.nan),
                'ci_high': np.full(n_timebins, np.nan),
                'boot_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'null_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'p_values': np.full(n_timebins, np.nan),
                'pre_window': pre_window,
                'n_trials_ctrl': nc, 'n_trials_opto': no,
                'skipped': True,
            })
            continue

        # --- shared baseline from pooled ctrl+opto trials ---
        baseline_vec = compute_pooled_baseline(
            [spc, spo], n_neurons, n_timebins, decoder_matrix, neuron_means,
            components, pre_mask)

        # --- observed deviations and delta ---
        E_ctrl = compute_deviation(spc, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components,
                                    pre_mask, baseline_override=baseline_vec)
        E_opto = compute_deviation(spo, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components,
                                    pre_mask, baseline_override=baseline_vec)
        observed = E_ctrl - E_opto

        base_seed = 5000 + seed_offset * 1000

        # --- bootstrap CI on delta ---
        print(f"    Bootstrapping delta ({n_bootstrap} iter, n_jobs={n_jobs})...")
        boot_seeds = _spawn_seeds(n_bootstrap, base_seed)
        boot_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_boot_deviation_diff_worker)(s, spc, spo, baseline_vec,
                                                  n_neurons, n_timebins,
                                                  decoder_matrix, neuron_means,
                                                  components, pre_mask)
            for s in boot_seeds)
        boot = np.array(boot_rows)
        alpha = (100 - ci) / 2
        ci_low  = np.nanpercentile(boot, alpha,       axis=0)
        ci_high = np.nanpercentile(boot, 100 - alpha, axis=0)

        # --- null via pooled trial shuffle ---
        sub_str = " (balanced null)" if subsample else ""
        print(f"    Permuting{sub_str} ({n_bootstrap} iter, n_jobs={n_jobs})...")
        pooled = _pool_ctrl_opto_sessions(spc, spo)
        null_seeds = _spawn_seeds(n_bootstrap, base_seed + 500)
        null_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_perm_deviation_diff_worker)(s, pooled, baseline_vec,
                                                  n_neurons, n_timebins,
                                                  decoder_matrix, neuron_means,
                                                  components, pre_mask, subsample)
            for s in null_seeds)
        null_dist = np.array(null_rows)

        # --- p-values: one-sided (E_ctrl > E_opto expected → delta > 0) ---
        n_perms = null_dist.shape[0]
        pvals = (np.sum(null_dist >= observed[np.newaxis, :], axis=0) + 1) / (n_perms + 1)

        # --- cluster test ---
        cr = cluster_permutation_test(observed, null_dist, time_centers,
                                       cluster_window=cluster_window,
                                       threshold_percentile=cluster_threshold_pctile,
                                       p_alpha=p_alpha)

        print(f"    Ctrl: {nc} trials | Opto: {no} trials")
        print(f"    Peak |delta| = {np.nanmax(np.abs(observed)):.3f} "
              f"at t={time_centers[np.nanargmax(np.abs(observed))]:.3f}s")
        print(f"    Sig. timepoints (p<{p_alpha}): {np.sum(pvals < p_alpha)}/{n_timebins}")
        _print_cluster_summary(cr, p_alpha=p_alpha)

        results.append({
            'label': label,
            'observed': observed,
            'deviation_ctrl': E_ctrl,
            'deviation_opto': E_opto,
            'ci_low': ci_low, 'ci_high': ci_high,
            'boot_distribution': boot,
            'null_distribution': null_dist,
            'p_values': pvals,
            'pre_window': pre_window,
            'baseline_vec': baseline_vec,
            'n_trials_ctrl': nc, 'n_trials_opto': no,
            'cluster_result': cr,
        })
    return results, psth_cache


# ============================================================================
# SAVE / LOAD (mirrors parent script's on-disk schema)
# ============================================================================
def _build_deviation_save_dict(results, time_centers, components, marg_key,
                                n_bootstrap, ci, pre_window,
                                cluster_window, cluster_threshold_pctile):
    sd = {}
    for i, res in enumerate(results):
        pf = f"cond{i}"
        sd[f"{pf}_label"] = res['label']
        for fld in ['deviation', 'ci_low', 'ci_high',
                    'boot_distribution', 'null_distribution', 'p_values']:
            sd[f"{pf}_{fld}"] = res[fld]
        sd[f"{pf}_pre_window"] = np.array(res['pre_window'])
        sd[f"{pf}_n_trials"]   = res['n_trials']
        sd[f"{pf}_skipped"]    = bool(res.get('skipped', False))
        if 'cluster_result' in res and res['cluster_result'] is not None:
            sd.update(_cluster_fields_to_save_dict(res['cluster_result'], pf))

    sd.update({
        'time_centers': time_centers,
        'components': np.array(components),
        'marg_key': marg_key,
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'pre_window':  np.array(pre_window),
        'cluster_window_start': np.nan if cluster_window is None or cluster_window[0] is None else cluster_window[0],
        'cluster_window_end':   np.nan if cluster_window is None or cluster_window[1] is None else cluster_window[1],
        'cluster_threshold_percentile': cluster_threshold_pctile,
    })
    return sd


def _build_deviation_change_save_dict(results, time_centers, components, marg_key,
                                       n_bootstrap, ci, pre_window,
                                       cluster_window, cluster_threshold_pctile,
                                       subsample_trials):
    sd = {}
    for i, res in enumerate(results):
        pf = f"pair{i}"
        sd[f"{pf}_label"] = res['label']
        for fld in ['observed', 'deviation_ctrl', 'deviation_opto',
                    'ci_low', 'ci_high',
                    'boot_distribution', 'null_distribution', 'p_values']:
            sd[f"{pf}_{fld}"] = res[fld]
        sd[f"{pf}_pre_window"]  = np.array(res['pre_window'])
        sd[f"{pf}_n_trials_ctrl"] = res['n_trials_ctrl']
        sd[f"{pf}_n_trials_opto"] = res['n_trials_opto']
        sd[f"{pf}_skipped"]       = bool(res.get('skipped', False))
        if 'baseline_vec' in res:
            sd[f"{pf}_baseline_vec"] = res['baseline_vec']
        if 'cluster_result' in res and res['cluster_result'] is not None:
            sd.update(_cluster_fields_to_save_dict(res['cluster_result'], pf))

    sd.update({
        'time_centers': time_centers,
        'components': np.array(components),
        'marg_key': marg_key,
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'pre_window':  np.array(pre_window),
        'subsample_trials': subsample_trials,
        'cluster_window_start': np.nan if cluster_window is None or cluster_window[0] is None else cluster_window[0],
        'cluster_window_end':   np.nan if cluster_window is None or cluster_window[1] is None else cluster_window[1],
        'cluster_threshold_percentile': cluster_threshold_pctile,
    })
    return sd


def _extract_cluster_from_npz(data, prefix):
    if f"{prefix}_cluster_mask" not in data: return {}
    out = {}
    for fld in ['cluster_mask','cluster_p_map','cluster_threshold_full',
                'cluster_max_null_masses','cluster_start_idx','cluster_end_idx',
                'cluster_t_start','cluster_t_end','cluster_mass','cluster_p_value']:
        out[fld] = data[f"{prefix}_{fld}"]
    out['cluster_window'] = tuple(data[f"{prefix}_cluster_window"])
    out['cluster_threshold_percentile'] = float(data[f"{prefix}_cluster_threshold_percentile"])
    return out


def _legacy_get(data, key):
    """Read `key` from an npz; fall back to the pre-rename 'excursion' name."""
    if key in data: return data[key]
    return data[key.replace('deviation', 'excursion')]


def load_existing_deviation_by_label(filepath):
    if not os.path.exists(filepath): return {}, {}
    data = np.load(filepath, allow_pickle=True)
    by_label = {}; i = 0
    while f"cond{i}_label" in data:
        pf = f"cond{i}"
        res = {
            'label':             str(data[f"{pf}_label"]),
            'deviation':         _legacy_get(data, f"{pf}_deviation"),
            'ci_low':            data[f"{pf}_ci_low"],
            'ci_high':           data[f"{pf}_ci_high"],
            'boot_distribution': data[f"{pf}_boot_distribution"],
            'null_distribution': data[f"{pf}_null_distribution"],
            'p_values':          data[f"{pf}_p_values"],
            'pre_window': tuple(data[f"{pf}_pre_window"]),
            'n_trials':   int(data[f"{pf}_n_trials"]),
            'skipped':    bool(data[f"{pf}_skipped"]),
        }
        res.update(_extract_cluster_from_npz(data, pf))
        by_label[res['label']] = res
        i += 1
    file_meta = {}
    for k in ['time_centers', 'pre_window', 'n_bootstrap', 'ci',
              'cluster_window_start', 'cluster_window_end',
              'cluster_threshold_percentile', 'components']:
        if k in data: file_meta[k] = data[k]
    return by_label, file_meta


def load_existing_deviation_change_by_label(filepath):
    if not os.path.exists(filepath): return {}, {}
    data = np.load(filepath, allow_pickle=True)
    by_label = {}; i = 0
    while f"pair{i}_label" in data:
        pf = f"pair{i}"
        res = {
            'label':             str(data[f"{pf}_label"]),
            'observed':          data[f"{pf}_observed"],
            'deviation_ctrl':    _legacy_get(data, f"{pf}_deviation_ctrl"),
            'deviation_opto':    _legacy_get(data, f"{pf}_deviation_opto"),
            'ci_low':            data[f"{pf}_ci_low"],
            'ci_high':           data[f"{pf}_ci_high"],
            'boot_distribution': data[f"{pf}_boot_distribution"],
            'null_distribution': data[f"{pf}_null_distribution"],
            'p_values':          data[f"{pf}_p_values"],
            'pre_window': tuple(data[f"{pf}_pre_window"]),
            'n_trials_ctrl': int(data[f"{pf}_n_trials_ctrl"]),
            'n_trials_opto': int(data[f"{pf}_n_trials_opto"]),
            'skipped':       bool(data[f"{pf}_skipped"]),
        }
        if f"{pf}_baseline_vec" in data:
            res['baseline_vec'] = data[f"{pf}_baseline_vec"]
        res.update(_extract_cluster_from_npz(data, pf))
        by_label[res['label']] = res
        i += 1
    file_meta = {}
    for k in ['time_centers', 'pre_window', 'n_bootstrap', 'ci',
              'subsample_trials', 'cluster_window_start', 'cluster_window_end',
              'cluster_threshold_percentile', 'components']:
        if k in data: file_meta[k] = data[k]
    return by_label, file_meta


def _check_compatibility(file_meta, current_meta, filename):
    if not file_meta: return True
    issues = []
    for key in ['n_bootstrap', 'ci', 'subsample_trials', 'cluster_threshold_percentile']:
        if key in file_meta and key in current_meta:
            v_old = file_meta[key]
            v_new = current_meta[key]
            try: v_old_s = v_old.item() if hasattr(v_old, 'item') and v_old.ndim == 0 else v_old
            except Exception: v_old_s = v_old
            if v_old_s != v_new:
                issues.append(f"    {key}: file={v_old_s} vs current={v_new}")
    for key_old, key_new in [('pre_window', 'pre_window')]:
        if key_old in file_meta and key_new in current_meta:
            v_old = tuple(file_meta[key_old].tolist()) if hasattr(file_meta[key_old], 'tolist') else tuple(file_meta[key_old])
            v_new = tuple(current_meta[key_new])
            if v_old != v_new:
                issues.append(f"    {key_old}: file={v_old} vs current={v_new}")
    if 'components' in file_meta and 'components' in current_meta:
        c_old = np.asarray(file_meta['components']).tolist()
        c_new = list(current_meta['components'])
        if c_old != c_new:
            issues.append(f"    components: file={c_old} vs current={c_new}")
    if 'cluster_window_start' in file_meta and 'cluster_window' in current_meta:
        cw_new = current_meta['cluster_window']
        f_start = file_meta['cluster_window_start']
        f_end   = file_meta.get('cluster_window_end', np.nan)
        try:    f_start_s = float(f_start) if not (hasattr(f_start, 'item') and np.isnan(f_start)) else None
        except (TypeError, ValueError): f_start_s = None
        try:    f_end_s = float(f_end) if not (hasattr(f_end, 'item') and np.isnan(f_end)) else None
        except (TypeError, ValueError): f_end_s = None
        n_start, n_end = (None, None) if cw_new is None else cw_new
        def _eq(a, b):
            if a is None and b is None: return True
            if a is None or b is None:  return False
            return a == b
        if not (_eq(f_start_s, n_start) and _eq(f_end_s, n_end)):
            issues.append(f"    cluster_window: file=({f_start_s}, {f_end_s}) vs current=({n_start}, {n_end})")
    if 'time_centers' in file_meta and 'time_centers' in current_meta:
        tc_old = np.asarray(file_meta['time_centers'])
        tc_new = np.asarray(current_meta['time_centers'])
        if tc_old.shape != tc_new.shape or not np.allclose(tc_old, tc_new):
            issues.append(f"    time_centers: shape or values differ "
                          f"(file={tc_old.shape}, current={tc_new.shape})")
    if issues:
        print(f"  WARNING: {filename} was generated with different settings:")
        for line in issues: print(line)
        print(f"  Merged results may be inconsistent. Consider --force {os.path.splitext(filename)[0]}.")
        return False
    return True


def _print_deviation_summary(results, time_centers):
    print(f"\n  {'Label':<45s} {'Trials':>10s} {'Peak E':>10s} {'Peak t':>8s} {'#sig clust':>12s}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")
    for res in results:
        if res.get('skipped', False):
            print(f"  {res['label']:<45s} {res.get('n_trials', '-'):>10} {'[skipped]'}"); continue
        e = res['deviation']
        cr = res.get('cluster_result', None)
        if cr is not None:
            n_sig = sum(1 for c in cr['observed_clusters'] if c['p_value'] < 0.05)
            n_tot = len(cr['observed_clusters'])
            sig_str = f"{n_sig}/{n_tot}"
        else:
            sig_str = "-"
        print(f"  {res['label']:<45s} {res['n_trials']:>10d} "
              f"{np.nanmax(e):>10.3f} "
              f"{time_centers[np.nanargmax(e)]:>8.3f} {sig_str:>12s}")
    print()


def _print_deviation_change_summary(results, time_centers):
    print(f"\n  {'Label':<45s} {'Ctrl':>6s} {'Opto':>6s} {'Peak |d|':>10s} {'Peak t':>8s} {'#sig clust':>12s}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*12}")
    for res in results:
        if res.get('skipped', False):
            print(f"  {res['label']:<45s} [skipped]"); continue
        obs = res['observed']
        cr = res.get('cluster_result', None)
        if cr is not None:
            n_sig = sum(1 for c in cr['observed_clusters'] if c['p_value'] < 0.05)
            n_tot = len(cr['observed_clusters'])
            sig_str = f"{n_sig}/{n_tot}"
        else:
            sig_str = "-"
        print(f"  {res['label']:<45s} {res['n_trials_ctrl']:>6d} {res['n_trials_opto']:>6d} "
              f"{np.nanmax(np.abs(obs)):>10.3f} "
              f"{time_centers[np.nanargmax(np.abs(obs))]:>8.3f} {sig_str:>12s}")
    print()


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================
def run_deviation_analysis(
        dpca_dir=DEFAULT_DPCA_DIR,
        path_root=DEFAULT_PATH_ROOT,
        experiment=DEFAULT_EXPERIMENT,
        animals=DEFAULT_ANIMALS,
        sessions_exclude=DEFAULT_SESSIONS_EXCLUDE,
        opto_lat=DEFAULT_OPTO_LAT,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
        ci=DEFAULT_CI,
        align_to=DEFAULT_ALIGN_TO,
        subsample_trials=DEFAULT_SUBSAMPLE_TRIALS,
        pre_window=DEFAULT_PRE_WINDOW,
        cluster_window=DEFAULT_CLUSTER_TIME_WINDOW,
        cluster_threshold_pctile=DEFAULT_CLUSTER_THRESHOLD_PCTILE,
        p_alpha=DEFAULT_P_ALPHA,
        n_jobs=DEFAULT_N_JOBS,
        parallel_backend=DEFAULT_PARALLEL_BACKEND,
        marg_components=None,
        output_dir=None,
        force_stems=None,
        output_tag=None,
        prune_orphans=False,
        run_test1=True,
        run_test2=True,
        marg_keys=None):
    """Run Test 1 (deviation vs baseline) and/or Test 2 (deviation change).

    Parameters
    ----------
    marg_components : list of int or None
        Components (dimensions) to include from each marginalization when
        computing the dPCA-space distance. Applied uniformly across all
        marginalizations available in the dPCA fit. Use this to loop over
        different component sets from a calling script, e.g.
            for comps in ([0], [0,1], [0,1,2]):
                run_deviation_analysis(..., marg_components=comps,
                                       output_tag=f"comps{''.join(map(str,comps))}")
        Defaults to DEFAULT_MARG_COMPONENTS = [0].
    align_to : str
        Trial alignment event passed through to PSTH computation.
    """
    force_stems = set(force_stems) if force_stems is not None else set()
    if marg_components is None:
        marg_components = list(DEFAULT_MARG_COMPONENTS)
    else:
        marg_components = list(marg_components)

    # --------------------------------------------------------------------
    # LOAD dPCA RESULTS
    # --------------------------------------------------------------------
    print(f"Using N_JOBS = {n_jobs}")
    if dpca_dir is None:
        dpca_dir = config.resolve_pp("dPCA", DEFAULT_DPCA_AREA, "subspace_results")
    print(f"Loading dPCA results from: {dpca_dir}")
    with open(os.path.join(dpca_dir, 'metadata.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)
    area = metadata['area']; window_dpca = tuple(metadata['window'])
    bin_size = metadata['bin_size']; step_size = metadata['step_size']
    if 'z_score' in metadata:
        z_score_data = metadata['z_score']
    else:
        print("WARNING: No 'z_score' in metadata. Defaulting to False.")
        z_score_data = False
    z_score_mode = 'custom_baseline' if z_score_data else None
    weights_npz = np.load(os.path.join(dpca_dir, 'dpca_weights.npz'), allow_pickle=True)
    D = weights_npz['D'].item()
    results_npz = np.load(os.path.join(dpca_dir, 'dpca_results.npz'), allow_pickle=True)
    neuron_ids_dpca = results_npz['neuron_ids']; X_dpca = results_npz['X']
    neuron_means = X_dpca.reshape(X_dpca.shape[0], -1).mean(axis=1)
    n_neurons = len(neuron_ids_dpca)
    print(f"Area: {area}\nNeurons: {n_neurons}\nMarginalizations: {sorted(D.keys())}")
    print(f"Marginalization components (applied to all margs): {marg_components}")

    # --------------------------------------------------------------------
    # LOAD EXPERIMENTAL DATA
    # --------------------------------------------------------------------
    print("\nLoading experimental data...")
    trialData, sessionData, spikeData = utils.load_data(path_root, experiment, animals)
    (trialData, sessionData, spikeData, _) = utils.filter_sessions_hitrate(
        trialData, sessionData=sessionData, spikeData=spikeData, min_hitrate=0.3)
    trialData    = trialData   [~trialData   ['session_ID'].isin(sessions_exclude)]
    spikeData    = spikeData   [~spikeData   ['session_ID'].isin(sessions_exclude)]
    sessionData  = sessionData [~sessionData ['session_ID'].isin(sessions_exclude)]
    spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=0.5, quality='good')
    spikeData = spikeData[spikeData['area'].isin(area)]
    trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)
    trialData_full = trialData.copy()
    print(f"Loaded {len(spikeData)} neurons, {len(trialData_full)} trials")
    print(f"Align to: {align_to}")
    print(f"Pre-stim baseline window:  {pre_window[0]} to {pre_window[1]} s")
    print(f"Subsample trials for balanced null: {subsample_trials}")

    spikeData_proj = spikeData[spikeData['cluster_ID'].isin(neuron_ids_dpca)].copy()
    neuron_order_map = {nid: i for i, nid in enumerate(neuron_ids_dpca)}
    spikeData_proj['_dpca_order'] = spikeData_proj['cluster_ID'].map(neuron_order_map)
    spikeData_proj = spikeData_proj.sort_values('_dpca_order').drop('_dpca_order', axis=1)
    assert np.array_equal(spikeData_proj['cluster_ID'].values, neuron_ids_dpca), \
        "Neuron order mismatch!"
    neuron_sessions = spikeData_proj['session_ID'].values
    print(f"Matched {len(spikeData_proj)} neurons to dPCA model")
    bins, time_centers = utils.custom_bins(bin_size=bin_size, step_size=step_size,
                                            window=window_dpca, causal=True)

    # --------------------------------------------------------------------
    # OUTPUT DIR
    # --------------------------------------------------------------------
    if output_dir is not None:
        save_dir = output_dir
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"output_dir does not exist: {save_dir}")
        print(f"\nReusing existing output directory: {save_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_parts = [f"deviation_{align_to}"]
        if subsample_trials: tag_parts.append("sub")
        tag_parts.append("comps" + "".join(map(str, marg_components)))  # <-- add this
        if output_tag:       tag_parts.append(str(output_tag))
        tag_parts.append(timestamp)
        dir_name = "_".join(tag_parts)
        save_dir = os.path.join(config.DATA_DIR_PP_NEW, "dPCA", "".join(area), "bsl_deviation", dir_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCreated new output directory: {save_dir}")

    # --------------------------------------------------------------------
    # RUN ANALYSES
    # --------------------------------------------------------------------
    print("\n" + "="*80 + "\nCOMPUTING deviation-FROM-BASELINE IN dPCA SPACE\n" + "="*80)

    current_meta = {
        'time_centers': time_centers,
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'subsample_trials': subsample_trials,
        'pre_window':  pre_window,
        'cluster_window': cluster_window,
        'cluster_threshold_percentile': cluster_threshold_pctile,
        'components': marg_components,
    }

    SC = get_single_conditions(opto_lat)
    PC = get_paired_conditions(opto_lat)
    all_results = {'test1': {}, 'test2': {}}
    all_psth_caches = {}

    for marg_key in sorted(D.keys()):
        if marg_keys is not None and marg_key not in marg_keys:
            continue
        components = list(marg_components)
        print(f"\n{'='*60}\nMarginalization: {marg_key} — components {components}\n{'='*60}")

        psth_cache = {}

        # =============================================================
        # TEST 1: deviation vs baseline
        # =============================================================
        if run_test1 and marg_key in SC and SC[marg_key]:
            t1_path = os.path.join(save_dir, f'deviation_vs_baseline_{marg_key}.npz')
            t1_stem = f"deviation_vs_baseline_{marg_key}"
            force_t1 = t1_stem in force_stems
            existing, file_meta = ({}, {})
            if not force_t1:
                existing, file_meta = load_existing_deviation_by_label(t1_path)
                if existing:
                    _check_compatibility(file_meta, current_meta, os.path.basename(t1_path))

            current_entries = []; seen = set()
            for fd, label in SC[marg_key]:
                if label in seen: continue
                seen.add(label); current_entries.append((fd, label))

            if force_t1:
                missing = current_entries
                print(f"\n  [Test 1] [FORCE] Recomputing all {len(missing)} conditions.")
            else:
                missing = [(fd, label) for (fd, label) in current_entries
                           if label not in existing]
                if existing and not missing:
                    print(f"\n  [Test 1] All {len(current_entries)} conditions cached. Skipping.")
                elif existing:
                    print(f"\n  [Test 1] {len(existing)} cached, "
                          f"{len(missing)} new condition(s) to compute.")
                else:
                    print(f"\n  [Test 1] No cache. Computing all {len(missing)} condition(s).")

            new_by_label = {}
            if missing:
                label_to_offset = {label: idx for idx, (_, label) in enumerate(current_entries)}
                seed_offsets = [label_to_offset[label] for (_, label) in missing]
                new_filters = [fd for fd, _ in missing]
                psth_cache = build_psth_cache(new_filters, spikeData_proj, trialData_full,
                                               neuron_sessions, bins, align_to, window_dpca,
                                               z_score_data, z_score_mode)
                new_results, psth_cache = run_deviation_vs_baseline(
                    spikeData_proj, trialData_full, neuron_sessions,
                    decoder_matrix=D[marg_key], neuron_means=neuron_means,
                    single_conditions=missing,
                    bins=bins, time_centers=time_centers,
                    align_to=align_to, window_dpca=window_dpca,
                    z_score_data=z_score_data, z_score_mode=z_score_mode,
                    components=components, n_bootstrap=n_bootstrap, ci=ci,
                    pre_window=pre_window,
                    cluster_window=cluster_window,
                    cluster_threshold_pctile=cluster_threshold_pctile,
                    p_alpha=p_alpha, n_jobs=n_jobs, backend=parallel_backend,
                    psth_cache=psth_cache, condition_seed_offsets=seed_offsets)
                for res in new_results: new_by_label[res['label']] = res

            merged = []
            for fd, label in current_entries:
                if label in new_by_label:   merged.append(new_by_label[label])
                elif label in existing:     merged.append(existing[label])
            if not force_t1 and not prune_orphans:
                orphans = [res for label, res in existing.items()
                           if label not in {l for _, l in current_entries}]
                if orphans:
                    print(f"  Keeping {len(orphans)} orphan condition(s). "
                          f"Use --prune-orphans to drop.")
                    merged.extend(orphans)

            all_results['test1'][marg_key] = merged
            if missing or force_t1:
                _print_deviation_summary(merged, time_centers)
                sd = _build_deviation_save_dict(
                    merged, time_centers, components, marg_key,
                    n_bootstrap, ci, pre_window,
                    cluster_window, cluster_threshold_pctile)
                np.savez(t1_path, **sd, allow_pickle=True)
                print(f"  Saved: {os.path.basename(t1_path)}")

        # =============================================================
        # TEST 2: deviation change (ctrl vs opto)
        # =============================================================
        if run_test2 and marg_key in PC and PC[marg_key]:
            t2_path = os.path.join(save_dir, f'deviation_change_{marg_key}.npz')
            t2_stem = f"deviation_change_{marg_key}"
            force_t2 = t2_stem in force_stems
            existing, file_meta = ({}, {})
            if not force_t2:
                existing, file_meta = load_existing_deviation_change_by_label(t2_path)
                if existing:
                    _check_compatibility(file_meta, current_meta, os.path.basename(t2_path))

            current_entries = []; seen = set()
            for fc, fo, label in PC[marg_key]:
                if label in seen: continue
                seen.add(label); current_entries.append((fc, fo, label))

            if force_t2:
                missing = current_entries
                print(f"\n  [Test 2] [FORCE] Recomputing all {len(missing)} pair(s).")
            else:
                missing = [(fc, fo, label) for (fc, fo, label) in current_entries
                           if label not in existing]
                if existing and not missing:
                    print(f"\n  [Test 2] All {len(current_entries)} pair(s) cached. Skipping.")
                elif existing:
                    print(f"\n  [Test 2] {len(existing)} cached, "
                          f"{len(missing)} new pair(s) to compute.")
                else:
                    print(f"\n  [Test 2] No cache. Computing all {len(missing)} pair(s).")

            new_by_label = {}
            if missing:
                label_to_offset = {label: idx for idx, (_, _, label) in enumerate(current_entries)}
                seed_offsets = [label_to_offset[label] for (_, _, label) in missing]
                new_filters = []
                for fc, fo, _ in missing: new_filters.extend([fc, fo])
                if not psth_cache:
                    psth_cache = build_psth_cache(new_filters, spikeData_proj, trialData_full,
                                                   neuron_sessions, bins, align_to, window_dpca,
                                                   z_score_data, z_score_mode)
                else:
                    for fd in new_filters:
                        psth_cache = ensure_in_cache(
                            psth_cache, fd, spikeData_proj, trialData_full,
                            neuron_sessions, bins, align_to, window_dpca,
                            z_score_data, z_score_mode)
                new_results, psth_cache = run_deviation_change(
                    spikeData_proj, trialData_full, neuron_sessions,
                    decoder_matrix=D[marg_key], neuron_means=neuron_means,
                    paired_conditions=missing,
                    bins=bins, time_centers=time_centers,
                    align_to=align_to, window_dpca=window_dpca,
                    z_score_data=z_score_data, z_score_mode=z_score_mode,
                    components=components, n_bootstrap=n_bootstrap, ci=ci,
                    pre_window=pre_window,
                    cluster_window=cluster_window,
                    cluster_threshold_pctile=cluster_threshold_pctile,
                    subsample=subsample_trials,
                    p_alpha=p_alpha, n_jobs=n_jobs, backend=parallel_backend,
                    psth_cache=psth_cache, pair_seed_offsets=seed_offsets)
                for res in new_results: new_by_label[res['label']] = res

            merged = []
            for fc, fo, label in current_entries:
                if label in new_by_label: merged.append(new_by_label[label])
                elif label in existing:   merged.append(existing[label])
            if not force_t2 and not prune_orphans:
                orphans = [res for label, res in existing.items()
                           if label not in {l for _, _, l in current_entries}]
                if orphans:
                    print(f"  Keeping {len(orphans)} orphan pair(s). "
                          f"Use --prune-orphans to drop.")
                    merged.extend(orphans)

            all_results['test2'][marg_key] = merged
            if missing or force_t2:
                _print_deviation_change_summary(merged, time_centers)
                sd = _build_deviation_change_save_dict(
                    merged, time_centers, components, marg_key,
                    n_bootstrap, ci, pre_window,
                    cluster_window, cluster_threshold_pctile, subsample_trials)
                np.savez(t2_path, **sd, allow_pickle=True)
                print(f"  Saved: {os.path.basename(t2_path)}")

        all_psth_caches[marg_key] = psth_cache

    # --------------------------------------------------------------------
    # METADATA
    # --------------------------------------------------------------------
    run_metadata = {
        'area': area, 'n_neurons': n_neurons, 'align_to': align_to,
        'window': list(window_dpca), 'bin_size': bin_size, 'step_size': step_size,
        'n_bootstrap': n_bootstrap, 'ci': ci, 'z_score_data': z_score_data,
        'metric': 'deviation_from_baseline',
        'marg_components': list(marg_components),
        'subsample_trials': subsample_trials,
        'pre_window':  list(pre_window),
        'cluster_window': [None if (cluster_window is None or cluster_window[0] is None) else cluster_window[0],
                           None if (cluster_window is None or cluster_window[1] is None) else cluster_window[1]],
        'cluster_threshold_percentile': cluster_threshold_pctile,
        'p_alpha': p_alpha,
        'n_jobs': n_jobs,
        'last_run': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(save_dir, 'metadata_deviation.yaml'), 'w') as f:
        yaml.dump(run_metadata, f, default_flow_style=False)
    print(f"\nDone! Results in: {save_dir}")

    return {
        'save_dir': save_dir,
        'all_results': all_results,
        'all_psth_caches': all_psth_caches,
        'run_metadata': run_metadata,
    }


# ============================================================================
# CLI
# ============================================================================
def _parse_window(s):
    if s is None: return None
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 2: raise argparse.ArgumentTypeError("Window must be 'start,end'")
    def _p(x):
        if x.lower() in ('none', 'null', ''): return None
        return float(x)
    return (_p(parts[0]), _p(parts[1]))


def _parse_components(s):
    """Parse a comma-separated list of integers, e.g. '0,1,2' -> [0,1,2]."""
    if s is None: return None
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if not parts: return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "marg-components must be a comma-separated list of integers, e.g. '0,1,2'")


def _main_cli():
    parser = argparse.ArgumentParser(
        description="Compute dPCA-space deviation-from-baseline tests. "
                    "Test 1: trajectory leaves baseline (circular-shift null). "
                    "Test 2: opto deviation smaller than ctrl deviation (flattening).")
    parser.add_argument('--dpca-dir',  type=str, default=DEFAULT_DPCA_DIR)
    parser.add_argument('--align-to',  type=str, default=DEFAULT_ALIGN_TO,
                        help="Trial alignment event (e.g. 'stimChange', 'firstLick').")
    parser.add_argument('--subsample-trials', action='store_true',
                        default=DEFAULT_SUBSAMPLE_TRIALS)
    parser.add_argument('--n-bootstrap', type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument('--pre-window',  type=_parse_window,
                        default=DEFAULT_PRE_WINDOW,
                        help="Pre-stim baseline window, 'start,end' in seconds. "
                             "Default: '-1,0'")
    parser.add_argument('--cluster-window', type=_parse_window,
                        default=DEFAULT_CLUSTER_TIME_WINDOW,
                        help="'start,end' in seconds, or 'None,None'")
    parser.add_argument('--cluster-threshold-pctile', type=float,
                        default=DEFAULT_CLUSTER_THRESHOLD_PCTILE)
    parser.add_argument('--p-alpha', type=float, default=DEFAULT_P_ALPHA)
    parser.add_argument('--marg-components', type=_parse_components, default=None,
                        help="Components (0-indexed) to include from every "
                             "marginalization, e.g. '0' or '0,1,2'. "
                             f"Default: {DEFAULT_MARG_COMPONENTS}")
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--output-tag', type=str, default=None)
    parser.add_argument('--force', nargs='*', default=[],
                        help="Force-recompute specific file stems "
                             "(e.g. 'deviation_vs_baseline_s deviation_change_s').")
    parser.add_argument('--prune-orphans', action='store_true')
    parser.add_argument('--no-test1', dest='run_test1', action='store_false', default=True)
    parser.add_argument('--no-test2', dest='run_test2', action='store_false', default=True)
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args()

    try:
        run_deviation_analysis(
            dpca_dir=args.dpca_dir,
            align_to=args.align_to,
            subsample_trials=args.subsample_trials,
            n_bootstrap=args.n_bootstrap,
            pre_window=args.pre_window,
            cluster_window=args.cluster_window,
            cluster_threshold_pctile=args.cluster_threshold_pctile,
            p_alpha=args.p_alpha,
            marg_components=args.marg_components,
            output_dir=args.output_dir,
            output_tag=args.output_tag,
            force_stems=args.force,
            prune_orphans=args.prune_orphans,
            run_test1=args.run_test1,
            run_test2=args.run_test2,
            n_jobs=args.n_jobs,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    _main_cli()