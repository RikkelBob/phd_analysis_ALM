import numpy as np
from pathlib import Path
from neural_analysis import utils
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import config

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

# Trial balancing: subsample all conditions to the lowest trial count per panel
balance_trial_counts = True
balance_seed = 42  # Random seed for reproducible subsampling

# ============================================================================
# FIGURE DEFINITIONS
# ============================================================================

HITS = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
     'label': 'Visual Hit', 'color': 'blue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
     'label': 'Auditory Hit', 'color': 'red', 'linestyle': '-'},
]

ERRORS = [
    {'filter': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
     'label': 'Visual Error', 'color': 'dodgerblue', 'linestyle': '--'},
    {'filter': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
     'label': 'Auditory Error', 'color': 'orangered', 'linestyle': '--'},
]

MISSES = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Visual Miss', 'color': 'navy', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Auditory Miss', 'color': 'darkred', 'linestyle': ':'},
]

CORRECT_REJECTIONS = [
    {'filter': {'trialType': 'P', 'noResponse': 1, 'hasphotostim': False},
     'label': 'Correct Rejection', 'color': 'green', 'linestyle': '-'},
]

FALSE_ALARMS = [
    {'filter': {'trialType': 'P', 'responseSide': 'L', 'hasphotostim': False},
     'label': 'False Alarm (left)', 'color': 'orange', 'linestyle': '-'},
    {'filter': {'trialType': 'P', 'responseSide': 'R', 'hasphotostim': False},
     'label': 'False Alarm (right)', 'color': 'goldenrod', 'linestyle': '-'},
]

HITS_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True},
     'label': 'Visual Hit Opto', 'color': 'lightblue', 'linestyle': '-'},
    {'filter': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True},
     'label': 'Auditory Hit Opto', 'color': 'pink', 'linestyle': '-'},
]

ERRORS_OPTO = [
    {'filter': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True},
     'label': 'Visual Error Opto', 'color': 'lightskyblue', 'linestyle': '--'},
    {'filter': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True},
     'label': 'Auditory Error Opto', 'color': 'lightsalmon', 'linestyle': '--'},
]

MISSES_OPTO = [
    {'filter': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True},
     'label': 'Visual Miss Opto', 'color': 'slategray', 'linestyle': ':'},
    {'filter': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True},
     'label': 'Auditory Miss Opto', 'color': 'rosybrown', 'linestyle': ':'},
]

CORRECT_REJECTIONS_OPTO = [
    {'filter': {'trialType': 'P', 'noResponse': 1, 'hasphotostim': True},
     'label': 'Correct Rejection Opto', 'color': 'lightgreen', 'linestyle': '-'},
]

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
# FIGURES LIST
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
# dPCA._zero_mean: X - mean(X.reshape(N, -1), axis=1)
# i.e., the mean across all conditions × timebins.
neuron_means = X_dpca.reshape(X_dpca.shape[0], -1).mean(axis=1)  # (n_neurons,)
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


def get_trial_count_for_alignment(trialData_subset, align_to):
    """
    Return the number of trials that would survive alignment-specific filtering.
    This mirrors the filtering done inside build_condition_average.
    """
    if align_to == 'firstLick':
        trialData_subset = trialData_subset[trialData_subset['responseSide'].isin(['L', 'R'])]
        trialData_subset = trialData_subset[trialData_subset['reactionTime'] > (0.1 * 1e6)]
    return len(trialData_subset)


def subsample_trials_per_session(trialData_subset, n_target, rng):
    """
    Subsample trials to n_target, balancing proportionally across sessions.

    For each session, the number of trials to keep is proportional to that
    session's share of the total, so the relative session contributions are
    preserved (important because different sessions contribute different neurons).

    Parameters
    ----------
    trialData_subset : DataFrame
        Trials for a single condition (already filtered).
    n_target : int
        Target total number of trials.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    trialData_subsampled : DataFrame
    """
    n_total = len(trialData_subset)
    if n_total <= n_target:
        return trialData_subset

    # Proportional allocation across sessions
    session_counts = trialData_subset.groupby('session_ID').size()
    fractions = session_counts / n_total
    # Initial allocation (floor), then distribute remainders by largest fraction
    allocated = np.floor(fractions * n_target).astype(int)
    remainder = n_target - allocated.sum()
    if remainder > 0:
        fractional_parts = (fractions * n_target) - allocated
        top_sessions = fractional_parts.nlargest(remainder).index
        allocated[top_sessions] += 1

    subsampled_parts = []
    for ses, n_keep in allocated.items():
        ses_trials = trialData_subset[trialData_subset['session_ID'] == ses]
        if n_keep >= len(ses_trials):
            subsampled_parts.append(ses_trials)
        else:
            idx = rng.choice(len(ses_trials), size=n_keep, replace=False)
            subsampled_parts.append(ses_trials.iloc[idx])

    return pd.concat(subsampled_parts)


def build_condition_average(spikeData_proj, trialData_subset, neuron_sessions,
                            bins, align_to, window, z_score_data, z_score_mode,
                            trialData_full):
    """
    Build a condition-averaged PSTH vector: (n_neurons, n_timebins).

    Replicates how the dPCA analysis script builds X:
    - Loop over sessions
    - Compute psth_matrix per session (only session neurons × session trials)
    - Average across trials for each neuron
    - Place into the correct neuron positions

    Returns
    -------
    X_cond : array (n_neurons, n_timebins)
        Condition-averaged firing rate per neuron. NaN for neurons with no trials.
    n_trials : int
        Total number of trials in this condition (after alignment filtering).
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

        # Same function as the dPCA analysis script
        psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
                                      z_score_data=z_score_data, align_to=align_to,
                                      window=window, trialData_full=trialData_full,
                                      z_score_mode=z_score_mode)
        # shape: (n_session_neurons, n_session_trials, n_timebins)

        for local_idx, global_idx in enumerate(neuron_indices):
            trial_data = psth_ses[local_idx, :, :]  # (n_trials, n_timebins)
            valid = ~np.isnan(trial_data[:, 0])
            if np.sum(valid) > 0:
                X_cond[global_idx, :] = np.nanmean(trial_data[valid, :], axis=0)

    return X_cond, n_trials


def project_condition_average(X_cond, decoder_matrix, neuron_means):
    """
    Project a condition-averaged PSTH through dPCA decoders.

    Replicates dPCA.transform():
        X_centered = X - mean(X.reshape(N,-1), axis=1)
        Z = D.T @ X_centered

    Uses neuron_means from the ORIGINAL dPCA fit (not from this condition).
    Neurons with NaN (no trials for this condition) are set to zero after
    centering, equivalent to assuming they fire at their grand mean.

    Returns
    -------
    projections : array (n_components, n_timebins)
    """
    n_neurons, n_timebins = X_cond.shape

    X_centered = X_cond - neuron_means[:, np.newaxis]

    # NaN neurons: no trials for this condition. Set to 0 (= grand mean).
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

if balance_trial_counts:
    print(f"Trial balancing ENABLED (seed={balance_seed})")
else:
    print("Trial balancing DISABLED")

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
# When balancing is off, caches are shared across figures as before.
# When balancing is on, condition averages depend on the panel's min trial count,
# so we use a separate cache keyed by (filter, align_to, n_target).
cond_avg_cache = {}   # (hashable_filter, align_to[, n_target]) -> (X_cond, n_trials)
proj_cache = {}       # (hashable_filter, align_to[, n_target], marg_key) -> projections

# Pre-filter trials for all conditions used across all figures, so we can
# look up trial counts quickly.
trial_subsets_cache = {}  # hashable_filter -> trialData_subset (before alignment filtering)

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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows),
                              squeeze=False)

    for row_idx, panel in enumerate(panels):
        components = panel['component']
        align_to = panel.get('align_to', 'stimChange')
        row_title = panel.get('title', f'Row {row_idx}')

        print(f"\n  Row {row_idx}: {row_title} (aligned to {align_to})")

        # ==================================================================
        # TRIAL BALANCING: determine min trial count across conditions
        # ==================================================================
        n_target = None  # None means no balancing
        if balance_trial_counts:
            panel_trial_counts = []
            for cond in panel['conditions']:
                filter_key = make_hashable_filter(cond['filter'])
                if filter_key not in trial_subsets_cache:
                    trial_subsets_cache[filter_key] = filter_trials(trialData_full, cond['filter'])
                td_sub = trial_subsets_cache[filter_key]
                n = get_trial_count_for_alignment(td_sub, align_to)
                panel_trial_counts.append(n)

            min_trials = min(panel_trial_counts)
            if min_trials == 0:
                print(f"    WARNING: a condition has 0 trials; balancing skipped for this panel.")
                n_target = None
            elif min_trials < max(panel_trial_counts):
                n_target = min_trials
                print(f"    Balancing trials to {n_target} "
                      f"(range across conditions: {min_trials}–{max(panel_trial_counts)})")
            else:
                print(f"    All conditions have {min_trials} trials; no subsampling needed.")

        rng = np.random.default_rng(balance_seed) if n_target is not None else None

        for cond in panel['conditions']:
            cond_filter = cond['filter']
            label = cond['label']
            color = cond['color']
            linestyle = cond.get('linestyle', '-')

            filter_key = make_hashable_filter(cond_filter)

            # Cache key includes n_target so balanced and unbalanced results
            # are stored separately (and different panels with different
            # n_target don't collide).
            avg_key = (filter_key, align_to, n_target)
            proj_key = (filter_key, align_to, n_target, marg_key)

            # --- Build condition average ---
            if avg_key not in cond_avg_cache:
                # Get (or reuse) the filtered trial subset
                if filter_key not in trial_subsets_cache:
                    trial_subsets_cache[filter_key] = filter_trials(trialData_full, cond_filter)
                trialData_subset = trial_subsets_cache[filter_key].copy()

                # Apply alignment-specific filtering BEFORE subsampling,
                # so the count matches what build_condition_average will see.
                if align_to == 'firstLick':
                    trialData_subset = trialData_subset[
                        trialData_subset['responseSide'].isin(['L', 'R'])]
                    trialData_subset = trialData_subset[
                        trialData_subset['reactionTime'] > (0.1 * 1e6)]

                # Subsample if balancing
                if n_target is not None and len(trialData_subset) > n_target:
                    trialData_subset = subsample_trials_per_session(
                        trialData_subset, n_target, rng)

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

            # --- Project once per (condition, alignment, n_target, marg) ---
            if proj_key not in proj_cache:
                proj_cache[proj_key] = project_condition_average(
                    X_cond.copy(), decoder_matrix, neuron_means
                )

            projections = proj_cache[proj_key]  # (n_components, n_timebins)

            # --- Plot ---
            for col_idx, comp in enumerate(components):
                ax = axes[row_idx, col_idx]

                if comp >= decoder_matrix.shape[1]:
                    if cond is panel['conditions'][0]:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes)
                    continue

                if n_trials == 0:
                    continue

                trajectory = projections[comp, :]

                ax.plot(time_centers, trajectory,
                        color=color, linestyle=linestyle, linewidth=2,
                        label=f"{label} (n={n_trials})")

        # --- Formatting ---
        for col_idx, comp in enumerate(components):
            ax = axes[row_idx, col_idx]

            ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.grid(False)

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
                ax.set_title(comp_title, fontsize=11, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\n\nProjection (a.u.)", fontsize=10)
            else:
                ax.set_ylabel('Projection (a.u.)', fontsize=10)

            if row_idx == n_rows - 1:
                if align_to == 'firstLick':
                    ax.set_xlabel('Time from first lick (s)', fontsize=10)
                else:
                    ax.set_xlabel('Time from stimulus (s)', fontsize=10)

            if col_idx == 0:
                ax.legend(fontsize=7, loc='best')

    fig.suptitle(f"{fig_title} — {', '.join(area)}",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_dir, f"{fig_name}_projections.png")
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {filepath}")
    plt.close(fig)

print(f"\nDone! All outputs saved to: {save_dir}")