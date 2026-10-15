import os
from matplotlib import pyplot as plt
import matplotlib
from neural_analysis import utils
from neural_analysis.ROC import ROC_utils
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from neural_analysis.plot_utils import paper_colors
from scipy.stats import binom
import pandas as pd
import yaml
import config

save_dir = config.FIGURE_DIR / "ROC"

sig_level = 0.01 #AUC vs shuffled AUC alpha
areas = ['lALM', 'rALM']
shuffled = False
only_neurons_good_sessions = True
auc_threshold = 0.6 # Not used
n_min_trials = None
p_alpha = 0.01 #Alpha for testing against baseline / chance / between-condition testing

# Which significance bars to plot (Against baseline: sig_difference_bsl; Against chance: sig_difference_chance
sig_type = 'sig_difference_chance'

# ── Font & style configuration ──────────────────────────────────────
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.size': 6,            # global default (ticks, legend, etc.)
    'axes.titlesize': 8,       # subplot titles
    'axes.labelsize': 8,       # axis labels
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})

# ── Subplot dimensions (inches) ────────────────────────────────────
SUBPLOT_W = 1
SUBPLOT_H = 20/25.4
H_PAD = 0.6
V_PAD = 0.7
LEFT_MARGIN = 0.6
RIGHT_MARGIN = 0.3
TOP_MARGIN = 0.3
BOT_MARGIN = 0.5
X_LIM_DEFAULT = [-0.5, 1.5]  # set to None for full range

# Location of stimulus-aligned ROC results
data_dir = config.resolve_pp("ROC/stim_aligned")
# Location of lick-aligned ROC results
data_dir_lick_aligned = config.resolve_pp("ROC/lick_aligned")

# Location of task-irrelevant decoding (hit/miss balanced)
data_dir_irr = config.resolve_pp("ROC/balanced_hit_miss/stim_aligned")

# Initialize dict to load results into
data = {
    'lALM': {
        'folder': os.path.join(data_dir, r"lALM/opto_off")
    },

    'rALM': {
        'folder': os.path.join(data_dir, r"rALM/opto_off")
    },

    'lALM_unilat': {
        'folder': os.path.join(data_dir, r"lALM/unilat")
    },

    'lALM_bilat': {
        'folder': os.path.join(data_dir, r"lALM/bilat")
    },

    'rALM_bilat': {
        'folder': os.path.join(data_dir, r"rALM/bilat")
    },

    'rALM_unilat': {
        'folder': os.path.join(data_dir, r"rALM/unilat")
    }
}

data_combined = {
    'lALM_rALM': {
        'data_keys': ['lALM', 'rALM']
    },

    'lALM_rALM_bilat': {
        'data_keys': ['lALM_bilat', 'rALM_bilat']
    }
}

data_lick_aligned = {
    'lALM': {
        'folder': os.path.join(data_dir_lick_aligned, r"lALM/opto_off")
    },

    'rALM': {
        'folder': os.path.join(data_dir_lick_aligned, r"rALM/opto_off")
    },

    'lALM_unilat': {
        'folder': os.path.join(data_dir_lick_aligned, r"lALM/unilat")
    },

    'lALM_bilat': {
        'folder': os.path.join(data_dir_lick_aligned, r"lALM/bilat")
    },

    'rALM_bilat': {
        'folder': os.path.join(data_dir_lick_aligned, r"rALM/bilat")
    }
}

data_combined_lick_aligned = {
    'lALM_rALM': {
        'data_keys': ['lALM', 'rALM']
    },

    'lALM_rALM_bilat': {
        'data_keys': ['lALM_bilat', 'rALM_bilat']
    }
}

data_task_irrelevant = {
    'lALM': {
        'folder': os.path.join(data_dir_irr, r"lALM/opto_off")
    },

    'rALM': {
        'folder': os.path.join(data_dir_irr, r"rALM/opto_off")
    }
}

data_combined_task_irrelevant = {
    'lALM_rALM': {
        'data_keys': ['lALM', 'rALM']
    }
}

_, sessionData, spikeData = utils.load_data(config.DATA_DIR_EXP,
                                  ['ALM_1', 'ALM_2', 'ALM_3'],
                                  animals=None,
                                  data_types=['spikeData', 'sessionData'],
                                  exclude_weird_trials=False,
                                  exclude_grace_violations=False,
                                  trim_last_trials=False)

# Only include neurons from "good" sessions (performance > 0.3 for max stim, both modalities)
clusters_incl = spikeData.copy()

if only_neurons_good_sessions:
    sessions_incl = utils.good_behavior_sessions()
    clusters_incl = clusters_incl[clusters_incl['session_ID'].isin(sessions_incl)]

# Create a separate inclusion list for opto plots (exclude ALM_1)
sessions_good_opto = sessionData[sessionData['experiment'] != 'ALM_1']['session_ID']
clusters_incl_opto = clusters_incl[clusters_incl['session_ID'].isin(sessions_good_opto)]

# ─────────────────────────────────────────────────────────────────────
# Lookup tables: cluster_ID -> session_ID, session_ID -> mousename
# Used to annotate panel titles with n_neurons / n_sessions / n_animals.
# ─────────────────────────────────────────────────────────────────────
cluster_to_session = dict(zip(spikeData['cluster_ID'], spikeData['session_ID']))
session_to_mouse = dict(zip(sessionData['session_ID'], sessionData['mousename']))


def count_neurons_sessions_animals(neuron_ids):
    """Given an iterable of cluster_IDs, return (n_neurons, n_sessions, n_animals)."""
    if neuron_ids is None:
        return 0, 0, 0
    neuron_ids = np.asarray(neuron_ids)
    n_neurons = int(neuron_ids.size)
    sessions = {cluster_to_session.get(c) for c in neuron_ids}
    sessions.discard(None)
    animals = {session_to_mouse.get(s) for s in sessions}
    animals.discard(None)
    return n_neurons, len(sessions), len(animals)


# ─────────────────────────────────────────────────────────────────────
# Helpers for significance-range extraction and YAML serialization
# ─────────────────────────────────────────────────────────────────────
def _extract_sig_ranges(sig_mask, time_axis):
    """
    Convert a boolean significance mask over bins into a list of contiguous
    time ranges. Each range is reported as the time of the first significant
    bin through the time of the last significant bin (bin-centre/left-edge
    convention inherited from `time_axis`).

    Returns a list of dicts: [{'start': float, 'end': float, 'n_bins': int}, ...]
    """
    sig_mask = np.asarray(sig_mask, dtype=bool)
    time_axis = np.asarray(time_axis, dtype=float)

    if sig_mask.size == 0 or not sig_mask.any():
        return []

    # Find edges of runs of True
    padded = np.concatenate([[False], sig_mask, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]   # inclusive start indices in sig_mask
    ends = np.where(diffs == -1)[0] - 1  # inclusive end indices in sig_mask

    ranges = []
    for s, e in zip(starts, ends):
        ranges.append({
            'start': float(time_axis[s]),
            'end': float(time_axis[e]),
            'n_bins': int(e - s + 1),
        })
    return ranges


def _py(obj):
    """Recursively coerce numpy scalars/arrays to native Python for clean YAML."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _summarize_bsl_test(entry, time_axis, alpha):
    """Build the YAML-ready summary for a single-area vs-baseline test."""
    sig_mask = entry['sig_difference_bsl']
    ranges = _extract_sig_ranges(sig_mask, time_axis)
    return {
        'test': "Fisher's exact, coding-vs-noncoding neurons per bin vs. "
                "mean pre-stimulus baseline; FDR (Benjamini-Hochberg) corrected",
        'alpha': float(alpha),
        'n_neurons': int(entry['n_neurons']),
        'n_coding_pre_baseline': float(entry['n_coding_pre']),
        'n_noncoding_pre_baseline': float(entry['n_non_coding_pre'])
            if 'n_non_coding_pre' in entry else float(entry.get('n_noncoding_pre')),
        'n_bins_total': int(len(sig_mask)),
        'n_bins_significant': int(np.sum(sig_mask)),
        'significant_time_ranges': ranges,
    }


def _summarize_between_test(n_coding_1, n_neurons_1, n_coding_2, n_neurons_2,
                            time_axis, alpha):
    """Recompute between-area Fisher test + FDR and package for YAML."""
    p_values_between = []
    for idx in range(len(n_coding_1)):
        c1 = n_coding_1[idx]
        c2 = n_coding_2[idx]
        table = np.array([[c1, n_neurons_1 - c1],
                          [c2, n_neurons_2 - c2]])
        _, p = fisher_exact(table)
        p_values_between.append(p)
    p_values_between = np.array(p_values_between)
    sig_between, p_adj_between, _, _ = multipletests(
        p_values_between, alpha=alpha, method='fdr_bh'
    )
    ranges = _extract_sig_ranges(sig_between, time_axis)
    return {
        'test': "Fisher's exact, coding-vs-noncoding neurons per bin between "
                "the two conditions/areas; FDR (Benjamini-Hochberg) corrected",
        'alpha': float(alpha),
        'n_neurons_condition_1': int(n_neurons_1),
        'n_neurons_condition_2': int(n_neurons_2),
        'n_bins_total': int(len(sig_between)),
        'n_bins_significant': int(np.sum(sig_between)),
        'significant_time_ranges': ranges,
    }


# Process results for each separate condition
for data_to_process in [data, data_lick_aligned, data_task_irrelevant]:
    for area_key in data_to_process:
        area_dir = data_to_process[area_key]['folder']

        if not os.path.isdir(area_dir):
            print(f"Directory {area_dir} does not exist, skipping area '{area_key}'.")
            continue

        conditions = [os.path.join(area_dir, name) for name in os.listdir(area_dir) if
                      os.path.isdir(os.path.join(area_dir, name))]

        # Iterate over each condition
        for condition_folder in conditions:
            condition = os.path.basename(condition_folder)

            # Load data for this condition
            try:
                auc_scores, neurons, p_values, params, filters, n_trials = ROC_utils.roc_load_results(condition_folder)
            except Exception as e:
                print(f"  Failed to load results for {area_key}/{condition}: {e}. Skipping.")
                continue

            # Filter neurons
            if n_min_trials is not None:
                mask_n_trials = np.all(n_trials > n_min_trials, axis=1)
                neurons = neurons[mask_n_trials]
                auc_scores = auc_scores[mask_n_trials]
                p_values = p_values[mask_n_trials]

            if clusters_incl is not None:
                is_opto_key = 'unilat' in area_key or 'bilat' in area_key
                incl = clusters_incl_opto if is_opto_key else clusters_incl
                neurons_mask = np.isin(neurons, incl['cluster_ID'])

                neurons = neurons[neurons_mask]
                auc_scores = auc_scores[neurons_mask]
                p_values = p_values[neurons_mask]

            # If false-positive rates exist, load them too
            fp_file = os.path.join(condition_folder, 'p_perm.npy')
            fp_std_file = os.path.join(condition_folder, 'p_perm_std.npy')

            fp_p = None
            fp_std = None

            if os.path.isfile(fp_file):
                fp_p = np.load(fp_file)
                if n_min_trials is not None:
                    fp_p = fp_p[mask_n_trials]
                if clusters_incl is not None:
                    fp_p = fp_p[neurons_mask]

            if os.path.isfile(fp_std_file):
                fp_std = np.load(fp_std_file)
                if n_min_trials is not None:
                    fp_std = fp_std[mask_n_trials]
                if clusters_incl is not None:
                    fp_std = fp_std[neurons_mask]

            significance = (p_values < sig_level)
            n_neurons = auc_scores.shape[0]

            if 't_bins' in params:
                time_axis = params['t_bins']
            else:
                start = params['window'][0]
                time_axis = start + np.arange(params['n_bins']) * params['step_size']

            pre_stimulus_bins = np.where(time_axis < 0)
            post_stimulus_bins = np.where(time_axis >= 0)

            n_coding_neurons = sum(significance)
            if n_neurons == 0:
                print(f"  {area_key}/{condition} has no neurons. Skipping.")
                continue

            fract_sig_neurons = n_coding_neurons / n_neurons

            # Average number of coding- and non-coding neurons in a bin pre-stimulus
            n_coding_pre = np.round(np.mean(n_coding_neurons[pre_stimulus_bins]))
            n_noncoding_pre = n_neurons - n_coding_pre

            p_values_bsl = []
            for coding_neurons_bin in n_coding_neurons:
                contingency_table_bsl = np.array([[n_coding_pre, n_noncoding_pre],
                                              [coding_neurons_bin, (n_neurons - coding_neurons_bin)]])
                odds_ratio, p_value = fisher_exact(contingency_table_bsl)
                p_values_bsl.append(p_value)

            p_values_bsl = np.array(p_values_bsl)

            # FDR-corrected significance testing
            sig_difference_bsl, p_adjusted, _, _ = multipletests(p_values_bsl, alpha=p_alpha, method='fdr_bh')

            # One-sided binomial test vs theoretical chance (sig_level)
            p_values_chance = binom.sf(n_coding_neurons - 1, n_neurons, sig_level)

            sig_difference_chance, p_adjusted_chance, _, _ = multipletests(
                p_values_chance, alpha=p_alpha, method='fdr_bh'
            )

            # Save data in dict
            data_to_process[area_key][condition] = {
                'auc_scores': auc_scores,
                'neurons': neurons,
                'p_values': p_values,
                'params': params,
                'filters': filters,
                'n_trials': n_trials,
                'fract_sig_neurons': fract_sig_neurons,
                'n_neurons': n_neurons,
                'n_coding_neurons': n_coding_neurons,
                'n_coding_pre': n_coding_pre,
                'n_non_coding_pre': n_noncoding_pre,
                'p_values_bsl': p_values_bsl,
                'pre_stimulus_bins': pre_stimulus_bins,
                'post_stimulus_bins': post_stimulus_bins,
                'sig_difference_bsl': sig_difference_bsl,
                'sig_difference_chance': sig_difference_chance,
                'p_adjusted': p_adjusted,
                'fp_p': fp_p,
                'fp_std': fp_std,
                'time_axis': np.asarray(time_axis),
            }


# ─────────────────────────────────────────────────────────────────────
# Enforce matched neuron sets across opto comparisons.
#
# For each (base_area, opto_area) pair (e.g. lALM vs lALM_unilat), create
# matched copies of both keys that only contain the intersection of
# cluster_IDs present in both. All derived stats are recomputed from
# the intersected arrays so panels compare apples to apples.
#
# We store the matched versions under new keys like 'lALM__for_unilat',
# 'lALM_unilat__matched', etc. The original keys remain untouched so that
# non-opto figures keep their existing (larger) populations.
# ─────────────────────────────────────────────────────────────────────
OPTO_PAIRS = [
    ('lALM', 'lALM_unilat'),
    ('lALM', 'lALM_bilat'),
    ('rALM', 'rALM_unilat'),
    ('rALM', 'rALM_bilat'),
]


def _recompute_entry_for_neuron_subset(entry, keep_mask, time_axis, sig_lvl, alpha):
    """
    Re-derive all per-condition stats for a subset of neurons selected by
    `keep_mask` (boolean array aligned to entry['neurons']).
    Returns a new dict that mirrors the original entry's shape.
    """
    auc_scores = entry['auc_scores'][keep_mask]
    neurons = entry['neurons'][keep_mask]
    p_values = entry['p_values'][keep_mask]
    fp_p = entry['fp_p'][keep_mask] if entry.get('fp_p') is not None else None
    fp_std = entry['fp_std'][keep_mask] if entry.get('fp_std') is not None else None

    n_neurons = auc_scores.shape[0]
    if n_neurons == 0:
        return None

    significance = (p_values < sig_lvl)
    n_coding_neurons = np.sum(significance, axis=0)
    fract_sig_neurons = n_coding_neurons / n_neurons

    pre_stimulus_bins = np.where(np.asarray(time_axis) < 0)
    post_stimulus_bins = np.where(np.asarray(time_axis) >= 0)

    n_coding_pre = np.round(np.mean(n_coding_neurons[pre_stimulus_bins]))
    n_noncoding_pre = n_neurons - n_coding_pre

    p_values_bsl = []
    for coding_neurons_bin in n_coding_neurons:
        contingency = np.array([[n_coding_pre, n_noncoding_pre],
                                [coding_neurons_bin, (n_neurons - coding_neurons_bin)]])
        _, p = fisher_exact(contingency)
        p_values_bsl.append(p)
    p_values_bsl = np.array(p_values_bsl)

    sig_difference_bsl, p_adjusted, _, _ = multipletests(
        p_values_bsl, alpha=alpha, method='fdr_bh'
    )

    p_values_chance = binom.sf(n_coding_neurons - 1, n_neurons, sig_lvl)
    sig_difference_chance, _, _, _ = multipletests(
        p_values_chance, alpha=alpha, method='fdr_bh'
    )

    return {
        'auc_scores': auc_scores,
        'neurons': neurons,
        'p_values': p_values,
        'params': entry['params'],
        'filters': entry.get('filters'),
        'n_trials': entry.get('n_trials'),
        'fract_sig_neurons': fract_sig_neurons,
        'n_neurons': n_neurons,
        'n_coding_neurons': n_coding_neurons,
        'n_coding_pre': n_coding_pre,
        'n_non_coding_pre': n_noncoding_pre,
        'p_values_bsl': p_values_bsl,
        'pre_stimulus_bins': pre_stimulus_bins,
        'post_stimulus_bins': post_stimulus_bins,
        'sig_difference_bsl': sig_difference_bsl,
        'sig_difference_chance': sig_difference_chance,
        'p_adjusted': p_adjusted,
        'fp_p': fp_p,
        'fp_std': fp_std,
        'time_axis': np.asarray(time_axis),
    }


def _match_opto_pair(data_source, base_area, opto_area):
    """
    For every condition that exists in BOTH base_area and opto_area, build
    matched-neuron versions and store them under new keys:
        '<base_area>__for_<suffix>'   (e.g. 'lALM__for_unilat')
        '<opto_area>__matched'        (e.g. 'lALM_unilat__matched')
    where <suffix> is whatever follows the underscore in opto_area
    ('unilat' or 'bilat').
    """
    suffix = opto_area.split('_', 1)[1]  # 'unilat' or 'bilat'
    base_matched_key = f"{base_area}__for_{suffix}"
    opto_matched_key = f"{opto_area}__matched"

    # Ensure containers exist
    data_source.setdefault(base_matched_key, {})
    data_source.setdefault(opto_matched_key, {})

    base_dict = data_source.get(base_area, {})
    opto_dict = data_source.get(opto_area, {})

    # Conditions present in both
    shared_conditions = [
        c for c in base_dict
        if c != 'folder' and c in opto_dict and c != 'folder'
    ]

    for condition in shared_conditions:
        base_entry = base_dict[condition]
        opto_entry = opto_dict[condition]

        base_neurons = np.asarray(base_entry['neurons'])
        opto_neurons = np.asarray(opto_entry['neurons'])

        # Intersection of cluster IDs
        shared_ids = np.intersect1d(base_neurons, opto_neurons)
        if shared_ids.size == 0:
            print(f"  No shared neurons between {base_area} and {opto_area} "
                  f"for condition '{condition}'. Skipping.")
            continue

        base_mask = np.isin(base_neurons, shared_ids)
        opto_mask = np.isin(opto_neurons, shared_ids)

        # Use each entry's own time_axis (should be identical anyway)
        ta_base = base_entry.get('time_axis')
        ta_opto = opto_entry.get('time_axis')

        new_base = _recompute_entry_for_neuron_subset(
            base_entry, base_mask, ta_base, sig_level, p_alpha
        )
        new_opto = _recompute_entry_for_neuron_subset(
            opto_entry, opto_mask, ta_opto, sig_level, p_alpha
        )
        if new_base is None or new_opto is None:
            continue

        data_source[base_matched_key][condition] = new_base
        data_source[opto_matched_key][condition] = new_opto

    n_shared = sum(1 for c in shared_conditions
                   if c in data_source[base_matched_key])
    print(f"  Matched {base_area} ↔ {opto_area}: "
          f"{n_shared} conditions, "
          f"keys '{base_matched_key}', '{opto_matched_key}'.")


for source in [data, data_lick_aligned]:
    for base_area, opto_area in OPTO_PAIRS:
        if base_area not in source or opto_area not in source:
            continue
        _match_opto_pair(source, base_area, opto_area)


def get_loaded_conditions(data_source):
    """Return the set of condition names found in any area_key of a data source."""
    conditions_found = set()
    for area_key in data_source:
        for key in data_source[area_key]:
            if key == 'folder':
                continue
            conditions_found.add(key)
    return conditions_found


# Combine data for lALM and rALM, plus matched combined keys for the
# bilateral-opto comparison so that lALM_rALM_bilat is properly matched too.
data_combined['lALM_rALM_for_bilat'] = {
    'data_keys': ['lALM__for_bilat', 'rALM__for_bilat']
}
data_combined['lALM_rALM_bilat__matched'] = {
    'data_keys': ['lALM_bilat__matched', 'rALM_bilat__matched']
}

data_combined_lick_aligned['lALM_rALM_for_bilat'] = {
    'data_keys': ['lALM__for_bilat', 'rALM__for_bilat']
}
data_combined_lick_aligned['lALM_rALM_bilat__matched'] = {
    'data_keys': ['lALM_bilat__matched', 'rALM_bilat__matched']
}

for data_source, data_to_combine in [(data, data_combined),
                                     (data_lick_aligned, data_combined_lick_aligned),
                                     (data_task_irrelevant, data_combined_task_irrelevant)]:

    all_conditions = get_loaded_conditions(data_source)

    for data_combination in data_to_combine:
        data_key_1 = data_to_combine[data_combination]['data_keys'][0]
        data_key_2 = data_to_combine[data_combination]['data_keys'][1]

        for condition in all_conditions:
            # Check that the condition exists in BOTH data keys
            if condition not in data_source.get(data_key_1, {}):
                print(f"  {condition} missing from {data_key_1}, skipping combine for {data_combination}.")
                continue
            if condition not in data_source.get(data_key_2, {}):
                print(f"  {condition} missing from {data_key_2}, skipping combine for {data_combination}.")
                continue

            ds1 = data_source[data_key_1][condition]
            ds2 = data_source[data_key_2][condition]

            n_neurons = ds1['n_neurons'] + ds2['n_neurons']
            n_coding_neurons = ds1['n_coding_neurons'] + ds2['n_coding_neurons']
            pre_stimulus_bins = ds1['pre_stimulus_bins']
            fract_sig_neurons = n_coding_neurons / n_neurons

            # Average number of coding- and non-coding neurons in a bin pre-stimulus
            n_coding_pre = np.round(np.mean(n_coding_neurons[pre_stimulus_bins]))
            n_noncoding_pre = n_neurons - n_coding_pre

            p_values_bsl = []
            for coding_neurons_bin in n_coding_neurons:
                contingency_table_bsl = np.array([[n_coding_pre, n_noncoding_pre],
                                                  [coding_neurons_bin, (n_neurons - coding_neurons_bin)]])
                odds_ratio, p_value = fisher_exact(contingency_table_bsl)
                p_values_bsl.append(p_value)

            p_values_bsl = np.array(p_values_bsl)

            # FDR-corrected significance testing
            sig_difference_bsl, p_adjusted, _, _ = multipletests(p_values_bsl, alpha=p_alpha, method='fdr_bh')

            # One-sided binomial test vs theoretical chance (sig_level)
            p_values_chance = binom.sf(n_coding_neurons - 1, n_neurons, sig_level)

            sig_difference_chance, p_adjusted_chance, _, _ = multipletests(
                p_values_chance, alpha=p_alpha, method='fdr_bh'
            )

            # Add false positive rates
            fp_1 = ds1.get('fp_p')
            fp_2 = ds2.get('fp_p')

            if fp_1 is not None and fp_2 is not None:
                combined_fp = np.concatenate([fp_1, fp_2], axis=0)
            else:
                combined_fp = None

            fp_std_1 = ds1.get('fp_std')
            fp_std_2 = ds2.get('fp_std')

            if fp_std_1 is not None and fp_std_2 is not None:
                combined_fp_std = np.concatenate([fp_std_1, fp_std_2], axis=0)
            else:
                combined_fp_std = None

            # Concatenate the neurons (cluster IDs) so we can later count sessions/animals
            neurons_1 = ds1.get('neurons')
            neurons_2 = ds2.get('neurons')
            if neurons_1 is not None and neurons_2 is not None:
                combined_neurons = np.concatenate([neurons_1, neurons_2])
            else:
                combined_neurons = None

            data_to_combine[data_combination][condition] = {
                'n_neurons': n_neurons,
                'n_coding_neurons': n_coding_neurons,
                'p_values_bsl': p_values_bsl,
                'n_coding_pre': n_coding_pre,
                'n_noncoding_pre': n_noncoding_pre,
                'fract_sig_neurons': fract_sig_neurons,
                'sig_difference_bsl': sig_difference_bsl,
                'sig_difference_chance': sig_difference_chance,
                'p_adjusted': p_adjusted,
                'fp_p': combined_fp,
                'fp_std': combined_fp_std,
                'neurons': combined_neurons,
                # Propagate a time axis so combined datasets can be summarized too
                'time_axis': ds1.get('time_axis'),
            }


def _resolve_time_axis(data_set, area_keys, condition_name):
    """Try to extract a time_axis from any available area_key for a given condition."""
    for ak in area_keys:
        entry = data_set.get(ak, {}).get(condition_name)
        if entry is None:
            continue
        if 'params' in entry:
            params = entry['params']
            if 't_bins' in params:
                return params['t_bins']
            else:
                start = params['window'][0]
                return start + np.arange(params['n_bins']) * params['step_size']
        # Combined datasets may store their own time_axis instead of params
        if 'time_axis' in entry and entry['time_axis'] is not None:
            return entry['time_axis']
    return None


figures_dict = {

    'lALM_vs_rALM_encoding': {
        'save_name': 'lALM_vs_rALM_encoding_paper',
        'layout': (1, 2),
        'panels': {
            'Stimulus presence (miss vs CR)': {
                'condition_name': ['visual_miss_cr', 'auditory_miss_cr'],
                'data': data,
                'area_keys': ['lALM', 'rALM'],
                'colors': ['blue', 'navy', 'red', 'pink'],
                'opto': False,
                'position': 0,
                'ylim': (0, 0.1)
            },

            'Hit coding (hit vs miss)': {
                'condition_name': ['visual_hit_miss', 'auditory_hit_miss'],
                'data': data,
                'area_keys': ['lALM', 'rALM'],
                'colors': ['blue', 'navy', 'red', 'pink'],
                'opto': False,
                'position': 1,
                'ylim': (0, 0.6)
            },
        }
    },

    'lALM_rALM': {
        'save_name': 'lALM_rALM_combined_paper',
        'layout': (1, 3),
        'panels': {
            'Hits vs misses': {
                'condition_name': ['visual_hit_miss', 'auditory_hit_miss'],
                'data': data_combined,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors('vis_max'), paper_colors('aud_max')],
                'opto': False,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Stimulus presence (miss vs CR)': {
                'condition_name': ['visual_miss_cr', 'auditory_miss_cr', 'visual_miss_auditory_miss'],
                'data': data_combined,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors('vis_max'), paper_colors('aud_max'), 'purple'],
                'opto': False,
                'position': 1,
                'ylim': (0, 0.1)
            },

            'Lick presence (FA vs CR)': {
                'condition_name': ['left_lick_FA_CR', 'right_lick_FA_CR'],
                'data': data_combined,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors("left_lick"), paper_colors("right_lick")],
                'opto': False,
                'position': 2,
                'ylim': (0, 0.1)
            },
        }
    },

    'lALM_unilat': {
        'save_name': 'lALM_unilat_paper',
        'layout': (5, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Visual hits vs misses': {
                'condition_name': 'visual_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Auditory hits vs misses': {
                'condition_name': 'auditory_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Visual hits vs FA': {
                'condition_name': 'visual_hit_FA',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.15)
            },

            'Auditory hits vs FA': {
                'condition_name': 'auditory_hit_FA',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.15)
            },

            'Visual presence': {
                'condition_name': 'visual_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 4,
                'ylim': (0, 0.1)
            },

            'Auditory presence': {
                'condition_name': 'auditory_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 5,
                'ylim': (0, 0.1)
            },

            'Visual vs aud misses': {
                'condition_name': 'visual_miss_auditory_miss',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 6,
                'ylim': (0, 0.1)
            },

            'Visual Hits vs FA opto (lick-aligned)': {
                'condition_name': ['visual_hit_FA'],
                'data': data_lick_aligned,
                'area_keys': ['lALM_unilat__matched', 'lALM__for_unilat'],
                'colors': [paper_colors('opto_purple'), paper_colors('vis_max')],
                'opto': True,
                'position': 7,
                'ylim': (0, 0.25),
                'xlim': (-1, 1)
            },

            'Auditory Hits vs FA opto (lick-aligned)': {
                'condition_name': ['auditory_hit_FA'],
                'data': data_lick_aligned,
                'area_keys': ['lALM_unilat__matched', 'lALM__for_unilat'],
                'colors': [paper_colors('opto_purple'), paper_colors('aud_max')],
                'opto': True,
                'position': 8,
                'ylim': (0, 0.15),
                'xlim': (-1, 1)
            },
        }
    },

    'lALM_bilat': {
        'save_name': 'lALM_bilat_paper',
        'layout': (4, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Visual hits vs misses': {
                'condition_name': 'visual_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Auditory hits vs misses': {
                'condition_name': 'auditory_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Visual hits vs FA': {
                'condition_name': 'visual_hit_FA',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.15)
            },

            'Auditory hits vs FA': {
                'condition_name': 'auditory_hit_FA',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.15)
            },

            'Visual presence': {
                'condition_name': 'visual_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 4,
                'ylim': (0, 0.1)
            },

            'Auditory presence': {
                'condition_name': 'auditory_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 5,
                'ylim': (0, 0.1)
            },

            'Visual vs aud misses': {
                'condition_name': 'visual_miss_auditory_miss',
                'data': data,
                'area_keys': ['lALM__for_bilat', 'lALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 6,
                'ylim': (0, 0.1)
            }
        }
    },

    'rALM_bilat': {
        'save_name': 'rALM_bilat_paper',
        'layout': (4, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Visual hits vs misses': {
                'condition_name': 'visual_hit_miss',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Auditory hits vs misses': {
                'condition_name': 'auditory_hit_miss',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Visual hits vs FA': {
                'condition_name': 'visual_hit_FA',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.15)
            },

            'Auditory hits vs FA': {
                'condition_name': 'auditory_hit_FA',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.15)
            },

            'Visual presence': {
                'condition_name': 'visual_miss_cr',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 4,
                'ylim': (0, 0.1)
            },

            'Auditory presence': {
                'condition_name': 'auditory_miss_cr',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 5,
                'ylim': (0, 0.1)
            },

            'Visual vs aud misses': {
                'condition_name': 'visual_miss_auditory_miss',
                'data': data,
                'area_keys': ['rALM__for_bilat', 'rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 6,
                'ylim': (0, 0.1)
            }
        }
    },

    'rALM_unilat': {
        'save_name': 'rALM_unilat_paper',
        'layout': (4, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Visual hits vs misses': {
                'condition_name': 'visual_hit_miss',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Auditory hits vs misses': {
                'condition_name': 'auditory_hit_miss',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Visual hits vs FA': {
                'condition_name': 'visual_hit_FA',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.15)
            },

            'Auditory hits vs FA': {
                'condition_name': 'auditory_hit_FA',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.15)
            },

            'Visual presence': {
                'condition_name': 'visual_miss_cr',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 4,
                'ylim': (0, 0.1)
            },

            'Auditory presence': {
                'condition_name': 'auditory_miss_cr',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 5,
                'ylim': (0, 0.1)
            },

            'Visual vs aud misses': {
                'condition_name': 'visual_miss_auditory_miss',
                'data': data,
                'area_keys': ['rALM__for_unilat', 'rALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 6,
                'ylim': (0, 0.1)
            }
        }
    },

    'lALM_rALM_bilat': {
        'save_name': 'lALM_rALM_bilat_paper',
        'layout': (4, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Visual hits vs misses': {
                'condition_name': 'visual_hit_miss',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Auditory hits vs misses': {
                'condition_name': 'auditory_hit_miss',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Visual hits vs FA': {
                'condition_name': 'visual_hit_FA',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.15)
            },

            'Auditory hits vs FA': {
                'condition_name': 'auditory_hit_FA',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.15)
            },

            'Visual presence': {
                'condition_name': 'visual_miss_cr',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 4,
                'ylim': (0, 0.1)
            },

            'Auditory presence': {
                'condition_name': 'auditory_miss_cr',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("aud_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 5,
                'ylim': (0, 0.1)
            },

            'Visual vs aud misses': {
                'condition_name': 'visual_miss_auditory_miss',
                'data': data_combined,
                'area_keys': ['lALM_rALM_for_bilat', 'lALM_rALM_bilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 6,
                'ylim': (0, 0.1)
            }
        }
    },

    'hit_fa_supp': {
            'save_name': 'hit_fa_supp',
            'layout': (2, 2),
            'panels': {
                'Hits vs FA (lick-aligned)': {
                    'condition_name': ['visual_hit_FA', 'auditory_hit_FA'],
                    'data': data_combined_lick_aligned,
                    'area_keys': ['lALM_rALM'],
                    'colors': [paper_colors('vis_max'), paper_colors('aud_max')],
                    'opto': False,
                    'position': 0,
                    'ylim': (0, 0.2),
                    'xlim': (-1, 1)
                },


                'Visual Hits vs FA opto (lick-aligned)': {
                    'condition_name': ['visual_hit_FA'],
                    'data': data_lick_aligned,
                    'area_keys': ['lALM_unilat__matched', 'lALM__for_unilat'],
                    'colors': [paper_colors('opto_purple'), paper_colors('vis_max')],
                    'opto': True,
                    'position': 2,
                    'ylim': (0, 0.2),
                    'xlim': (-1, 1)
                },

                'Auditory Hits vs FA opto (lick-aligned)': {
                    'condition_name': ['auditory_hit_FA'],
                    'data': data_lick_aligned,
                    'area_keys': ['lALM_unilat__matched', 'lALM__for_unilat'],
                    'colors': [paper_colors('opto_purple'), paper_colors('aud_max')],
                    'opto': True,
                    'position': 2,
                    'ylim': (0, 0.2)
                },

            }
        },

    'ALM_split': {
        'save_name': 'ALM_split_supp',
        'layout': (2, 2),
        'panels': {
            'Stimulus presence lALM': {
                'condition_name': ['visual_miss_cr', 'auditory_miss_cr'],
                'data': data,
                'area_keys': ['lALM'],
                'colors': [paper_colors("vis_max"), paper_colors("aud_max")],
                'opto': False,
                'position': 0,
                'ylim': (0, 0.1)
            },

            'Stimulus presence rALM': {
                'condition_name': ['visual_miss_cr', 'auditory_miss_cr'],
                'data': data,
                'area_keys': ['rALM'],
                'colors': [paper_colors("vis_max"), paper_colors("aud_max")],
                'opto': False,
                'position': 1,
                'ylim': (0, 0.1)
            },

            'Hit coding lALM': {
                'condition_name': ['visual_hit_miss', 'auditory_hit_miss'],
                'data': data,
                'area_keys': ['lALM'],
                'colors': [paper_colors("vis_max"), paper_colors("aud_max")],
                'opto': False,
                'position': 2,
                'ylim': (0, 0.6)
            },

            'Hit coding rALM': {
                'condition_name': ['visual_hit_miss', 'auditory_hit_miss'],
                'data': data,
                'area_keys': ['rALM'],
                'colors': [paper_colors("vis_max"), paper_colors("aud_max")],
                'opto': False,
                'position': 3,
                'ylim': (0, 0.6)
            }
        }
    },


    'task-irrelevant stimulus': {
        'save_name': 'task-irrelevant-properties',
        'layout': (1, 5),
        'panels': {
            'Visual side': {
                'condition_name': 'visual_side_all',
                'data': data_combined_task_irrelevant,
                'area_keys': ['lALM_rALM'],
                'colors': paper_colors("vis_max"),
                'opto': False,
                'position': 0,
                'ylim': (-0.01, 0.05)
            },

            'Visual orientation': {
                'condition_name': ['visual_ori_left_all', 'visual_ori_right_all'],
                'data': data_combined_task_irrelevant,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors("vis_ori"), 'darkorange'],
                'opto': False,
                'position': 1,
                'ylim': (-0.01, 0.05)
            },

            'Auditory pitch': {
               'condition_name': 'auditory_pitch_all',
               'data': data_combined_task_irrelevant,
               'area_keys': ['lALM_rALM'],
               'colors': paper_colors("aud_max"),
               'opto': False,
               'position': 2,
                'ylim': (-0.01, 0.05)
            }
        }
    },

# ── New: left/right visual coding, lALM+rALM combined ──────────────
    'visual_side_split': {
        'save_name': 'visual_side_split_lALM_rALM_paper',
        'layout': (1, 2),
        'panels': {
            'Visual hits vs misses (L vs R)': {
                'condition_name': ['left_visual_hit_miss', 'right_visual_hit_miss'],
                'data': data_combined,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors("vis_max"), 'darkorange'],
                'opto': False,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Visual presence (L vs R)': {
                'condition_name': ['left_visual_miss_cr', 'right_visual_miss_cr'],
                'data': data_combined,
                'area_keys': ['lALM_rALM'],
                'colors': [paper_colors("vis_max"), 'darkorange'],
                'opto': False,
                'position': 1,
                'ylim': (0, 0.1)
            },
        }
    },

    # ── New: left/right visual coding, lALM vs lALM unilateral opto ────
    'visual_side_split_lALM_unilat': {
        'save_name': 'visual_side_split_lALM_unilat_paper',
        'layout': (2, 2),
        'plot_jaccard_idx': False,
        'panels': {
            'Left visual hits vs misses': {
                'condition_name': 'left_visual_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.6)
            },

            'Right visual hits vs misses': {
                'condition_name': 'right_visual_hit_miss',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.6)
            },

            'Left visual presence': {
                'condition_name': 'left_visual_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.1)
            },

            'Right visual presence': {
                'condition_name': 'right_visual_miss_cr',
                'data': data,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("vis_max"), paper_colors("opto_purple")],
                'opto': True,
                'position': 3,
                'ylim': (0, 0.1)
            },
        }
    },

}

# ── Drop-in: lick-aligned lick-coding figures ─────────────────────────
# Conditions: left_lick_no_lick, right_lick_no_lick, lick_no_lick
# All use the lick-aligned data sources.
# Missing conditions are skipped automatically by the plotting loop.
figures_dict.update({

    # 1) lALM + rALM pooled into one population
    'lick_coding_combined': {
        'save_name': 'lick_coding_lALM_rALM_combined_paper',
        'layout': (1, 3),
        'panels': {
            'Left lick vs no lick': {
                'condition_name': 'left_lick_no_lick',
                'data': data_combined_lick_aligned,
                'area_keys': ['lALM_rALM'],
                'colors': paper_colors("left_lick"),
                'opto': False,
                'position': 0,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Right lick vs no lick': {
                'condition_name': 'right_lick_no_lick',
                'data': data_combined_lick_aligned,
                'area_keys': ['lALM_rALM'],
                'colors': paper_colors("right_lick"),
                'opto': False,
                'position': 1,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Lick vs no lick': {
                'condition_name': 'lick_no_lick',
                'data': data_combined_lick_aligned,
                'area_keys': ['lALM_rALM'],
                'colors': 'black',
                'opto': False,
                'position': 2,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
        }
    },

    # 2) lALM and rALM as separate traces (two lines + between-area Fisher test)
    'lick_coding_lALM_rALM_separate': {
        'save_name': 'lick_coding_lALM_rALM_separate_paper',
        'layout': (1, 3),
        'panels': {
            'Left lick vs no lick': {
                'condition_name': 'left_lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM', 'rALM'],          # colors[0]=lALM, colors[1]=rALM
                'colors': ['navy', 'crimson'],          # swap for your area colors if you have them
                'opto': False,
                'position': 0,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Right lick vs no lick': {
                'condition_name': 'right_lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM', 'rALM'],
                'colors': ['navy', 'crimson'],
                'opto': False,
                'position': 1,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Lick vs no lick': {
                'condition_name': 'lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM', 'rALM'],
                'colors': ['navy', 'crimson'],
                'opto': False,
                'position': 2,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
        }
    },

    # 3) lALM control vs lALM unilateral opto (matched neuron sets)
    'lick_coding_lALM_unilat': {
        'save_name': 'lick_coding_lALM_unilat_paper',
        'layout': (1, 3),
        'plot_jaccard_idx': False,
        'panels': {
            'Left lick vs no lick': {
                'condition_name': 'left_lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("left_lick"), paper_colors("opto_purple")],
                'opto': True,
                'position': 0,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Right lick vs no lick': {
                'condition_name': 'right_lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': [paper_colors("right_lick"), paper_colors("opto_purple")],
                'opto': True,
                'position': 1,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
            'Lick vs no lick': {
                'condition_name': 'lick_no_lick',
                'data': data_lick_aligned,
                'area_keys': ['lALM__for_unilat', 'lALM_unilat__matched'],
                'colors': ['black', paper_colors("opto_purple")],
                'opto': True,
                'position': 2,
                'ylim': (0, 0.3),
                'xlim': (-1, 1)
            },
        }
    },

})

# Plotting loop
for figure_name, figure_config in figures_dict.items():
    print(f"Creating figure: {figure_name}")

    # ── Pre-check: can we plot ANY panel in this figure? ───────────
    any_panel_plottable = False
    for panel_name, panel_details in figure_config['panels'].items():
        condition_names = panel_details['condition_name']
        if isinstance(condition_names, str):
            condition_names = [condition_names]
        data_set = panel_details['data']
        area_keys = panel_details['area_keys']

        for cn in condition_names:
            if any(cn in data_set.get(ak, {}) for ak in area_keys):
                any_panel_plottable = True
                break
        if any_panel_plottable:
            break

    if not any_panel_plottable:
        print(f"  No data available for any panel in '{figure_name}'. Skipping entire figure.")
        continue

    # Initialize plot with specified layout
    nrows, ncols = figure_config['layout']

    # ── Compute figure size so each subplot is exactly SUBPLOT_W × SUBPLOT_H ───
    fig_w = LEFT_MARGIN + ncols * SUBPLOT_W + (ncols - 1) * H_PAD + RIGHT_MARGIN
    fig_h = TOP_MARGIN + nrows * SUBPLOT_H + (nrows - 1) * V_PAD + BOT_MARGIN

    fig, axes_plot = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    if isinstance(axes_plot, np.ndarray):
        axes_plot = axes_plot.flatten()
    else:
        axes_plot = [axes_plot]

    # ── Force each Axes to be exactly SUBPLOT_W × SUBPLOT_H ────────
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx < len(axes_plot):
                left = (LEFT_MARGIN + col * (SUBPLOT_W + H_PAD)) / fig_w
                bottom = (BOT_MARGIN + (nrows - 1 - row) * (SUBPLOT_H + V_PAD)) / fig_h
                width = SUBPLOT_W / fig_w
                height = SUBPLOT_H / fig_h
                axes_plot[idx].set_position([left, bottom, width, height])

    # Whether to plot jaccard idx
    plot_jaccard_idx = figure_config.get('plot_jaccard_idx', False)

    # Track which axes actually received plot data (to hide empty ones)
    axes_used = set()
    last_im = None  # for colorbar if jaccard is plotted

    # ─── Stats collector for this figure ────────────────────────────
    figure_stats = {
        'figure_name': figure_name,
        'save_name': figure_config['save_name'],
        'sig_level_auc_vs_shuffled': float(sig_level),
        'p_alpha_fisher_fdr': float(p_alpha),
        'fdr_method': 'benjamini-hochberg',
        'panels': {},
    }

    # Loop over all panels in this figure
    for panel_name, panel_details in figure_config['panels'].items():
        print(f"  Plotting panel: {panel_name}")

        # Convert condition_name to list if it's a string
        condition_names = panel_details['condition_name']
        if isinstance(condition_names, str):
            condition_names = [condition_names]

        data_set = panel_details['data']
        area_keys = panel_details['area_keys']
        colors = panel_details['colors']

        if not isinstance(colors, list):
            colors = [colors]

        opto = panel_details['opto']
        ax_idx = panel_details['position']
        plot_chance_level = panel_details.get('plot_chance_level', True)

        # Get the appropriate axis
        ax = axes_plot[ax_idx]

        # ── Resolve time_axis for this panel ───────────────────────
        time_axis = None
        for cn in condition_names:
            time_axis = _resolve_time_axis(data_set, area_keys, cn)
            if time_axis is not None:
                break

        # Also try to get it from the underlying per-area data dicts if this is a combined dataset
        if time_axis is None:
            for source_dict in [data, data_lick_aligned]:
                for ak in source_dict:
                    for key, val in source_dict[ak].items():
                        if isinstance(val, dict) and 'params' in val:
                            params = val['params']
                            if 't_bins' in params:
                                time_axis = params['t_bins']
                            else:
                                start = params['window'][0]
                                time_axis = start + np.arange(params['n_bins']) * params['step_size']
                            break
                    if time_axis is not None:
                        break
                if time_axis is not None:
                    break

        if time_axis is None:
            print(f"    Could not determine time_axis for panel '{panel_name}'. Skipping.")
            continue

        bars = []
        sig_colors = []
        color_idx = 0
        n_neurons_check = []
        panel_has_data = False

        # Panel-level stats container
        panel_stats = {
            'position': int(ax_idx),
            'area_keys': list(area_keys),
            'conditions': {},
            'time_axis_bounds': {
                'min': float(np.min(time_axis)),
                'max': float(np.max(time_axis)),
                'n_bins': int(len(time_axis)),
            },
        }

        # Loop over each condition to plot and possibly calculate stats
        for condition_name in condition_names:
            # Check if condition exists in at least one area_key
            available_area_keys = [ak for ak in area_keys if condition_name in data_set.get(ak, {})]

            if not available_area_keys:
                print(f"    {condition_name} not found in any area_key ({area_keys}), skipping.")
                # Advance color_idx to keep colors aligned
                color_idx += len(area_keys)
                continue

            # Container for this condition's stats
            cond_stats = {
                'vs_baseline': {},
                'between_areas': None,
            }

            # Grab n_neurons (only from available area_keys)
            for area_key in area_keys:
                if condition_name in data_set.get(area_key, {}):
                    n_neurons_check.append(data_set[area_key][condition_name]['n_neurons'])

            # Collect per-area vs-baseline stats
            for area_key in area_keys:
                if condition_name in data_set.get(area_key, {}):
                    entry = data_set[area_key][condition_name]
                    cond_stats['vs_baseline'][area_key] = _summarize_bsl_test(
                        entry, time_axis, p_alpha
                    )

            # Calculate between-area stats if comparing two areas
            has_both = False
            if len(area_keys) == 2:
                area_key_1 = area_keys[0]
                area_key_2 = area_keys[1]

                has_both = (condition_name in data_set.get(area_key_1, {}) and
                            condition_name in data_set.get(area_key_2, {}))

                if has_both:
                    dataset_1 = data_set[area_key_1][condition_name]
                    dataset_2 = data_set[area_key_2][condition_name]

                    p_values_between = []
                    for idx in range(len(dataset_1['n_coding_neurons'])):
                        n_coding_1 = dataset_1['n_coding_neurons'][idx]
                        n_coding_2 = dataset_2['n_coding_neurons'][idx]
                        n_noncoding_1 = dataset_1['n_neurons'] - n_coding_1
                        n_noncoding_2 = dataset_2['n_neurons'] - n_coding_2

                        contingency_table_between = np.array([
                            [n_coding_1, n_noncoding_1],
                            [n_coding_2, n_noncoding_2]
                        ])
                        odds_ratio, p_value = fisher_exact(contingency_table_between)
                        p_values_between.append(p_value)

                    p_values_between = np.array(p_values_between)
                    sig_difference_between, p_adjusted_between, _, _ = multipletests(
                        p_values_between, alpha=p_alpha, method='fdr_bh'
                    )

                    # Record between-area stats
                    between_summary = _summarize_between_test(
                        dataset_1['n_coding_neurons'],
                        dataset_1['n_neurons'],
                        dataset_2['n_coding_neurons'],
                        dataset_2['n_neurons'],
                        time_axis,
                        p_alpha,
                    )
                    between_summary['comparison'] = f"{area_key_1}_vs_{area_key_2}"
                    cond_stats['between_areas'] = between_summary

                    # Calculate time-resolved jaccard index
                    if plot_jaccard_idx:
                        jac1 = ROC_utils.build_jaccard_array(dataset_1['neurons'], dataset_1['p_values'], sig_level)
                        jac2 = ROC_utils.build_jaccard_array(dataset_2['neurons'], dataset_2['p_values'], sig_level)
                        jac_array = ROC_utils.calc_time_resolved_jaccard(jac2, jac1)
                else:
                    sig_difference_between = None
                    jac_array = None if plot_jaccard_idx else None

            # Plot data for each area (only those that have data)
            area_color = {}   # remember the color used per area, for this condition
            for area_idx, area_key in enumerate(area_keys):
                if condition_name not in data_set.get(area_key, {}):
                    color_idx += 1
                    continue

                trace_color = colors[color_idx % len(colors)]
                area_color[area_key] = trace_color
                label = f"{area_key}_{condition_name}"
                ax.plot(
                        time_axis,
                        data_set[area_key][condition_name]['fract_sig_neurons'],
                        color=trace_color,
                        linewidth=1,
                        alpha=0.9,
                        label=label
                )
                color_idx += 1
                panel_has_data = True

            # Add significance bars for this condition
            for area_key in area_keys:
                if condition_name in data_set.get(area_key, {}):
                    bars.append(data_set[area_key][condition_name][sig_type])
                    sig_colors.append(area_color.get(area_key, 'black'))

            if len(area_keys) == 2 and has_both and sig_difference_between is not None:
                bars.append(sig_difference_between)
                sig_colors.append('black')  # between-area bar

            panel_stats['conditions'][condition_name] = cond_stats

        if not panel_has_data:
            print(f"    No data plotted for panel '{panel_name}'. Leaving axis empty.")
            ax.set_visible(False)
            continue

        axes_used.add(ax_idx)

        # Chance-level CI (record alongside stats)
        if plot_chance_level and len(n_neurons_check) > 0:
            n_neurons_check_arr = np.array(n_neurons_check)
            if not np.all(n_neurons_check_arr == n_neurons_check_arr[0]):
                print(f"WARNING: NOT ALL CONDITIONS: {panel_details['area_keys']} - HAVE THE SAME NUMBER OF NEURONS - CI OF CHANCE LEVEL NOT ACCURATE")

            n_neurons = n_neurons_check_arr[0]
            ci_lower = binom.ppf(0.005, n_neurons, 0.01) / n_neurons
            ci_upper = binom.ppf(0.995, n_neurons, 0.01) / n_neurons
            print(f"n_neurons: {n_neurons} - ci_lower: {ci_lower}, ci_upper: {ci_upper}")
            panel_stats['chance_level'] = {
                'expected_fraction': 0.01,
                'ci_lower_99pct': float(ci_lower),
                'ci_upper_99pct': float(ci_upper),
                'n_neurons_used': int(n_neurons),
                'all_conditions_same_n': bool(np.all(n_neurons_check_arr == n_neurons_check_arr[0])),
            }



        # Add the now-complete panel stats to the figure-level collector
        figure_stats['panels'][panel_name] = panel_stats

        ax = ROC_utils.add_sig_lines_ax(
            ax, bars,
            colors=sig_colors,
            linewidth=3,
            alpha=0.5,
            timeline=time_axis
        )
        if plot_chance_level and len(n_neurons_check) > 0:
            n_neurons_check_arr = np.array(n_neurons_check)
            n_neurons = n_neurons_check_arr[0]
            ci_lower = binom.ppf(0.005, n_neurons, 0.01) / n_neurons
            ci_upper = binom.ppf(0.995, n_neurons, 0.01) / n_neurons
            ax.axhline(y=0.01, color='grey', linewidth=1, linestyle='--')
            ax.axhspan(ci_lower, ci_upper, alpha=0.4, color='grey', linewidth=0)

        # Format plot appearance
        ax.patch.set_alpha(0)
        ax.grid(False)
        ax.legend(fontsize=6)

        # ── Build title with n_neurons / n_sessions / n_animals ──
        # Collect all unique cluster_IDs that contributed to this panel.
        # Taking the union avoids double-counting when the same neurons
        # appear under different conditions or matched opto variants.
        all_neurons_panel = []
        for cn in condition_names:
            for ak in area_keys:
                if cn in data_set.get(ak, {}):
                    nids = data_set[ak][cn].get('neurons')
                    if nids is not None:
                        all_neurons_panel.append(np.asarray(nids))
        if all_neurons_panel:
            all_neurons_panel = np.unique(np.concatenate(all_neurons_panel))
            n_n, n_s, n_a = count_neurons_sessions_animals(all_neurons_panel)
            title_str = f"{panel_name}\nn={n_n} neurons, {n_s} sessions, {n_a} animals"
            # Store in panel stats too
            panel_stats['neuron_session_animal_counts'] = {
                'n_neurons_unique': int(n_n),
                'n_sessions': int(n_s),
                'n_animals': int(n_a),
            }
        else:
            title_str = panel_name
        ax.set_title(title_str, fontsize=8)

        if 'ylim' in panel_details:
            ax.set_ylim(panel_details['ylim'])

        # Format axes
        ax = ROC_utils.format_ROC_plot_ax(ax, round_up=False)

        ax.set_ylabel("")
        ax.set_xlabel("")

        # Add optogenetic shading if applicable
        if opto:
            ROC_utils.add_opto_shading(ax)

        if 'xlim' in panel_details:
            ax.set_xlim(panel_details['xlim'])
        elif X_LIM_DEFAULT is not None:
            ax.set_xlim(X_LIM_DEFAULT)

        if plot_jaccard_idx and len(area_keys) == 2:
            jac_to_plot = jac_array if has_both else None
            if jac_to_plot is not None:
                y_min, y_max = ax.get_ylim()
                extent = [time_axis.min(), time_axis.max(), y_min, y_min + y_max/10]
                last_im = ax.imshow(jac_to_plot.reshape(1, -1), cmap='viridis',
                           aspect='auto', extent=extent, vmin=0, vmax=0.5)

    # Hide unused axes
    for i, ax in enumerate(axes_plot):
        if i not in axes_used:
            ax.set_visible(False)

    if plot_jaccard_idx and last_im is not None:
        plt.colorbar(last_im, ax=axes_plot[max(axes_used)], orientation='vertical', label='Jaccard Index')

    # Remove backgrounds — do NOT call tight_layout (positions are set manually)
    fig.patch.set_alpha(0)

    # Save the figure
    save_name = figure_config['save_name']
    output_file = os.path.join(save_dir, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(os.path.join(output_file + '.png'), dpi=1000)
    fig.savefig(os.path.join(output_file + '.svg'), dpi=72)

    # ── Save significance test results to YAML ────────────────────
    yaml_path = output_file + '_stats.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(
            _py(figure_stats),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    print(f"  Saved significance stats to: {yaml_path}")

    plt.close(fig)


def format_ticks(value, _):
    return f"{int(value)}" if value == 0 else f"{value:.2f}".rstrip("0").rstrip(".")