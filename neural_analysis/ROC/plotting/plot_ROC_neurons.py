"""
ROC-based neural coding analysis.

Identifies neurons that significantly decode between trial conditions (e.g. hit vs miss)
using ROC analysis. Computes overlap between modalities, generates PSTHs for significant
neurons, and summarises coding neuron counts/fractions per area and contrast.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ROC_utils
from neural_analysis import utils
from neural_analysis.plot_utils import paper_colors
from matplotlib_venn import venn2
from matplotlib.collections import PolyCollection
import config

# =============================================================================
# Helper functions
# =============================================================================

def clip_sig_dict(sig_dict, t_bins, window, roc_window=None, atol=1e-9):
    """Trim each mask to bins inside `window`, but only if `window` differs
    from `roc_window`. When they match, return the dict unchanged."""
    if roc_window is not None and \
       np.isclose(window[0], roc_window[0], atol=atol) and \
       np.isclose(window[1], roc_window[1], atol=atol):
        return sig_dict
    t_bins = np.asarray(t_bins)
    in_win = (t_bins >= window[0] - atol) & (t_bins <= window[1] + atol)
    return {nid: np.asarray(mask)[in_win] for nid, mask in sig_dict.items()}


def select_sig_bins(mask, t_bins, window, atol=1e-9):
    # match requested window bins with existing ROC data (computed from -1 to 1.5)
    t_bins = np.asarray(t_bins)
    in_win = (t_bins >= window[0] - atol) & (t_bins <= window[1] + atol)
    return mask[in_win], t_bins[in_win]


def filter_neurons(cond, cluster_IDs):
    """Filter a condition dict to only include neurons whose ID is in cluster_IDs.

    Parameters
    ----------
    cond : dict
        Condition dict with keys 'neurons', 'auc_scores', 'p_values', 'n_trials'.
    cluster_IDs : array-like
        Cluster IDs to keep.

    Returns
    -------
    cond : dict
        The same dict with neuron-indexed fields filtered in place.
    """
    if cond['neurons'] is None:
        return cond

    cluster_set = set(str(c) for c in cluster_IDs)
    mask = np.array([str(n) in cluster_set for n in cond['neurons']])

    cond['neurons'] = cond['neurons'][mask]
    cond['auc_scores'] = cond['auc_scores'][mask] if cond['auc_scores'] is not None else None
    cond['p_values'] = cond['p_values'][mask] if cond['p_values'] is not None else None
    cond['n_trials'] = cond['n_trials'][mask] if cond['n_trials'] is not None else None

    return cond


def find_neurons_with_low_p_values(p_values, neurons, auc_scores,
                                   min_consecutive_values=10, p_value_threshold=0.01):
    """Return neurons whose p-value timecourse drops below threshold for enough consecutive bins."""
    if p_values is None:
        return {}

    matching_neurons = {}
    for i, neuron_p_values in enumerate(p_values):
        count = 0
        for p in neuron_p_values:
            if p < p_value_threshold:
                count += 1
                if count >= min_consecutive_values:
                    matching_neurons[str(neurons[i])] = neuron_p_values < p_value_threshold
                    break
            else:
                count = 0
    return matching_neurons


def append_if_unique(df_list, new_df):
    """Append dataframe only if it doesn't already exist in the list."""
    for existing_df in df_list:
        if existing_df.equals(new_df):
            return
    df_list.append(new_df)


def contrast_has_opto(contrast_details):
    """Check if any condition in a contrast references hasphotostim=1 in its filters."""
    for condition in contrast_details.values():
        for filt in condition.get('filters_trialData', []):
            if filt.get('hasphotostim') == 1:
                return True
    return False


def contrast_opto_type(contrast_details):
    """Check if any condition in a contrast references hasphotostim=1 in its filters."""
    for condition in contrast_details.values():
        for filt in condition.get('filters_trialData', []):
            if filt.get('optoLat') == 'unilat':
                return 'unilat'
            elif filt.get('optoLat') == 'bilat':
                return 'bilat'
    return 'opto_off'


def resolve_alignment(condition_details):
    """Return the align_to string ('stimChange' or 'firstLick') for a condition."""
    alignment = condition_details.get('alignment', 'stim_aligned')
    mapping = {'lick_aligned': 'firstLick', 'stim_aligned': 'stimChange'}
    if alignment not in mapping:
        print(f"  WARNING: '{alignment}' is unexpected, defaulting to stimChange")
        return 'stimChange'
    return mapping[alignment]


def filter_opto_trials(trialData_1, trialData_2, opto_flag):
    """Filter trial dataframes by opto status."""
    if opto_flag == 'opto_off':
        return (trialData_1[trialData_1['hasphotostim'] == 0],
                trialData_2[trialData_2['hasphotostim'] == 0])
    else:
        return (trialData_1[trialData_1['hasphotostim'] == 1],
                trialData_2[trialData_2['hasphotostim'] == 1])


def subset_spike_data(spikeData_in, condition_details, debug=False, plot_subset=True):
    """Optionally limit spikeData to selected neurons and/or debug subset."""
    result = spikeData_in
    if debug:
        result = result.head(2)
    if plot_subset:
        neuron_subset = condition_details.get('selected_neurons', None)
        if neuron_subset is not None:
            result = result[result['cluster_ID'].isin(neuron_subset)]
        else:
            result = None
    return result


def safe_frac(numerator, denominator):
    """Compute fraction, returning 0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0


def build_summary_row(contrast, condition, area, is_opto, n_total_area, n_roc, n_sig,
                      neuron_ids=None, neuron_ids_unique=None, neuron_ids_overlap=None,
                      n_unique=None, n_overlap=None):
    """Create a single summary dict.

    Parameters
    ----------
    neuron_ids : set or None
        All significantly coding neuron cluster_IDs for this row.
    neuron_ids_unique : dict or None
        Dict mapping condition name -> set of cluster_IDs uniquely coding that
        condition. Only populated for two-condition contrasts.
        E.g. {'visual': {'n1', 'n2'}, 'auditory': {'n3'}}.
    neuron_ids_overlap : set or None
        Set of cluster_IDs that are significant in both conditions.
        Only populated for two-condition contrasts.
    n_unique, n_overlap : int or None
        Counts for two-condition contrasts.
    """
    return {
        'contrast': contrast,
        'condition': condition,
        'area': area,
        'is_opto': is_opto,
        'n_total_area': n_total_area,
        'n_total_roc': n_roc,
        'n_coding': n_sig,
        'frac_of_area': safe_frac(n_sig, n_total_area),
        'frac_of_roc': safe_frac(n_sig, n_roc),
        'neuron_ids': neuron_ids if neuron_ids is not None else set(),
        'neuron_ids_unique': neuron_ids_unique,   # dict{cond_name: set} or None
        'neuron_ids_overlap': neuron_ids_overlap,  # set or None
        'n_unique': n_unique,
        'n_overlap': n_overlap,
        'frac_unique': safe_frac(n_unique, n_roc) if n_unique is not None else None,
        'frac_overlap': safe_frac(n_overlap, n_roc) if n_overlap is not None else None,
        'frac_unique_of_coding': safe_frac(n_unique, n_sig) if n_unique is not None else None,
        'frac_overlap_of_coding': safe_frac(n_overlap, n_sig) if n_overlap is not None else None,
    }


# =============================================================================
# Overlap functions
# =============================================================================

def overlap_coefficient(a, b):
    """Szymkiewicz-Simpson overlap coefficient.

    Returns the fraction of the smaller set that is contained in the larger.
    1.0 = the smaller set is a perfect subset of the larger.
    0.0 = no overlap.

    Parameters
    ----------
    a, b : set or array-like
        Two collections of neuron IDs.

    Returns
    -------
    float
    """
    a, b = set(a), set(b)
    min_size = min(len(a), len(b))
    if min_size == 0:
        return 0.0
    return len(a & b) / min_size


def compute_pairwise_overlap(summary_df, row_filter=None, group_by='area'):
    """Compute pairwise overlap coefficients between coding neuron populations.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The summary dataframe. Must contain a 'neuron_ids' column with sets.
    row_filter : dict or None
        Key-value pairs to filter summary_df before computing overlaps.
        E.g. {'area': 'lALM', 'is_opto': False}.
        Keys can map to a single value or a list.
    group_by : str or list of str or None
        Column(s) to group by before computing pairwise overlaps.
        Overlaps are only computed between rows within the same group.
        If None, all (filtered) rows are compared to each other.

    Returns
    -------
    overlap_df : pd.DataFrame
        One row per pair.
    """
    df = summary_df.copy()

    # Apply filters
    if row_filter is not None:
        for col, val in row_filter.items():
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    if df.empty:
        print("No rows match the filter criteria.")
        return pd.DataFrame()

    # Determine grouping
    if group_by is None:
        groups = [('all', df)]
    else:
        if isinstance(group_by, str):
            group_by = [group_by]
        groups = df.groupby(group_by)

    results = []
    for group_key, group_df in groups:
        if isinstance(group_key, str):
            group_key = (group_key,)

        indices = group_df.index.tolist()

        for idx_a, idx_b in combinations(indices, 2):
            row_a = group_df.loc[idx_a]
            row_b = group_df.loc[idx_b]

            set_a = set(row_a['neuron_ids'])
            set_b = set(row_b['neuron_ids'])

            intersection = set_a & set_b
            union = set_a | set_b

            result = {}
            if group_by is not None:
                for col, val in zip(group_by, group_key):
                    result[col] = val

            result.update({
                'contrast_1': row_a['contrast'],
                'condition_1': row_a['condition'],
                'contrast_2': row_b['contrast'],
                'condition_2': row_b['condition'],
                'n_1': len(set_a),
                'n_2': len(set_b),
                'n_intersection': len(intersection),
                'overlap_coef': overlap_coefficient(set_a, set_b),
                'jaccard': len(intersection) / len(union) if len(union) > 0 else 0.0,
            })
            results.append(result)

    return pd.DataFrame(results)


def plot_overlap_heatmap(overlap_df, metric='overlap_coef', title=None, save_path=None,
                         figsize=(10, 8), cmap='viridis', vmin=0, vmax=1, fmt='.2f'):
    """Plot a symmetric heatmap of pairwise neuron population overlap.

    Parameters
    ----------
    overlap_df : pd.DataFrame
        Output of compute_pairwise_overlap().
    metric : str
        Column to plot. 'overlap_coef' or 'jaccard'.
    title : str or None
        Plot title. Auto-generated if None.
    save_path : str or None
        If provided, saves figure to this path.
    """
    if overlap_df.empty:
        print("Empty overlap_df, nothing to plot.")
        return

    def _label(row, suffix):
        contrast = row[f'contrast_{suffix}']
        condition = row[f'condition_{suffix}']
        n = row[f'n_{suffix}']
        if condition in ('_combined', 'all'):
            return f"{contrast}\n(n={n})"
        return f"{contrast} / {condition}\n(n={n})"

    seen = {}
    for _, row in overlap_df.iterrows():
        for suffix in ('1', '2'):
            key = (row[f'contrast_{suffix}'], row[f'condition_{suffix}'])
            if key not in seen:
                seen[key] = _label(row, suffix)
    keys = list(seen.keys())
    labels = [seen[k] for k in keys]

    n = len(keys)
    matrix = np.eye(n)

    key_to_idx = {k: i for i, k in enumerate(keys)}
    for _, row in overlap_df.iterrows():
        i = key_to_idx[(row['contrast_1'], row['condition_1'])]
        j = key_to_idx[(row['contrast_2'], row['condition_2'])]
        matrix[i, j] = row[metric]
        matrix[j, i] = row[metric]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                square=True,
                ax=ax)

    metric_name = 'Overlap Coefficient' if metric == 'overlap_coef' else 'Jaccard Index'
    ax.set_title(title or f'{metric_name} — Neuronal Population Overlap')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# =============================================================================
# PSTH helpers
# =============================================================================

def make_psth_kwargs(cfg):
    """Return the shared keyword arguments for generate_psth_master_optimized."""
    return dict(
        bin_size=cfg['bin_size'],
        smoothing_width=cfg['smoothing_width'],
        use_causal_bins=cfg['use_causal_window'],
        svg=cfg['gen_svg'],
        step_size=cfg['step_size'],
        window=cfg['window'],
        plot_licks='first_only_correct',
        sort_order='firstLick',
        calc_single_trial=cfg['calc_single_trial'],
        height_ratios=cfg['height_ratios'],
        figsize=cfg['figsize'],
        z_score_psth=cfg['z_score_psth'],
        raster_dot_width=cfg['raster_dot_width']
    )


def plot_psth(spikeData_plot, trialData_list, output_dir, labels, colors,
              highlight_dict, highlight_colors, align_to, psth_kwargs,
              highlight_plot_style='bar', y_lim=None, line_styles=None):
    """Wrapper around generate_psth_master_optimized with directory creation."""
    os.makedirs(output_dir, exist_ok=True)
    utils.generate_psth_master_optimized(
        spikeData_plot, trialData_list,
        output_dir=output_dir,
        labels=labels,
        colors=colors,
        highlight_dict_conditions=highlight_dict,
        highlight_colors=highlight_colors,
        align_to=align_to,
        highlight_plot_style=highlight_plot_style,
        y_lim=y_lim,
        linetypes=line_styles,
        **psth_kwargs,
    )


def make_condition(condition_dir, filters_trialData, labels, colors,
                   selected_neurons=None, alignment=None, y_lim=None):
    """Build a single condition dict with sensible defaults for ROC result fields."""
    cond = {
        'condition_dir': condition_dir,
        'auc_scores': None,
        'neurons': None,
        'p_values': None,
        'params': None,
        'filters': None,
        'n_trials': None,
        'neurons_sig_timepoints': None,
        'filters_trialData': filters_trialData,
        'labels': labels,
        'colors': colors,
        'y_lim': y_lim
    }
    if selected_neurons is not None:
        cond['selected_neurons'] = selected_neurons
    if alignment is not None:
        cond['alignment'] = alignment
    return cond


# =============================================================================
# Contrast definitions
# =============================================================================
# Conditions in details section are used to retrieve ROC data and calculate unique and overlap coding
# To add plots, use the 'additional_plots' section

contrast_dict = {
    'hit_miss': {
        'details': {
            'visual': make_condition(
                'visual_hit_miss',
                [{'trialType': 'X', 'correctResponse': 1},
                 {'trialType': 'X', 'noResponse': 1}],
                ['Visual hit', 'Visual miss'],
                [paper_colors('vis_hit'), paper_colors('vis_hit')],  #paper_colors('vis_miss')],
                selected_neurons=['022202402251602111', '022202402281126167', '2066202207091445130', '022202402251602111'],
                y_lim=(-5, 25)
            ),
            'auditory': make_condition(
                'auditory_hit_miss',
                [{'trialType': 'Y', 'correctResponse': 1},
                 {'trialType': 'Y', 'noResponse': 1}],
                ['Auditory hit', 'Auditory miss'],
                [paper_colors('aud_hit'), paper_colors('aud_hit')],  #paper_colors('aud_miss')],
                selected_neurons=['2066202207131138215', '022202402281126167'],
                y_lim=(-15, 75)
            ),
        },
        'plot_subset': True,
        'neurons_overlap': ['015202304221515117', '02120240201152521',
                            '02120240130150836', '02620240313155316',
                            '0262024030614122150', '2063202206281123154'],
        'plot_additional': True,
        'line_styles': ['-', (0, (1,3))],
        'additional_plots': {
            #'opto': {
            #    'neurons': 'overlap',
            #    'filters_trialData': [
            #        {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 1},
            #        {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 1},
            #        {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
            #    ],
            #    'labels': ['Auditory hit (opto)', 'Visual hit (opto)',
            #                'Auditory hit', 'Visual hit'],
            #    'colors': [paper_colors('aud_thr'), paper_colors('vis_thr'),
            #               paper_colors('aud_max'), paper_colors('vis_max')],
            #},

            #'auditory_hit_neurons_all_trials': {
            #    'neurons': 'auditory',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #    ],
            #    'labels': ['Visual hit', 'Visual miss', 'Auditory hit', 'Auditory miss'],
            #    'colors': [paper_colors('vis_hit'), paper_colors('vis_miss'),
            #               paper_colors('aud_hit'), paper_colors('aud_miss')],
            #    'plot_subset': False
            #},
#
            #'visual_hit_neurons_all_trials': {
            #    'neurons': 'visual',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #    ],
            #    'labels': ['Visual hit', 'Visual miss', 'Auditory hit', 'Auditory miss'],
            #    'colors': [paper_colors('vis_hit'), paper_colors('vis_miss'),
            #               paper_colors('aud_hit'), paper_colors('aud_miss')],
            #    'plot_subset': False
            #},
#
            #'overlap_hit_neurons_all_trials': {
            #    'neurons': 'overlap',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #    ],
            #    'labels': ['Visual hit', 'Visual miss', 'Auditory hit', 'Auditory miss'],
            #    'colors': [paper_colors('vis_hit'), paper_colors('vis_miss'),
            #               paper_colors('aud_hit'), paper_colors('aud_miss')],
            #    'plot_subset': False
            #},

            'auditory_hit_neurons_visual_trials': {
                'neurons': 'auditory',
                'filters_trialData': [
                    {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
                ],
                'labels': ['Visual hit', 'Visual miss'],
                'colors': [paper_colors('vis_hit'), paper_colors('vis_hit')],
                'plot_subset': True,
                'y_lim': (-15, 75),
                'sig_bars': 'visual'
            },
#
            'visual_hit_neurons_auditory_trials': {
                'neurons': 'visual',
                'filters_trialData': [
                    {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
                ],
                'labels': ['Auditory hit', 'Auditory miss'],
                'colors': [ paper_colors('aud_hit'), paper_colors('aud_hit')],
                'plot_subset': True,
                'y_lim': (-5, 25),
                'sig_bars': 'auditory'
            },
#
            'overlap_hit_neurons_visual_trials': {
                'neurons': ['0262024030614122150'],
                'filters_trialData': [
                    {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Visual hit', 'Visual miss'],
                'colors': [paper_colors('vis_hit'), paper_colors('vis_hit')],
                'plot_subset': True,
                'y_lim': (-40, 200),
                'sig_bars': 'visual'
            },
#
            'overlap_hit_neurons_auditory_trials': {
                'neurons': ['0262024030614122150'],
                'filters_trialData': [
                    {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Auditory hit', 'Auditory miss'],
                'colors': [ paper_colors('aud_hit'), paper_colors('aud_hit')],
                'plot_subset': True,
                'y_lim': (-40, 200),
                'sig_bars': 'auditory'
            },
        },
    },

    'miss_cr': {
        'details': {
            'visual': make_condition(
                'visual_miss_cr',
                [{'trialType': 'X', 'noResponse': 1},
                 {'trialType': 'P', 'noResponse': 1}],
                ['Visual miss', 'Correct rejection'],
                [paper_colors('vis_hit'), 'black'],
                selected_neurons=['007202304061549118'],
                y_lim=(-13, 65)
            ),
            'auditory': make_condition(
                'auditory_miss_cr',
                [{'trialType': 'Y', 'noResponse': 1},
                 {'trialType': 'P', 'noResponse': 1}],
                ['Auditory miss', 'Correct rejection'],
                [paper_colors('aud_hit'), 'black'],
                selected_neurons=['021202401291344332'],
                y_lim=(-13, 65)
            ),
        },
        'plot_subset': True,
        'neurons_overlap': ['2063202206281123154'],
        'plot_additional': True,
        'additional_plots': {

            #'auditory_stim_neurons_all_trials': {
            #    'neurons': 'auditory',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
            #    ],
            #    'labels': ['Visual miss', 'Auditory miss', 'Correct rejection'],
            #    'colors': [paper_colors('vis_miss'), paper_colors('aud_miss'), paper_colors('probe')],
            #    'plot_subset': False
            #},
#
            #'visual_stim_neurons_all_trials': {
            #    'neurons': 'visual',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
            #    ],
            #    'labels': ['Visual miss', 'Auditory miss', 'Correct rejection'],
            #    'colors': [paper_colors('vis_miss'), paper_colors('aud_miss'), paper_colors('probe')],
            #    'plot_subset': False
            #},
#
            #'overlap_stim_neurons_all_trials': {
            #    'neurons': 'overlap',
            #    'filters_trialData': [
            #        {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
            #        {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
            #    ],
            #    'labels': ['Visual miss', 'Auditory miss', 'Correct rejection'],
            #    'colors': [paper_colors('vis_miss'), paper_colors('aud_miss'), paper_colors('probe')],
            #    'plot_subset': False
            #},

            'auditory_stim_neurons_visual_trials': {
                'neurons': 'auditory',
                'filters_trialData': [
                    {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Visual miss', 'Correct rejection'],
                'colors': [paper_colors('vis_hit'), 'black'],
                'plot_subset': True,
                'y_lim': (-13, 65),
                'sig_bars': 'visual'
            },

            'visual_stim_neurons_auditory_trials': {
                'neurons': 'visual',
                'filters_trialData': [
                    {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Auditory miss', 'Correct rejection'],
                'colors': [paper_colors('aud_hit'), 'black'],
                'plot_subset': True,
                'y_lim': (-13, 65),
                'sig_bars': 'auditory'
            },


            'overlap_stim_neurons_visual_trials': {
                'neurons': ['2063202206281123154'],
                'filters_trialData': [
                    {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Visual miss', 'Correct rejection'],
                'colors': [paper_colors('vis_hit'), 'black'],
                'plot_subset': True,
                'y_lim': (-9, 45),
                'sig_bars': 'visual'
            },

            'overlap_stim_neurons_auditory_trials': {
                'neurons': ['2063202206281123154'],
                'filters_trialData': [
                    {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 0},
                    {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 0}
                ],
                'labels': ['Auditory miss', 'Correct rejection'],
                'colors': [paper_colors('aud_hit'), 'black'],
                'plot_subset': True,
                'y_lim': (-9, 45),
                'sig_bars': 'auditory'
            }
        }
    },

    #'hit_fa_lick_aligned': {
    #    'details': {
    #        'visual': make_condition(
    #            'visual_hit_FA',
    #            [{'trialType': 'X', 'correctResponse': 1},
    #             {'trialType': 'P', 'responseSide': 'L'}],
    #            ['Visual hit', 'False alarm (L)'],
    #            [paper_colors('vis_hit'), paper_colors('probe')],
    #            alignment='lick_aligned'
    #        ),
    #        'auditory': make_condition(
    #            'auditory_hit_FA',
    #            [{'trialType': 'Y', 'correctResponse': 1},
    #             {'trialType': 'P', 'responseSide': 'R'}],
    #            ['Auditory hit', 'False alarm (R)'],
    #            [paper_colors('aud_hit'), paper_colors('probe')],
    #            alignment='lick_aligned'
    #        ),
    #    },
    #},

    #'fa_cr': {
    #    'details': {
    #        'left': make_condition(
    #            'left_lick_FA_CR',
    #            [{'trialType': 'P', 'responseSide': 'L'},
    #             {'trialType': 'P', 'noResponse': 1}],
    #            ['False alarm (L)', 'Correct rejection'],
    #            [paper_colors('left_lick'), paper_colors('probe')],
    #        ),
    #        'right': make_condition(
    #            'right_lick_FA_CR',
    #            [{'trialType': 'P', 'responseSide': 'R'},
    #             {'trialType': 'P', 'noResponse': 1}],
    #            ['False alarm (R)', 'Correct rejection'],
    #            [paper_colors('right_lick'), paper_colors('probe')],
    #        ),
    #    },
    #},

    #'visual_auditory': {
    #    'details': {
    #        'hits': make_condition(
    #            'visual_hit_auditory_hit',
    #            [{'trialType': 'X', 'correctResponse': 1},
    #             {'trialType': 'Y', 'correctResponse': 1}],
    #            ['Visual hit', 'Auditory hit'],
    #            [paper_colors('vis_hit'), paper_colors('aud_hit')],
    #        ),
    #        'misses': make_condition(
    #            'visual_miss_auditory_miss',
    #            [{'trialType': 'X', 'noResponse': 1},
    #             {'trialType': 'Y', 'noResponse': 1}],
    #            ['Visual miss', 'Auditory miss'],
    #            [paper_colors('vis_miss'), paper_colors('aud_miss')],
    #            selected_neurons=['2065202207131544132', '00820230412155829'],
    #        ),
    #    },
    #},

    #'visual_auditory_hit_lick_aligned': {
    #    'details': {
    #        'hits': make_condition(
    #            'visual_hit_auditory_hit',
    #            [{'trialType': 'X', 'correctResponse': 1},
    #             {'trialType': 'Y', 'correctResponse': 1}],
    #            ['Visual hit', 'Auditory hit'],
    #            [paper_colors('vis_hit'), paper_colors('aud_hit')],
    #            alignment='lick_aligned',
    #            selected_neurons=['026202403141415231', '01520230421154813',
    #                              '026202403131553221'],
    #        ),
    #    },
    #},

    #'visual_side': {
    #    'details': {
    #        'all': make_condition(
    #            'visual_side_all',
    #            [{'trialType': 'X', 'visualChangeSide': 1},
    #             {'trialType': 'X', 'visualChangeSide': 2}],
    #            ['Visual left change', 'Visual right change'],
    #            [paper_colors('left_ALM'), paper_colors('right_ALM')],
    #        ),
    #    },
    #},

    #'auditory_pitch': {
    #    'details': {
    #        'all': make_condition(
    #            'auditory_pitch_all',
    #            [{'trialType': 'Y', 'audioOctPostChange_clust': 1},
    #             {'trialType': 'Y', 'audioOctPostChange_clust': 2}],
    #            ['Visual left change', 'Visual right change'],
    #            [paper_colors('vis_side'), 'darkgreen'],
    #        ),
    #    },
    #},

    #'lick_presence': {
    #    'details': {
    #        'left licks': make_condition(
    #            'left_lick_FA_CR',
    #            [{'trialType': 'P', 'responseSide': 'L'},
    #             {'trialType': 'P', 'noResponse': 1}],
    #            ['False alarm (L)', 'Correct rejection'],
    #            [paper_colors('left_lick'), paper_colors('probe')],
    #            selected_neurons=['022202402291342210', '02220240225160212'],
    #        ),
    #        'right licks': make_condition(
    #            'right_lick_FA_CR',
    #            [{'trialType': 'P', 'responseSide': 'R'},
    #             {'trialType': 'P', 'noResponse': 1}],
    #            ['False alarm (R)', 'Correct rejection'],
    #            [paper_colors('right_lick'), paper_colors('probe')],
    #            selected_neurons=[],
    #        ),
    #    },
    #},

    #'hit_fa_lick_aligned': {
    #    'details': {
    #        'visual': make_condition(
    #            'visual_hit_fa',
    #            [{'trialType': 'X', 'correctResponse': 1},
    #             {'trialType': 'P', 'responseSide': 'L'}],
    #            ['Visual hit', 'False alarm (L)'],
    #            [paper_colors('vis_hit'), paper_colors('probe')],
    #            alignment='lick_aligned',
    #        ),
    #        'auditory': make_condition(
    #            'auditory_hit_fa',
    #            [{'trialType': 'Y', 'correctResponse': 1},
    #             {'trialType': 'P', 'responseSide': 'R'}],
    #            ['Auditory hit', 'False alarm (R)'],
    #            [paper_colors('aud_hit'), paper_colors('probe')],
    #            alignment='lick_aligned',
    #        ),
    #    },
    #},

    #'left_right_lick_lick_aligned': {
    #    'details': {
    #        'left vs right': make_condition(
    #            'left_right_lick',
    #            [{'trialType': 'P', 'responseSide': 'L'},
    #             {'trialType': 'P', 'responseSide': 'R'}],
    #            ['Left lick', 'Right lick'],
    #            [paper_colors('left_lick'), paper_colors('right_lick')],
    #            alignment='lick_aligned',
    #        ),
    #    },
    #},

    #'hit_miss_opto': {
    #    'details': {
    #        'visual': make_condition(
    #            'visual_hit_miss',
    #            [{'trialType': 'X', 'correctResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'},
    #             {'trialType': 'X', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'}],
    #            ['Visual hit, opto', 'Visual miss, opto'],
    #            [paper_colors('vis_hit'), paper_colors('vis_miss')],
    #            selected_neurons=['007202304061549118', '0232024012014232108'],
    #        ),
    #        'auditory': make_condition(
    #            'auditory_hit_miss',
    #            [{'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'},
    #             {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'}],
    #            ['Auditory hit, opto', 'Auditory miss, opto'],
    #            [paper_colors('aud_hit'), paper_colors('aud_miss')],
    #            selected_neurons=['02620240306141224', '02620240312142126'],
    #        ),
    #    },
    #    'opto': 'unilat',
    #    'select_neurons_overlap': ['015202304221515117', '02120240201152521',
    #                               '02120240130150836', '02620240313155316'],
    #},
#
    #'miss_cr_opto': {
    #    'details': {
    #        'visual': make_condition(
    #            'visual_miss_cr',
    #            [{'trialType': 'X', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'},
    #             {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'}],
    #            ['Visual miss, opto', 'Correct rejection, opto'],
    #            [paper_colors('vis_max'), paper_colors('probe')],
    #            selected_neurons=['2065202207131544128', '02120240129134438'],
    #        ),
    #        'auditory': make_condition(
    #            'auditory_miss_cr',
    #            [{'trialType': 'Y', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'},
    #             {'trialType': 'P', 'noResponse': 1, 'hasphotostim': 1, 'optoLat': 'unilat'}],
    #            ['Auditory miss, opto', 'Correct rejection, opto'],
    #            [paper_colors('aud_max'), paper_colors('probe')],
    #            selected_neurons=['021202401291344332', '02620240312142117'],
    #        ),
    #    },
    #    'opto': 'unilat',
    #    'neurons_overlap': ['021202401311421263', '2063202206281123154'],
    #},

}

# =============================================================================
# Configuration
# =============================================================================

# Paths
data_dir = config.DATA_DIR_EXP

# Location of stimulus-aligned ROC results
roc_data_dir = config.resolve_pp("ROC")


output_dir_psth = config.FIGURE_DIR / "coding_neurons/PSTH"
output_dir = config.FIGURE_DIR / "coding_neurons/venn"

for dir_i in [output_dir_psth, output_dir]:
    if not os.path.isdir(dir_i):
        dir_i.mkdir(parents=True, exist_ok=True)

# Data loading and filtering
experiment = ["ALM_1", "ALM_2", "ALM_3"]
opto_experiments = ["ALM_2", "ALM_3"]
animals = "all"
min_fire = 0.5
quality = 'good'

# Analysis parameters
areas = ['lALM', 'rALM']
min_consecutive_values = 5
stimulus_cluster_method = 'optimized_per_session'

# PSTH / plotting parameters
psth_cfg = {
    'window': [-0.5, 1.5],
    'bin_size': 0.200,
    'step_size': 0.01,
    'smoothing_width': None,
    'height_ratios': (1.5, 1),
    'figsize': (1, 1.5),
    'use_causal_window': True,
    'calc_single_trial': True,
    'gen_svg': True,
    'z_score_psth': True,
    'raster_dot_width': 0.3
}

min_trials_additional_plots = 10
# DEBUG


# Flags
do_plot = True
debug = False
# contrast_dict = contrast_dict_test  # Uncomment for testing

# =============================================================================
# Data loading and preprocessing
# =============================================================================

trialData, sessionData, spikeData = utils.load_data(data_dir, experiment, animals)
trialData, sessionData, spikeData, removed_sessions = utils.filter_sessions_hitrate(trialData, sessionData, spikeData,
                                                                                    min_hitrate=0.3)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)

# Cluster stimuli for task-irrelevant stimulus information decoding
trialData.loc[:, 'visualOriPostChangeLeft_clust'] = utils.cluster_samples(
    trialData, 'visualOriPostChangeLeft',
    n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=15)

trialData.loc[:, 'visualOriPostChangeRight_clust'] = utils.cluster_samples(
    trialData, 'visualOriPostChangeRight',
    n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=15)

trialData.loc[:, 'audioOctPostChange_clust'] = utils.cluster_samples(
    trialData, 'audioOctPostChange',
    n_clusters=2, method=stimulus_cluster_method, modality='aud', bin_size=1 / 24)

# Opto-restricted subsets (ALM_2, ALM_3 only)
opto_sessions = sessionData[sessionData['experiment'].isin(opto_experiments)]['session_ID']
spikeData_opto = spikeData[spikeData['session_ID'].isin(opto_sessions)]
trialData_opto = trialData[trialData['session_ID'].isin(opto_sessions)]

# Precompute total neuron counts per area
total_neurons_area = {
    area: spikeData[spikeData['area'] == area]['cluster_ID'].nunique()
    for area in areas
}
total_neurons_area_opto = {
    area: spikeData_opto[spikeData_opto['area'] == area]['cluster_ID'].nunique()
    for area in areas
}

# Shared PSTH keyword arguments
psth_kwargs = make_psth_kwargs(psth_cfg)

# Area → analysis directory mapping
AREA_TO_ANALYSIS_DIR = {
    'lALM': 'ROC_ALM', 'rALM': 'ROC_ALM',
    'rV1': 'ROC_V1', 'V1': 'ROC_V1',
}

CONTRASTS_TO_PLOT = ['hit_miss', 'miss_cr']

# =============================================================================
# Main analysis loop
# =============================================================================

summary_rows = []

for contrast_name in contrast_dict:
    contrast_dict[contrast_name]['results'] = {}

for area in areas:
    analysis_dir = AREA_TO_ANALYSIS_DIR.get(area)
    if analysis_dir is None:
        raise NotADirectoryError(f"{area} is not valid — no corresponding analysis found")
    print(f"\n{'='*60}\nArea: {area}\n{'='*60}")

    for contrast_name, contrast_cfg in contrast_dict.items():
        if contrast_name not in CONTRASTS_TO_PLOT:
            continue

        # DEBUGGING
        #if contrast_name == 'hit_fa_lick_aligned':
        #    pass
        #else:
        #    continue

        print(f"\n--- Contrast: {contrast_name} ---")
        conditions = contrast_cfg['details']
        is_opto = contrast_has_opto(conditions)
        opto_flag_trials = contrast_opto_type(conditions)
        if is_opto and opto_flag_trials == 'opto_off':
            print(f"WARNING: Detected opto trials in conditions, but ROC data is being loaded for non-opto run")
        opto_flag_ROC = contrast_cfg.get('opto', 'opto_off')
        plot_subset = contrast_cfg.get('plot_subset', False)
        skip_plot = contrast_cfg.get('skip_plot', False)

        # Select appropriate data based on opto status
        if is_opto:
            spikeData_use, trialData_use = spikeData_opto, trialData_opto
            n_total_area = total_neurons_area_opto[area]
            print(f"  Opto contrast → restricting to {opto_experiments}")
        else:
            spikeData_use, trialData_use = spikeData, trialData
            n_total_area = total_neurons_area[area]

        # ── Phase 1: Load ROC results and identify significant neurons ───
        labels = []
        neurons_list = []
        all_sig_neurons = set()
        all_roc_neurons = set()

        for cond_name, cond in conditions.items():
            print(f"  condition: {cond_name}")
            alignment = cond.get('alignment', 'stim_aligned')

            roc_dir = os.path.join(roc_data_dir, alignment, analysis_dir, area,
                                   opto_flag_ROC, cond['condition_dir'])

            (cond['auc_scores'], cond['neurons'], cond['p_values'],
             cond['params'], cond['filters'], cond['n_trials']) = \
                ROC_utils.roc_load_results(roc_dir)

            # Filter ROC neurons based on previous filtering
            cond = ROC_utils.filter_neurons(cond, cluster_IDs=spikeData_use['cluster_ID'])

            # Determine which neurons will be considered significantly coding based on consecutive low p-value bins
            cond['neurons_sig_timepoints'] = find_neurons_with_low_p_values(
                cond['p_values'], cond['neurons'], cond['auc_scores'],
                min_consecutive_values=min_consecutive_values)

            neurons_list.append(cond['neurons_sig_timepoints'].keys())
            labels.append(cond_name)

            all_sig_neurons.update(cond['neurons_sig_timepoints'].keys())
            if cond['neurons'] is not None:
                all_roc_neurons.update(str(n) for n in cond['neurons'])

        # ── Summary rows (deferred for 2-cond contrasts until overlap is known) ──
        has_two_conditions = (len(labels) == 2)

        if has_two_conditions:
            overlap = ROC_utils.compute_neuron_overlap(neurons_list, labels)
            n_ovlp = len(overlap['overlap_all'])

            # Build the shared unique/overlap neuron ID structures
            neuron_ids_unique = {
                cond_name: set(overlap['unique_to_' + cond_name])
                for cond_name in conditions
            }
            neuron_ids_overlap = set(overlap['overlap_all'])

            # ── Store in contrast_dict for later use ──
            contrast_dict[contrast_name]['results'][area] = {
                'unique': neuron_ids_unique,
                'overlap': neuron_ids_overlap,
                'all_sig': {
                    cond_name: set(cond['neurons_sig_timepoints'].keys())
                    for cond_name, cond in conditions.items()
                },
                #'all_sig_bool': {
                #    cond_name: cond['neurons_sig_timepoints']
                #    for cond_name, cond in conditions.items()
                #}
            }

            # Per-condition rows
            for cond_name, cond in conditions.items():
                n_roc = len(cond['neurons']) if cond['neurons'] is not None else 0
                n_sig = len(cond['neurons_sig_timepoints'])
                n_unique = len(overlap['unique_to_' + cond_name])
                summary_rows.append(build_summary_row(
                    contrast_name, cond_name, area, is_opto, n_total_area, n_roc, n_sig,
                    neuron_ids=set(cond['neurons_sig_timepoints'].keys()),
                    neuron_ids_unique=neuron_ids_unique,
                    neuron_ids_overlap=neuron_ids_overlap,
                    n_unique=n_unique, n_overlap=n_ovlp))

            # Combined row
            n_all_unique = sum(len(overlap['unique_to_' + l]) for l in labels)
            summary_rows.append(build_summary_row(
                contrast_name, '_combined', area, is_opto,
                n_total_area, len(all_roc_neurons), len(all_sig_neurons),
                neuron_ids=set(all_sig_neurons),
                neuron_ids_unique=neuron_ids_unique,
                neuron_ids_overlap=neuron_ids_overlap,
                n_unique=n_all_unique, n_overlap=n_ovlp))
        else:
            # Single-condition: no overlap to compute
            contrast_dict[contrast_name]['results'][area] = {
                'unique': None,
                'overlap': None,
                'all_sig': {
                    cond_name: set(cond['neurons_sig_timepoints'].keys())
                    for cond_name, cond in conditions.items()
                },
            }
            for cond_name, cond in conditions.items():
                n_roc = len(cond['neurons']) if cond['neurons'] is not None else 0
                n_sig = len(cond['neurons_sig_timepoints'])
                summary_rows.append(build_summary_row(
                    contrast_name, cond_name, area, is_opto, n_total_area, n_roc, n_sig,
                    neuron_ids=set(cond['neurons_sig_timepoints'].keys())))
            summary_rows.append(build_summary_row(
                contrast_name, '_combined', area, is_opto,
                n_total_area, len(all_roc_neurons), len(all_sig_neurons),
                neuron_ids=set(all_sig_neurons)))

        # ── Phase 2a: Two-condition contrasts — overlap and unique coding ─ PLOTTING
        if has_two_conditions:

            trialData_overlap_list = []
            neurons_overlap_list = []
            neurons_overlap_dict = {}
            labels_overlap_list = []
            colors_overlap = []
            colors_overlap_highlight = []
            neurons_unique_dict = {}

            for cond_name, cond in conditions.items():
                #opto_flag = cond.get('opto', 'opto_off')
                opto_flag = opto_flag_ROC
                align_to = resolve_alignment(cond)

                # Split neurons into unique-to-condition and overlapping
                neurons_unique = {
                    k: v for k, v in cond['neurons_sig_timepoints'].items()
                    if k in overlap['unique_to_' + cond_name]
                }
                neurons_ovlp = {
                    k: v for k, v in cond['neurons_sig_timepoints'].items()
                    if k in overlap['overlap_all']
                }
                neurons_overlap_list.append(neurons_ovlp)
                neurons_overlap_dict[cond_name] = neurons_ovlp

                neurons_unique_dict[cond_name] = neurons_unique

                # Filter trial data
                td1 = utils.filter_trialData(trialData_use, cond['filters_trialData'][0])
                td2 = utils.filter_trialData(trialData_use, cond['filters_trialData'][1])

                if opto_flag == 'opto_off':
                    td1 = td1[td1['hasphotostim'] == 0]
                    td2 = td2[td2['hasphotostim'] == 0]

                #td1, td2 = filter_opto_trials(td1, td2, opto_flag)

                append_if_unique(trialData_overlap_list, td1)
                append_if_unique(trialData_overlap_list, td2)
                labels_overlap_list.extend(cond['labels'])
                colors_overlap_highlight.append(cond['colors'][0])
                colors_overlap.extend(cond['colors'])

                # Plot uniquely coding neurons
                if do_plot and not skip_plot:
                    print(f"  Plotting PSTHs for {contrast_name}/{cond_name}")
                    sd_cond = spikeData_use[spikeData_use['cluster_ID'].isin(neurons_unique)]
                    sd_cond = subset_spike_data(sd_cond, cond, debug, plot_subset)
                    if sd_cond is None:
                        print(f"plot_subset is {plot_subset} but no subset was specified - Skipping condition plots")
                        continue
                    subset_dir = 'subset' if plot_subset else ""

                    highlight_clipped = clip_sig_dict(
                        neurons_unique, cond['params']['t_bins'], psth_cfg['window'], roc_window=cond['params']['window'])

                    plot_psth(
                        sd_cond, [td1, td2],
                        output_dir=os.path.join(output_dir_psth, contrast_name, cond_name, area, subset_dir),
                        labels=cond['labels'], colors=cond['colors'],
                        highlight_dict={cond_name: highlight_clipped},
                        highlight_colors=cond['colors'],
                        align_to=align_to, y_lim=cond.get('y_lim', None), psth_kwargs=psth_kwargs,
                        line_styles=contrast_cfg.get('line_styles', None)
                    )

            # Plot overlap neurons
            if do_plot:
                print("  Plotting overlap")
                neurons_overlap_ids = [n for sub in neurons_overlap_list for n in sub]
                sd_ovlp = spikeData_use[spikeData_use['cluster_ID'].isin(neurons_overlap_ids)]

                if debug:
                    sd_ovlp = sd_ovlp.head(2)
                if plot_subset:
                    neuron_subset = contrast_cfg.get('neurons_overlap', [])
                    sd_ovlp = sd_ovlp[sd_ovlp['cluster_ID'].isin(neuron_subset)]

                plot_psth(
                    sd_ovlp, trialData_overlap_list,
                    output_dir=os.path.join(output_dir_psth, contrast_name, 'overlap', area),
                    labels=labels_overlap_list, colors=colors_overlap,
                    highlight_dict=neurons_overlap_dict,
                    highlight_colors=colors_overlap_highlight,
                    align_to=align_to, psth_kwargs=psth_kwargs,
                    highlight_plot_style='br',
                )

        # ── Phase 2b: Single-condition contrasts ─────────────────────────
        else:
            cond = next(iter(conditions.values()))
            cond_name = next(iter(conditions.keys()))
            #opto_flag = cond.get('opto', 'opto_off')
            opto_flag = opto_flag_ROC
            align_to = resolve_alignment(cond)
            neurons_coding = cond['neurons_sig_timepoints']

            td1 = utils.filter_trialData(trialData_use, cond['filters_trialData'][0])
            td2 = utils.filter_trialData(trialData_use, cond['filters_trialData'][1])
            #td1, td2 = filter_opto_trials(td1, td2, opto_flag)

            if do_plot:
                sd_cond = spikeData_use[spikeData_use['cluster_ID'].isin(neurons_coding.keys())]
                sd_cond = subset_spike_data(sd_cond, cond, debug, plot_subset)

                neurons_coding_clipped = clip_sig_dict(
                    neurons_coding, cond['params']['t_bins'], psth_cfg['window'], roc_window=cond['params']['window'])

                plot_psth(
                    sd_cond, [td1, td2],
                    output_dir=os.path.join(output_dir_psth, contrast_name, area),
                    labels=cond['labels'], colors=cond['colors'],
                    highlight_dict={cond_name: neurons_coding_clipped},
                    highlight_colors=cond['colors'],
                    align_to=align_to, psth_kwargs=psth_kwargs,
                )

        print('0262024030614122150' in conditions['visual']['neurons_sig_timepoints'])
        print('0262024030614122150' in conditions['auditory']['neurons_sig_timepoints'])

        # ── Phase 3: Additional plots (e.g. opto comparisons) ───────────
        plot_additional = contrast_cfg.get('plot_additional', False)
        if do_plot and plot_additional and 'additional_plots' in contrast_cfg:
            for plot_name, plot_details in contrast_cfg['additional_plots'].items():
                print(f"Plotting additional plots: {plot_name}")
                # Additional plot may reference opto even if parent contrast doesn't
                additional_is_opto = any(
                    f.get('hasphotostim') == 1 for f in plot_details['filters_trialData'])
                if additional_is_opto and not is_opto:
                    sd_add, td_add = spikeData_opto, trialData_opto
                else:
                    sd_add, td_add = spikeData_use, trialData_use

                td_list = [utils.filter_trialData(td_add, f)
                           for f in plot_details['filters_trialData']]

                sd_add = utils.filter_neurons_min_trials(sd_add, td_list, min_trials_additional_plots)
                if len(sd_add) == 0:
                    print(f"WARNING: No neurons can be plotted for {plot_name} because "
                          f"min_trials_additional_plots is too high {min_trials_additional_plots}")
                if isinstance(plot_details['neurons'], list):
                    sd_plot = sd_add[sd_add['cluster_ID'].isin(plot_details['neurons'])]
                elif plot_details['neurons'] == 'overlap':
                    overlap_neurons = contrast_cfg['results'][area]['overlap']
                    sd_plot = sd_add[sd_add['cluster_ID'].isin(overlap_neurons)]
                elif plot_details['neurons'] == 'all':
                    sd_plot = sd_add
                elif plot_details['neurons'] in contrast_cfg['details'].keys():
                    if plot_details['plot_subset']:
                        neurons_add = contrast_cfg['details'][plot_details['neurons']]['selected_neurons']
                    else:
                        unique_neurons = contrast_cfg['results'][area]['unique']
                        neurons_add = unique_neurons[plot_details['neurons']]
                    sd_plot = sd_add[sd_add['cluster_ID'].isin(neurons_add)]

                if any(sd_plot['area'] != area):
                    continue

                y_lim = plot_details.get('y_lim', None)
                sig_type = plot_details.get('sig_bars', None)

                # Get significant timepoints
                if sig_type is None:
                    sig_highlight = None
                    sig_colors = None
                elif sig_type == 'overlap':
                    overlap_ids = contrast_cfg['results'][area]['overlap']
                    sig_colors = [colors_overlap_highlight[0]]  # or whichever color fits
                else:
                    sig_highlight = {
                        sig_type: clip_sig_dict(
                            conditions[sig_type]['neurons_sig_timepoints'],
                            conditions[sig_type]['params']['t_bins'],
                            psth_cfg['window'], roc_window=cond['params']['window'])
                    }
                    sig_colors = conditions[sig_type]['colors']

                line_styles = contrast_cfg.get('line_styles', None)

                # Plot
                plot_psth(
                    sd_plot, td_list,
                    output_dir=os.path.join(output_dir_psth, 'additional_plots', plot_name),
                    labels=plot_details['labels'], colors=plot_details['colors'],
                    highlight_dict=sig_highlight, highlight_colors=sig_colors,
                    align_to=align_to, psth_kwargs=psth_kwargs, y_lim=y_lim, line_styles=line_styles
                )


# =============================================================================
# Summary table
# =============================================================================

summary_df = pd.DataFrame(summary_rows)

# Add combined-across-areas rows (lALM + rALM pooled)
combined_rows = []
for (contrast, condition), grp in summary_df.groupby(['contrast', 'condition']):
    n_total = grp['n_total_area'].sum()
    n_roc = grp['n_total_roc'].sum()
    n_coding = grp['n_coding'].sum()

    # Pool neuron_ids across areas
    pooled_neuron_ids = set()
    for _, row in grp.iterrows():
        pooled_neuron_ids.update(row['neuron_ids'])

    # Pool neuron_ids_unique across areas: merge dicts, union sets per condition key
    has_unique = grp['neuron_ids_unique'].notna().any()
    if has_unique:
        pooled_unique = {}
        for _, row in grp.iterrows():
            if row['neuron_ids_unique'] is not None:
                for cond_name, id_set in row['neuron_ids_unique'].items():
                    if cond_name not in pooled_unique:
                        pooled_unique[cond_name] = set()
                    pooled_unique[cond_name].update(id_set)
    else:
        pooled_unique = None

    # Pool neuron_ids_overlap across areas
    has_overlap_ids = grp['neuron_ids_overlap'].notna().any()
    if has_overlap_ids:
        pooled_overlap_ids = set()
        for _, row in grp.iterrows():
            if row['neuron_ids_overlap'] is not None:
                pooled_overlap_ids.update(row['neuron_ids_overlap'])
    else:
        pooled_overlap_ids = None

    # Sum count columns if present
    has_overlap = grp['n_unique'].notna().any()
    if has_overlap:
        n_unique = int(grp['n_unique'].sum())
        n_overlap = int(grp['n_overlap'].sum())
    else:
        n_unique = None
        n_overlap = None

    combined_rows.append({
        'contrast': contrast,
        'condition': condition,
        'area': 'combined',
        'is_opto': grp['is_opto'].iloc[0],
        'n_total_area': n_total,
        'n_total_roc': n_roc,
        'n_coding': n_coding,
        'frac_of_area': safe_frac(n_coding, n_total),
        'frac_of_roc': safe_frac(n_coding, n_roc),
        'neuron_ids': pooled_neuron_ids,
        'neuron_ids_unique': pooled_unique,
        'neuron_ids_overlap': pooled_overlap_ids,
        'n_unique': n_unique,
        'n_overlap': n_overlap,
        'frac_unique': safe_frac(n_unique, n_roc) if n_unique is not None else None,
        'frac_overlap': safe_frac(n_overlap, n_roc) if n_overlap is not None else None,
        'frac_unique_of_coding': safe_frac(n_unique, n_coding) if n_unique is not None else None,
        'frac_overlap_of_coding': safe_frac(n_overlap, n_coding) if n_overlap is not None else None,
    })

contrasts_to_print = ['hit_miss', 'hit_miss_opto', 'miss_cr', 'miss_cr_opto']
summary_df = pd.concat([summary_df, pd.DataFrame(combined_rows)], ignore_index=True)
summary_df = summary_df.sort_values(['contrast', 'condition', 'area']).reset_index(drop=True)

# For printing, hide the neuron_ids columns (too wide for console)
cols_to_hide = ['neuron_ids', 'neuron_ids_unique', 'neuron_ids_overlap']
summary_df_print = summary_df[
    (summary_df['contrast'].isin(contrasts_to_print))
]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)
print("\n" + "=" * 120)
print("SUMMARY: Coding neuron counts and fractions")
print("=" * 120)
print(summary_df_print.drop(columns=cols_to_hide).to_string(index=False))
print("=" * 120)


# =============================================================================
# Pairwise overlap analysis
# =============================================================================

# Overlap between all non-opto _combined rows, computed per area
overlap_results = compute_pairwise_overlap(
    summary_df,
    row_filter={'is_opto': False, 'condition': '_combined'},
    group_by='area'
)

print("\n" + "=" * 120)
print("PAIRWISE OVERLAP COEFFICIENTS (non-opto, _combined, per area)")
print("=" * 120)
if not overlap_results.empty:
    print(overlap_results.to_string(index=False))
print("=" * 120)

overlap_input_df = summary_df[summary_df['contrast'].isin(['hit_miss', 'miss_cr'])]

# Example: heatmap per area
plot_heatmaps = False
if plot_heatmaps:
    for area in ['lALM', 'rALM', 'combined']:
        overlap_df = compute_pairwise_overlap(
            overlap_input_df,
            row_filter={'area': area},
            group_by=None
        )
        plot_overlap_heatmap(
            overlap_df,
            metric='overlap_coef',
            title=f'Overlap Coefficient — {area}',
            save_path=os.path.join(output_dir, f'overlap_heatmap_{area}.png'),
        )

plot_venn = True
if plot_venn:
    output_dir_venn = os.path.join(output_dir, 'venn')
    if not os.path.isdir(output_dir_venn):
        os.makedirs(output_dir_venn
                    )
    for area in ['lALM', 'rALM', 'combined']:
        venn_plots = {
            # How many stimulus coding neurons are overlapping between modalities?
            'miss_cr': {
                'data':
                    ((summary_df[(summary_df['contrast'] == 'miss_cr') &
                                   (summary_df['condition'] == 'visual') &
                                   (summary_df['area'] == area)]['neuron_ids'].iloc[0]),

                        (summary_df[(summary_df['contrast'] == 'miss_cr') &
                                    (summary_df['condition'] == 'auditory') &
                                    (summary_df['area'] == area)]['neuron_ids'].iloc[0])),
                'labels': ['Visual', 'Auditory']
            },

            # How many hit/miss neurons are overlapping between modalities?
            'hit_miss': {
                'data':
                    ((summary_df[(summary_df['contrast'] == 'hit_miss') &
                                   (summary_df['condition'] == 'visual') &
                                   (summary_df['area'] == area)]['neuron_ids'].iloc[0]),

                        (summary_df[(summary_df['contrast'] == 'hit_miss') &
                                    (summary_df['condition'] == 'auditory') &
                                    (summary_df['area'] == area)]['neuron_ids'].iloc[0])),
                'labels': ['Visual', 'Auditory']
            },

            # Are neurons encoding hit/miss the same as those encoding miss/CR?
            'hit_miss_miss_cr_overlap_combined': {
                'data':
                    ((summary_df[(summary_df['contrast'] == 'hit_miss') &
                                    (summary_df['condition'] == '_combined') &
                                    (summary_df['area'] == area)]['neuron_ids'].iloc[0]),

                        (summary_df[(summary_df['contrast'] == 'miss_cr') &
                                    (summary_df['condition'] == '_combined') &
                                    (summary_df['area'] == area)]['neuron_ids'].iloc[0])),
                'labels': ['Hit/Miss', 'Miss/CR']
            },

            'hit_miss_miss_cr_overlap_visual': {
                'data':
                    ((summary_df[(summary_df['contrast'] == 'hit_miss') &
                                 (summary_df['condition'] == 'visual') &
                                 (summary_df['area'] == area)]['neuron_ids'].iloc[0]),

                     (summary_df[(summary_df['contrast'] == 'miss_cr') &
                                 (summary_df['condition'] == 'visual') &
                                 (summary_df['area'] == area)]['neuron_ids'].iloc[0])),
                'labels': ['Hit/Miss', 'Miss/CR']
            },

            'hit_miss_miss_cr_auditory': {
                'data':
                    ((summary_df[(summary_df['contrast'] == 'hit_miss') &
                                 (summary_df['condition'] == 'auditory') &
                                 (summary_df['area'] == area)]['neuron_ids'].iloc[0]),

                     (summary_df[(summary_df['contrast'] == 'miss_cr') &
                                 (summary_df['condition'] == 'auditory') &
                                 (summary_df['area'] == area)]['neuron_ids'].iloc[0])),
                'labels': ['Hit/Miss', 'Miss/CR']
            }
        }

        for key, value in venn_plots.items():
            set_A, set_B = value['data']
            labels = value['labels']
            plt.figure(figsize=(2, 2))
            v = venn2((set_A, set_B), set_labels=labels)
            plt.title(key)
            plt.savefig(os.path.join(output_dir_venn, f'venn_{key}_{area}.png'), format='png', dpi=300)
            plt.savefig(os.path.join(output_dir_venn, f'venn_{key}_{area}.svg'), format='svg', dpi=300)

# =============================================================================
# Violin plots: depth distribution of coding neurons
# =============================================================================

output_dir_violin = os.path.join(output_dir, 'violin_depth')
os.makedirs(output_dir_violin, exist_ok=True)

def get_depths_for_neuron_ids(neuron_ids, spikeData):
    """Look up depth for a set of neuron cluster_IDs from spikeData."""
    neuron_ids_str = set(str(n) for n in neuron_ids)
    mask = spikeData['cluster_ID'].astype(str).isin(neuron_ids_str)
    df = spikeData.loc[mask, ['cluster_ID', 'depth']].drop_duplicates(subset='cluster_ID')
    df['depth'] = np.abs(np.int16(df['depth'].values - 1000))
    return df


for contrast_name, category_label in [('hit_miss', 'Hit / Miss'), ('miss_cr', 'Miss / CR')]:
    area = 'combined'

    rows = summary_df[
        (summary_df['contrast'] == contrast_name) &
        (summary_df['area'] == area) &
        (summary_df['condition'].isin(['visual', 'auditory']))
    ]

    vis_ids = rows.loc[rows['condition'] == 'visual', 'neuron_ids']
    aud_ids = rows.loc[rows['condition'] == 'auditory', 'neuron_ids']
    vis_ids = vis_ids.iloc[0] if len(vis_ids) > 0 else set()
    aud_ids = aud_ids.iloc[0] if len(aud_ids) > 0 else set()
    overlap_ids = vis_ids & aud_ids
    vis_unique = vis_ids - overlap_ids
    aud_unique = aud_ids - overlap_ids

    dfs = []
    for label, ids in [('Visual', vis_unique), ('Auditory', aud_unique), ('Overlap', overlap_ids)]:
        depth_df = get_depths_for_neuron_ids(ids, spikeData)
        if len(depth_df) > 0:
            depth_df = depth_df.copy()
            depth_df['group'] = label
            dfs.append(depth_df)

    if len(dfs) == 0:
        print(f'No neurons found for {contrast_name} — skipping violin plot')
        continue

    plot_df = pd.concat(dfs, ignore_index=True)

    violin_w, violin_h = 0.6, 1.0
    left, bottom = 0.35, 0.25
    fig = plt.figure(figsize=(violin_w / (1 - left), violin_h / (1 - bottom)))
    ax = fig.add_axes([left, bottom, 1 - left, 1 - bottom])

    sns.violinplot(
        data=plot_df, x='group', y='depth', ax=ax,
        order=['Visual', 'Auditory', 'Overlap'],
        palette=[paper_colors('vis_hit'), paper_colors('aud_hit'), 'grey'],
        inner='box', cut=0
    )

    for art in ax.collections:
        if isinstance(art, PolyCollection):
            art.set_edgecolor('none')

    # Restore visibility of the inner box/whisker elements
    for line in ax.lines:
        line.set_linewidth(1)

    ax.set_title(f'Depth — {category_label} coding neurons (combined)')
    ax.set_xlabel('')
    ax.set_ylabel('Depth (µm)')
    ax.invert_yaxis()
    ax.set_ylim([1050, -50])

    #plt.tight_layout()
    fig.savefig(os.path.join(output_dir_violin, f'violin_depth_{contrast_name}_combined.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir_violin, f'violin_depth_{contrast_name}_combined.svg'), dpi=300, bbox_inches='tight')

# =============================================================================
# Left/right visual stimulus encoding neuron counts (lALM vs rALM)
# =============================================================================

lr_conditions = [
    'right_visual_hit_miss',
    'left_visual_hit_miss',
    'right_visual_miss_cr',
    'left_visual_miss_cr',
]

lr_alignment = 'stim_aligned'   # stim-aligned contrasts
lr_opto_flag = 'opto_off'       # non-opto run

# lr_counts[area][condition_dir] = n significantly coding neurons
lr_counts = {area: {} for area in areas}
lr_neuron_ids = {area: {} for area in areas}   # <-- add this

for area in areas:
    analysis_dir = AREA_TO_ANALYSIS_DIR.get(area)
    if analysis_dir is None:
        raise NotADirectoryError(f"{area} is not valid — no corresponding analysis found")

    for cond_dir in lr_conditions:
        roc_dir = os.path.join(roc_data_dir, lr_alignment, analysis_dir, area,
                               lr_opto_flag, cond_dir)

        # Load ROC results
        cond = {'neurons': None, 'auc_scores': None, 'p_values': None,
                'params': None, 'filters': None, 'n_trials': None}
        (cond['auc_scores'], cond['neurons'], cond['p_values'],
         cond['params'], cond['filters'], cond['n_trials']) = \
            ROC_utils.roc_load_results(roc_dir)

        # Restrict to neurons surviving previous filtering for this area
        cond = ROC_utils.filter_neurons(cond, cluster_IDs=spikeData['cluster_ID'])

        # Identify significantly coding neurons (same criterion as main loop)
        sig = find_neurons_with_low_p_values(
            cond['p_values'], cond['neurons'], cond['auc_scores'],
            min_consecutive_values=min_consecutive_values)

        lr_neuron_ids[area][cond_dir] = set(sig.keys())   # <-- add this
        lr_counts[area][cond_dir] = len(sig)

# ── Print ──
print("\n" + "=" * 60)
print("Left/right visual encoding neuron counts")
print("=" * 60)
for area in areas:
    print(f"\n{area}:")
    for cond_dir in lr_conditions:
        print(f"  {cond_dir:25s} {lr_counts[area][cond_dir]:4d}")
# ── Bar plot ──
import matplotlib as mpl

# Make text stay as editable text in the SVG, set Arial 6pt globally for this figure
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6

# Desired x-axis order
lr_plot_order = [
    'left_visual_hit_miss',
    'right_visual_hit_miss',
    'left_visual_miss_cr',
    'right_visual_miss_cr',
]
# Bar order within each group
bar_areas = ['lALM', 'rALM']

fig, ax = plt.subplots(figsize=(2.5, 1.5))
x = np.arange(len(lr_plot_order))
width = 0.35

for i, area in enumerate(bar_areas):
    n_total = total_neurons_area[area]
    counts = [lr_counts[area][c] for c in lr_plot_order]
    pcts = [100 * safe_frac(n, n_total) for n in counts]
    bars = ax.bar(x + (i - 0.5) * width, pcts, width, label=area, linewidth=1)
    for rect, n, p in zip(bars, counts, pcts):
        ax.text(rect.get_x() + rect.get_width() / 2, p,
                f"{n}\n{p:.1f}%", ha='center', va='bottom', fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels(lr_plot_order, rotation=30, ha='right', fontsize=6)
ax.set_ylabel('% of neurons in area', fontsize=6)
ax.set_title('Left/right visual stimulus encoding neurons', fontsize=6)
ax.tick_params(labelsize=6, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.legend(title='Area', fontsize=6, title_fontsize=6)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'lr_visual_encoding.svg'), format='svg', bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'lr_visual_encoding.png'), dpi=300, bbox_inches='tight')

# =============================================================================
# Left/right visual stimulus encoding: overlap (Venn) per area
# =============================================================================
# Complementary to the bar plot above. Uses the neuron-ID sets already
# collected in the lr_counts loop — no reloading.

output_dir_lr_venn = os.path.join(output_dir, 'venn_lr_visual')
os.makedirs(output_dir_lr_venn, exist_ok=True)

# Venn comparisons: left vs right, one per contrast
lr_venn_pairs = {
    'hit_miss': ('left_visual_hit_miss', 'right_visual_hit_miss'),
    'miss_cr':  ('left_visual_miss_cr',  'right_visual_miss_cr'),
}

for area in areas:
    for contrast_label, (left_cond, right_cond) in lr_venn_pairs.items():
        set_L = lr_neuron_ids[area][left_cond]
        set_R = lr_neuron_ids[area][right_cond]

        plt.figure(figsize=(2, 2))
        v = venn2((set_L, set_R), set_labels=['Left', 'Right'])

        # Optional colouring to match the rest of the script
        if v.get_patch_by_id('10') is not None:
            v.get_patch_by_id('10').set_color(paper_colors('left_ALM'))
            v.get_patch_by_id('10').set_alpha(0.6)
        if v.get_patch_by_id('01') is not None:
            v.get_patch_by_id('01').set_color(paper_colors('right_ALM'))
            v.get_patch_by_id('01').set_alpha(0.6)
        if v.get_patch_by_id('11') is not None:
            v.get_patch_by_id('11').set_color('grey')
            v.get_patch_by_id('11').set_alpha(0.6)

        plt.title(f'{contrast_label} — {area}')
        fname = f'venn_lr_visual_{contrast_label}_{area}'
        plt.savefig(os.path.join(output_dir_lr_venn, fname + '.png'),
                    format='png', dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir_lr_venn, fname + '.svg'),
                    format='svg', dpi=300, bbox_inches='tight')
        plt.close()
print('end')