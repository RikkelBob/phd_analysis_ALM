import os
# ============================================================================
# BLAS thread control — MUST be set before importing numpy
# ============================================================================
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from neural_analysis import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from neural_analysis.SVM.utils_svm import sample_exhaustive
import shutil
import yaml
import sys
import argparse
import numpy as np
from joblib import Parallel, delayed
import config

# ============================================================================
# Parallel pseudopopulation workers (module-level for loky pickling)
# ============================================================================


def _iteration_worker(seed, data_contrast, lookup, label_col,
                      n_trials_per_class, bins, n_splits, estimator):
    """Run one full iteration: sample, fit, score all bins (real + shuffled)."""
    rng = np.random.default_rng(seed)
    cv_local = StratifiedKFold(n_splits=n_splits)

    matrix, _, labels, labels_shuffled = sample_data(
        data_contrast, lookup, label_col,
        n_trials_per_class=n_trials_per_class,
        sample_mode="all", n_neurons=None, rng=rng
    )

    results_real = []
    results_shuf = []
    for idx_bin, bin_edges in enumerate(bins):
        bin_matrix = matrix[:, :, idx_bin].T

        r_real = cross_validate(
            estimator=estimator, X=bin_matrix, y=labels,
            cv=cv_local, scoring="balanced_accuracy", n_jobs=1,
        )
        r_shuf = cross_validate(
            estimator=estimator, X=bin_matrix, y=labels_shuffled,
            cv=cv_local, scoring="balanced_accuracy", n_jobs=1,
        )
        results_real.append({'bin': bin_edges, 'scores': r_real})
        results_shuf.append({'bin': bin_edges, 'scores': r_shuf})

    return results_real, results_shuf


def _batch_worker(seeds, data_contrast, lookup, label_col,
                  n_trials_per_class, bins, n_splits, estimator):
    """Run a batch of iterations in one subprocess to amortize startup."""
    return [
        _iteration_worker(s, data_contrast, lookup, label_col,
                          n_trials_per_class, bins, n_splits, estimator)
        for s in seeds
    ]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run SVM analysis with specified brain area')
    parser.add_argument('--area', nargs='+', default=None,
                        help='Brain area(s) to analyze (e.g., lALM rALM)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='L2 penalty (regularization strength) for SGDClassifier')
    return parser.parse_args()


def filter_trialData(trialData, conditions):
    filtered_df = trialData.copy()

    # Apply each filter condition (except decoder_variable)
    for key, value in conditions.items():
        if key in ['decoder_variable', 'align_to', 'k_folds']:
            continue

        if isinstance(value, list):
            # If value is a list, filter rows where column is in that list
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        else:
            # Otherwise, filter for exact match
            filtered_df = filtered_df[filtered_df[key] == value]

    return filtered_df


def sample_data(data_contrast, lookup, label_col, n_trials_per_class, n_neurons=None,
                sample_mode="all", rng=None):
    """Sample neurons and trials to build a pseudopopulation matrix.

    Args:
        rng: numpy Generator. If None, uses default_rng() (fresh state each call).
    """
    if rng is None:
        rng = np.random.default_rng()

    all_sampled_neurons = []
    matrix = []

    # Sample neurons
    if sample_mode == "all":
        all_sampled_neurons = data_contrast['neuron_IDs'].explode().tolist()
    elif sample_mode == "per_session":
        if n_neurons is None:
            print("n_neurons need to specified when using sample_mode == total_n")
            return
        for _, ses in data_contrast.iterrows():
            sample_neurons = sample_exhaustive(ses['neuron_IDs'], size=n_neurons)
            all_sampled_neurons.extend(sample_neurons)
    elif sample_mode == "total_n":
        if n_neurons is None:
            print("n_neurons need to specified when using sample_mode == total_n")
            return
        all_neurons = data_contrast['neuron_IDs'].explode().tolist()
        all_sampled_neurons = sample_exhaustive(all_neurons, size=n_neurons)

    # Derive class labels once, from data_contrast itself
    if isinstance(data_contrast, pd.Series):
        all_classes = pd.unique(data_contrast['trial_labels_reset'])
    else:
        all_classes = pd.unique(data_contrast['trial_labels_reset'].explode())

    for neuron in all_sampled_neurons:
        ses_neuron = lookup[neuron]
        if isinstance(data_contrast, pd.Series):
            spike_data_ses = data_contrast.to_frame().T
        else:
            spike_data_ses = data_contrast[data_contrast['session_ID'] == ses_neuron]
        idx_neuron = spike_data_ses['neuron_IDs'].iloc[0] == neuron
        psth_neuron_cls = []
        labels = []

        for cls in all_classes:
            bool_trials_cls = spike_data_ses['trial_labels_reset'].iloc[0] == cls
            idx_trials_cls = bool_trials_cls[bool_trials_cls].index
            sample_trials = sample_exhaustive(idx_trials_cls, n_trials_per_class)
            psth_cls = spike_data_ses['psth_ses'].iloc[0][idx_neuron, sample_trials, :]
            psth_neuron_cls.append(psth_cls)
            labels.extend([cls] * n_trials_per_class)

        psth_neuron = np.concatenate(psth_neuron_cls)
        matrix.append(psth_neuron)

    matrix = np.stack(matrix)
    labels = np.array(labels)
    labels = le.fit_transform(labels)
    labels_shuffled = rng.permutation(labels)

    return matrix, all_sampled_neurons, labels, labels_shuffled


def get_decoder_counts(df, analysis_dict):
    """
    For each analysis in analysis_dict, filter the dataframe and count rows by decoder_variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The main dataframe containing all data
    analysis_dict : dict
        Dictionary of dictionaries, where each sub-dict contains:
        - filtering conditions (column names and their values)
        - 'decoder_variable': the column to count by

    Returns:
    --------
    dict : Dictionary with analysis names as keys and count dictionaries as values
    """
    results = {}

    for analysis_name, conditions in analysis_dict.items():
        # Extract the decoder variable
        decoder_var = conditions['decoder_variable']

        filtered_df = filter_trialData(df, conditions)

        # Count occurrences of each value in the decoder variable
        counts = filtered_df[decoder_var].value_counts().to_dict()

        results[analysis_name] = counts

    return results


def get_sessions_with_min_trials(df, analysis_dict, min_trials_default=10):
    """
    For each analysis, find sessions that have at least min_trials rows for EACH
    unique value of the decoder_variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The main dataframe containing all data
    analysis_dict : dict
        Dictionary of dictionaries, where each sub-dict contains:
        - filtering conditions (column names and their values)
        - 'decoder_variable': the column to count by
    min_trials : int
        Minimum number of trials required for each decoder variable value

    Returns:
    --------
    dict : Dictionary with analysis names as keys and sub-dictionaries as values.
           Each sub-dict contains:
           - 'count': number of qualifying sessions
           - 'session_IDs': list of qualifying session IDs
           - 'n_subjects': number of unique subjects (mousenames)
    """

    results = {}

    for analysis_name, conditions in analysis_dict.items():
        if 'k_folds' in conditions:
            min_trials = conditions['k_folds']
        else:
            min_trials = min_trials_default

        # Extract the decoder variable
        decoder_var = conditions['decoder_variable']

        # Start with the full dataframe
        filtered_df = df.copy()

        # Drop nan rows
        filtered_df = filtered_df.dropna(subset=decoder_var)

        # Apply each filter condition (except decoder_variable)
        for key, value in conditions.items():
            if key in ['decoder_variable', 'align_to', 'k_folds']:
                continue

            if isinstance(value, list):
                # If value is a list, filter rows where column is in that list
                filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                # Otherwise, filter for exact match
                filtered_df = filtered_df[filtered_df[key] == value]

        # Group by session_ID and decoder_variable, count rows
        session_counts = filtered_df.groupby(['session_ID', decoder_var]).size().reset_index(name='count')

        # Pivot to get decoder values as columns
        pivot_counts = session_counts.pivot(index='session_ID', columns=decoder_var, values='count').fillna(0)

        # Find sessions where ALL decoder variable values have at least min_trials
        qualifying_sessions = pivot_counts[(pivot_counts >= min_trials).all(axis=1)]

        # Get the list of session IDs
        session_ids = qualifying_sessions.index.tolist()

        # Get unique subjects from the qualifying sessions
        unique_subjects = filtered_df[filtered_df['session_ID'].isin(session_ids)]['mousename'].nunique()

        results[analysis_name] = {
            'count': len(session_ids),
            'session_IDs': session_ids,
            'n_subjects': unique_subjects,
            'trials_per_label': pivot_counts
        }

    return results


analyses = {
    #'vis_presence_hit_fa': {
    #    'trialType': ['X', 'P'],
    #    'responseSide': 'L',
    #    'hasphotostim': False,
    #    'decoder_variable': 'trialType',
    #    'k_folds': 5
    #},

    'vis_presence_miss_cr': {
        'trialType': ['X', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'trialType'
    },

    'vis_presence_miss_cr_max': {
        'trialType': ['X', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'visualOriChange_norm': [0, 2],
        'decoder_variable': 'trialType'
    },

    #'aud_presence_hit_fa': {
    #    'trialType': ['Y', 'P'],
    #    'responseSide': 'R',
    #    'hasphotostim': False,
    #    'decoder_variable': 'trialType',
    #    'k_folds': 5
    #},
#
    'aud_presence_miss_cr': {
        'trialType': ['Y', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'trialType'
    },

    'aud_presence_miss_cr_max': {
        'trialType': ['Y', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'audioOctChange_norm': [0, 2],
        'decoder_variable': 'trialType'
    },
#
    #'vis_vs_aud_hits': {
    #    'trialType': ['Y', 'X'],
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'trialType'
    #},

    #'vis_vs_aud_miss': {
    #    'trialType': ['Y', 'X'],
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'trialType'
    #},
#
    #'vis_side_hits': {
    #    'trialType': 'X',
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualChangeSide'
    #},
#
    #'vis_side_misses': {
    #    'trialType': 'X',
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualChangeSide'
    #},

    #'vis_side_all': {
    #    'trialType': 'X',
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualChangeSide'
    #},
#
    #'lick_direction': {
    #    'trialType': 'P',
    #    'noResponse': 0,
    #    'hasphotostim': False,
    #    'decoder_variable': 'responseSide'
    #},
#
    #'vis_hits_errors': {
    #    'trialType': 'X',
    #    'noResponse': 0,
    #    'hasphotostim': False,
    #    'decoder_variable': 'responseSide'
    #},
#
    #'aud_hits_errors': {
    #    'trialType': 'Y',
    #    'noResponse': 0,
    #    'hasphotostim': False,
    #    'decoder_variable': 'responseSide'
    #},
#
    #'lick_presence_left': {
    #    'trialType': 'P',
    #    'responseSide': ['L', 'N'],
    #    'hasphotostim': False,
    #    'decoder_variable': 'responseSide',
    #    'k_folds': 5
    #},
#
    #'lick_presence_right': {
    #    'trialType': 'P',
    #    'responseSide': ['R', 'N'],
    #    'hasphotostim': False,
    #    'decoder_variable': 'responseSide',
    #    'k_folds': 5
    #},

    #'vis_ori_left_hits': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_left_misses': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust',
    #    'k_folds': 5
    #},
#
    #'vis_ori_left_all': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_left_all_max': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'visualOriChange_norm': 2,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_right_hits': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust',
    #    'k_folds': 5
    #},
#
    #'vis_ori_right_misses': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_right_all': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_right_all_max': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'visualOriChange_norm': 2,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust',
    #    'k_folds': 5
    #},

    #'aud_pitch_hits': {
    #    'trialType': 'Y',
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust',
    #    'k_folds': 5
    #},
#
    #'aud_pitch_misses': {
    #    'trialType': 'Y',
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust',
    #    'k_folds': 5
    #},

    #'aud_pitch_all': {
    #    'trialType': 'Y',
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust',
    #    'k_folds': 5
    #},

    #'aud_pitch_all_max': {
    #    'trialType': 'Y',
    #    'audioOctChange_norm': 2,
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust',
    #    'k_folds': 5
    #},

    #'vis_ori_left_hits_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust_session',
    #    'k_folds': 5
    #},
#
    #'vis_ori_left_misses_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust_session',
    #    'k_folds': 5
    #},

    #'vis_ori_left_all_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeLeft_clust_session',
    #    'k_folds': 5
    #},

    #'vis_ori_right_hits_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust_session',
    #    'k_folds': 5
    #},
#
    #'vis_ori_right_misses_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust_session',
    #    'k_folds': 5
    #},

    #'vis_ori_right_all_session': {
    #    'trialType': 'X',
    #    'visualChangeSide': 2,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualOriPostChangeRight_clust_session',
    #    'k_folds': 5
    #},

    #'aud_pitch_hits_session': {
    #    'trialType': 'Y',
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust_session',
    #    'k_folds': 5
    #},
#
    #'aud_pitch_misses_session': {
    #    'trialType': 'Y',
    #    'noResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust_session',
    #    'k_folds': 5
    #},

    #'aud_pitch_all_session': {
    #    'trialType': 'Y',
    #    'hasphotostim': False,
    #    'decoder_variable': 'audioOctPostChange_clust_session',
    #    'k_folds': 5
    #},
#
    'vis_hit_miss': {
        'trialType': 'X',
        'hasphotostim': False,
        'responseSide': ['N', 'L'],
        'decoder_variable': 'responseSide'
    },
#
    'aud_hit_miss': {
        'trialType': 'Y',
        'hasphotostim': False,
        'responseSide': ['N', 'R'],
        'decoder_variable': 'responseSide'
    }

}

analyses_opto = {
    #'vis_presence_hit_fa': {
    #    'trialType': ['X', 'P'],
    #    'responseSide': 'L',
    #    'hasphotostim': True,
    #    'decoder_variable': 'trialType'
    #},

    'vis_presence_miss_cr': {
        'trialType': ['X', 'P'],
        'noResponse': 1,
        'hasphotostim': True,
        'decoder_variable': 'trialType'
    },

    #'aud_presence_hit_fa': {
    #    'trialType': ['Y', 'P'],
    #    'responseSide': 'R',
    #    'hasphotostim': True,
    #    'decoder_variable': 'trialType'
    #},

    'aud_presence_miss_cr': {
        'trialType': ['Y', 'P'],
        'noResponse': 1,
        'hasphotostim': True,
        'decoder_variable': 'trialType'
    },

    #'vis_vs_aud_hits': {
    #    'trialType': ['Y', 'X'],
    #    'correctResponse': 1,
    #    'hasphotostim': True,
    #    'decoder_variable': 'trialType'
    #},

    #'vis_vs_aud_miss': {
    #    'trialType': ['Y', 'X'],
    #    'noResponse': 1,
    #    'hasphotostim': True,
    #    'decoder_variable': 'trialType'
    #},

    'lick_direction': {
        'trialType': 'P',
        'noResponse': 0,
        'hasphotostim': True,
        'decoder_variable': 'responseSide'
    },

    'vis_hit_miss': {
        'trialType': 'X',
        'hasphotostim': True,
        'responseSide': ['N', 'L'],
        'decoder_variable': 'responseSide'
    },

    'aud_hit_miss': {
        'trialType': 'Y',
        'hasphotostim': True,
        'responseSide': ['N', 'R'],
        'decoder_variable': 'responseSide'
    }

}

analyses_lick_aligned = {
    'vis_presence_hit_fa': {
        'trialType': ['X', 'P'],
        'responseSide': 'L',
        'hasphotostim': False,
        'decoder_variable': 'trialType',
        'align_to': 'firstLick'
    },

    'aud_presence_hit_fa': {
        'trialType': ['Y', 'P'],
        'responseSide': 'R',
        'hasphotostim': False,
        'decoder_variable': 'trialType',
        'align_to': 'firstLick'
    },

    #'vis_vs_aud_hits': {
    #    'trialType': ['Y', 'X'],
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'trialType',
    #    'align_to': 'firstLick'
    #},

    #'vis_side_hits': {
    #    'trialType': 'X',
    #    'correctResponse': 1,
    #    'hasphotostim': False,
    #    'decoder_variable': 'visualChangeSide',
    #    'align_to': 'firstLick'
    #},

    'lick_direction': {
        'trialType': 'P',
        'noResponse': 0,
        'hasphotostim': False,
        'decoder_variable': 'responseSide'
    },

    'vis_hits_errors': {
        'trialType': 'X',
        'noResponse': 0,
        'hasphotostim': False,
        'decoder_variable': 'responseSide'
    },

    'aud_hits_errors': {
        'trialType': 'Y',
        'noResponse': 0,
        'hasphotostim': False,
        'decoder_variable': 'responseSide'
    }
}

analyses_lick_aligned_opto = {
    'vis_presence_hit_fa': {
        'trialType': ['X', 'P'],
        'responseSide': 'L',
        'hasphotostim': True,
        'decoder_variable': 'trialType',
        'align_to': 'firstLick'
    },

    'aud_presence_hit_fa': {
        'trialType': ['Y', 'P'],
        'responseSide': 'R',
        'hasphotostim': True,
        'decoder_variable': 'trialType',
        'align_to': 'firstLick'
    },

    'vis_vs_aud_hits': {
        'trialType': ['Y', 'X'],
        'correctResponse': 1,
        'hasphotostim': True,
        'decoder_variable': 'trialType',
        'align_to': 'firstLick'
    },

    #'vis_side_hits': {
    #    'trialType': 'X',
    #    'correctResponse': 1,
    #    'hasphotostim': True,
    #    'decoder_variable': 'visualChangeSide',
    #    'align_to': 'firstLick'
    #},

    'lick_direction': {
        'trialType': 'P',
        'noResponse': 0,
        'hasphotostim': True,
        'decoder_variable': 'responseSide',
        'align_to': 'firstLick'
    },

    'vis_hits_errors': {
        'trialType': 'X',
        'noResponse': 0,
        'hasphotostim': True,
        'decoder_variable': 'responseSide'
    },

    'aud_hits_errors': {
        'trialType': 'Y',
        'noResponse': 0,
        'hasphotostim': True,
        'decoder_variable': 'responseSide'
    }
}

analyses_debug = {

    'vis_ori_left_hits': {
        'trialType': 'X',
        'visualChangeSide': 1,
        'correctResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'visualOriPostChangeLeft_clust',
        'k_folds': 5
    },

    'vis_ori_left_misses': {
        'trialType': 'X',
        'visualChangeSide': 1,
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'visualOriPostChangeLeft_clust',
        'k_folds': 5
    },

    'vis_ori_right_hits': {
        'trialType': 'X',
        'visualChangeSide': 2,
        'correctResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'visualOriPostChangeRight_clust',
        'k_folds': 5
    },

    'vis_ori_right_misses': {
        'trialType': 'X',
        'visualChangeSide': 2,
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'visualOriPostChangeRight_clust',
        'k_folds': 5
    },

    'aud_pitch_hits': {
        'trialType': 'Y',
        'correctResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'audioOctPostChange_clust',
        'k_folds': 5
    },

    'aud_pitch_misses': {
        'trialType': 'Y',
        'correctResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'audioOctPostChange_clust',
        'k_folds': 5
    }

}


# Parse arguments if provided, otherwise use defaults
if len(sys.argv) > 1:
    args = parse_arguments()
    area = args.area if args.area is not None else ['lALM', 'rALM']
    alpha_value = args.alpha
else:
    # Default values when running as standalone
    area = ['lALM']
    alpha_value = 0.01

np.random.seed(42)
le = LabelEncoder()

# Parameters for data loading
data_dir = config.DATA_DIR_EXP
output_dir_base = config.DATA_DIR_PP_NEW / "SVM" / "decode_from_neurons" / f"{alpha_value:g}"  # same layout as the shipped data
timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
output_dir = output_dir_base / f"megascript_output_{'_'.join(area)}_alpha{alpha_value}_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# Parameters for data inclusion
min_fire = 0.5
quality = 'good'  # 'good', 'mua', None
experiment = ["ALM_1", "ALM_2", "ALM_3"]
animals = "all"
script_name = os.path.splitext(os.path.basename(__file__))[0]
normalize_stimuli = True
sessions_exclude = ['008202304121458', '006202306141358', '021202402021327', '022202403051220',
                    '026202403191125', '026202403201103']

n_neurons_min_ses = 10  # How many neurons a single sessions needs to be included in the session-level analysis
window = (-1, 1.5)
bins, time = utils.custom_bins(bin_size=0.2, step_size=0.05, window=window, causal=True)
use_all_neurons = True  # Whether to use all available neurons for pseudopopulation
n_iterations = 2  # Number of iterations for pseudopopulation SVM

session_level_analysis = False # Perform SVM on individual sessions
pseudopopulation_analysis = True # Perform SVM on pseudopopulation across sessions

# Decoder
k_folds = 5
n_trials_per_class = 20  # Minimum number of trials per class for a session to be included
cv = StratifiedKFold(n_splits=k_folds)
estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SGDClassifier(max_iter=10**6,
                          tol=1e-4,
                          penalty='l2',
                          alpha=alpha_value))  # Use the parsed alpha value
])

print(output_dir)

# Load behavioral-, session- and spike-data
trialData, sessionData, spikeData = utils.load_data(data_dir, experiment, animals)

# Filter sessions with hitrate < 0.3 for any modality
(trialData, sessionData, spikeData, _) = utils.filter_sessions_hitrate(trialData,
                                                                              sessionData=sessionData,
                                                                              spikeData=spikeData,
                                                                              min_hitrate=0.3)

# Manually exclude some sessions
trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
spikeData = spikeData[~spikeData['session_ID'].isin(sessions_exclude)]
sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]

# Only use good neurons from specified area(s)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=0.5, quality='good')
spikeData = spikeData[spikeData['area'].isin(area)]


# Cluster stimuli (orientation, pitch) for task-irrelevant stimulus information decoding
# Pseudopopulation level - this finds stimuli across sessions
stimulus_cluster_method = 'optimized_resample'

trialData.loc[:, 'visualOriPostChangeLeft_clust'] =\
    utils.cluster_samples(trialData, 'visualOriPostChangeLeft',
                          n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=45)

trialData.loc[:, 'visualOriPostChangeRight_clust'] =\
    utils.cluster_samples(trialData, 'visualOriPostChangeRight',
                          n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=45)

trialData.loc[:, 'audioOctPostChange_clust'] =\
    utils.cluster_samples(trialData, 'audioOctPostChange',
                          n_clusters=2, method=stimulus_cluster_method, modality='aud', bin_size=1/8)

# Session level - this finds stimuli within sessions to allow for better within-session decoding
stimulus_cluster_method_session = 'optimized_per_session'

trialData.loc[:, 'visualOriPostChangeLeft_clust_session'] =\
    utils.cluster_samples(trialData, 'visualOriPostChangeLeft',
                          n_clusters=2, method=stimulus_cluster_method_session, modality='vis', bin_size=15)

trialData.loc[:, 'visualOriPostChangeRight_clust_session'] =\
    utils.cluster_samples(trialData, 'visualOriPostChangeRight',
                          n_clusters=2, method=stimulus_cluster_method_session, modality='vis', bin_size=15)

trialData.loc[:, 'audioOctPostChange_clust_session'] =\
    utils.cluster_samples(trialData, 'audioOctPostChange',
                          n_clusters=2, method=stimulus_cluster_method_session, modality='aud', bin_size=1/24)

# Fix empty entries for no response
trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)

# Identify unilateral opto sessions
sessionData['unilat_opto'] = sessionData['optoarea'].apply(utils.check_only_rALM)

for contrast_set_name, contrast_set in [("opto off", analyses),
                                        ("opto on", analyses_opto),
                                        ("lick aligned", analyses_lick_aligned),
                                        ("lick aligned opto", analyses_lick_aligned_opto)]:
    print(contrast_set_name)

    # For opto analyses, only include unilateral (=rALM silenced) sessions
    if contrast_set_name in ["opto on", "lick aligned opto"]:
        sessions_opto = sessionData[sessionData['unilat_opto'] == True]['session_ID']
        sessionData_analysis = sessionData[sessionData['session_ID'].isin(sessions_opto)]
        trialData_analysis = trialData[trialData['session_ID'].isin(sessions_opto)]
        spikeData_analysis = spikeData[spikeData['session_ID'].isin(sessions_opto)]
    else:
        sessionData_analysis = sessionData
        trialData_analysis = trialData
        spikeData_analysis = spikeData

    # For lick-aligned analyses, exclude trials with RT < 100ms or no response
    if contrast_set_name in ["lick aligned", "lick aligned opto"]:
        trialData_analysis = trialData_analysis[trialData_analysis['noResponse'] == 0]
        trialData_analysis = trialData_analysis[trialData_analysis['reactionTime'] > (0.1*1e6)]

    # Define session inclusion based on n_trial per class
    inclusion = get_sessions_with_min_trials(trialData_analysis, contrast_set, min_trials_default=n_trials_per_class)
    metadata = {}

    # Initialize output dict
    output = {}
    output_ses = {}

    # Prepare output dictionaries
    for name in contrast_set:
        output['results_' + name] = []
        output['results_shuffled_' + name] = []

        output_ses['results_' + name] = []
        output_ses['results_shuffled_' + name] = []

    # Run each contrast (e.g. visual presence, hits)
    for contrast_idx, (name, conditions) in enumerate(contrast_set.items()):
        print(name)
        data_contrast = pd.DataFrame(columns=['session_ID', 'psth_ses', 'neuron_IDs', 'trial_labels'])
        sessions_contrast = inclusion[name]['session_IDs']  # Sessions to include
        label_col = conditions['decoder_variable']

        if len(sessions_contrast) == 0:
            print(f"Contrast {name} contains no sessions. Skipping.")
            continue

        trialData_contrast = filter_trialData(trialData_analysis, conditions)
        trialData_contrast = trialData_contrast.dropna(subset=label_col)
        spikeData_contrast = spikeData_analysis[spikeData_analysis['session_ID'].isin(sessions_contrast)]
        n_neurons_contrast = len(spikeData_contrast)

        metadata_contrast = {
            'name': name,
            'conditions': conditions,
            'n_neurons': n_neurons_contrast,
            'sessions': sessions_contrast,
            'n_sessions': len(sessions_contrast),
            'n_trials_per_class':  conditions['k_folds'] if 'k_folds' in conditions else n_trials_per_class,
            'n_neurons_per_session': spikeData_contrast['session_ID'].value_counts().values.tolist(),
            'n_neurons_min_ses': n_neurons_min_ses,
            'sessions_single_ses': []
        }

        print(f"{name} has {len(sessions_contrast)} sessions")

        if 'align_to' in conditions:
            align_to = conditions['align_to']
        else:
            align_to = 'stimChange'

        z_score_mode = 'custom_baseline' if align_to == 'firstLick' else None

        # Build PSTH matrices per session
        for ses in sessions_contrast:
            spikeData_ses = spikeData_contrast[spikeData_contrast['session_ID'] == ses]
            trialData_ses = trialData_contrast[trialData_contrast['session_ID'] == ses]

            if len(spikeData_ses) == 0:
                print(f"Session {ses} contains no neurons, skipping")
                continue

            if len(trialData_ses) == 0:
                print(f"Session {ses} contains no trials, skipping")
                continue

            # Calculate neuron PSTHs
            psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
                                         z_score_data=True, align_to=align_to, window=window, z_score_mode=z_score_mode,
                                         trialData_full=trialData_ses)  # neurons x trials x bins

            # Collect session data
            data_ses = {'session_ID': ses,
                        'psth_ses': [psth_ses],
                        'neuron_IDs': [spikeData_ses['cluster_ID']],
                        'trial_labels': [trialData_ses[label_col]],
                        'trial_labels_reset': [trialData_ses[label_col].reset_index(drop=True)]}

            # Concatenate data
            data_contrast = pd.concat([data_contrast, pd.DataFrame(data_ses)], ignore_index=True)

        # Build lookup table
        lookup = data_contrast.explode('neuron_IDs').set_index('neuron_IDs')['session_ID']

        # Cross-validation setup
        if 'k_folds' in conditions:
            cv = StratifiedKFold(n_splits=conditions['k_folds'])
        else:
            cv = StratifiedKFold(n_splits=k_folds)

        # SESSION-LEVEL - Run decoder separately for every session
        if session_level_analysis:
            n_trials_per_class_single_ses = []

            # Filter data_contrast to exclude session with too few neurons
            data_contrast_ses = data_contrast[data_contrast['neuron_IDs'].apply(len) >= n_neurons_min_ses]
            metadata_contrast['sessions_single_ses'] = data_contrast_ses['session_ID'].tolist()

            for _, ses in data_contrast_ses.iterrows():
                # Per session, find the class with the fewest trials, and balance other classes, instead of a flat number
                n_trials_per_class_ses = np.min(ses['trial_labels'].value_counts())
                n_trials_per_class_single_ses.append(int(n_trials_per_class_ses))

                matrix_session, all_sampled_neurons, labels, labels_shuffled = (
                    sample_data(ses, lookup, label_col,
                                n_trials_per_class=n_trials_per_class_ses, sample_mode="all", n_neurons=None))

                for idx_bin, bin in enumerate(bins):
                    bin_matrix = matrix_session[:, :, idx_bin].T

                    results_bin = cross_validate(
                        estimator=estimator,
                        X=bin_matrix,
                        y=labels,
                        cv=cv,
                        scoring="balanced_accuracy",
                        n_jobs=-1
                    )

                    results_bin_shuffled = cross_validate(
                        estimator=estimator,
                        X=bin_matrix,
                        y=labels_shuffled,
                        cv=cv,
                        scoring="balanced_accuracy",
                        n_jobs=-1
                    )

                    # print(results_bin)
                    # print(results_bin_shuffled)

                    result_id = {'bin': bin, 'session_ID': ses['session_ID']}
                    output_ses['results_' + name].append({**result_id, 'scores': results_bin})
                    output_ses['results_shuffled_' + name].append({**result_id, 'scores': results_bin_shuffled})

            metadata_contrast['n_trials_per_class_single_ses'] = n_trials_per_class_single_ses

        # PSEUDOPOPULATION - Run n_iterations of the decoder using a randomly sampled pseudopopulation
        if pseudopopulation_analysis:
            n_splits = conditions.get('k_folds', k_folds)

            seed_seq = np.random.SeedSequence(42 + contrast_idx)  # see Issue 2
            seeds = seed_seq.spawn(n_iterations)

            batch_size = 10
            batches = [seeds[i:i + batch_size] for i in range(0, n_iterations, batch_size)]

            print(f"    Running {n_iterations} iterations in {len(batches)} batches "
                  f"(batch_size={batch_size}, n_jobs=-1)...")

            batch_results = Parallel(n_jobs=-1, backend='loky', verbose=5)(
                delayed(_batch_worker)(
                    batch_seeds, data_contrast, lookup, label_col,
                    n_trials_per_class, bins, n_splits, estimator,
                )
                for batch_seeds in batches
            )

            iter_results = [item for batch in batch_results for item in batch]

            for iteration, (real_list, shuf_list) in enumerate(iter_results):
                for r in real_list:
                    output['results_' + name].append({
                        'bin': r['bin'], 'iteration': iteration, 'scores': r['scores'],
                    })
                for r in shuf_list:
                    output['results_shuffled_' + name].append({
                        'bin': r['bin'], 'iteration': iteration, 'scores': r['scores'],
                    })
        # Add metadata
        metadata[name] = metadata_contrast

    save_dir = os.path.join(output_dir, contrast_set_name)

    # Ensure directories
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(os.path.join(save_dir, 'pseudopop')):
        os.makedirs(os.path.join(save_dir, 'pseudopop'))
    if not os.path.isdir(os.path.join(save_dir, 'session_level')):
        os.makedirs(os.path.join(save_dir, 'session_level'))

    # PSEUDOPOPULATION RESULTS
    # Save results as npy file
    np.save(os.path.join(save_dir, 'pseudopop', 'output.npy'), output)

    # SESSION-LEVEL RESULTS
    # Save results as npy file
    np.save(os.path.join(save_dir, 'session_level', 'output.npy'), output_ses)

    # METADATA
    # Save this script
    shutil.copy(__file__, os.path.join(save_dir, 'script.py'))
    # Save metadata as .yaml
    with open(os.path.join(save_dir, 'metadata.yaml'), 'w') as file:
        yaml.dump(metadata, file)