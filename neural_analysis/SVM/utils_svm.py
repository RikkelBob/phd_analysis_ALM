import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from itertools import combinations
from typing import List, Union, Tuple
from scipy.stats import wilcoxon, ttest_rel, false_discovery_control
from mne.stats import permutation_cluster_test, combine_adjacency, permutation_cluster_1samp_test
from pingouin import ttest
import scipy
from scipy.ndimage import label as ndi_label

def cluster_samples(samples, n_clusters=4, method='gmm', normalize=False):
    """
    Clusters orientations (in degrees) into `n_clusters` using the specified method.

    Parameters:
        samples: pd.Series or np.ndarray of degrees [0-360)
        n_clusters: number of clusters
        method: 'gmm', 'kmeans', or 'quantile'

    Returns:
        labels: np.ndarray of cluster labels (0-based)
    """
    #samples = samples.dropna().values if isinstance(samples, pd.Series) else samples

    if normalize:
        samples = (samples - 13) / (14 - 13) * 2 * np.pi
        X_circle = np.vstack((np.cos(samples), np.sin(samples))).T
    else:
        angles_rad = np.deg2rad(samples % 360)
        X_circle = np.vstack((np.cos(angles_rad), np.sin(angles_rad))).T

    if method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        labels = model.fit_predict(X_circle)

    elif method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = model.fit_predict(X_circle)

    elif method == 'quantile':
        angles_deg = np.rad2deg(angles_rad % (2 * np.pi))
        quantiles = np.percentile(angles_deg, np.linspace(0, 100, n_clusters + 1))
        labels = pd.cut(angles_deg, bins=quantiles, labels=False, include_lowest=True)
        labels = labels.to_numpy()

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels

def sessions_valid_clusters(trialData, label_col, n_min_trials):
    """

    Args:
        trialData:
        label_col: column containing labels
        n_min_trials: minimum number of trials per label

    Returns:
        valid_sessions: list of session_IDs for sessions that have enough trials for all labels
    """
    valid_sessions = []

    # Get all unique cluster values in the dataset
    all_cluster_values = trialData[label_col].unique()

    # Check each session
    for session_id in trialData['session_ID'].unique():
        trialData_ses = trialData[trialData['session_ID'] == session_id]

        # Count trials per cluster value in this session
        cluster_counts = trialData_ses[label_col].value_counts()

        # Check if this session has all cluster values
        has_all_clusters = set(cluster_counts.index) == set(all_cluster_values)

        # Check if all cluster values have enough trials
        all_meet_threshold = (cluster_counts >= n_min_trials).all()

        if has_all_clusters and all_meet_threshold:
            valid_sessions.append(session_id)

    return valid_sessions


def sample_exhaustive(arr, size, rng=None):
    """Sample without replacement until exhausted, then with replacement"""
    if rng is None:
        rng = np.random.default_rng()

    arr = np.array(arr)

    if size <= len(arr):
        # Can sample without replacement
        return rng.choice(arr, size=size, replace=False)
    else:
        # Need to exhaust without replacement, then continue with replacement
        without_replacement = rng.choice(arr, size=len(arr), replace=False)
        remaining_size = size - len(arr)
        with_replacement = rng.choice(arr, size=remaining_size, replace=True)
        return np.concatenate([without_replacement, with_replacement])


def find_sessions_with_criteria_vectorized(df: pd.DataFrame,
                                           column_name: str,
                                           n_stimuli: int,
                                           x_times: int,
                                           bin_size: Union[int, float, None] = None) -> Tuple[
    List[Union[str, int]], pd.DataFrame]:
    """
    Vectorized version of the session filtering function for better performance.
    Finds sessions where at least n_stimuli of the SAME stimuli each appear at least x_times.
    If no common stimuli exist across all sessions, finds the stimulus set that includes
    the maximum number of sessions.

    Parameters and returns are the same as the basic version.
    """

    if 'session_ID' not in df.columns:
        raise ValueError("DataFrame must contain 'session_ID' column")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Create a working copy and apply binning if specified
    working_df = df.copy()

    if bin_size is not None:
        bin_column = f"{column_name}_bin"

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(working_df[column_name]):
            # Numeric binning (now compatible with floats)
            min_val = working_df[column_name].min()
            max_val = working_df[column_name].max()

            # Calculate the number of bins needed
            bin_start = (min_val // bin_size) * bin_size
            bin_end = ((max_val // bin_size) + 1) * bin_size

            # Use numpy arange for float compatibility
            import numpy as np
            bins = np.arange(bin_start, bin_end + bin_size, bin_size)

            working_df[bin_column] = pd.cut(working_df[column_name],
                                            bins=bins,
                                            right=False,
                                            include_lowest=True)
            working_df[bin_column] = working_df[bin_column].astype(str)
        else:
            # Categorical binning
            unique_values = sorted(working_df[column_name].unique())
            bin_mapping = {}

            for i, value in enumerate(unique_values):
                bin_number = i // int(bin_size)
                bin_start = bin_number * int(bin_size)
                bin_end = min(bin_start + int(bin_size) - 1, len(unique_values) - 1)
                bin_label = f"bin_{bin_start}-{bin_end}"
                bin_mapping[value] = bin_label

            working_df[bin_column] = working_df[column_name].map(bin_mapping)

        analysis_column = bin_column
    else:
        analysis_column = column_name

    # Count value occurrences within each session
    value_counts_by_session = (working_df.groupby(['session_ID', analysis_column])
                               .size()
                               .reset_index(name='count'))

    # Find stimuli that appear at least x_times in each session
    qualifying_stimuli_by_session = value_counts_by_session[value_counts_by_session['count'] >= x_times]

    # Count how many qualifying stimuli each session has
    stimuli_count_by_session = qualifying_stimuli_by_session.groupby('session_ID').size()

    # Find sessions with at least n_stimuli qualifying stimuli
    potential_sessions = stimuli_count_by_session[stimuli_count_by_session >= n_stimuli].index.tolist()

    if not potential_sessions:
        return [], pd.DataFrame()

    # Create a mapping of session to qualifying stimuli
    session_stimulus_sets = {}
    for session in potential_sessions:
        session_stimuli = set(qualifying_stimuli_by_session[
                                  qualifying_stimuli_by_session['session_ID'] == session
                                  ][analysis_column].tolist())
        session_stimulus_sets[session] = session_stimuli

    # Find common stimuli across all potential sessions
    common_stimuli = set.intersection(*session_stimulus_sets.values())

    # Check if we have enough common stimuli across ALL sessions
    if len(common_stimuli) >= n_stimuli:
        # Original behavior: all sessions share enough common stimuli
        final_qualifying_sessions = potential_sessions
        selected_stimuli = common_stimuli
    else:
        # New behavior: find the stimulus set that maximizes the number of sessions

        # Collect all unique stimuli across all sessions
        all_stimuli = set()
        for stimuli_set in session_stimulus_sets.values():
            all_stimuli.update(stimuli_set)

        # For each combination of n_stimuli, count how many sessions have all of them
        best_stimuli_set = None
        best_session_count = 0
        best_sessions = []

        # Generate all possible combinations of n_stimuli from all available stimuli
        for stimulus_combination in combinations(all_stimuli, n_stimuli):
            stimulus_set = set(stimulus_combination)

            # Count sessions that have all stimuli in this combination
            matching_sessions = []
            for session, session_stimuli in session_stimulus_sets.items():
                if stimulus_set.issubset(session_stimuli):
                    matching_sessions.append(session)

            # Update best if this combination covers more sessions
            if len(matching_sessions) > best_session_count:
                best_session_count = len(matching_sessions)
                best_stimuli_set = stimulus_set
                best_sessions = matching_sessions

        # If we found no valid combination (shouldn't happen if potential_sessions exists)
        if best_stimuli_set is None:
            return [], pd.DataFrame()

        final_qualifying_sessions = best_sessions
        selected_stimuli = best_stimuli_set
        print(selected_stimuli)

    # Filter to only rows from qualifying sessions with selected stimuli
    mask = (working_df['session_ID'].isin(final_qualifying_sessions)) & \
           (working_df[analysis_column].isin(selected_stimuli))
    qualifying_data = working_df[mask].copy()

    # Add stimulus labels (base 1) - exactly n_stimuli unique labels
    selected_stimuli_list = sorted(list(selected_stimuli))
    stimulus_to_label = {stimulus: i + 1 for i, stimulus in enumerate(selected_stimuli_list)}
    qualifying_data['stimulus_label'] = qualifying_data[analysis_column].map(stimulus_to_label)

    return final_qualifying_sessions, qualifying_data

def extract_condition_df(output, condition_key, it='iteration'):
    """
    Extract decoding results for a specific condition from loaded output dict.

    Parameters:
        output (dict): Loaded from np.load(..., allow_pickle=True).item()
        condition_key (str): Key for decoding results, e.g., 'results_aud'

    Returns:
        pd.DataFrame: Columns include 'bin_start', 'bin_end', 'iteration', 'mean_score'
    """

    if condition_key not in output:
        return None

    results = output[condition_key]
    data = []

    for entry in results:
        bin_window = tuple(entry['bin'])  # (start, end)
        ID = entry[it]
        scores = entry['scores']['test_score']
        mean_score = np.mean(scores)

        data.append({
            'bin_start': bin_window[0],
            'bin_end': bin_window[1],
            'ID': ID,
            'mean_score': mean_score,
            'all_scores': scores
        })

    return pd.DataFrame(data)

def pct_against_chance(data, shuffled_data, n_conditions=2, n_permutation=10000):
    #threshold = {'start': 0, 'step': 0.02}  # Start at 0, step by 0.2
    pval = 0.05  # arbitrary
    dfn = n_conditions - 1  # degrees of freedom numerator
    n_observations = data.shape[0]
    dfd = n_observations - n_conditions  # degrees of freedom denominator
    thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
    print(thresh)
    n_times = data.shape[-1]
    temporal_adj = combine_adjacency(n_times)

    diff = data - shuffled_data
    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
        diff,
        n_permutations=10000,
        tail=1,
        n_jobs=-1
    )
    #f_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
    #    [data, shuffled_data],  # List with single condition
    #    n_permutations=n_permutation,
    #    tail=1,  # One-tailed (accuracy > chance)
    #    n_jobs=-1,  # Use all CPUs
    #    seed=None,
    #    threshold=thresh,
    #    adjacency=None
    #)

    return t_obs, clusters, cluster_p_values, h0


def SVM_stats(data_array, baseline_indices=None, data_array_test=None,
              stat_test='ttest', fdr_test='bh', test_against='baseline', sig_thr=None):

    if test_against == 'data' and data_array_test is None:
        print("Testing against another condition requires data_array_test")
        return

    if test_against == 'baseline' and baseline_indices is None:
        print("Testing against baseline requires baseline_indices")
        return

    if sig_thr is None:
        if stat_test == 'bf':
            sig_thr = 3
        else:  # assuming p_values (incl. fraction)
            sig_thr = 0.005

    if test_against == 'baseline':
        baseline_values = data_array[:, baseline_indices]
        baseline_values_mean = np.mean(baseline_values, axis=1).reshape(-1, 1)  # mean over bins per session (size n_sessions)
        baseline_repeated = np.tile(baseline_values_mean, (1, data_array.shape[1]))  # repeat for every bin to make testing easier
        data_array_test = baseline_repeated
    elif test_against == 'data':
        data_array_test = data_array_test
    else:
        print(f"{test_against} is not valid")
        return

    # Stat test
    if stat_test == 'fraction':
        # Per-bin p-value: fraction of iterations where data_array <= data_array_test.
        # +1 / +1 keeps p > 0 even when all iterations are above the comparison.
        n_iter = data_array.shape[0]
        p_values = (np.sum(data_array <= data_array_test, axis=0) + 1) / (n_iter + 1)
        p_values = list(p_values)  # match the list type used by other branches

    elif stat_test != 'pct':
        p_values = []

        if stat_test == 'wilcoxon':
            w_stats = []
            for bin in range(data_array.shape[-1]):
                w_stat, p_value = wilcoxon(data_array[:, bin], data_array_test[:, bin])
                w_stats.append(w_stat)
                p_values.append(p_value)
        elif stat_test == 'ttest':
            t_stats = []
            for bin in range(data_array.shape[-1]):
                t_statistic, p_value = ttest_rel(data_array[:, bin], data_array_test[:, bin])
                t_stats.append(t_statistic)
                p_values.append(p_value)
        elif stat_test == 'bf':
            bf_list = []
            for bin in range(data_array.shape[-1]):
                bf_stats_bin = ttest(data_array[:, bin], data_array_test[:, bin], paired=True, alternative='two-sided',
                                     correction='auto', confidence=0.95, r=1)
                bf_list.append(bf_stats_bin)
            bf_stats = pd.concat(bf_list)
            test_scores = bf_stats['BF10'].values.astype(float)

    # Multiple comparison correction: cluster-based permutation testing or bh FDR
    if stat_test != 'bf':
        if stat_test == 'pct':
            t_obs, clusters, cluster_p_values, h0 = pct_against_chance(data_array, data_array_test)

            significance = np.zeros(data_array.shape[1], dtype=bool)
            test_scores = np.zeros(data_array.shape[1], dtype=float)
            for cluster, p_val in zip(clusters, cluster_p_values):
                test_scores[cluster[0]] = p_val
                if p_val < 0.05:
                    significance[cluster[0]] = True

        elif fdr_test == 'bh':
            p_values_adjusted = false_discovery_control(p_values)
            test_scores = p_values_adjusted

    if stat_test == 'bf':
        significance = test_scores > sig_thr
    else:
        significance = test_scores < sig_thr

    return test_scores, significance


# ============================================================================
# CLUSTER-BASED PERMUTATION TEST (Maris & Oostenveld 2007)
# ============================================================================
def _build_cluster_time_mask(time_centers, cluster_window):
    if cluster_window is None: return np.ones(len(time_centers), dtype=bool)
    start, end = cluster_window
    mask = np.ones(len(time_centers), dtype=bool)
    if start is not None: mask &= (time_centers >= start)
    if end is not None: mask &= (time_centers <= end)
    return mask


def _find_clusters_1d(trace_windowed, threshold_windowed):
    above = trace_windowed > threshold_windowed
    if not np.any(above): return []
    labeled, n_clusters = ndi_label(above)
    clusters = []
    for k in range(1, n_clusters + 1):
        mask = labeled == k; idx = np.where(mask)[0]
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
    threshold_windowed = np.nanpercentile(null_distribution[:, win_mask], threshold_percentile, axis=0)
    threshold_full[win_mask] = threshold_windowed
    obs_clusters_win = _find_clusters_1d(observed[win_mask], threshold_windowed)
    max_null_masses = np.zeros(n_perms)
    for p in range(n_perms):
        clusters_p = _find_clusters_1d(null_distribution[p, win_mask], threshold_windowed)
        if clusters_p: max_null_masses[p] = max(c[2] for c in clusters_p)
    observed_clusters, cluster_mask, p_map = [], np.zeros(n_timebins, dtype=bool), np.full(n_timebins, np.nan)
    for (s_win, e_win, mass) in obs_clusters_win:
        p = (np.sum(max_null_masses >= mass) + 1) / (n_perms + 1)
        s_full, e_full = int(win_idx[s_win]), int(win_idx[e_win])
        observed_clusters.append({'start_idx': s_full, 'end_idx': e_full,
            't_start': float(time_centers[s_full]), 't_end': float(time_centers[e_full]),
            'mass': float(mass), 'p_value': float(p)})
        if p < p_alpha: cluster_mask[s_full:e_full+1] = True; p_map[s_full:e_full+1] = p
    return {'observed_clusters': observed_clusters, 'max_null_masses': max_null_masses,
            'threshold_full': threshold_full, 'cluster_mask': cluster_mask, 'p_map': p_map,
            'cluster_window': cluster_window if cluster_window is not None else (None, None),
            'threshold_percentile': float(threshold_percentile)}
