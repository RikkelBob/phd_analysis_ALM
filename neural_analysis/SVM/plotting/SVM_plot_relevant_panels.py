import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_analysis.SVM.utils_svm import SVM_stats, cluster_permutation_test
from neural_analysis.plot_utils import add_sig_lines_ax, add_opto_shading
from neural_analysis.plot_utils import paper_colors, add_t0_line
import matplotlib
import traceback
from statsmodels.stats.multitest import multipletests
import config

# ============================================================================
# Permutation tests against the shuffle distribution
# ============================================================================
def perm_test_vs_shuffle(data_real, data_shuffle, fdr_alpha=0.05, sig_thr_p=None):
    """Per-bin one-sided permutation p-values: P(shuffle >= real_mean).

    Parameters
    ----------
    data_real : np.ndarray, shape (n_iter, n_bins)
        Real decoding scores per iteration and time bin.
    data_shuffle : np.ndarray, shape (n_iter_shuf, n_bins)
        Shuffled decoding scores.
    fdr_alpha : float or None
        If set, apply Benjamini-Hochberg FDR correction at this level.
    sig_thr_p : float or None
        If set, use a fixed p-value threshold instead of FDR.

    Returns
    -------
    p_per_bin : np.ndarray, shape (n_bins,)
    significance : np.ndarray of bool, shape (n_bins,)
    """
    real_mean = data_real.mean(axis=0)
    n_shuf = data_shuffle.shape[0]
    p_per_bin = (np.sum(data_shuffle >= real_mean[None, :], axis=0) + 1) / (n_shuf + 1)

    if sig_thr_p is not None:
        significance = p_per_bin < sig_thr_p
    elif fdr_alpha is not None:
        significance, _, _, _ = multipletests(p_per_bin, alpha=fdr_alpha, method='fdr_bh')
    else:
        significance = p_per_bin < 0.05

    return p_per_bin, significance


def cluster_test_vs_shuffle(data_real, data_shuffle, timeline,
                            cluster_window=None, threshold_pctile=95.0, p_alpha=0.05):
    """Cluster-based permutation test on real vs shuffle decoding scores.

    Wraps `cluster_permutation_test` from utils_svm so the returned shape
    matches what perm_test_vs_shuffle produces.

    Returns
    -------
    significance : np.ndarray of bool, shape (n_bins,)
        Cluster mask (True = bin in a significant cluster).
    cluster_result : dict
        Full output from cluster_permutation_test.
    """
    observed = data_real.mean(axis=0)
    timeline = np.asarray(timeline)
    cr = cluster_permutation_test(
        observed, data_shuffle, timeline,
        cluster_window=cluster_window,
        threshold_percentile=threshold_pctile,
        p_alpha=p_alpha,
    )
    return cr['cluster_mask'], cr


def mean_vs_distribution_test(data_ref, data_test, timeline,
                              direction='less', stat_method='cluster',
                              cluster_window=None, threshold_pctile=5.0,
                              p_alpha=0.05, fdr_alpha=0.05, sig_thr_p=None):
    """One-sided mean-vs-distribution permutation test.

    Tests whether the mean of `data_test` falls in the tail of the reference
    distribution `data_ref`. This is the same machinery as a real-vs-shuffle
    test, but with one real condition acting as the reference.

    Designed for the question: *"Did the perturbation push accuracy outside
    what we'd expect from the baseline iteration distribution?"*

    Robust to n_iter: as n_iter grows, the mean of `data_test` converges to
    its true value, while the spread of `data_ref` stabilizes at the
    pseudopopulation-sampling variability (data-driven, not iteration-driven).
    So p-values stay meaningful rather than asymptotically going to 0.

    Parameters
    ----------
    data_ref : np.ndarray, shape (n_iter_ref, n_bins)
        Reference distribution (e.g. opto off iterations).
    data_test : np.ndarray, shape (n_iter_test, n_bins)
        Distribution whose mean is being tested (e.g. opto on iterations).
    direction : {'less', 'greater', 'two_sided'}
        Tail of data_ref to test against. 'less' = test_mean below ref bulk
        (e.g. did opto reduce decoding?). 'greater' = test_mean above ref bulk.
    stat_method : {'cluster', 'perm_vs_shuffle'}
        'cluster' = cluster-mass permutation test (recommended).
        'perm_vs_shuffle' = per-bin p-values with FDR correction.
    threshold_pctile : float
        Cluster-forming percentile of `data_ref`. For direction='less',
        this is the *lower* percentile (e.g. 5.0). For direction='greater',
        this is the *upper* percentile (e.g. 95.0). 'two_sided' uses both.
    cluster_window : (t_min, t_max) or None
        Restrict clusters to this time window.

    Returns
    -------
    significance : np.ndarray of bool, shape (n_bins,)
    test_scores : dict
        Diagnostic info (observed_mean, threshold, cluster pvalues, etc.).
    """
    data_ref = np.asarray(data_ref)
    data_test = np.asarray(data_test)
    timeline = np.asarray(timeline)

    if data_ref.shape[1] != data_test.shape[1]:
        raise ValueError(
            f"Bin counts differ: data_ref has {data_ref.shape[1]} bins, "
            f"data_test has {data_test.shape[1]}"
        )

    test_mean = data_test.mean(axis=0)

    # Compute cluster-forming threshold(s) from the reference distribution.
    if direction == 'less':
        thr_lo = np.percentile(data_ref, threshold_pctile, axis=0)
        thr_hi = None
    elif direction == 'greater':
        thr_lo = None
        thr_hi = np.percentile(data_ref, 100 - threshold_pctile, axis=0)
    elif direction == 'two_sided':
        thr_lo = np.percentile(data_ref, threshold_pctile / 2.0, axis=0)
        thr_hi = np.percentile(data_ref, 100 - threshold_pctile / 2.0, axis=0)
    else:
        raise ValueError(f"Unknown direction {direction!r}")

    # Per-bin "extremeness" of a trace relative to the reference distribution.
    def _extreme_mask(trace):
        if direction == 'less':
            return trace < thr_lo
        elif direction == 'greater':
            return trace > thr_hi
        else:  # two_sided
            return (trace < thr_lo) | (trace > thr_hi)

    # Statistic magnitude per bin (signed distance from threshold,
    # zeroed outside the tail). Used to build cluster masses.
    def _stat(trace):
        s = np.zeros_like(trace)
        if direction in ('less', 'two_sided'):
            below = trace < thr_lo
            s[below] = thr_lo[below] - trace[below]  # positive when below
        if direction in ('greater', 'two_sided'):
            above = trace > thr_hi
            s[above] = trace[above] - thr_hi[above]  # positive when above
        return s

    if stat_method == 'perm_vs_shuffle':
        # Per-bin one-sided permutation p-values from the reference distribution.
        n_ref = data_ref.shape[0]
        if direction == 'less':
            p_per_bin = (np.sum(data_ref <= test_mean[None, :], axis=0) + 1) / (n_ref + 1)
        elif direction == 'greater':
            p_per_bin = (np.sum(data_ref >= test_mean[None, :], axis=0) + 1) / (n_ref + 1)
        else:  # two_sided
            p_lo = (np.sum(data_ref <= test_mean[None, :], axis=0) + 1) / (n_ref + 1)
            p_hi = (np.sum(data_ref >= test_mean[None, :], axis=0) + 1) / (n_ref + 1)
            p_per_bin = 2.0 * np.minimum(p_lo, p_hi)
            p_per_bin = np.minimum(p_per_bin, 1.0)

        if sig_thr_p is not None:
            significance = p_per_bin < sig_thr_p
        elif fdr_alpha is not None:
            significance, _, _, _ = multipletests(p_per_bin, alpha=fdr_alpha, method='fdr_bh')
        else:
            significance = p_per_bin < 0.05

        return significance, {
            'p_per_bin': p_per_bin,
            'test_mean': test_mean,
            'threshold_low': thr_lo,
            'threshold_high': thr_hi,
            'direction': direction,
        }

    elif stat_method == 'cluster':
        # Cluster-mass permutation test.
        # Null distribution: treat each reference iteration as a hypothetical
        # "test mean" against the rest of the reference distribution.

        def _find_clusters(trace, t):
            """Return list of (mass, indices) for runs where trace is in the tail."""
            mask = _extreme_mask(trace)
            stat = _stat(trace)
            clusters = []
            in_cluster = False
            start = 0
            for j, a in enumerate(mask):
                if a and not in_cluster:
                    start = j
                    in_cluster = True
                elif not a and in_cluster:
                    if cluster_window is None or (
                        t[start] >= cluster_window[0] and t[j - 1] <= cluster_window[1]
                    ):
                        idxs = np.arange(start, j)
                        mass = stat[idxs].sum()
                        clusters.append((mass, idxs))
                    in_cluster = False
            if in_cluster:
                if cluster_window is None or (
                    t[start] >= cluster_window[0] and t[-1] <= cluster_window[1]
                ):
                    idxs = np.arange(start, len(mask))
                    mass = stat[idxs].sum()
                    clusters.append((mass, idxs))
            return clusters

        # Observed clusters in the test mean trace
        obs_clusters = _find_clusters(test_mean, timeline)

        # Null distribution: each ref iteration treated as a candidate test trace.
        n_ref = data_ref.shape[0]
        max_null_mass = np.zeros(n_ref)
        for i in range(n_ref):
            null_trace = data_ref[i]
            null_clusters = _find_clusters(null_trace, timeline)
            if null_clusters:
                max_null_mass[i] = max(c[0] for c in null_clusters)

        # p-value per observed cluster
        cluster_mask = np.zeros_like(test_mean, dtype=bool)
        cluster_pvals = []
        for mass, idxs in obs_clusters:
            p = (np.sum(max_null_mass >= mass) + 1) / (n_ref + 1)
            cluster_pvals.append(p)
            if p < p_alpha:
                cluster_mask[idxs] = True

        return cluster_mask, {
            'cluster_mask': cluster_mask,
            'test_mean': test_mean,
            'threshold_low': thr_lo,
            'threshold_high': thr_hi,
            'cluster_pvalues': cluster_pvals,
            'clusters': obs_clusters,
            'max_null_mass': max_null_mass,
            'direction': direction,
        }

    else:
        raise ValueError(f"Unknown stat_method {stat_method!r}")


def between_condition_test(data_a, data_b, timeline, stat_method='cluster',
                           fdr_alpha=0.05, sig_thr_p=None,
                           cluster_window=None, threshold_pctile=95.0,
                           p_alpha=0.05, n_perm=10000, two_sided=True,
                           seed=None):
    """Permutation test comparing two real conditions (e.g. opto off vs opto on).

    Implements a permutation test where the labels (condition A vs B) are
    shuffled across iterations. Supports either a cluster-based test or a
    per-bin permutation test with FDR correction.

    Parameters
    ----------
    data_a, data_b : np.ndarray, shape (n_iter, n_bins)
        Decoding scores from each condition.
    timeline : array-like
        Bin centers / edges (length n_bins). Used by the cluster test.
    stat_method : {'cluster', 'perm_vs_shuffle'}
        Which test to run. 'perm_vs_shuffle' here means "per-bin permutation
        with FDR" — same idea as for shuffle, but with labels permuted across
        the pooled (A, B) distribution.
    two_sided : bool
        Per-bin test only. Cluster test is one-sided on |t|-like statistics.

    Returns
    -------
    significance : np.ndarray of bool, shape (n_bins,)
    test_scores : dict or np.ndarray
        Per-bin p-values for 'perm_vs_shuffle', or cluster_result dict for 'cluster'.
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    timeline = np.asarray(timeline)

    if data_a.shape[1] != data_b.shape[1]:
        raise ValueError(
            f"Bin counts differ: data_a has {data_a.shape[1]} bins, "
            f"data_b has {data_b.shape[1]}"
        )

    rng = np.random.default_rng(seed)

    if stat_method == 'cluster':
        # Build a null distribution of cluster masses by permuting labels.
        # Observed test statistic = mean(A) - mean(B) per bin.
        obs_diff = data_a.mean(axis=0) - data_b.mean(axis=0)

        n_a = data_a.shape[0]
        pooled = np.concatenate([data_a, data_b], axis=0)
        n_total = pooled.shape[0]

        # Build null distribution of permuted mean-differences.
        null_diffs = np.empty((n_perm, data_a.shape[1]))
        for i in range(n_perm):
            perm_idx = rng.permutation(n_total)
            a_perm = pooled[perm_idx[:n_a]]
            b_perm = pooled[perm_idx[n_a:]]
            null_diffs[i] = a_perm.mean(axis=0) - b_perm.mean(axis=0)

        # Cluster-forming threshold from the null distribution of |diffs|.
        thresh = np.percentile(np.abs(null_diffs), threshold_pctile)

        def _find_clusters(diff_arr, thr, t):
            """Return list of (mass, indices) for supra-threshold runs."""
            above = np.abs(diff_arr) > thr
            clusters = []
            in_cluster = False
            start = 0
            for j, a in enumerate(above):
                if a and not in_cluster:
                    start = j
                    in_cluster = True
                elif not a and in_cluster:
                    if cluster_window is None or (
                        t[start] >= cluster_window[0] and t[j - 1] <= cluster_window[1]
                    ):
                        idxs = np.arange(start, j)
                        mass = np.abs(diff_arr[idxs]).sum()
                        clusters.append((mass, idxs))
                    in_cluster = False
            if in_cluster:
                if cluster_window is None or (
                    t[start] >= cluster_window[0] and t[-1] <= cluster_window[1]
                ):
                    idxs = np.arange(start, len(above))
                    mass = np.abs(diff_arr[idxs]).sum()
                    clusters.append((mass, idxs))
            return clusters

        # Observed clusters
        obs_clusters = _find_clusters(obs_diff, thresh, timeline)

        # Null distribution of max cluster mass
        max_null_mass = np.zeros(n_perm)
        for i in range(n_perm):
            null_clusters = _find_clusters(null_diffs[i], thresh, timeline)
            if null_clusters:
                max_null_mass[i] = max(c[0] for c in null_clusters)

        # p-value per observed cluster
        cluster_mask = np.zeros_like(obs_diff, dtype=bool)
        cluster_pvals = []
        for mass, idxs in obs_clusters:
            p = (np.sum(max_null_mass >= mass) + 1) / (n_perm + 1)
            cluster_pvals.append(p)
            if p < p_alpha:
                cluster_mask[idxs] = True

        return cluster_mask, {
            'cluster_mask': cluster_mask,
            'observed_diff': obs_diff,
            'threshold': thresh,
            'cluster_pvalues': cluster_pvals,
            'clusters': obs_clusters,
            'max_null_mass': max_null_mass,
        }

    elif stat_method == 'perm_vs_shuffle':
        # Per-bin permutation: P(|null_diff| >= |obs_diff|) under label-shuffle
        obs_diff = data_a.mean(axis=0) - data_b.mean(axis=0)
        n_a = data_a.shape[0]
        pooled = np.concatenate([data_a, data_b], axis=0)
        n_total = pooled.shape[0]

        null_diffs = np.empty((n_perm, data_a.shape[1]))
        for i in range(n_perm):
            perm_idx = rng.permutation(n_total)
            a_perm = pooled[perm_idx[:n_a]]
            b_perm = pooled[perm_idx[n_a:]]
            null_diffs[i] = a_perm.mean(axis=0) - b_perm.mean(axis=0)

        if two_sided:
            p_per_bin = (np.sum(np.abs(null_diffs) >= np.abs(obs_diff)[None, :], axis=0)
                         + 1) / (n_perm + 1)
        else:
            p_per_bin = (np.sum(null_diffs >= obs_diff[None, :], axis=0)
                         + 1) / (n_perm + 1)

        if sig_thr_p is not None:
            significance = p_per_bin < sig_thr_p
        elif fdr_alpha is not None:
            significance, _, _, _ = multipletests(p_per_bin, alpha=fdr_alpha, method='fdr_bh')
        else:
            significance = p_per_bin < 0.05

        return significance, p_per_bin

    else:
        raise ValueError(f"Unknown stat_method {stat_method!r} for between_condition_test")


def bootstrap_ci(scores, n_bootstrap=10000, confidence=0.95):
    n = len(scores)
    if n < 2:
        return (np.nan, np.nan)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (ci_lower, ci_upper)


def extract_condition_df(output, condition_key, it='iteration'):
    """
    Extract decoding results for a specific condition from loaded output dict.
    Returns None on any failure or if the condition is missing/empty.
    """
    if output is None:
        return None
    if not isinstance(output, dict):
        return None
    if condition_key not in output:
        return None

    try:
        results = output[condition_key]
        if results is None or len(results) == 0:
            return None

        data = []
        for entry in results:
            try:
                bin_window = tuple(entry['bin'])
                iteration = entry[it]
                scores = entry['scores']['test_score']
                mean_score = np.mean(scores)

                data.append({
                    'bin_start': bin_window[0],
                    'bin_end': bin_window[1],
                    'iteration': iteration,
                    'mean_score': mean_score,
                    'all_scores': scores
                })
            except (KeyError, TypeError, ValueError) as e:
                # Skip malformed entries
                continue

        if len(data) == 0:
            return None

        return pd.DataFrame(data)
    except Exception as e:
        print(f"    extract_condition_df failed for {condition_key}: {e}")
        return None


def plot_decoding_over_time(mean_scores, error_scores, bins, color=None,
                            label='results', ax=None, error_type='std', set_label_x=True,
                            linewidth=1.5, alpha=0.9, linestyle='-'):
    """
    Plot decoding performance with a shaded error band.

    error_scores formats:
        - 'std', 'sem'   : 1D array of half-widths (band = mean ± err)
        - 'percentile'   : (low_array, high_array) — absolute lower/upper bounds
        - 'ci'           : iterable of (low, high) tuples per bin (legacy)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(bins, mean_scores, label=label, color=color, linewidth=linewidth,
            alpha=alpha, linestyle=linestyle)

    if error_type == 'percentile':
        lower_bounds, upper_bounds = error_scores
        ax.fill_between(bins, lower_bounds, upper_bounds, alpha=0.2,
                        color=color, edgecolor=None)
    elif error_type == 'ci':
        lower_bounds, upper_bounds = zip(*error_scores)
        ax.fill_between(bins, lower_bounds, upper_bounds, alpha=0.15, color=color)
    else:
        ax.fill_between(bins, mean_scores - error_scores, mean_scores + error_scores,
                        alpha=0.25, color=color, edgecolor=None)

    if set_label_x:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Decoding accuracy, +- {error_type}")

    ax.tick_params(axis='x', labelsize=6, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=0, rotation=60)

    return ax


def parse_SVM_data(figure_details, data, it='iteration'):
    data_list = []
    timeline = None

    for data_name in figure_details['data_names']:
        for condition in figure_details['conditions']:
            if data_name not in data or data[data_name].get('results') is None:
                print(f"  data source {data_name} not loaded, skipping")
                return None, None

            df = extract_condition_df(data[data_name]['results'], condition, it=it)

            if df is None or len(df) == 0:
                print(f"{condition} is empty, skipping")
                return None, None

            grouped = df.groupby(['bin_start', 'bin_end'])['mean_score']

            if timeline is None:
                timeline = [end for start, end in grouped.groups]

            data_array = np.transpose(np.array([group.values for _, group in grouped]))
            data_list.append(data_array)

    return data_list, timeline


def parse_line_data(line_config, data_sources, it='iteration'):
    """
    Parse data for a single line based on configuration.
    Returns None if the data source isn't loaded, the condition is missing,
    or anything else goes wrong — so the calling loop can simply skip the line.
    """
    try:
        data_source_key = line_config['data_source']
        condition = line_config['condition']
        plot_mean = line_config.get('plot_mean')
        subtract_chance = line_config.get('subtract_chance', False)

        # Check data source exists and is loaded
        if data_source_key not in data_sources:
            print(f"    Data source '{data_source_key}' not in data_sources dict")
            return None

        source = data_sources[data_source_key]
        if source.get('results') is None:
            print(f"    Data source '{data_source_key}' has no loaded results")
            return None

        # Extract data
        df = extract_condition_df(source['results'], condition, it=it)

        if df is None or len(df) == 0:
            print(f"    {condition} from {data_source_key} is empty or missing")
            return None

        grouped = df.groupby(['bin_start', 'bin_end'])['mean_score']
        timeline = [end for start, end in grouped.groups]

        try:
            data_array = np.transpose(np.array([group.values for _, group in grouped]))
        except Exception as e:
            print(f"    Could not stack {condition} from {data_source_key}: {e}")
            return None

        if data_array.size == 0:
            print(f"    Empty data array for {condition} from {data_source_key}")
            return None

        # ── Load shuffle data (if requested for subtraction or stats) ──
        shuffle_array = None
        shuffle_condition = line_config.get(
            'shuffle_condition',
            condition.replace("results_", "results_shuffled_", 1)
        )

        if shuffle_condition and 'shuffled' not in condition.lower():
            shuffle_df = extract_condition_df(
                source['results'],
                shuffle_condition,
                it=it
            )
            if shuffle_df is not None and len(shuffle_df) > 0:
                try:
                    shuffle_grouped = shuffle_df.groupby(['bin_start', 'bin_end'])['mean_score']
                    shuffle_array = np.transpose(
                        np.array([group.values for _, group in shuffle_grouped])
                    )
                    if shuffle_array.size == 0:
                        shuffle_array = None
                except Exception as e:
                    print(f"    Could not stack shuffle {shuffle_condition}: {e}")
                    shuffle_array = None

        # ── Subtract chance level (mean over time of shuffle distribution) ─
        chance_level = 0.5
        if subtract_chance:
            if shuffle_array is not None:
                chance_level = float(np.mean(shuffle_array))
                data_array = data_array - chance_level
                print(f"    Subtracted chance level {chance_level:.4f} from "
                      f"{data_source_key}/{condition}")
            else:
                print(f"    WARNING: subtract_chance requested for {condition} "
                      f"but no shuffle data found. Using 0.5 as fallback.")
                data_array = data_array - 0.5
                chance_level = 0.5

        # Percentile interval across iterations. Default is 95% (2.5/97.5).
        # Override per line via 'percentile_range' = (low, high).
        pct_low, pct_high = line_config.get('percentile_range', (2.5, 97.5))
        pct_lo_vals = np.percentile(data_array, pct_low, axis=0)
        pct_hi_vals = np.percentile(data_array, pct_high, axis=0)

        result = {
            'data': data_array,
            'timeline': timeline,
            'mean': np.mean(data_array, axis=0),
            'std': np.std(data_array, axis=0),
            'sem': np.std(data_array, axis=0) / np.sqrt(data_array.shape[0]),
            'percentile_low': pct_lo_vals,
            'percentile_high': pct_hi_vals,
            'percentile_range': (pct_low, pct_high),
            'significance': None,
            'test_scores': None,
            'plot_mean': plot_mean,
            'chance_level': chance_level,
            'chance_subtracted': subtract_chance,
        }

        # Compute statistics if requested
        compute_stats = line_config.get('compute_stats', True)

        if compute_stats and 'shuffled' not in condition.lower():
            # New: stat_method dispatches between cluster permutation, per-bin
            # permutation against shuffle, and the original SVM_stats. Default
            # is the cluster permutation test, which matches the reference
            # script (`stat_method = 'cluster'`).
            stat_method = line_config.get('stat_method', 'cluster')
            stat_test = line_config.get('stat_test', 'bf')
            test_against = line_config.get('test_against', 'shuffle')
            sig_thr = line_config.get('sig_thr', 50)
            fdr_alpha = line_config.get('fdr_alpha', 0.05)
            p_alpha = line_config.get('p_alpha', 0.05)
            threshold_pctile = line_config.get('threshold_pctile', 95.0)
            cluster_window = line_config.get('cluster_window', None)
            sig_thr_p = line_config.get('sig_thr_p', None)

            try:
                # Baseline tests always go through SVM_stats regardless of
                # stat_method, since baseline tests don't involve shuffle.
                if test_against == 'baseline':
                    baseline_indices = np.array(timeline) <= 0
                    test_scores, significance = SVM_stats(
                        data_array, baseline_indices,
                        stat_test=stat_test, fdr_test='bh',
                        test_against='baseline', sig_thr=sig_thr
                    )
                    result['significance'] = significance
                    result['test_scores'] = test_scores

                elif test_against == 'shuffle':
                    if shuffle_array is None:
                        print(f"    Shuffled data was not found for "
                              f"{shuffle_condition} — skipping stats")
                    else:
                        # If we subtracted chance from the real data, do the
                        # same to the shuffle so the comparison stays on the
                        # same scale.
                        shuffle_for_stats = shuffle_array
                        if subtract_chance:
                            shuffle_for_stats = shuffle_array - chance_level

                        if stat_method == 'cluster':
                            significance, cluster_result = cluster_test_vs_shuffle(
                                data_array, shuffle_for_stats, timeline,
                                cluster_window=cluster_window,
                                threshold_pctile=threshold_pctile,
                                p_alpha=p_alpha,
                            )
                            result['significance'] = significance
                            result['test_scores'] = cluster_result

                        elif stat_method == 'perm_vs_shuffle':
                            test_scores, significance = perm_test_vs_shuffle(
                                data_array, shuffle_for_stats,
                                fdr_alpha=fdr_alpha, sig_thr_p=sig_thr_p,
                            )
                            result['significance'] = significance
                            result['test_scores'] = test_scores

                        else:  # 'svm_stats' or anything legacy
                            test_scores, significance = SVM_stats(
                                data_array, data_array_test=shuffle_for_stats,
                                stat_test=stat_test, fdr_test='bh',
                                test_against='data', sig_thr=sig_thr
                            )
                            result['significance'] = significance
                            result['test_scores'] = test_scores
            except Exception as e:
                print(f"    Stats computation failed for {condition} "
                      f"(stat_method={stat_method}): {e}")
                traceback.print_exc()
                # Keep result, just without significance

        return result

    except Exception as e:
        print(f"    parse_line_data failed unexpectedly: {e}")
        traceback.print_exc()
        return None


def find_subdir(root, must_contain, must_not_contain=None):
    """Safely find a subdirectory matching name constraints. Returns None if not found."""
    must_not_contain = must_not_contain or []
    try:
        if not root.exists():
            return None
        candidates = [
            d for d in root.iterdir()
            if d.is_dir()
            and all(s in d.name for s in must_contain)
            and not any(s in d.name for s in must_not_contain)
        ]
        if len(candidates) == 0:
            return None
        return candidates[0]
    except Exception as e:
        print(f"  find_subdir failed for {must_contain}: {e}")
        return None


def main(penalty):

    # Directory setup
    dir_root = config.resolve_pp("SVM", "decode_from_neurons", str(penalty))

    if not dir_root.exists():
        print(f"Root directory does not exist: {dir_root}")
        return

    dir_lALM = find_subdir(dir_root, must_contain=["lALM"], must_not_contain=["rALM"])
    dir_rALM = find_subdir(dir_root, must_contain=["rALM"], must_not_contain=["lALM"])
    dir_lALM_rALM = find_subdir(dir_root, must_contain=["lALM_rALM"])

    print(f"  dir_lALM: {dir_lALM}")
    print(f"  dir_rALM: {dir_rALM}")
    print(f"  dir_lALM_rALM: {dir_lALM_rALM}")

    save_dir = config.FIGURE_DIR / "SVM" / "pseudopop" / str(penalty)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # ── Font & style configuration ──────────────────────────────────────
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

    # ── Subplot dimensions ──────────────────────────────────────────────
    SUBPLOT_W = 19.7 / 25.4
    SUBPLOT_H = 0.6
    H_PAD = 0.6
    V_PAD = 0.7
    LEFT_MARGIN = 0.6
    RIGHT_MARGIN = 0.3
    TOP_MARGIN = 0.3
    BOT_MARGIN = 0.5

    def _safe_join(base_dir, *parts):
        """Return a path, or None if base_dir is None."""
        if base_dir is None:
            return None
        return os.path.join(base_dir, *parts)

    # Data sources dictionary - PSEUDOPOP
    data_sources_pseudopop = {
        'lALM': {
            'path': _safe_join(dir_lALM, 'opto off', 'pseudopop'),
            'results': None
        },
        'lALM_opto': {
            'path': _safe_join(dir_lALM, 'opto on', 'pseudopop'),
            'results': None
        },
        'lALM_lick_aligned': {
            'path': _safe_join(dir_lALM, 'lick aligned', 'pseudopop'),
            'results': None
        },
        'lALM_lick_aligned_opto': {
            'path': _safe_join(dir_lALM, 'lick aligned opto', 'pseudopop'),
            'results': None
        },
        'rALM': {
            'path': _safe_join(dir_rALM, 'opto off', 'pseudopop'),
            'results': None
        },
        'rALM_lick_aligned': {
            'path': _safe_join(dir_rALM, 'lick aligned', 'pseudopop'),
            'results': None
        },
        'rALM_lick_aligned_opto': {
            'path': _safe_join(dir_rALM, 'lick aligned opto', 'pseudopop'),
            'results': None
        },
        'lALM_rALM': {
            'path': _safe_join(dir_lALM_rALM, 'opto off', 'pseudopop'),
            'results': None
        },
        'lALM_rALM_lick_aligned': {
            'path': _safe_join(dir_lALM_rALM, 'lick aligned', 'pseudopop'),
            'results': None
        },
        'lALM_rALM_lick_aligned_opto': {
            'path': _safe_join(dir_lALM_rALM, 'lick aligned opto', 'pseudopop'),
            'results': None
        }
    }

    # Data sources dictionary - SESSION LEVEL
    data_sources_session = {
        'lALM': {
            'path': _safe_join(dir_lALM, 'opto off', 'session_level'),
            'results': None
        },
        'lALM_opto': {
            'path': _safe_join(dir_lALM, 'opto on', 'session_level'),
            'results': None
        },
        'lALM_lick_aligned': {
            'path': _safe_join(dir_lALM, 'lick aligned', 'session_level'),
            'results': None
        },
        'lALM_lick_aligned_opto': {
            'path': _safe_join(dir_lALM, 'lick aligned opto', 'session_level'),
            'results': None
        },
        'rALM': {
            'path': _safe_join(dir_rALM, 'opto off', 'session_level'),
            'results': None
        },
        'rALM_opto': {
            'path': _safe_join(dir_rALM, 'opto on', 'session_level'),
            'results': None
        },
        'rALM_lick_aligned': {
            'path': _safe_join(dir_rALM, 'lick aligned', 'session_level'),
            'results': None
        },
        'rALM_lick_aligned_opto': {
            'path': _safe_join(dir_rALM, 'lick aligned opto', 'session_level'),
            'results': None
        },
        'lALM_rALM': {
            'path': _safe_join(dir_lALM_rALM, 'opto off', 'session_level'),
            'results': None
        },
        'lALM_rALM_opto': {
            'path': _safe_join(dir_lALM_rALM, 'opto on', 'session_level'),
            'results': None
        },
        'lALM_rALM_lick_aligned': {
            'path': _safe_join(dir_lALM_rALM, 'lick aligned', 'session_level'),
            'results': None
        },
        'lALM_rALM_lick_aligned_opto': {
            'path': _safe_join(dir_lALM_rALM, 'lick aligned opto', 'session_level'),
            'results': None
        }
    }

    def _try_load(key, source_dict):
        """Attempt to load output.npy for one entry, leaving 'results' as None on any failure."""
        path = source_dict.get('path')
        if path is None:
            print(f"  {key}: path is None (parent dir not found), skipping")
            return
        full_path = os.path.join(dir_root, path, 'output.npy')
        if not os.path.exists(full_path):
            print(f"  {key}: file not found at {full_path}")
            return
        try:
            loaded_data = np.load(full_path, allow_pickle=True).item()
            if loaded_data is None:
                print(f"  {key}: loaded data is None")
            else:
                source_dict['results'] = loaded_data
        except Exception as e:
            print(f"  Could not load {key}: {e}")

    # Preload all data
    print("Preloading pseudopop data")
    for key in data_sources_pseudopop:
        _try_load(key, data_sources_pseudopop[key])

    print("Preloading session-level data")
    for key in data_sources_session:
        _try_load(key, data_sources_session[key])

    # Define multi-panel figures
    figures_dict = {
        'stimulus_coding_main': {
            'save_name': 'main_figure',
            'layout': (4, 2),
            'data_type': 'pseudopop',
            'panels': {
                'Visual hit vs miss': {
                    'position': 0,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Visual hit vs miss',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_vis_hit_miss',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual hit vs miss'
                },

                'Auditory hit vs miss': {
                    'position': 1,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_aud_hit_miss',
                            'label': 'Auditory hit vs miss',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_aud_hit_miss',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory hit vs miss'
                },

                'Visual hit vs FA': {
                    'position': 4,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Visual presence',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_vis_presence_hit_fa',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual hit vs FA'
                },

                'Auditory hit vs FA': {
                    'position': 5,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Auditory presence',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_aud_presence_hit_fa',
                            'label': paper_colors('shuffle'),
                            'color': 'grey',
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory hit vs FA'
                },

                'Visual presence miss vs CR': {
                    'position': 2,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_presence_miss_cr',
                            'label': 'Visual presence',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_vis_presence_miss_cr',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual stimulus presence (miss vs CR)'
                },

                'Auditory presence miss vs CR': {
                    'position': 3,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_aud_presence_miss_cr',
                            'label': 'Auditory presence',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_shuffled_aud_presence_miss_cr',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory stimulus presence (miss vs CR)'
                },

                'Visual hit vs FA (lick-aligned)': {
                    'position': 6,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM_lick_aligned',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Visual presence',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'shuffle_condition': 'results_shuffled_vis_presence_hit_fa'
                        },
                        {
                            'data_source': 'lALM_rALM_lick_aligned',
                            'condition': 'results_shuffled_vis_presence_hit_fa',
                            'label': 'Shuffled',
                            'color': paper_colors('shuffle'),
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual hit vs FA (lick-aligned)'
                },

                'Auditory hit vs FA (lick-aligned)': {
                    'position': 7,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM_lick_aligned',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Auditory presence',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'shuffle_condition': 'results_shuffled_aud_presence_hit_fa'
                        },
                        {
                            'data_source': 'lALM_rALM_lick_aligned',
                            'condition': 'results_shuffled_aud_presence_hit_fa',
                            'label': paper_colors('shuffle'),
                            'color': 'grey',
                            'linewidth': 1,
                            'alpha': 0.7,
                            'linestyle': '--',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory hit vs FA (lick-aligned)'
                },
            }
        },

        'opto_comparison': {
            'save_name': 'opto_comparison_figure',
            'layout': (5, 2),
            'data_type': 'pseudopop',
            'panels': {
                'Visual presence opto': {
                    'position': 2,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_vis_presence_miss_cr',
                            'label': 'Control',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_vis_presence_miss_cr',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Visual stimulus presence (miss/CR)'
                },

                'Auditory presence opto': {
                    'position': 3,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_aud_presence_miss_cr',
                            'label': 'Control',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_aud_presence_miss_cr',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Auditory stimulus presence (miss/CR)'
                },

                'Visual hit/fa opto': {
                    'position': 4,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Control',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Visual hit/FA'
                },

                'Auditory hit/fa opto': {
                    'position': 5,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Control',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Auditory hit/FA'
                },

                'Visual hit/miss opto': {
                    'position': 0,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Control',
                            'color': paper_colors("vis_max"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Opto on',
                            'color': paper_colors("opto_purple"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Visual hit/miss'
                },

                'Auditory hit/miss opto': {
                    'position': 1,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_aud_hit_miss',
                            'label': 'Control',
                            'color': paper_colors("aud_max"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_aud_hit_miss',
                            'label': 'Opto on',
                            'color': paper_colors("opto_purple"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': True,
                    'title': 'Auditory hit/miss'
                },

                'Visual hit vs FA (lick-aligned, opto)': {
                    'position': 6,
                    'lines': [
                        {
                            'data_source': 'lALM_lick_aligned',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Visual presence',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'shuffle_condition': 'results_shuffled_vis_presence_hit_fa'
                        },
                        {
                            'data_source': 'lALM_lick_aligned_opto',
                            'condition': 'results_vis_presence_hit_fa',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual hit vs FA (lick-aligned)'
                },

                'Auditory hit vs FA (lick-aligned, opto)': {
                    'position': 7,
                    'lines': [
                        {
                            'data_source': 'lALM_lick_aligned',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Auditory presence',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'shuffle_condition': 'results_shuffled_aud_presence_hit_fa'
                        },
                        {
                            'data_source': 'lALM_lick_aligned_opto',
                            'condition': 'results_aud_presence_hit_fa',
                            'label': 'Opto on',
                            'color': paper_colors('opto_purple'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'error_type': 'percentile'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory hit vs FA (lick-aligned)'
                },
            }
        },

        'opto_hit_miss_chance_subtracted': {
            'save_name': 'opto_hit_miss_chance_subtracted',
            'layout': (1, 2),
            'data_type': 'pseudopop',
            'chance_subtracted': True,
            'panels': {
                'Visual hit/miss opto (chance-subtracted)': {
                    'position': 0,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Control',
                            'color': paper_colors("vis_max"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'subtract_chance': True,
                            'shuffle_condition': 'results_shuffled_vis_hit_miss',
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Opto on',
                            'color': paper_colors("opto_purple"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'subtract_chance': True,
                            'shuffle_condition': 'results_shuffled_vis_hit_miss',
                        },
                    ],
                    'ylim': [-0.15, 0.55],
                    'add_opto_shading': True,
                    'title': 'Visual hit/miss (accuracy − chance)',
                },

                'Auditory hit/miss opto (chance-subtracted)': {
                    'position': 1,
                    'lines': [
                        {
                            'data_source': 'lALM',
                            'condition': 'results_aud_hit_miss',
                            'label': 'Control',
                            'color': paper_colors("aud_max"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'subtract_chance': True,
                            'shuffle_condition': 'results_shuffled_aud_hit_miss',
                        },
                        {
                            'data_source': 'lALM_opto',
                            'condition': 'results_aud_hit_miss',
                            'label': 'Opto on',
                            'color': paper_colors("opto_purple"),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': True,
                            'test_against': 'shuffle',
                            'sig_thr': 50,
                            'error_type': 'percentile',
                            'subtract_chance': True,
                            'shuffle_condition': 'results_shuffled_aud_hit_miss',
                        },
                    ],
                    'ylim': [-0.15, 0.55],
                    'add_opto_shading': True,
                    'title': 'Auditory hit/miss (accuracy − chance)',
                },
            },
        },

        'session_level_example': {
            'save_name': 'session_level_figure',
            'layout': (2, 2),
            'data_type': 'session',
            'panels': {
                'Visual hit vs miss - mean': {
                    'position': 0,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Mean ± SEM',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'plot_individual': True,
                            'plot_mean': True,
                            'error_type': 'sem'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual hit vs miss (mean)'
                },

                'Auditory hit vs miss - Mean': {
                    'position': 1,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_hit_miss',
                            'label': 'Mean ± SEM',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'plot_individual': True,
                            'plot_mean': True,
                            'error_type': 'sem'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory hit vs miss (individual sessions)'
                },

                'Visual stimulus presence - mean': {
                    'position': 2,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_vis_presence_miss_cr',
                            'label': 'Mean ± SEM',
                            'color': paper_colors('vis_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'plot_individual': True,
                            'plot_mean': True,
                            'error_type': 'sem'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Visual presence (mean)'
                },

                'Auditory stimulus presence - Mean': {
                    'position': 3,
                    'lines': [
                        {
                            'data_source': 'lALM_rALM',
                            'condition': 'results_aud_presence_miss_cr',
                            'label': 'Mean ± SEM',
                            'color': paper_colors('aud_max'),
                            'linewidth': 1,
                            'alpha': 0.9,
                            'linestyle': '-',
                            'compute_stats': False,
                            'plot_individual': True,
                            'plot_mean': True,
                            'error_type': 'sem'
                        }
                    ],
                    'ylim': [0.2, 1.05],
                    'add_opto_shading': False,
                    'title': 'Auditory presence (individual sessions)'
                },
            }
        }
    }

    # ── Auto-inject between-condition tests into opto comparison figures ──
    # Every panel of these figures will get a permutation test between
    # line 0 (control) and line 1 (opto on). Skipped if a panel already
    # defines its own between_condition_test(s) or has fewer than 2 lines.
    OPTO_FIGURES_WITH_BETWEEN_TEST = {
        'opto_comparison',
        'opto_hit_miss_chance_subtracted',
    }
    DEFAULT_BETWEEN_TEST = {
        # Test type: 'mean_vs_distribution' tests whether mean(line_b) is in
        # the tail of distribution(line_a). With direction='less', this asks
        # "is opto-on accuracy below the opto-off iteration distribution"
        # — directly testing whether opto deteriorates encoding.
        # Robust to n_iter (unlike label_shuffle of mean-differences).
        'test_type': 'mean_vs_distribution',
        'line_a': 0,       # reference distribution (control / opto off)
        'line_b': 1,       # tested mean trace (opto on)
        'direction': 'less',   # opto reduces decoding accuracy
        'stat_method': 'cluster',
        'threshold_pctile': 5.0,  # 5th percentile of opto-off as cluster threshold
        'p_alpha': 0.05,
        'color': 'black',
    }
    for _fig_name in OPTO_FIGURES_WITH_BETWEEN_TEST:
        _fig_cfg = figures_dict.get(_fig_name)
        if _fig_cfg is None:
            continue
        for _panel_name, _panel_cfg in _fig_cfg.get('panels', {}).items():
            if 'between_condition_test' in _panel_cfg or 'between_condition_tests' in _panel_cfg:
                continue  # user already specified one, don't overwrite
            if len(_panel_cfg.get('lines', [])) < 2:
                continue
            _panel_cfg['between_condition_test'] = dict(DEFAULT_BETWEEN_TEST)

    # Plotting loop
    print("Creating multi-panel figures")
    for figure_name, figure_config in figures_dict.items():
        print(f"Creating figure: {figure_name}")

        try:
            # Determine which data source to use
            data_type = figure_config.get('data_type', 'pseudopop')
            if data_type == 'session':
                data_sources = data_sources_session
                it = 'session_ID'
            else:
                data_sources = data_sources_pseudopop
                it = 'iteration'

            is_chance_subtracted = figure_config.get('chance_subtracted', False)

            nrows, ncols = figure_config['layout']

            # ── Compute figure size ────────────────────────────────────
            fig_w = LEFT_MARGIN + ncols * SUBPLOT_W + (ncols - 1) * H_PAD + RIGHT_MARGIN
            fig_h = TOP_MARGIN + nrows * SUBPLOT_H + (nrows - 1) * V_PAD + BOT_MARGIN

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
            axes = axes.flatten() if nrows * ncols > 1 else [axes]

            # ── Force each Axes to fixed size ──────────────────────────
            for row in range(nrows):
                for col in range(ncols):
                    idx = row * ncols + col
                    left = (LEFT_MARGIN + col * (SUBPLOT_W + H_PAD)) / fig_w
                    bottom = (BOT_MARGIN + (nrows - 1 - row) * (SUBPLOT_H + V_PAD)) / fig_h
                    width = SUBPLOT_W / fig_w
                    height = SUBPLOT_H / fig_h
                    axes[idx].set_position([left, bottom, width, height])

            # Process each panel
            for panel_name, panel_config in figure_config['panels'].items():
                print(f"  Plotting panel: {panel_name}")

                try:
                    ax_idx = panel_config['position']
                    if ax_idx >= len(axes):
                        print(f"    Panel position {ax_idx} out of range, skipping")
                        continue
                    ax = axes[ax_idx]

                    significance_bars = []
                    sig_colors = []
                    panel_has_chance_subtracted = False
                    panel_has_any_data = False

                    # Keep parsed line data + their line configs so we can
                    # run between-condition tests (e.g. opto off vs opto on)
                    # after all lines are processed.
                    parsed_lines = []

                    # Plot each line
                    for line_config in panel_config['lines']:
                        try:
                            parsed_data = parse_line_data(line_config, data_sources, it=it)

                            if parsed_data is None:
                                print(f"    Skipping line: {line_config.get('label', 'unnamed')}")
                                parsed_lines.append(None)
                                continue

                            parsed_lines.append((line_config, parsed_data))
                            panel_has_any_data = True

                            if parsed_data.get('chance_subtracted', False):
                                panel_has_chance_subtracted = True

                            plot_mean = line_config.get('plot_mean', True)
                            plot_individual = line_config.get('plot_individual')
                            error_type = line_config.get('error_type', 'std')

                            # Plot individual sessions/iterations
                            if plot_individual:
                                for i in range(parsed_data['data'].shape[0]):
                                    ax.plot(
                                        parsed_data['timeline'],
                                        parsed_data['data'][i, :],
                                        color='grey',
                                        linewidth=0.5,
                                        alpha=0.3,
                                        linestyle='-',
                                        label=line_config.get('label', '') if i == 0 and not plot_mean else ''
                                    )

                            # Plot mean with error bars
                            if plot_mean:
                                if error_type == 'sem':
                                    error_scores = parsed_data['sem']
                                elif error_type == 'std':
                                    error_scores = parsed_data['std']
                                elif error_type == 'percentile':
                                    error_scores = (parsed_data['percentile_low'],
                                                    parsed_data['percentile_high'])
                                elif error_type == 'ci':
                                    error_scores = parsed_data.get('ci', parsed_data['std'])
                                else:
                                    error_scores = parsed_data['std']

                                ax = plot_decoding_over_time(
                                    mean_scores=parsed_data['mean'],
                                    error_scores=error_scores,
                                    bins=parsed_data['timeline'],
                                    color=line_config.get('color', 'black'),
                                    label=line_config.get('label', ''),
                                    ax=ax,
                                    error_type=error_type,
                                    set_label_x=(ax_idx >= (nrows - 1) * ncols),
                                    linewidth=line_config.get('linewidth', 1.5),
                                    alpha=line_config.get('alpha', 0.9),
                                    linestyle=line_config.get('linestyle', '-')
                                )

                            # Store significance
                            if plot_mean and parsed_data['significance'] is not None:
                                significance_bars.append(parsed_data['significance'])
                                sig_colors.append(line_config.get('color', 'black'))

                        except Exception as e:
                            print(f"    Failed to plot line {line_config.get('label', 'unnamed')}: {e}")
                            traceback.print_exc()
                            parsed_lines.append(None)
                            continue

                    # ── Between-condition test (e.g. opto off vs opto on) ──
                    # The panel can specify one or more pairwise tests via the
                    # 'between_condition_tests' key. Each entry is a dict with:
                    #   'line_a': index (or label) of the first line
                    #   'line_b': index (or label) of the second line
                    #   'stat_method': 'cluster' (default) or 'perm_vs_shuffle'
                    #   'color': color for the sig bar (default 'black')
                    #   plus optional 'n_perm', 'p_alpha', 'fdr_alpha',
                    #   'threshold_pctile', 'cluster_window', 'sig_thr_p',
                    #   'two_sided', 'seed'.
                    bc_tests = panel_config.get('between_condition_tests', [])
                    # Backwards-friendly singular form
                    if 'between_condition_test' in panel_config:
                        bc_tests = bc_tests + [panel_config['between_condition_test']]

                    def _resolve_line(ref, parsed_lines, line_configs):
                        """Resolve a line reference (int index or label string)
                        to a parsed_data dict, or None on failure."""
                        if isinstance(ref, int):
                            if 0 <= ref < len(parsed_lines) and parsed_lines[ref] is not None:
                                return parsed_lines[ref][1]
                            return None
                        # Otherwise assume it's a label string
                        for entry in parsed_lines:
                            if entry is None:
                                continue
                            cfg, pd_ = entry
                            if cfg.get('label') == ref:
                                return pd_
                        return None

                    for bc in bc_tests:
                        try:
                            pd_a = _resolve_line(bc.get('line_a', 0), parsed_lines, None)
                            pd_b = _resolve_line(bc.get('line_b', 1), parsed_lines, None)

                            if pd_a is None or pd_b is None:
                                print(f"    Between-condition test skipped: "
                                      f"one or both lines unavailable")
                                continue

                            # Timelines must match for the test to make sense
                            if (len(pd_a['timeline']) != len(pd_b['timeline'])
                                or not np.allclose(pd_a['timeline'], pd_b['timeline'])):
                                print(f"    Between-condition test skipped: "
                                      f"timelines differ between lines")
                                continue

                            # test_type dispatches between:
                            #   'mean_vs_distribution' — one-sided test of
                            #       mean(line_b) against distribution(line_a).
                            #       Robust to n_iter. Default for opto figures.
                            #   'label_shuffle' — symmetric permutation of
                            #       mean-differences across A and B.
                            bc_test_type = bc.get('test_type', 'mean_vs_distribution')
                            bc_method = bc.get('stat_method', 'cluster')

                            if bc_test_type == 'mean_vs_distribution':
                                # line_a is reference, line_b is tested.
                                # direction='less' = "is line_b below line_a's
                                # iteration distribution" (e.g. opto reducing
                                # decoding accuracy).
                                significance_bc, test_scores_bc = mean_vs_distribution_test(
                                    data_ref=pd_a['data'],
                                    data_test=pd_b['data'],
                                    timeline=pd_a['timeline'],
                                    direction=bc.get('direction', 'less'),
                                    stat_method=bc_method,
                                    cluster_window=bc.get('cluster_window', None),
                                    threshold_pctile=bc.get('threshold_pctile', 5.0),
                                    p_alpha=bc.get('p_alpha', 0.05),
                                    fdr_alpha=bc.get('fdr_alpha', 0.05),
                                    sig_thr_p=bc.get('sig_thr_p', None),
                                )
                            else:  # 'label_shuffle' (legacy)
                                significance_bc, test_scores_bc = between_condition_test(
                                    pd_a['data'], pd_b['data'], pd_a['timeline'],
                                    stat_method=bc_method,
                                    fdr_alpha=bc.get('fdr_alpha', 0.05),
                                    sig_thr_p=bc.get('sig_thr_p', None),
                                    cluster_window=bc.get('cluster_window', None),
                                    threshold_pctile=bc.get('threshold_pctile', 95.0),
                                    p_alpha=bc.get('p_alpha', 0.05),
                                    n_perm=bc.get('n_perm', 10000),
                                    two_sided=bc.get('two_sided', True),
                                    seed=bc.get('seed', None),
                                )

                            significance_bars.append(significance_bc)
                            sig_colors.append(bc.get('color', 'black'))
                            print(f"    Between-condition test "
                                  f"({bc_test_type}/{bc_method}) "
                                  f"done for panel {panel_name}")

                        except Exception as e:
                            print(f"    Between-condition test failed for "
                                  f"panel {panel_name}: {e}")
                            traceback.print_exc()

                    # Format the axis (always do this, even with no data)
                    try:
                        if is_chance_subtracted or panel_has_chance_subtracted:
                            if 'ylim' in panel_config:
                                ax.set_ylim(panel_config['ylim'])
                            else:
                                ax.set_ylim([-0.15, 0.55])
                            ax.axhline(0, color='black', linewidth=0.5, linestyle=':', alpha=0.6)
                            ax.set_ylabel('Accuracy − chance')
                        else:
                            override_ylim = True
                            if override_ylim:
                                ax.set_ylim([0.35, 1.05])
                            elif 'ylim' in panel_config:
                                ax.set_ylim(panel_config['ylim'])

                        if 'title' in panel_config:
                            ax.set_title(panel_config['title'], fontsize=8)

                        ax.grid(False)

                        # Annotate empty panels
                        if not panel_has_any_data:
                            ax.text(
                                0.5, 0.5, 'no data',
                                transform=ax.transAxes,
                                ha='center', va='center',
                                fontsize=7, color='lightgray',
                                style='italic'
                            )

                        # Add significance bars (only if any)
                        if significance_bars:
                            try:
                                add_sig_lines_ax(ax, significance_bars, colors=sig_colors)
                            except Exception as e:
                                print(f"    add_sig_lines_ax failed: {e}")

                        # Add opto shading if requested
                        if panel_config.get('add_opto_shading', False):
                            try:
                                add_opto_shading(ax)
                            except Exception as e:
                                print(f"    add_opto_shading failed: {e}")

                        try:
                            add_t0_line(ax)
                        except Exception as e:
                            print(f"    add_t0_line failed: {e}")

                    except Exception as e:
                        print(f"    Axis formatting failed for panel {panel_name}: {e}")
                        traceback.print_exc()

                except Exception as e:
                    print(f"  Panel {panel_name} failed entirely: {e}")
                    traceback.print_exc()
                    continue

            # Clean up unused axes
            try:
                used_positions = {p['position'] for p in figure_config['panels'].values()}
                for idx in range(len(axes)):
                    if idx not in used_positions:
                        axes[idx].axis('off')
            except Exception as e:
                print(f"  Cleanup of unused axes failed: {e}")

            # Save figure
            try:
                save_name = figure_config['save_name']
                plt.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=1000)
                plt.savefig(os.path.join(save_dir, f'{save_name}.svg'), dpi=1000)
                print(f"Saved: {save_name}")
            except Exception as e:
                print(f"  Failed to save figure {figure_name}: {e}")
                traceback.print_exc()

            plt.close(fig)

        except Exception as e:
            print(f"Figure {figure_name} failed entirely: {e}")
            traceback.print_exc()
            try:
                plt.close('all')
            except Exception:
                pass
            continue

    print("All figures created successfully!")


if __name__ == "__main__":
    for penalty in ['0.01', '0.1', '1', '10']:
        try:
            main(penalty)
        except Exception as e:
            print(f"main() failed for penalty={penalty}: {e}")
            traceback.print_exc()
            continue