"""
Plot motion decoding over time with a paired cluster-based permutation test.

Same figures as SVM_motion_plotting.py, but real and shuffled accuracy are compared
within session (one-sample test on real - shuffled, one-tailed for real > shuffled)
instead of as two independent groups. On the group plot, the blue band is the 95% CI of
the across-session mean and the grey band is the 95% interval of the shuffle null for
that mean (both were SEM across sessions before).
"""
import os
import sys
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mne
from matplotlib.patches import Rectangle
import config

from neural_analysis.SVM.utils_svm import pct_against_chance
from neural_analysis.SVM.plotting.SVM_motion_plotting import (
    mm_to_inches,
    extract_condition_df,
    extract_shuffled_condition_df,
    plot_decoding_over_time,
    improvement_from_baseline,
    print_session_summary,
)

RNG_SEED = 42
ALPHA = 0.05
NULL_PCTILES = (2.5, 97.5)  # shuffle band on the group plot (central 95% of the null)
CI_LEVEL = 0.95  # band around the real accuracy on the group plot

mne.set_log_level('WARNING')  # no progress bars from the permutation test


def session_by_bin_matrix(df):
    """Pivot a results DataFrame to a (n_sessions, n_bins) table of mean scores."""
    return df.pivot_table(index='session_ID', columns='bin_start', values='mean_score', aggfunc='mean')


def paired_session_matrices(df, df_shuffled):
    """
    Build real and shuffled (n_sessions, n_bins) tables with identical row and column order,
    so that row i of both tables is the same session.
    """
    real = session_by_bin_matrix(df)
    shuffled = session_by_bin_matrix(df_shuffled)

    if set(real.index) != set(shuffled.index):
        raise ValueError(f"Real and shuffled results contain different sessions: "
                         f"{sorted(set(real.index) ^ set(shuffled.index))}")
    if set(real.columns) != set(shuffled.columns):
        raise ValueError("Real and shuffled results contain different time bins")

    shuffled = shuffled.reindex(index=real.index, columns=real.columns)

    if real.isna().any().any() or shuffled.isna().any().any():
        raise ValueError("Some sessions are missing time bins")

    return real, shuffled


def group_null_interval(df_shuffled, sessions, bins, pctiles=NULL_PCTILES):
    """
    Percentiles of the shuffle null for the across-session mean accuracy, per time bin.

    Shuffle k is averaged across sessions to give one null value of the group mean. Shuffles
    are independent between sessions, so this is the null distribution of the plotted mean.
    The interval is per bin (not corrected across bins); the cluster test is the inference.

    Returns:
        np.ndarray, shape (len(pctiles), n_bins)
    """
    lookup = df_shuffled.set_index(['session_ID', 'bin_start'])['all_scores']
    # (n_sessions, n_bins, n_shuffles)
    null = np.stack([np.stack([lookup[(ses, b)] for b in bins]) for ses in sessions])
    group_null = null.mean(axis=0)
    return np.percentile(group_null, pctiles, axis=1)


def group_mean_ci(real, level=CI_LEVEL):
    """
    Across-session mean accuracy and its t-based confidence interval, per time bin.

    Parameters:
        real: (n_sessions, n_bins) table of mean scores

    Returns:
        mean, lower, upper: np.ndarray, each of shape (n_bins,)
    """
    n_sessions = real.shape[0]
    mean = real.mean(axis=0).values
    half_width = scipy.stats.t.ppf(1 - (1 - level) / 2, n_sessions - 1) * real.sem(axis=0).values
    return mean, mean - half_width, mean + half_width


def pct_paired(real, shuffled):
    """
    Paired cluster-based permutation test of real > shuffled across sessions.

    Uses pct_against_chance from utils_svm (one-sample test on real - shuffled, tail=1).
    That function takes no seed, so the global numpy seed is set to keep p-values repeatable.
    """
    np.random.seed(RNG_SEED)
    t_obs, clusters, cluster_p_values, h0 = pct_against_chance(np.asarray(real), np.asarray(shuffled))
    return t_obs, clusters, cluster_p_values, h0


def main(data_dir, exclude_sessions=None):
    """
    Parameters:
        data_dir (str): Path to data directory containing output.npy
        exclude_sessions (list): List of session_IDs (as strings) to exclude from plotting
    """
    if exclude_sessions is None:
        exclude_sessions = []
    else:
        print(f"\nExcluding sessions: {exclude_sessions}\n")

    plt.rcParams.update({'font.size': 10})
    plt.rcParams['svg.fonttype'] = 'none'
    save_dir = config.FIGURE_DIR / "SVM" / "decode_from_motion"

    window = [-1, 1.5]
    bins_time = np.linspace(window[0], window[1], 201)
    bins_time = bins_time + 0.5 * (1 / 80)
    bins_time = bins_time[:-1]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    conditions = [('au_presence', 'Auditory Presence (CR vs Miss)'),
                  ('vis_presence', 'Visual Presence (CR vs Miss'),
                  ('trial_type', 'Trial Type (X,Y,P)'),
                  ('response_side', 'Response Side (L,R,N)'),
                  ('response_presence', 'Lick Presence'),
                  ('vis_side_miss', 'Visual Side misses'),
                  ('vis_side_hit', 'Visual Side Hits'),
                  ('vis_side', 'Visual side all trials'),
                  ('vis_ori_left_all', 'Visual orientation left all trials'),
                  ('vis_ori_right_all', 'Visual orientation right all trials'),
                  ('vis_ori_left_miss', 'Visual orientation left misses'),
                  ('vis_ori_right_miss', 'Visual orientation right misses'),
                  ('vis_ori_left_hit', 'Visual orientation left hits'),
                  ('vis_ori_right_hit', 'Visual orientation right hits'),
                  ('au_pitch_all', 'Auditory pitch all trials'),
                  ('au_pitch_miss', 'Auditory pitch misses'),
                  ('au_pitch_hit', 'Auditory pitch hits'),
                  ('vis_side_opto', 'Visual side opto'),
                  ('vis_side_opto_control', 'Visual side opto control'),
                  ('au_pitch_all_opto', 'Auditory pitch opto'),
                  ('au_pitch_all_opto_control', 'Auditory pitch opto control'),
                  ('vis_hit_aud_hit', 'Visual hit vs auditory hit'),
                  ('vis_hit_miss', 'Visual hit vs miss'),
                  ('aud_hit_miss', 'Auditory hit vs miss')
                  ]

    res = np.load(os.path.join(data_dir, 'output.npy'), allow_pickle=True).item()

    errorbars_type = 'sem'
    exclude_set = set(str(s) for s in exclude_sessions)

    for name, title in conditions:
        cond = 'results_' + name
        cond_shuffled = 'results_shuffled_' + name
        print(cond)
        df = extract_condition_df(res, cond)
        df_shuffled = extract_shuffled_condition_df(res, cond_shuffled)

        if df is None or df_shuffled is None or df.empty or df_shuffled.empty:
            print(f"{cond} or {cond_shuffled} are empty, skipping")
            continue

        df['session_ID'] = df['session_ID'].astype(str)
        df_shuffled['session_ID'] = df_shuffled['session_ID'].astype(str)

        # --- Print per-session summary BEFORE filtering ---
        print_session_summary(df, df_shuffled, cond)

        # --- Exclude sessions ---
        if exclude_set:
            n_before = df['session_ID'].nunique()
            df = df[~df['session_ID'].isin(exclude_set)]
            df_shuffled = df_shuffled[~df_shuffled['session_ID'].isin(exclude_set)]
            n_after = df['session_ID'].nunique()

            if n_before != n_after:
                print(f"  Excluded {n_before - n_after} session(s), {n_after} remaining")

            if df.empty or df_shuffled.empty:
                print(f"  No sessions remaining after exclusion, skipping {cond}")
                continue

        # --- Paired matrices: row i is the same session in both ---
        real, shuffled = paired_session_matrices(df, df_shuffled)
        n_sessions = real.shape[0]
        print(f"  {n_sessions} sessions, {real.shape[1]} bins")

        # — Configuration (in mm) —
        subplot_w_mm = 40
        subplot_h_mm = 30
        h_gap_mm = 15
        left_margin_mm = 15
        right_margin_mm = 5
        top_margin_mm = 10
        bottom_margin_mm = 12

        n_rows = 3
        fig_w_mm = left_margin_mm + subplot_w_mm + right_margin_mm
        fig_h_mm = top_margin_mm + bottom_margin_mm + n_rows * subplot_h_mm + (n_rows - 1) * h_gap_mm

        fig_w = mm_to_inches(fig_w_mm)
        fig_h = mm_to_inches(fig_h_mm)

        fig = plt.figure(figsize=(fig_w, fig_h))

        axes = []
        for row in range(n_rows):
            left = left_margin_mm / fig_w_mm
            bottom = (bottom_margin_mm + (n_rows - 1 - row) * (subplot_h_mm + h_gap_mm)) / fig_h_mm
            width = subplot_w_mm / fig_w_mm
            height = subplot_h_mm / fig_h_mm
            axes.append(fig.add_axes([left, bottom, width, height]))

        ax, ax_ses, ax_heat = axes

        for ses_id, df_ses in df.groupby('session_ID'):
            ax_ses = plot_decoding_over_time(df_ses, 'Not shuffled', error=errorbars_type, ax=ax_ses, alpha=0.25,
                                             x_time=bins_time)
            ax_ses.set_ylim(0.1, 1)

        ax_heat.pcolormesh(real.values, vmin=0, vmax=1)

        # Shuffled: mean and 95% interval of the null for the group mean
        null_lo, null_hi = group_null_interval(df_shuffled, real.index, real.columns)
        ax.plot(bins_time, shuffled.mean(axis=0), label='Shuffled', color='gray')
        ax.fill_between(bins_time, null_lo, null_hi, alpha=0.3, color='gray', linewidth=0)

        # Not shuffled: mean and 95% CI of the across-session mean
        real_mean, real_lo, real_hi = group_mean_ci(real)
        ax.plot(bins_time, real_mean, label='Not shuffled', color='blue')
        ax.fill_between(bins_time, real_lo, real_hi, alpha=0.3, color='blue', linewidth=0)

        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Decoding accuracy (balanced)")
        ax.set_ylim(0.4, 1)

        # --- Paired cluster-based permutation test (real > shuffled) ---
        if n_sessions < 2:
            print("  Fewer than 2 sessions, skipping cluster test")
        else:
            t_obs, clusters, cluster_p_values, h0 = pct_paired(real, shuffled)

            for clust, p_val in zip(clusters, cluster_p_values):
                if p_val >= ALPHA:
                    continue

                clust_idx = np.asarray(clust[0] if isinstance(clust, tuple) else clust)
                if clust_idx.dtype == bool:
                    clust_idx = np.where(clust_idx)[0]

                x_start = bins_time[clust_idx.min()]
                x_end = bins_time[clust_idx.max()]
                print(f"  Significant cluster: {x_start:.3f} to {x_end:.3f} s, p = {p_val:.4f}")

                y_min, y_max = ax.get_ylim()
                rect_height = 0.05 * (y_max - y_min)

                rect = Rectangle((x_start, y_min),
                                 x_end - x_start, rect_height,
                                 facecolor='red',
                                 alpha=0.3,
                                 edgecolor='red',
                                 linewidth=1)
                ax.add_patch(rect)

        ax.set_title(f"Decoding: {title}")
        ax_ses.set_title(f"Decoding: {title}")

        fig.set_dpi(600)
        fig.savefig(os.path.join(save_dir, cond) + '.png')
        fig.savefig(os.path.join(save_dir, cond) + '.svg')
        plt.close('all')

        # improvement from baseline
        score, bin_end = improvement_from_baseline(df)
        score_shuffled, _ = improvement_from_baseline(df_shuffled)

        plt.plot(bin_end, score)
        plt.plot(bin_end, score_shuffled)
        plt.title(cond)

        if not os.path.isdir(os.path.join(save_dir, 'pre_stim_improvement')):
            os.mkdir(os.path.join(save_dir, 'pre_stim_improvement'))

        plt.gcf().set_size_inches(2, 2)
        plt.gcf().set_dpi(600)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pre_stim_improvement', cond + '.png'))
        plt.savefig(os.path.join(save_dir, 'pre_stim_improvement', cond + '.svg'))
        plt.close('all')

    print("end of script")


if __name__ == "__main__":
    # --- EXCLUDE SESSIONS HERE ---
    exclude = ['006202306051450', '006202306061601', '023202401221608', '023202401201423']

    # Or pass via command line: python script.py /path/to/data --exclude ses1 ses2 ses3
    if '--exclude' in sys.argv:
        idx = sys.argv.index('--exclude')
        exclude = sys.argv[idx + 1:]
        sys.argv = sys.argv[:idx]

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = config.resolve_pp("SVM", "decode_from_motion")

    main(data_dir, exclude_sessions=exclude)
