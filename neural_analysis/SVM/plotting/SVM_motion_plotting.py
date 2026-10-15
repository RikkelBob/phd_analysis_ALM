import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
from mne.stats import permutation_cluster_test
from matplotlib.patches import Rectangle
import config

def mm_to_inches(mm):
    return mm / 25.4


def pct_against_chance(data, shuffled_data, n_conditions=2):
    f_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
        [data, shuffled_data],  # List with single condition
        n_permutations=1000,
        tail=1,  # One-tailed (accuracy > chance)
        n_jobs=-1,  # Use all CPUs
        seed=42,
        threshold=None
    )

    return f_obs, clusters, cluster_p_values, h0


def extract_condition_df(output, condition_key):
    """
    Extract REAL decoding results for a specific condition from loaded output dict.

    Real-score entries have structure:
        {'bin': (start, end), 'scores': {'test_score': array(...)}, 'session_ID': ...}

    Returns:
        pd.DataFrame with columns: 'bin_start', 'bin_end', 'mean_score',
                                   'all_scores', 'session_ID'
    """

    if condition_key not in output:
        return None

    results = output[condition_key]
    data = []

    for entry in results:
        bin_window = tuple(entry['bin'])
        scores = entry['scores']['test_score']
        mean_score = np.mean(scores)
        session_ID = entry['session_ID']

        data.append({
            'bin_start': bin_window[0],
            'bin_end': bin_window[1],
            'mean_score': mean_score,
            'all_scores': scores,
            'session_ID': session_ID
        })

    return pd.DataFrame(data)


def extract_shuffled_condition_df(output, condition_key):
    """
    Extract SHUFFLED decoding results for a specific condition.

    Shuffled entries have structure:
        {'bin': (start, end), 'shuffled_scores': array(N_SHUFFLES,), 'session_ID': ...}

    Each entry's `shuffled_scores` is the full null distribution (1000 permutation means)
    for that session/bin. We treat the per-session/per-bin mean over shuffles as the null
    point estimate, so this DataFrame has the same shape and downstream usage as the
    real-score one.

    Returns:
        pd.DataFrame with columns: 'bin_start', 'bin_end', 'mean_score',
                                   'all_scores', 'session_ID'
    """

    if condition_key not in output:
        return None

    results = output[condition_key]
    data = []

    for entry in results:
        bin_window = tuple(entry['bin'])
        shuffled_scores = np.asarray(entry['shuffled_scores'])
        mean_score = np.mean(shuffled_scores)
        session_ID = entry['session_ID']

        data.append({
            'bin_start': bin_window[0],
            'bin_end': bin_window[1],
            'mean_score': mean_score,
            'all_scores': shuffled_scores,
            'session_ID': session_ID
        })

    return pd.DataFrame(data)


def plot_decoding_over_time(df, label='results', ax=None, error='std', alpha=1, color=None, x_time=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    grouped = df.groupby(['bin_start', 'bin_end'])['mean_score']
    print(grouped.size().describe())  # should equal n_sessions at every bin
    bin_end = [end for start, end in grouped.groups]
    if x_time is None:
        x_time = bin_end

    mean_scores = grouped.mean()
    if error == 'std':
        error_scores = grouped.std()
    elif error == 'sem':
        error_scores = grouped.sem()

    ax.plot(x_time, mean_scores, label=label, alpha=alpha, color=color)
    ax.fill_between(x_time, mean_scores - error_scores, mean_scores + error_scores, alpha=0.3, color=color, linewidth=0)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding accuracy (balanced)")
    return ax


def calculate_std(score_arrays):
    all_values = np.concatenate(score_arrays.values)
    return np.std(all_values)


def improvement_from_baseline(df):
    grouped_mean = df.groupby(['bin_start', 'bin_end'])['mean_score'].mean().reset_index()
    bin_end = grouped_mean['bin_end']
    pre_stim_mask = (grouped_mean['bin_start'] < 1e-10) & (grouped_mean['bin_end'] < 1e-10)
    pre_stim_data = grouped_mean.loc[pre_stim_mask, 'mean_score']
    pre_stim_mean = pre_stim_data.mean()
    max_possible_improvement = 1 - pre_stim_mean
    score = (grouped_mean['mean_score'] - pre_stim_mean) / max_possible_improvement
    return score, bin_end


def print_session_summary(df, df_shuffled, condition_name):
    """
    Print mean accuracy per session for a given condition.
    """
    print(f"\n{'='*60}")
    print(f"  Session summary for: {condition_name}")
    print(f"{'='*60}")
    print(f"  {'Session ID':<20} {'Mean Acc':>10} {'Mean Shuf':>10} {'Diff':>10}")
    print(f"  {'-'*50}")

    session_ids = df['session_ID'].unique()
    for ses_id in sorted(session_ids, key=str):
        ses_mean = df[df['session_ID'] == ses_id]['mean_score'].mean()
        ses_shuf_mean = np.nan
        if df_shuffled is not None and ses_id in df_shuffled['session_ID'].values:
            ses_shuf_mean = df_shuffled[df_shuffled['session_ID'] == ses_id]['mean_score'].mean()
        diff = ses_mean - ses_shuf_mean if not np.isnan(ses_shuf_mean) else np.nan
        print(f"  {str(ses_id):<20} {ses_mean:>10.4f} {ses_shuf_mean:>10.4f} {diff:>10.4f}")

    overall_mean = df['mean_score'].mean()
    print(f"  {'-'*50}")
    print(f"  {'OVERALL':<20} {overall_mean:>10.4f}")
    print()


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
    cbp_test = True
    save_dir = config.FIGURE_DIR / "SVM" / "decode_from_motion"

    window = [-1, 1.5]
    bins_time = np.linspace(window[0], window[1], 201)
    bins_time = bins_time + 0.5 * (1 / 80)
    bins_time = bins_time[:-1]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    conditions = [('results_au_presence', 'results_shuffled_au_presence'),
                  ('results_vis_presence', 'results_shuffled_vis_presence'),
                  ('results_trial_type', 'results_shuffled_trial_type'),
                  ('results_response_side', 'results_shuffled_response_side'),
                  ('results_response_presence', 'results_shuffled_response_presence'),
                  ('results_vis_side_miss', 'results_shuffled_vis_side_miss'),
                  ('results_vis_side_hit', 'results_shuffled_vis_side_hit'),
                  ('results_vis_side', 'results_shuffled_vis_side'),
                  ('results_vis_ori_left_all', 'results_shuffled_vis_ori_left_all'),
                  ('results_vis_ori_right_all', 'results_shuffled_vis_ori_right_all'),
                  ('results_vis_ori_left_miss', 'results_shuffled_vis_ori_left_miss'),
                  ('results_vis_ori_right_miss', 'results_shuffled_vis_ori_right_miss'),
                  ('results_vis_ori_left_hit', 'results_shuffled_vis_ori_left_hit'),
                  ('results_vis_ori_right_hit', 'results_shuffled_vis_ori_right_hit'),
                  ('results_au_pitch_all', 'results_shuffled_au_pitch_all'),
                  ('results_au_pitch_miss', 'results_shuffled_au_pitch_miss'),
                  ('results_au_pitch_hit', 'results_shuffled_au_pitch_hit'),
                  ('results_vis_side_opto', 'results_shuffled_vis_side_opto'),
                  ('results_vis_side_opto_control', 'results_shuffled_vis_side_opto_control'),
                  ('results_au_pitch_all_opto', 'results_shuffled_au_pitch_all_opto'),
                  ('results_au_pitch_all_opto_control', 'results_shuffled_au_pitch_all_opto_control'),
                  ('results_vis_hit_aud_hit', 'results_shuffled_vis_hit_aud_hit'),
                  ('results_vis_hit_miss', 'results_shuffled_vis_hit_miss'),
                  ('results_aud_hit_miss', 'results_shuffled_aud_hit_miss')
                  ]

    decoding_titles = ['Auditory Presence (CR vs Miss)', 'Visual Presence (CR vs Miss',
                       'Trial Type (X,Y,P)', 'Response Side (L,R,N)', 'Lick Presence',
                       'Visual Side misses', 'Visual Side Hits', 'Visual side all trials',
                       'Visual orientation left all trials', 'Visual orientation right all trials',
                       'Visual orientation left misses', 'Visual orientation right misses',
                       'Visual orientation left hits', 'Visual orientation right hits',
                       'Auditory pitch all trials', 'Auditory pitch misses', 'Auditory pitch hits',
                       'Visual side opto', 'Visual side opto control',
                       'Auditory pitch opto', 'Auditory pitch opto control',
                       'Visual hit vs auditory hit',
                       'Visual hit vs miss',
                       'Auditory hit vs miss']

    res = np.load(os.path.join(data_dir, 'output.npy'), allow_pickle=True).item()

    i = 0
    errorbars_type = 'sem'

    for cond, cond_shuffled in conditions:
        print(cond)
        df = extract_condition_df(res, cond)
        df_shuffled = extract_shuffled_condition_df(res, cond_shuffled)

        if df is None or df_shuffled is None or df.empty or df_shuffled.empty:
            print(f"{cond} or {cond_shuffled} are empty, skipping")
            i += 1
            continue

        # --- Print per-session summary BEFORE filtering ---
        print_session_summary(df, df_shuffled, cond)

        # --- Exclude sessions ---
        if exclude_sessions:
            df['session_ID'] = df['session_ID'].astype(str)
            df_shuffled['session_ID'] = df_shuffled['session_ID'].astype(str)
            exclude_set = set(str(s) for s in exclude_sessions)

            n_before = df['session_ID'].nunique()
            df = df[~df['session_ID'].isin(exclude_set)]
            df_shuffled = df_shuffled[~df_shuffled['session_ID'].isin(exclude_set)]
            n_after = df['session_ID'].nunique()

            if n_before != n_after:
                print(f"  Excluded {n_before - n_after} session(s), {n_after} remaining")

            if df.empty or df_shuffled.empty:
                print(f"  No sessions remaining after exclusion, skipping {cond}")
                i += 1
                continue

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

        accuracy_per_session = []
        accuracy_per_session_shuffled = []

        g = df.groupby(['bin_start', 'bin_end'])['mean_score']
        print(df['session_ID'].nunique(), 'sessions')
        print(g.size().describe())  # mean should = n_sessions
        print('SD at peak:', g.std().max(), 'SEM at peak:', g.sem().max())

        for ses_id, df_ses in df.groupby('session_ID'):
            ax_ses = plot_decoding_over_time(df_ses, 'Not shuffled', error=errorbars_type, ax=ax_ses, alpha=0.25,
                                             x_time=bins_time)
            ax_ses.set_ylim(0.1, 1)
            grouped = df_ses.groupby(['bin_start', 'bin_end'])['mean_score']
            bin_end = [end for start, end in grouped.groups]
            session_row = grouped.mean().values
            accuracy_per_session.append(session_row)

        for ses_id, df_ses in df_shuffled.groupby('session_ID'):
            grouped = df_ses.groupby(['bin_start', 'bin_end'])['mean_score']
            bin_end = [end for start, end in grouped.groups]
            session_row = grouped.mean().values
            accuracy_per_session_shuffled.append(session_row)

        if cbp_test:
            f_obs, clusters, cluster_p_values, h0 = pct_against_chance(
                np.array(accuracy_per_session),
                np.array(accuracy_per_session_shuffled)
            )

        heatmap_data = np.vstack(accuracy_per_session)
        ax_heat.pcolormesh(heatmap_data, vmin=0, vmax=1)

        ax = plot_decoding_over_time(df_shuffled, 'Shuffled', ax=ax, error=errorbars_type, color='gray',
                                     x_time=bins_time)
        ax = plot_decoding_over_time(df, 'Not shuffled', error=errorbars_type, ax=ax, color='blue',
                                     x_time=bins_time)
        ax.set_ylim(0.4, 1)

        if cbp_test:
            alpha = 0.05
            for idx, clust in enumerate(clusters):
                if cluster_p_values[idx] < alpha:
                    print('we got there')
                    clust_idx = np.asarray(clust[0] if isinstance(clust, tuple) else clust)
                    if clust_idx.dtype == bool:
                        clust_idx = np.where(clust_idx)[0]

                    x_start = bins_time[clust_idx.min()]
                    x_end = bins_time[clust_idx.max()]
                    width = x_end - x_start

                    y_min, y_max = ax.get_ylim()
                    figure_height = y_max - y_min
                    rect_height = 0.05 * figure_height
                    rect_y_position = y_min

                    # DEBUG
                    print(type(clust), np.asarray(clust).dtype, np.asarray(clust).shape)
                    print('min/max idx:', np.min(clust), np.max(clust))
                    print('x_start, x_end:', x_start, x_end)
                    print('len(bin_end):', len(bin_end), 'xlim:', ax.get_xlim())

                    rect = Rectangle((x_start, rect_y_position),
                                     width, rect_height,
                                     facecolor='red',
                                     alpha=0.3,
                                     edgecolor='red',
                                     linewidth=1)
                    ax.add_patch(rect)

        ax.set_title(f"Decoding: {decoding_titles[i]}")
        ax_ses.set_title(f"Decoding: {decoding_titles[i]}")
        i += 1

        fig.set_dpi(600)
        fig.tight_layout()
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

    print("end of script")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = config.resolve_pp("SVM", "decode_from_motion")

    # --- EXCLUDE SESSIONS HERE ---
    exclude = ['006202306051450', '006202306061601', '023202401221608', '023202401201423']

    # Or pass via command line: python script.py /path/to/data --exclude ses1 ses2 ses3
    if '--exclude' in sys.argv:
        idx = sys.argv.index('--exclude')
        exclude = sys.argv[idx + 1:]
        sys.argv = sys.argv[:idx]

    main(data_dir, exclude_sessions=exclude)