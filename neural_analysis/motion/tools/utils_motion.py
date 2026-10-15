import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import convolve1d
import os
from scipy.stats import norm

def extract_frames(timeline, frames_array, ref_times, window, n_components, sampling_interval=None):
    """
    Extracts frames from motSVD_face surrounding each reference time within the given window.

    Parameters:
    timeline (list or np.array): List of timestamps in microseconds.
    frames_array (np.array): Array of shape (1, n_timepoints, n_components).
    ref_times (list or np.array): List of reference times in microseconds.
    window (tuple): Time window (start, end) in seconds.
    n_components (int): Number of components to extract.

    Returns:
    np.array: 3D matrix with shape (n_refpoints, timepoints, n_components)
    """
    timeline = np.array(timeline)
    frames_array = np.array(frames_array)
    ref_times = np.array(ref_times)
    good_trials = np.ones(len(ref_times), dtype=bool)
    window_micro = (window[0] * 1e6, window[1] * 1e6)

    # Check if reference points fall within video timeline
    if not np.any((ref_times >= np.min(timeline)) & (ref_times <= np.max(timeline))):
        raise ValueError("Not all reference points fall within video timeline")

    # Compute expected number of timepoints based on sampling interval
    if sampling_interval is None:
        sampling_interval = np.median(np.diff(timeline))  # Assume uniform sampling

    num_timepoints = int(np.round((window_micro[1] - window_micro[0]) / (sampling_interval*1e6)))
    if n_components is None or n_components in [0, 1]:
        extracted_data = np.full((len(ref_times), num_timepoints), np.nan)
    else:
        extracted_data = np.full((len(ref_times), num_timepoints, n_components), np.nan)

    for i, ref_time in enumerate(ref_times):
        start_time = ref_time + window_micro[0]
        end_time = ref_time + window_micro[1]

        mask = (timeline >= start_time) & (timeline <= end_time)
        if n_components is None or n_components in [0, 1]:
            selected_frames = frames_array[mask]
        else:
            selected_frames = frames_array[mask, :n_components]

        # Ensure exact number of timepoints by indexing properly
        if selected_frames.shape[0] == num_timepoints:
            extracted_data[i] = selected_frames
        else:
            extracted_data[i] = np.nan
            good_trials[i] = False

        #elif selected_frames.shape[0] < num_timepoints:
        #    extracted_data[i, :selected_frames.shape[0], :] = selected_frames
        #else:
        #    extracted_data[i, :, :] = selected_frames[:num_timepoints, :]
    #if np.any(np.isnan(extracted_data)):
    #    print(f"Some trials did not have all frames "
    #          f"- Trial removed. Make sure to use the returned labels for trial ID!")

    extracted_data = extracted_data[good_trials, ...]
    return extracted_data, good_trials


def plot_summed_motion(trial_aligned_motion, labels=None, smooth_sigma=None):
    """
    Plots the mean of trial_aligned_motion across reference points and components.
    If labels are provided, it plots separate lines for each unique label,
    including the number of trials per condition in the legend.

    Parameters:
    trial_aligned_motion (np.array): 3D matrix (n_refpoints, timepoints, n_components)
    labels (np.array, optional): 1D array of length n_refpoints, indicating label types.
    """
    kernel = np.ones(3) / 3
    fig, ax = plt.subplots(figsize=(8, 6))
    values = np.square(trial_aligned_motion)
    mean_out = []
    sem_out = []

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            num_trials = np.sum(mask)  # Count the number of trials per condition
            sum_values = np.sum(values[mask], axis=2)

            if smooth_sigma is not None:
                sum_values = scipy.ndimage.gaussian_filter1d(sum_values, axis=1, sigma=smooth_sigma)

            # Remove rows with any NaN
            sum_values = sum_values[~np.isnan(sum_values).any(axis=1)]
            #sum_values = convolve1d(sum_values, kernel, mode='constant', axis=1)

            # Calculate mean and SEM
            mean_values = np.mean(sum_values, axis=0)
            sem_values = np.std(sum_values, axis=0) / np.sqrt(num_trials)  # SEM

            mean_out.append(mean_values)
            sem_out.append(sem_values)
            plt.plot(mean_values, label=f'Label {label} (n={num_trials})')
            plt.fill_between(range(len(mean_values)), mean_values - sem_values, mean_values + sem_values, alpha=0.3)

        plt.legend()
    else:
        sum_values = np.sum(values, axis=(0, 2))
        if smooth_sigma is not None:
            sum_values = scipy.ndimage.gaussian_filter1d(sum_values, axis=1, sigma=smooth_sigma)
        #sum_values = convolve1d(sum_values, kernel, mode='constant', axis=1)
        mean_values = np.mean(sum_values, axis=0)
        sem_values = np.std(values, axis=0) / np.sqrt(values.shape[0])  # SEM

        plt.plot(sum_values, label=f"All Data (n={values.shape[0]})")
        plt.fill_between(range(len(mean_values)), mean_values - sem_values, mean_values + sem_values, alpha=0.3)

    plt.xlabel('Timepoints')
    plt.ylabel('Summed Motion')
    plt.title('Summed Motion Across Trials')
    plt.legend()

    return fig, mean_out, sem_out, sum_values


def sum_motion(motSVD, n_components):
    """
    Plots the mean of trial_aligned_motion across reference points and components.
    If labels are provided, it plots separate lines for each unique label.

    Parameters:
    trial_aligned_motion (np.array): 3D matrix (n_refpoints, timepoints, n_components)
    labels (np.array, optional): 1D array of length n_refpoints, indicating label types.
    """
    data = motSVD[:, 0:n_components]
    data = np.square(data)
    summed_data = np.sum(data, axis=1)

    return summed_data

def sum_motion_3D(motSVD):
    """
    Squares and sums across the last dimension of a 3D matrix motSVD
    with shape (n_refpoints, timepoints, n_components).

    Parameters:
    motSVD (np.ndarray): 3D array of shape (n_refpoints, timepoints, n_components)

    Returns:
    np.ndarray: A 2D array of shape (n_refpoints, timepoints) where each entry
                is the sum of squared values across all components.
    """
    # Square the entire array
    squared = np.square(motSVD)
    # Sum over the last dimension (axis=2), flattening all components
    summed = np.sum(squared, axis=2)
    return summed


def plot_neuron_heatmap(data, time_axis, trial_axis, neuron_axis):
    """
    Averages over trials for each neuron and plots a heatmap.

    Parameters:
    - data: 3D numpy array with dimensions (neurons, trials, time) in any order
    - time_axis: int, index of the time dimension
    - trial_axis: int, index of the trial dimension
    - neuron_axis: int, index of the neuron dimension
    """
    # Ensure axes are in the correct order (neurons, trials, time)
    data = np.moveaxis(data, [neuron_axis, trial_axis, time_axis], [0, 1, 2])

    # Average over trials (axis 1 after reordering)
    avg_data = np.mean(data, axis=1)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_data, cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    plt.title('Neuron Activity Heatmap')
    plt.show()

    return avg_data


def plot_still_frames(motion, time, threshold):
    plt.figure()

    still_frames = motion < threshold
    still_frames_idx = np.where(still_frames)[0]
    streaks = np.split(still_frames_idx, np.where(np.diff(still_frames_idx) > 1)[0] + 1)
    n_still_frames = len(still_frames_idx)

    plt.plot(time, motion)
    for streak in streaks:
        if len(streak) > 0:
            plt.plot([time[streak[0]], time[streak[-1]]], [0, 0], 'r-', linewidth=2)

    print(f"Number of still frames: {n_still_frames} ({(n_still_frames / len(motion)) * 100} %)")
    plt.show()


def z_score_motion(motion_data, baseline_window, data_window):
    """Z-score motion data using baseline window."""
    num_timepoints = motion_data.shape[1]
    time_axis = np.linspace(data_window[0], data_window[1], num_timepoints)

    baseline_indices = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    baseline_data = motion_data[:, baseline_indices]

    z_scored_data = (motion_data - np.mean(baseline_data)) / np.std(baseline_data)
    return z_scored_data


def plot_grand_average(motion_data_dict, title, save_path, timeline, sigma=1,
                       per_label=True, z_score=False, label_ignore=None, legend=None,
                       window=(-1, 1.5), baseline_window=(-1, -0.2)):
    """
    Plots the grand average motion data.
    NOTE: z_score parameter is kept for compatibility but should be False
    as data should already be z-scored if needed.

    Parameters:
    - motion_data_dict: dict[label -> list of motion traces]
    - title: str, plot title
    - save_path: str, full path to save the plot
    - sigma: float, Gaussian smoothing sigma
    - per_label: bool, if True, plot each label separately; if False, average across all
    - z_score: bool, should be False (data is pre-processed)
    - window: tuple, data window for z-scoring
    - baseline_window: list, baseline window for z-scoring
    """
    if label_ignore is None:
        label_ignore = []

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.5, 10))
    i = 0

    if per_label:
        for label, traces in motion_data_dict.items():
            if label in label_ignore:
                continue
            traces = np.array(traces)

            # Data should already be z-scored if needed - don't z-score again
            mean_data = np.mean(traces, axis=0)
            std_data = np.std(traces, axis=0)
            sem_data = std_data / np.sqrt(traces.shape[0])  # Standard error of mean

            smoothed_mean = scipy.ndimage.gaussian_filter1d(mean_data, sigma)
            smoothed_sem = scipy.ndimage.gaussian_filter1d(sem_data, sigma)

            label_name = legend[i] if legend and i < len(legend) else f'Label {label}'
            plt.plot(timeline, smoothed_mean, label=f'{label_name} (n={traces.shape[0]})',
                     color=colors[i], linewidth=2)
            plt.fill_between(timeline,
                             smoothed_mean - smoothed_sem,
                             smoothed_mean + smoothed_sem,
                             alpha=0.3, color=colors[i])
            i += 1
    else:
        all_traces = []
        for label, traces in motion_data_dict.items():
            if label not in label_ignore:
                all_traces.extend(traces)

        if not all_traces:
            print(f"[Warning] No data to plot for: {title}")
            return

        all_traces = np.array(all_traces)

        # Data should already be z-scored if needed - don't z-score again
        mean_data = np.mean(all_traces, axis=0)
        std_data = np.std(all_traces, axis=0)
        sem_data = std_data / np.sqrt(all_traces.shape[0])

        smoothed_mean = scipy.ndimage.gaussian_filter1d(mean_data, sigma)
        smoothed_sem = scipy.ndimage.gaussian_filter1d(sem_data, sigma)

        plt.plot(timeline, smoothed_mean, label=f'Grand Average (n={all_traces.shape[0]})', linewidth=2)
        plt.fill_between(timeline,
                         smoothed_mean - smoothed_sem,
                         smoothed_mean + smoothed_sem,
                         alpha=0.3, label='±SEM')

    plt.xlabel('Time from stimulus (s)', fontsize=12)
    plt.ylabel('Motion Energy (a.u.)' + (' (z-scored)' if z_score else ''), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Stimulus onset')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()

    # Save both PNG and SVG versions
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".svg", bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}.png and .svg")


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_session_data(session, sessionData, trialData, config, SVD_dict_per_session):
    """Process motion data for a single session."""
    session_folder = os.path.basename(session['session'])
    sessionData_ses = sessionData[sessionData['folderName'] == session_folder]

    if sessionData_ses.empty:
        return None

    sampling_interval = 1 / 25 if (sessionData_ses['experiment'] == 'ALM_1').iloc[0] else 1 / 80
    session_ID = sessionData_ses['session_ID'].item()
    trialData_ses = trialData[trialData['session_ID'] == session_ID]

    # Store data for plotting
    session_plot_data = {}

    # Process each condition
    for condition_name, condition_params in config.conditions.items():
        trial_type = condition_params['trial_type']
        no_response = condition_params['no_response']

        # Select trials based on trial type
        trialData_cond = trialData_ses[
            (trialData_ses['trialType'].isin(trial_type)) &
            (trialData_ses['noResponse'] == no_response)
            ]

        if trialData_cond.empty:
            continue

        # Extract motion data
        trial_aligned_motion, good_trials = extract_frames(
            session['timeline'],
            session['motSVD_face'][1],
            trialData_cond['stimChange'],
            config.window,
            config.n_components,
            sampling_interval
        )

        # Filter trial data for good trials
        trialData_cond = trialData_cond[good_trials]

        # Create labels based on condition definitions
        labels = np.zeros(len(trialData_cond))
        for label_value, label_func in condition_params['labels'].items():
            mask = label_func(trialData_cond)
            labels[mask] = label_value

        # Sum motion across SVD components
        summed_motion = sum_motion_3D(trial_aligned_motion)

        # Only keep trials with valid labels (exclude label 0)
        valid_trials = labels > 0
        if not np.any(valid_trials):
            print(f"Warning: No valid trials for {condition_name} in session {session_ID}")
            continue

        # Filter data to only include valid labeled trials
        summed_motion = summed_motion[valid_trials]
        trial_aligned_motion = trial_aligned_motion[valid_trials]
        labels = labels[valid_trials]

        # Store data for plotting
        session_plot_data[condition_name] = {
            'summed_motion': summed_motion,
            'trial_aligned_motion': trial_aligned_motion,
            'labels': labels,
            'condition_params': condition_params
        }

        # Save trial-level data
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label not in SVD_dict_per_session[condition_name]:
                SVD_dict_per_session[condition_name][label] = []
            SVD_dict_per_session[condition_name][label].append(
                summed_motion[labels == label]
            )

    # Plot all conditions together if requested
    if config.plot_per_session and session_plot_data:
        plot_session_data_combined(session_plot_data, session_ID, config)

    return session_ID


def plot_session_data_combined(session_plot_data, session_ID, config):
    """Plot data for a single session with all conditions combined."""
    time_axis = np.linspace(config.window[0], config.window[1], config.num_timepoints)

    # Create figure with 2 rows per condition
    n_conditions = len(session_plot_data)
    fig, axes = plt.subplots(n_conditions * 2, 1, figsize=(12, 5 * n_conditions))

    if n_conditions == 1:
        axes = [axes[0], axes[1]]

    ax_idx = 0
    for condition_name, data in session_plot_data.items():
        ax1 = axes[ax_idx]
        ax2 = axes[ax_idx + 1]

        summed_motion = data['summed_motion']
        trial_aligned_motion = data['trial_aligned_motion']
        labels = data['labels']
        condition_params = data['condition_params']

        # Plot mean motion for each label group (should only be 1 and 2)
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            if np.sum(label_mask) == 0:
                continue

            label_motion = summed_motion[label_mask]
            mean_data = np.mean(label_motion, axis=0)
            std_data = np.std(label_motion, axis=0)
            smoothed_mean = scipy.ndimage.gaussian_filter1d(mean_data, config.smooth_sigma)
            smoothed_std = scipy.ndimage.gaussian_filter1d(std_data, config.smooth_sigma)

            # Get legend label
            legend_labels = condition_params.get('legend', [])
            legend_label = legend_labels[int(label) - 1] if len(legend_labels) >= label else f'Label {label}'

            ax1.plot(time_axis, smoothed_mean, color=colors[i],
                     label=f'{legend_label} (n={np.sum(label_mask)})')
            ax1.fill_between(time_axis,
                             smoothed_mean - smoothed_std,
                             smoothed_mean + smoothed_std,
                             alpha=0.2, color=colors[i])

        ax1.set_title(f'{session_ID} - {condition_name}')
        ax1.set_xlabel('Time from stimulus (s)')
        ax1.set_ylabel('Motion Energy (a.u.)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Plot snake motion with labels
        plot_snake_motion(trial_aligned_motion, labels=labels, ax=ax2)
        ax2.set_title(f'Trial-by-trial motion - {condition_name}')

        ax_idx += 2

    plt.tight_layout()

    # Save figure
    motion_dir = os.path.join(config.output_dir, "session_plots")
    ensure_dir(os.path.join(motion_dir, 'png'))
    ensure_dir(os.path.join(motion_dir, 'svg'))

    plt.savefig(os.path.join(motion_dir, 'png', f'{session_ID}.png'))
    plt.savefig(os.path.join(motion_dir, 'svg', f'{session_ID}.svg'))
    plt.close()


def plot_snake_motion(trial_aligned_motion, labels=None, ax=None):
    """
    Plots a heatmap (snake plot) of trial_aligned_motion where each row represents a trial,
    and values are the mean over all components. If labels are provided, the data is sorted
    based on labels, and demarcation lines are added to separate label groups.

    Parameters:
    trial_aligned_motion (np.array): 3D matrix (n_refpoints, timepoints, n_components)
    labels (np.array, optional): 1D array of length n_refpoints, indicating label types.
    """
    if ax == None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()  # get current axes

    #trial_aligned_motion = np.abs(trial_aligned_motion)
    mean_over_components = np.mean(trial_aligned_motion, axis=2)  # Collapse components

    if labels is not None:
        if not len(labels) == trial_aligned_motion.shape[0]:
            print("Number of labels is not equal not number of trials")
            return

        sorted_indices = np.argsort(labels)  # Sort indices by labels
        mean_over_components = mean_over_components[sorted_indices]  # Reorder trials
        sorted_labels = labels.iloc[sorted_indices]

        # Find where label changes occur
        unique_labels, label_change_indices = np.unique(sorted_labels, return_index=True)

    ax.imshow(mean_over_components, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add demarcation lines for label groups
    if labels is not None:
        for idx in label_change_indices[1:]:  # Skip the first (0) index
            ax.axhline(y=idx - 0.5, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Timepoints')
    ax.set_ylabel('Trials (Sorted by Label)')
    ax.set_title('Snake Plot of Motion Data')
    #plt.colorbar(label='Mean Motion')

    return ax


def compute_grand_averages(SVD_dict_per_session, config):
    """Compute grand averages across all sessions."""
    grand_average = {}
    grand_average_z = {}

    for condition_name in SVD_dict_per_session:
        if condition_name not in grand_average:
            grand_average[condition_name] = {}
            grand_average_z[condition_name] = {}

        for label in SVD_dict_per_session[condition_name]:
            all_trials = []
            all_trials_z = []

            print(f"Processing {condition_name} - Label: {label}")

            # Z-score each session independently, then concatenate
            for ses in SVD_dict_per_session[condition_name][label]:
                # Each session's baseline should be computed from its own data
                ses_z = z_score_motion(
                    ses,
                    baseline_window=config.baseline_window,
                    data_window=config.window
                )
                all_trials.append(ses)
                all_trials_z.append(ses_z)

            if all_trials:
                # Concatenate raw trials
                all_trials_concat = np.concatenate(all_trials, axis=0)
                grand_average[condition_name][label] = all_trials_concat

                # Concatenate z-scored trials (already z-scored per session)
                all_trials_concat_z = np.concatenate(all_trials_z, axis=0)
                grand_average_z[condition_name][label] = all_trials_concat_z

    return grand_average, grand_average_z


def causal_gaussian_kernel(sigma_sec, fps):
    sigma_frames = sigma_sec * fps
    size = int(sigma_frames)
    size = max(size, 1)
    x = np.arange(0, size)
    kernel = norm.pdf(x, loc=0, scale=sigma_frames)
    kernel /= kernel.sum()
    return kernel[::-1]


def bad_motion_sessions():
    sessions = ['006202306031528',
                '006202306051450',
                '006202306061601',
                '021202401311421',
                '023202401201423'
                ]

    return sessions
