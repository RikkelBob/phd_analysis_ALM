from neural_analysis import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import neural_analysis.motion.tools.utils_motion as utils_motion
from scipy.ndimage import convolve1d
from subspaceID import funcs
import shutil
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

import config

# Function to handle click event
def onclick(event):
    global selected_bin
    if event.xdata is not None:
        selected_bin = event.xdata
        print(f"Selected bin: {selected_bin:.2f}")


def causal_gaussian_kernel(sigma_sec, fps):
    sigma_frames = sigma_sec * fps
    size = int(sigma_frames)
    size = max(size, 1)
    x = np.arange(0, size)
    kernel = norm.pdf(x, loc=0, scale=sigma_frames)
    kernel /= kernel.sum()
    return kernel[::-1]


# Set data paths
root_path_video = config.DATA_DIR_PP / "motion" / "facemap"
data_dir = config.DATA_DIR_EXP
areas = ['lALM', 'rALM']
area_subdir = "".join(areas)
output_dir = config.DATA_DIR_PP_NEW / "null_potent_subspace" / "subspace_results" / area_subdir

# Load movement threshold values from existing output_dir
load_thresholds = True
subdir_subspace = "lALMrALM"  # Subdirectory of previous run
threshold_dir = config.DATA_DIR_PP / "null_potent_subspace/subspace_results" / subdir_subspace
window = (-2, 2)
#kernel = np.ones(3) / 3  # Moving average kernel of size 3 (Smoothing kernel for motion and spiking)

min_neurons = 10
min_fire = 0.5

experiments = ['ALM_1', 'ALM_2', 'ALM_3']

all_session_paths = utils.get_session_paths(root_path_video)

# Load behavioral and neural data
trialData, sessionData, spikeData = utils.load_data(path_root=data_dir, experiment=experiments, animals='all')
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, 'good')
spikeData = spikeData[spikeData['area'].isin(areas)]

ses_all = []

for idx, sessionData_ses in sessionData.iterrows():
    ses = sessionData_ses['session_ID']
    ses_save_folder = os.path.join(output_dir, str(ses))

    if os.path.isdir(ses_save_folder):
        print(f"Folder {ses_save_folder} already exists, skipping.")
        ses_all.append(ses)
        continue

    # Find the video folder associated with the session
    session_path = [s for s in all_session_paths if sessionData_ses['folderName'] in s]
    print(session_path)

    if not session_path:
        continue

    if not os.path.isfile(os.path.join(session_path[0], 'realigned_timeline_nlx_ts.npy')):
        print(f"{ses} has no realigned timeline. Skipping")
        continue

    if sessionData_ses['experiment'] == 'ALM_1':
        sampling_interval = 1 / 25
    else:
        sampling_interval = 1 / 80

    kernel = causal_gaussian_kernel(0.2, 1/sampling_interval)
    print(f"KERNEL IS LENGTH {len(kernel)}")
    # Filter data for specific session
    trialData_ses = trialData[(trialData['session_ID'] == ses) & (trialData['hasphotostim'] == False)]
    spikeData_ses = spikeData[spikeData['session_ID'] == ses]

    # Check if there are enough neurons
    if len(spikeData_ses) < min_neurons:
        print(f"{ses} has fewer than {min_neurons} neurons. Skipping")
        continue

    # Define spike data binning
    bins, time_bins = utils.custom_bins(bin_size=sampling_interval*2, step_size=sampling_interval, window=window, causal=True)
    # Remove first bin (motion frame 0 is always > start_time, motion from -1 always < end_time
    bins = bins[1:]

    # Get trial-aligned spiking data (3D psth) from spikeData and trialData
    matrix = utils.psth_matrix_2(spikeData_ses,
                                 trialData_ses,
                                 window=window,
                                 bins=bins,
                                 z_score_data=True)

    # Sanity check plot
    # utils_motion.plot_neuron_heatmap(matrix, time_axis=2, trial_axis=1, neuron_axis=0)

    # Reshape to match subspaceID requirements (bins x trials x neurons)
    matrix = matrix.transpose(2, 1, 0)

    # Sanity check plot
    # utils_motion.plot_neuron_heatmap(matrix, time_axis=0, trial_axis=1, neuron_axis=2)

    # Load the motion file
    motion_type = 'motion'
    n_components = 500

    if motion_type == 'motion':
        motion = utils.get_motion(session_path, timeline_file="realigned_timeline_nlx_ts.npy")
        motion_raw = motion[0]['motion'][1]  # n_frames x n_components
        motion_smooth = np.convolve(motion_raw, kernel, mode='full')[:len(motion_raw)]
        motion_median = np.median(motion_smooth)
        n_components = 1

    elif motion_type == 'motSVD':
        motion = utils.get_motSVD(session_path, timeline_file="realigned_timeline_nlx_ts.npy")

        motion_raw = motion[0]['motSVD_face'][1]  # n_frames x n_components
        motion_summed = utils_motion.sum_motion(motion_raw, n_components=n_components)
        motion_smooth = np.convolve(motion_summed, kernel, mode='full')[:len(motion_raw)]
        motion_median = np.median(motion_smooth)
    else:
        print(f"ERROR: Invalid motion_type: {motion_type}")
        break

    motion_timeline = motion[0]['timeline']

    # Get trial-aligned motion from motion data
    trial_aligned_motion, good_trials = utils_motion.extract_frames(timeline=motion_timeline,
                                                                    frames_array=motion_smooth,
                                                                    ref_times=trialData_ses['stimChange'],
                                                                    window=window,
                                                                    n_components=1,
                                                                    sampling_interval=sampling_interval)


    # Exclude trials that contains nans (this means there were missing frames)
    valid_rows = ~np.isnan(trial_aligned_motion[:, :]).any(axis=1)
    trial_aligned_motion = trial_aligned_motion[valid_rows, :]

    # Also exclude these trials from spike data
    matrix = matrix[:, good_trials, :]
    matrix = matrix[:, valid_rows, :]

    # Get number of  bins, trials, neurons and time axis
    n_bins = np.shape(trial_aligned_motion)[1]
    n_trials = np.shape(trial_aligned_motion)[0]
    n_neurons = len(spikeData_ses)
    time = np.linspace(window[0],window[1], n_bins)

    # Process motion data (Smoothed, squared sums)
    #if motion_type == 'motSVD':
    #    trial_aligned_motion = utils_motion.sum_motion_3D(trial_aligned_motion)

    #trial_aligned_motion = convolve1d(trial_aligned_motion, kernel, mode='constant', axis=1)
    trial_aligned_motion = np.transpose(trial_aligned_motion)

    # Plotting for debugging
    #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    #sns.heatmap(trial_aligned_motion.T, cmap='viridis', ax=ax1)
    #ax1.set_xlabel('Time')
    #ax1.set_ylabel('Trials')
    #ax1.set_title('Motion heatmap')
###
    #sns.heatmap(np.mean(matrix, 1).T, cmap='viridis', ax=ax2)
    #ax2.set_xlabel('Time')
    #ax2.set_ylabel('Neurons')
    #ax2.set_title('Spiking heatmap')
    #plt.show()

    # Determine threshold
    #motion_raw = motion[0]['motSVD_face'][1] # n_frames x n_components
    #motion_summed = utils_motion.sum_motion(motion_raw, n_components=n_components)

    #motion_smooth = np.convolve(motion_summed, kernel, mode='same')
    #motion_median = np.median(motion_smooth)
    

    # Load pre-selected motion threshold values or manually select per session
    if load_thresholds and os.path.isfile(os.path.join(threshold_dir, str(ses), 'threshold.npy')):
        threshold = np.load(os.path.join(threshold_dir, str(ses), 'threshold.npy'))
    else:
        plt.close()  # In case any plots are left open
        selected_bin = None
        histfig, (ax, ax_trace) = plt.subplots(2, 1, figsize=(10, 8))

        # Calculate bounds for central 90%
        upper_bound = np.percentile(motion_smooth, 80)
        n, bins, patches = ax.hist(motion_smooth, bins=200, alpha=0.75, range=(np.min(motion_smooth), upper_bound))
        # Connect the click event
        cid = histfig.canvas.mpl_connect('button_press_event', onclick)

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Interactive Histogram (Click to Select a Bin)')
        ax.grid(True)

        # Plot a random 30-second trace of motion_smooth below the histogram
        framerate = 1 / sampling_interval
        n_frames_90s = int(90 * framerate)
        max_start = len(motion_smooth) - n_frames_90s
        random_start = np.random.randint(0, max_start)
        random_end = random_start + n_frames_90s

        trace_frames = np.arange(random_start, random_end)
        trace_time = trace_frames / framerate  # convert to seconds

        ax_trace.plot(trace_time, motion_smooth[random_start:random_end], color='steelblue', linewidth=0.8)
        ax_trace.set_xlabel('Time (s)')
        ax_trace.set_ylabel('Motion')
        ax_trace.set_title(f'Motion trace (random 30s window, frames {random_start}–{random_end})')
        ax_trace.grid(True)

        plt.tight_layout()
        plt.show()

        threshold = selected_bin

    print(threshold)

    time_full = np.arange(0, len(motion_smooth)) / sampling_interval

    #utils_motion.plot_still_frames(motion_smooth, time_full, threshold)

    move_mask = (trial_aligned_motion > threshold).astype(bool)
    # Which bins in trial aligned motion data have movement?
    #if load_thresholds:
    #    mask_file = os.path.join(threshold_dir, str(ses), 'move_mask.npy')
    #    if os.path.isfile(mask_file) and load_mask:
    #        move_mask = np.load(mask_file)
    #    else:
    #        print(f"No threshold file found for {ses}, skipping")
    #        continue
    #else:
    #    move_mask = (trial_aligned_motion > threshold).astype(bool)

    N_potent = matrix[move_mask, :]
    N_null = matrix[~move_mask, :]

    covNull = np.cov(N_null.T)
    covPotent = np.cov(N_potent.T)

    print("covNull shape:", covNull.shape)
    print("covPotent shape:", covPotent.shape)
    print("covNull NaNs or infs:", np.isnan(covNull).any(), np.isinf(covNull).any())
    print("covPotent NaNs or infs:", np.isnan(covPotent).any(), np.isinf(covPotent).any())

    dNull = int(np.floor(len(spikeData_ses) / 2))
    dPotent = int(np.floor(len(spikeData_ses) / 2))

    print(dNull)
    print(dPotent)

    dMax = np.max([dNull, dPotent])

    np.random.default_rng(seed=101)

    Q, Ppotent, Pnull = funcs.orthogonal_subspaces(len(spikeData_ses), dNull, dPotent, covNull, covPotent)
    # % Q is a matrix of size (nNeurons, nDimensions),
    # % where nDimensions = dNull + dPotent.
    # % P is a cell array, where
    # % Q*P{1} = Qpotent (the potent subspace)
    # % Q*P{2} = Qnull (the null subspace)

    # % Q contains the dimensions of both the null and potent subspaces. Here, we
    # % extract the columns corresponding to null and potent subspaces.

    Qpotent = Q @ Ppotent  # size = (nNeurons,dPotent)
    Qnull = Q @ Pnull  # size = (nNeurons,dNull)


    # Make a folder to save stuff in

    if not os.path.isdir(ses_save_folder):
        os.makedirs(ses_save_folder)
    else:
        continue

    # Save relevant information
    np.save(os.path.join(ses_save_folder, 'threshold.npy'), threshold)
    np.save(os.path.join(ses_save_folder, 'move_mask.npy'), move_mask)
    np.save(os.path.join(ses_save_folder, 'trials.npy'), trialData_ses['trialType'])
    np.savez(os.path.join(ses_save_folder, 'weight_matrices.npz'), Qpotent=Qpotent, Qnull=Qnull)
    np.save(os.path.join(ses_save_folder, 'neurons.npy'), spikeData_ses['cluster_ID'])

    # % now that we have null and potent subspaces, we can project our neural
    # % activity onto the subspaces separately.
    potent_proj = (matrix @ Qpotent)#.reshape((n_bins, n_trials, dPotent))  # (time, trials, potent dims)
    null_proj = (matrix @ Qnull)#.reshape((n_bins, n_trials, dNull))  # (time, trials, null dims)

    # get the sum squared magnitude across dimensions (for plotting)
    potent_ssm = np.sum((potent_proj ** 2), axis=2)
    null_ssm = np.sum((null_proj ** 2), axis=2)

    trials = np.arange(potent_ssm.shape[1])  # length = trial count

    f, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 5))
    axs[0].set_xlabel("Time from go cue (s)")
    axs[0].set_ylabel("Trials")

    # # % plot motion energy
    ax = axs[0]
    ax.set_title('Motion energy')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    ax.tick_params('both', length=5, width=1, which='major')
    ax.pcolormesh(time, trials, trial_aligned_motion.T, shading='nearest')
    ax.vlines(0, 0, n_trials, colors='w', linestyle='--')

    # # % plot potent
    ax = axs[1]
    ax.set_title('Potent')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    ax.tick_params('both', length=5, width=1, which='major')
    ax.pcolormesh(time, trials, np.transpose(potent_ssm), shading='nearest')
    ax.vlines(0, 0, n_trials, colors='w', linestyle='--')

    # # % plot null
    ax = axs[2]
    ax.set_title('Null')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    ax.tick_params('both', length=5, width=1, which='major')
    ax.pcolormesh(time, trials, np.transpose(null_ssm), shading='nearest')
    ax.vlines(0, 0, n_trials, colors='w', linestyle='--')

    # save figure
    plt.savefig(os.path.join(ses_save_folder, ("plot" + '.png')))

    # Save script
    shutil.copy(__file__, os.path.join(output_dir, 'script.py'))

    ses_all.append(ses)

n_sessions = len(ses_all)
n_mice = len(sessionData[sessionData['session_ID'].isin(ses_all)]['mousename'].unique())
print(f"Finished subspace decomposition for {n_sessions} sessions from {n_mice} unique animals")