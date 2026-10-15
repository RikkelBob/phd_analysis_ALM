import scipy.io as sio
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy
import warnings
import glasbey
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d
import neo
import shutil
import re
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import List, Union, Tuple
from itertools import combinations
from neural_analysis import plot_utils
from numba import jit
from matplotlib.collections import LineCollection
from matplotlib import colormaps
import config

def exclude_neurons(spikeData,
                    sessionData,
                    min_fire,
                    quality):

    for session in sessionData['session_ID']:
        current_ses = sessionData[sessionData['session_ID'] == session]
        dur = int((current_ses['t_stop'].iloc[0] - current_ses['t_start'].iloc[0]) / 1e6)
        spikes_ses = spikeData[spikeData['session_ID'] == session].copy()

        # Filter by FR
        spikes_ses = spikes_ses[spikes_ses['fr'] > min_fire]

        if quality is not None:
            spikes_ses = spikes_ses[spikes_ses['group'] == quality]

        if not 'spikeData_excl' in locals():
            spikeData_excl = spikes_ses.copy()
        else:
            spikeData_excl = pd.concat([spikeData_excl, spikes_ses], axis=0)

    return spikeData_excl.reset_index()


def find_sessions_with_files(path_root, experiment, data_types, animals=None):
    required_files = [s + '.mat' for s in data_types]
    all_sessions = []

    for exp in experiment:
        print("Experiment: " + str(exp))

        if animals is None or animals == "all":
            animals_in_exp = os.listdir(os.path.join(path_root, exp))
            print("Loading all animals - ")
            for anml in animals_in_exp:
                print(anml)
        else:
            # Only include animals that exist in the current experiment's directory
            animals_in_exp = [anml for anml in animals if os.path.exists(os.path.join(path_root, exp, anml))]

        for anml in animals_in_exp:
            if anml == '.DS_Store':
                continue

            sessions = os.listdir(os.path.join(path_root, exp, anml))

            for ses in sessions:
                session_path = os.path.join(path_root, exp, anml, ses)

                if all(os.path.exists(os.path.join(session_path, file)) for file in required_files):
                    all_sessions.append(session_path)

    return all_sessions

def load_data(path_root,
              experiment,
              animals='all',
              data_types=['trialData', 'sessionData', 'spikeData'],
              exclude_weird_trials=True,
              exclude_grace_violations=True,
              trim_last_trials=True):

    trialData = 0
    sessionData = 0
    spikeData = 0
    print("Loading data from ")

    all_sessions = find_sessions_with_files(path_root, experiment, data_types, animals)

    for ses in all_sessions:
        for type in data_types:
            # load file
            loaded_data_ext = sio.loadmat(os.path.join(ses, type), squeeze_me=True)
            # then extract just the data
            loaded_data = loaded_data_ext[type]
            loaded_data_dct = {x: loaded_data[x].item() for x in loaded_data.dtype.names}

            if type == 'sessionData':
                for var in loaded_data_dct:
                    if loaded_data_dct[var].__class__ is np.ndarray:
                        loaded_data_dct[var] = [loaded_data_dct[var]]

            if type == 'spikeData':
                for key, values in loaded_data_dct.items():
                    # Count the number of values for each key
                    num_values = len(values)

                    # Print the key and the number of values
                    #print(f"Key: {key}, Number of values: {num_values}")
            loaded_data_df = pd.DataFrame(loaded_data_dct)

            if type == 'trialData':
                if not isinstance(trialData, pd.DataFrame):
                    if trim_last_trials:
                        loaded_data_df = loaded_data_df.iloc[:-20]
                    trialData = loaded_data_df
                else:
                    if trim_last_trials:
                        loaded_data_df = loaded_data_df.iloc[:-20]
                    trialData = pd.concat([trialData, loaded_data_df], axis=0)
            elif type == 'sessionData':
                if not isinstance(sessionData, pd.DataFrame):
                    sessionData = loaded_data_df
                else:
                    for var in loaded_data_dct:
                        if loaded_data_dct[var].__class__ is np.ndarray:
                            loaded_data_dct[var] = [loaded_data_dct[var]]
                    sessionData = pd.concat([sessionData, loaded_data_df], axis=0)
            elif type == 'spikeData':

                if not isinstance(spikeData, pd.DataFrame):
                    spikeData = loaded_data_df
                else:
                    spikeData = pd.concat([spikeData, loaded_data_df], axis=0)

    if isinstance(trialData, pd.DataFrame):
        trialData.reset_index(inplace=True)
    if isinstance(sessionData, pd.DataFrame):
        sessionData.reset_index(inplace=True)
        if sessionData['session_ID'].duplicated().any():
            warnings.warn('Duplicate sessions found in sessionData!')
    if isinstance(spikeData, pd.DataFrame):
        spikeData.reset_index(inplace=True)
        if 'cluster_ID' in spikeData.columns:
            if spikeData['cluster_ID'].duplicated().any():
                warnings.warn('Duplicate neurons found in spikeData!')
        elif 'cell_ID' in spikeData.columns:
            if spikeData['cell_ID'].duplicated().any():
                warnings.warn('Duplicate neurons found in spikeData!')

    if exclude_weird_trials:
        trialData = trialData[~((trialData['noResponse'] == 1) & (~trialData['reactionTime'].isna()))]

    if exclude_grace_violations:
        if 'reactionTime' not in trialData.keys():
            print(f"No reactionTime found in trialData, cannot exclude trials based on graceperiod violations")

        elif np.all(np.isnan(trialData['reactionTime'])):
            print(f"WARNING: Something is wrong with reactionTimes (all nan) - Including all trials")

        else:
            if np.max(trialData['reactionTime'] < 2):
                print("WARNING: Reaction time in seconds instead of microseconds!")
                trialData = trialData[~(trialData['reactionTime'] < 0.1)]
            else:
                trialData = trialData[~(trialData['reactionTime'] < 1e5)]

    # Add trial_ID
    if isinstance(trialData, pd.DataFrame):
        trialData['trial_ID'] = trialData['session_ID'].astype(str) + '_' + trialData['trialNum'].astype(str)

    return(trialData, sessionData, spikeData)


def smooth_psth(psth,
                bin_size,
                width):


    """
    Apply Gaussian smoothing to the PSTH data.

    Parameters:
    psth (np.array): The PSTH data (spike counts per bin).
    bin_size (float): The size of the bins for the histogram (in seconds).
    width (float): The width (standard deviation) of the Gaussian filter (in seconds).

    Returns:
    np.array: The smoothed PSTH data.
    """
    # Calculate the sigma for the Gaussian filter in terms of bins
    sigma = width / bin_size

    # Apply Gaussian smoothing
    smoothed_psth = gaussian_filter1d(psth, sigma)

    return smoothed_psth


def filter_df(dataframe,
              **filters):
    """
    Filter neurons in spike_data based on specified criteria.

    Parameters:
    spike_data (pd.DataFrame): DataFrame containing spike times and metadata for each neuron.
    **filters: Key-value pairs where key is the column name and value is the value to filter by.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    filtered_data = dataframe.copy()

    for column, value in filters.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]
        else:
            print(f"Column {column} does not exist in data")

    return filtered_data


def filter_df_adv(dataframe, **filters):
    """
    Filter dataframe based on specified criteria. - Advanced version, requires specification of operator.
    Example usage: filter_df_adv(df, spike_rate=('>', 10), quality=('==', 'good'))

    Parameters:
    dataframe (pd.DataFrame): DataFrame to be filtered.
    **filters: Key-value pairs where key is the column name and value is a tuple specifying
               the operator and the value to filter by. The tuple should be in the form (operator, value).

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """

    query_str = []
    for column, condition in filters.items():
        if column in dataframe.columns:
            operator, value = condition
            if isinstance(value, str):
                query_str.append(f"({column} {operator} '{value}' | {column}.isnull())")
            else:
                query_str.append(f"({column} {operator} {value} | {column}.isnull())")
        else:
            print(f"Column {column} does not exist in data")

    if query_str:
        query_str = " & ".join(query_str)
        return dataframe.query(query_str)
    else:
        return dataframe


def filter_df_2(dataframe,
                **filters):
    """
    Filter rows in the DataFrame based on specified criteria.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing data to be filtered.
    **filters: Key-value pairs where key is the column name and value is the value to filter by.
               If value is a list, it will filter rows where the column's value is in the list.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    filtered_data = dataframe.copy()

    for column, value in filters.items():
        if column in filtered_data.columns:
            if isinstance(value, list):
                filtered_data = filtered_data[filtered_data[column].isin(value)]
            else:
                filtered_data = filtered_data[filtered_data[column] == value]
        else:
            print(f"Column {column} does not exist in data")

    return filtered_data


def calc_raster(spike_data,
                trial_data,
                window=(-0.5, 1.0),
                align_to='stimChange',
                sort_order=None):
    """
    Calculate the PSTH for a given set of trials.

    Parameters:
    spike_data (pd.DataFrame): DataFrame containing spike times and metadata for each neuron.
    trial_data (pd.DataFrame): DataFrame containing trial information.
    bin_size (float): Size of the bins for the histogram (in seconds).
    window (tuple): Time window around the stimulus change to consider (in seconds).
    smoothing_width (float, optional): The width (standard deviation) of the Gaussian filter for smoothing (in seconds).

    Returns:
    np.array: The PSTH.
    np.array: The SEM.
    np.array: The bin edges.
    """

    neuron_ids = []
    n_stim_times_total = 0
    order = []

    data_container = spike_data if isinstance(spike_data, pd.DataFrame) else pd.DataFrame([spike_data])

    for index, neuron in data_container.iterrows():
        session_id = neuron['session_ID']
        neuron_id = neuron['cluster_id']
        spiketimes = neuron['ts']

        neuron_ids.append(neuron_id)

        # Filter trials for the same session
        trials = trial_data[trial_data['session_ID'] == session_id]

        if trials.empty:
            continue

        # Order raster data if requested
        if sort_order is not None:
            # Calculate the sort column
            sort_column = trials['firstLick'] - trials['stimChange']

            # Separate trials with valid sort values from those without
            valid_sort_mask = ~sort_column.isna()
            trials_with_sort = trials[valid_sort_mask]
            trials_without_sort = trials[~valid_sort_mask]

            # Sort the trials that have valid sort values
            sorted_idx = sort_column[valid_sort_mask].sort_values().index
            trials_with_sort_sorted = trials_with_sort.loc[sorted_idx]

            # Get unsorted indices and append to sorted trials to save order
            unsorted_idx = trials_without_sort.index
            order = sorted_idx.append(unsorted_idx)

            # Concatenate: sorted trials first, then trials without sort values
            trials = pd.concat([trials_with_sort_sorted, trials_without_sort])

        # Get the stimulus change times for all trials
        stim_times = trials[align_to].values

        # Get rid of trials that do not have the alignment timestamps (e.g., miss trials aligned to lick)
        stim_times = stim_times[stim_times != 0]
        stim_times = stim_times[~np.isnan(stim_times)]

        n_stim_times_total += len(stim_times)

        # Initialize lists to collect spike data for PSTH
        aligned_spikes_neuron = []
        raster_data = []

        # Filter and align spikes to stimulus change times for each trial
        for trial_index, stim_time in enumerate(stim_times):
            # Filter spikes within the window
            trial_spikes = spiketimes[
                (spiketimes >= (stim_time + window[0] * 1e6)) & (spiketimes <= (stim_time + window[1] * 1e6))]
            # Align spikes to the stimulus change time
            aligned_spikes = trial_spikes - stim_time
            raster_data.append((trial_index, aligned_spikes))  # Apply trial_offset here

    return raster_data, order


def calc_psth_single(spike_data_neuron,
                     trial_data,
                     window=(-0.5, 1),
                     bin_size=0.005,
                     smoothing_width=None,
                     bins=None,
                     align_to='stimChange'):

    spike_data_neuron = spike_data_neuron if isinstance(spike_data_neuron, pd.DataFrame) else pd.DataFrame([spike_data_neuron])

    if not (np.array(spike_data_neuron.shape) == 1).any():
        print("More than one neuron passed to calc_psth_single. Quitting.")
        return

    if bins is None: # Assuming non-overlapping, non-causal bins
        # Calculate bins
        n_bins = int((window[1] - window[0]) / bin_size)
        bins = np.linspace(window[0], window[1], n_bins + 1) * 1e6
    else:
        # Check and correct order of magnitude for bins. should be in microseconds
        if not np.any((bins < -1e3).any() or (bins > 1e3).any()):
            bins = bins * 1e6
        n_bins = len(bins)
        if bin_size is None:
            if bins.ndim == 2:
                bin_size = (bins[0, 1] - bins[0, 0]) /1e6
            else:
                bin_size = (bins[1] - bins[0]) /1e6


    #if bins.ndim == 2: # if bins are provided as start:stop for each bin (probably causal bins), redefine window to include all necessary spikes
    #    window = (bins[0][0], bins[-1][-1])

    #n_bins = len(bins)

    # Filter trials for the same session
    session_id = spike_data_neuron['session_ID'].iloc[0]
    trials = trial_data[trial_data['session_ID'] == session_id]
    spiketimes = spike_data_neuron['ts'].iloc[0]

    # Get the stimulus change times for all trials
    stim_times = trials[align_to].values

    # Initialize lists to collect spike data for PSTH
    all_aligned_spikes = []
    binned_spikes_per_trial = np.empty((0, n_bins))

    # In case causal bins are used, spikes from before window[0] need to be included
    if bins.ndim == 2:
        window_trial = (bins[0][0], bins[-1][-1])
    else:
        window_trial = (bins[0], bins[-1])

    # Filter and align spikes to stimulus change times for each trial
    for stim_time in stim_times:
        # Filter spikes within the window
        spiketimes = np.atleast_1d(spiketimes)

        trial_spikes = spiketimes[
            (spiketimes >= (stim_time + window_trial[0])) &
            (spiketimes <= (stim_time + window_trial[1]))]

        # Align spikes to the stimulus change time
        aligned_spikes = trial_spikes - stim_time
        if bins.ndim == 1:
            binned_spikes, _ = np.histogram(aligned_spikes, bins=bins)
        else:
            binned_spikes = custom_histogram_searchsort(aligned_spikes, bins=bins)

        binned_spikes_per_trial = np.vstack([binned_spikes_per_trial, binned_spikes])

        all_aligned_spikes.extend(aligned_spikes)

    # Convert list to numpy array for histogram calculation
    all_aligned_spikes = np.array(all_aligned_spikes)
    psth_per_trial = binned_spikes_per_trial / bin_size

    # Generate histogram averaged over trials
    if bins.ndim == 1:
        hist, _ = np.histogram(all_aligned_spikes, bins=bins)
    else:
        hist = custom_histogram(all_aligned_spikes, bins=bins)
    psth = hist / (len(stim_times) * bin_size)

    if smoothing_width is not None:
        psth = smooth_psth(psth, bin_size, smoothing_width)

    return psth, psth_per_trial


def load_matlab_struct(filepath):
    import scipy.io
    import pandas as pd
    import numpy as np
    from scipy.io.matlab import mat_struct
    def mat_to_dict(matobj):
        """Recursively converts mat_struct to dictionary."""
        if isinstance(matobj, mat_struct):
            result = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                result[strg] = mat_to_dict(elem)
            return result
        elif isinstance(matobj, np.ndarray):
            if matobj.dtype.names is not None:
                result = {name: mat_to_dict(matobj[name]) for name in matobj.dtype.names}
                return result
            else:
                return [mat_to_dict(e) for e in matobj]
        else:
            return matobj

    def convert_to_dataframe(d):
        """Convert a nested dictionary to a DataFrame."""
        if isinstance(d, list):
            return pd.DataFrame(d)
        elif isinstance(d, dict):
            df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
            return df
        else:
            return pd.DataFrame([d])

    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    data = {key: mat_to_dict(value) for key, value in mat.items() if not key.startswith('__')}

    dataframes = {key: convert_to_dataframe(value) for key, value in data.items()}

    return dataframes


def exclude_session_ids(session_ids, *dataframes):
    """
    Excludes rows with the specified session_ids from multiple dataframes and concatenates them.

    Parameters:
    session_ids (list of str/int): The session IDs to be excluded.
    dataframes (list of pd.DataFrame): Variable number of dataframes to be processed.

    Returns:
    pd.DataFrame: Concatenated dataframe with the specified session_ids excluded.
    """
    result = []
    for df in dataframes:
        if df is not None and 'session_ID' in df.columns:
            df_filtered = df[~df['session_ID'].isin(session_ids)]
            result.append(df_filtered)
        else:
            result.append(df)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()

def filter_sessions_hitrate(trialData, sessionData=None, spikeData=None, min_hitrate=0.3, verbose=False):
    """
    Filters trialData, sessionData, spikeData dataframes to only include sessions with specified minimum
    hit rates for both sensory modalities
    Args:
        data:
        min_hitrate:

    Returns:

    """
    sessions_exclude = []
    sessionData_filtered = []
    spikeData_filtered = []
    trialData_filtered = []

    all_sessions = trialData['session_ID'].unique()

    for session in all_sessions:
        trialData_ses = filter_df(trialData, session_ID=session)

        # Determine hit rates for max saliency stimuli
        hits_vis = filter_df(trialData_ses, trialType='X', correctResponse=True,
                             visualOriChange_norm=2, hasphotostim=0)
        hits_aud = filter_df(trialData_ses, trialType='Y', correctResponse=True,
                             audioOctChange_norm=2, hasphotostim=0)
        trials_vis = filter_df(trialData_ses, trialType='X', visualOriChange_norm=2,
                               hasphotostim=0)
        trials_aud = filter_df(trialData_ses, trialType='Y', audioOctChange_norm=2,
                               hasphotostim=0)

        n_hits_vis = hits_vis.shape[0]
        n_hits_aud = hits_aud.shape[0]
        n_vis = trials_vis.shape[0]
        n_aud = trials_aud.shape[0]

        hit_rate_vis = n_hits_vis / n_vis
        hit_rate_aud = n_hits_aud / n_aud

        if any(value < min_hitrate for value in [hit_rate_vis, hit_rate_aud]):
            if verbose:
                print(str(session) + " has a visual hit rate of " + str(hit_rate_vis)
                      + " and an auditory hit rate of " + str(hit_rate_aud) + " - Excluding")
            sessions_exclude.append(session)

    if sessionData is not None:
        sessionData_filtered = exclude_session_ids(sessions_exclude, sessionData)
    if spikeData is not None:
        spikeData_filtered = exclude_session_ids(sessions_exclude, spikeData)
    if trialData is not None:
        trialData_filtered = exclude_session_ids(sessions_exclude, trialData)

    return trialData_filtered, sessionData_filtered, spikeData_filtered, sessions_exclude


def resp_units(spikeData, trialData, window=1, verbose=False, n_resamples=100):
    """ Uses - https://github.com/JorritMontijn/zetapy
    """
    from zetapy import zetatest, zetatstest
    import pandas as pd
    from tqdm import tqdm

    spikeData_out = pd.DataFrame()

    if verbose:
        print("Finding repsonsive neurons in " + spikeData.iloc[0].area + " for " + str(len(spikeData)) + " clusters")

    for index, neuron in tqdm(spikeData.iterrows()):
        trials = trialData[trialData['session_ID'] == neuron.session_ID]
        neuron['zeta_p'], neuron['zeta'], neuron['dRate'] = zetatest(neuron['ts']/1e6,
                                                                     trials['stimChange']/1e6,
                                                                     dblUseMaxDur=window, intResampNum=n_resamples)

        neuron = pd.DataFrame([neuron])
        spikeData_out = pd.concat([spikeData_out, neuron], ignore_index=True)

    return spikeData_out


def find_modulated_neurons(fr_list, neuron_ids=None, method='wilcoxon', p_threshold=0.05):
    """
    Identifies neurons significantly modulated post-stimulus.

    Parameters:
    - fr_list (list): List of arrays, one per neuron. Each array is (n_trials, 2)
      where [:, 0] = pre-stimulus firing rate, [:, 1] = post-stimulus firing rate.
    - neuron_ids (list): Optional neuron IDs. Defaults to indices.
    - method (str): 'wilcoxon' or 'ttest'.
    - p_threshold (float): Significance threshold after FDR correction.

    Returns:
    - inhibited_neuron_ids (list)
    - excited_neuron_ids (list)
    """
    from scipy.stats import wilcoxon, ttest_rel
    from statsmodels.stats.multitest import multipletests

    if neuron_ids is None:
        neuron_ids = list(range(len(fr_list)))

    p_values = []
    directions = []
    valid_mask = []

    for i, fr in enumerate(fr_list):
        pre = fr[:, 0]
        post = fr[:, 1]
        diff = post - pre

        # Skip neurons with no variance (all-zero or constant)
        if np.all(diff == 0):
            valid_mask.append(False)
            p_values.append(1.0)
            directions.append(0)
            continue

        if method == 'wilcoxon':
            stat, p = wilcoxon(pre, post)
        elif method == 'ttest':
            stat, p = ttest_rel(pre, post)

        else:
            raise ValueError("method must be 'wilcoxon' or 'ttest'")

        p_values.append(p)
        directions.append(np.sign(np.mean(post) - np.mean(pre)))
        valid_mask.append(True)

    # FDR correction across all neurons
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=p_threshold, method='fdr_bh')

    inhibited = [neuron_ids[i] for i in range(len(fr_list))
                 if rejected[i] and directions[i] < 0]
    excited = [neuron_ids[i] for i in range(len(fr_list))
               if rejected[i] and directions[i] > 0]
    unaffected = [neuron_ids[i] for i in range(len(fr_list))
                  if not rejected[i]]

    return inhibited, excited, unaffected


def plot_pairwise_effects_multiple_conditions(dataframe, condition_column, conditions, animal_column,
                                              selected_value_type, ax=None, alpha_marker=0.3, alpha_line=0.3):
    """
    Plots effect plots of a performance metric (selected by 'value' column) across multiple conditions for each session,
    with unique colors for each animal and a mean effect line.

    Parameters:
    dataframe (pd.DataFrame): Input DataFrame with sessions, conditions, performance metric, and animal information.
    condition_column (str): Column name representing the condition (e.g., 'opto', 'probe').
    conditions (list): List of condition values to compare (e.g., [True, False, AnotherValue]).
    animal_column (str): Column representing the animal ID for color-coding.
    selected_value_type (str): Specific value type to filter the data by (e.g., 'responserate').
    ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, creates a new figure.
    alpha_marker (float): Alpha value for markers (default: 0.3).
    alpha_line (float): Alpha value for lines (default: 0.3).

    Returns:
    matplotlib.pyplot: The pyplot module.
    """

    if selected_value_type == 'responserate':
        y_min = 0
        y_max = 1

    elif selected_value_type == 'prime':
        y_min = - 2
        y_max = 3.5

    # Filter the dataframe to only include rows with the selected value type
    filtered_df = dataframe[dataframe['value_type'] == selected_value_type]

    # Further filter to include only the desired conditions
    filtered_df = filtered_df[filtered_df[condition_column].isin(conditions)]

    # Set up a color palette for animals
    unique_animals = filtered_df[animal_column].unique()
    unique_sessions = filtered_df['session'].unique()
    palette = gen_palette(len(unique_animals))
    animal_colors = dict(zip(unique_animals, palette))

    print(f"{len(unique_animals)} unique animals")
    print(f"{len(unique_sessions)} unique sessions")

    # Create a new figure if ax is not provided, otherwise set it as current
    if ax is None:
        plt.figure(figsize=(5, 6))
    else:
        plt.sca(ax)

    # Plot each animal's data with its unique color, connecting the conditions for each session
    for animal in unique_animals:
        animal_data = filtered_df[filtered_df[animal_column] == animal]

        # Group by session to ensure each session's conditions are connected by a line
        for session, session_data in animal_data.groupby('session'):
            plt.plot(session_data[condition_column], session_data['value'],
                     color=animal_colors[animal], alpha=alpha_line, linewidth=1)

            plt.scatter(session_data[condition_column], session_data['value'],
                        color=animal_colors[animal], alpha=alpha_marker, edgecolors='none')

    # Calculate the mean effect across all animals and sessions
    mean_values = filtered_df.groupby(condition_column)['value'].mean().reindex(conditions)

    # Plot the mean effect as a thicker, fully opaque line
    plt.plot(conditions, mean_values, marker='o', color='black', linewidth=3, alpha=1.0)

    # Customize the plot
    plt.xlabel('Condition')
    plt.ylabel(selected_value_type)
    plt.title(f'Effect Plot: Value for {selected_value_type} Comparison across Conditions')
    plt.xticks(conditions, [str(condition) for condition in conditions])  # Ensure x-ticks are labeled correctly
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(y_min, y_max)

    ax = plt.gca()
    ax.grid(False)

    return plt

def transform_df_long(df):
    # First, identify the columns that contain session-level information (id_vars)
    id_vars = ['session', 'animal', 'selection', 'selectionLegend', 'nTrials', 'opto']

    # Exclude id_vars and any indexing columns from the data to be melted
    value_vars = [col for col in df.columns if col not in id_vars and not col.startswith(('level_0', 'index', 'plot_colors'))]

    # Melt the DataFrame from wide to long format
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')

    # Function to parse the 'variable' column
    def parse_variable(var_name):
        # Initialize default values
        modality = 'none'
        value_type = None
        saliency = 'all'
        direction = 'all'
        opto_bool = False
        is_2ADC = False

        # Split the variable name by underscores
        parts = var_name.split('_')

        # Check for 'opto' in parts
        if 'opto' in parts:
            opto_bool = True
            parts.remove('opto')

        # Check for '2ADC' in parts
        if '2ADC' in parts:
            is_2ADC = True
            parts.remove('2ADC')

        # Identify value type (e.g., 'd_prime', 'c', 'hitrate', etc.)
        value_types = ['prime', 'c', 'hitrate', 'errorrate', 'FArate']
        for vt in value_types:
            if vt in parts:
                value_type = vt
                parts.remove(vt)
                break

        # Identify modality (e.g., 'visual', 'auditory')
        modalities = ['visual', 'auditory']
        for mod in modalities:
            if mod in parts:
                modality = mod
                parts.remove(mod)
                break

        # Check for 'left' or 'right' in parts
        if 'left' in parts:
            direction = 'left'
            parts.remove('left')
        elif 'right' in parts:
            direction = 'right'
            parts.remove('right')

        # Check for saliency (e.g., 'thr', 'max')
        if 'thr' in parts:
            saliency = 'thr'
            parts.remove('thr')
        elif 'max' in parts:
            saliency = 'max'
            parts.remove('max')

        # Any remaining parts can be ignored or handled as needed

        return pd.Series({
            'modality': modality,
            'value_type': value_type,
            'saliency': saliency,
            'direction': direction,
            'opto_condition': opto_bool,
            'is_2ADC': is_2ADC
        })

    # Apply the parsing function to the 'variable' column
    parsed_columns = df_long['variable'].apply(parse_variable)
    df_long = pd.concat([df_long, parsed_columns], axis=1)
    df_long.loc[df_long['opto_condition'] == False, 'opto'] = 'none'
    df_long['opto'] = pd.Categorical(df_long['opto'], categories=["none", "unilateral", "bilateral"])

    left_responses = ((df_long['value_type']=='FArate') & (df_long['direction']=='left') |
                      ((df_long['value_type'])=='hitrate') & (df_long['modality'] == 'visual') |
                      ((df_long['value_type']) == 'errorrate') & (df_long['modality'] == 'auditory') |
                      ((df_long['value_type']) == 'prime') & (df_long['modality'] == 'visual') |
                      ((df_long['value_type']) == 'c') & (df_long['modality'] == 'visual'))

    right_responses = ((df_long['value_type']=='FArate') & (df_long['direction']=='right') |
                      ((df_long['value_type'])=='hitrate') & (df_long['modality'] == 'auditory') |
                      ((df_long['value_type']) == 'errorrate') & (df_long['modality'] == 'visual') |
                      ((df_long['value_type']) == 'prime') & (df_long['modality'] == 'auditory')|
                      ((df_long['value_type']) == 'c') & (df_long['modality'] == 'auditory'))

    df_long.loc[left_responses, 'response_direction'] = 'left'
    df_long.loc[right_responses, 'response_direction'] = 'right'

    df_long['direction'].iloc[df_long['value_type'].isin(['FArate'])] = 'none'
    df_long['saliency'].iloc[df_long['value_type'].isin(['FArate'])] = 'probe'
    df_long['value_type'].iloc[df_long['value_type'].isin(['FArate','errorrate', 'hitrate'])] = 'responserate'

    return df_long



def plot_mean_lines_with_errorbars_ax(dataframe, condition_column, conditions, grouping_column, groups, selected_value_type,
                                   errorbars='sem', alpha_marker=0.5, alpha_line=0.5, animal_column='animal',
                                   error_type='bars', ax=None, show_legend=True, group_colors=None):
    """
    Plots mean lines with error bars (STD or SEM) for a performance metric ('value' column) across multiple conditions.
    It allows comparison of values for several different conditions (e.g., opto, no opto) across a second set of conditions (e.g., saliency).

    Parameters:
    dataframe (pd.DataFrame): Input DataFrame with sessions, conditions, performance metric, and animal information.
    condition_column (str): Column name representing the condition for the x-axis (e.g., 'saliency').
    conditions (list): List of condition values to compare (for x-axis, e.g., [low_saliency, high_saliency]).
    grouping_column (str): Column representing the grouping condition (e.g., opto or trial type).
    groups (list): List of groups to compare for the grouping column (e.g., [no_opto, opto]).
    animal_column (str): Column representing the animal ID for color-coding.
    selected_value_type (str): Specific value type to filter the data by (e.g., 'responserate').
    errorbars (str): Type of error bars ('sem' for standard error, 'std' for standard deviation).
    alpha_marker (float): Opacity of the markers for individual points.
    alpha_line (float): Opacity of the lines connecting individual points.
    error_type (str): Type of error visualization ('bars' or 'scatter').
    ax (matplotlib.axes.Axes): Axes object to plot on. If None, creates new figure.

    Returns:
    ax (matplotlib.axes.Axes): The axes object with the plot.
    """
    if selected_value_type == 'responserate':
        y_min = 0
        y_max = 1.05
    elif selected_value_type == 'prime':
        y_min = -1
        y_max = 3.5
    elif selected_value_type == 'c':
        y_min = -1.5
        y_max = 2.5

    # Filter the dataframe to only include rows with the selected value type
    filtered_df = dataframe[dataframe['value_type'] == selected_value_type]

    # Further filter to include only the desired conditions for the x-axis and groups for grouping column
    filtered_df = filtered_df[filtered_df[condition_column].isin(conditions)]
    filtered_df = filtered_df[filtered_df[grouping_column].isin(groups)]

    # Replace inf values with 5
    filtered_df['value'].replace([np.inf, -np.inf], 5, inplace=True)

    # Set up a color palette for the groups
    if group_colors is None:
        palette = gen_palette(len(groups))
        group_colors = dict(zip(groups, palette))

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    if error_type == 'bars':
        # Loop through each group (e.g., opto vs no opto)
        for group in groups:
            group_data = filtered_df[filtered_df[grouping_column] == group]

            # Calculate mean and error bars for each condition on the x-axis
            if errorbars == 'sem':
                error = group_data.groupby(condition_column)['value'].sem().reindex(conditions)
            elif errorbars == 'std':
                error = group_data.groupby(condition_column)['value'].std().reindex(conditions)
            else:
                raise ValueError("Invalid value for 'errorbars'. Choose 'sem' or 'std'.")

            mean_values = group_data.groupby(condition_column)['value'].mean().reindex(conditions)

            # Plot mean line with error bars
            ax.errorbar(conditions, mean_values, yerr=error, label=f'{grouping_column} = {group}',
                       color=group_colors[group], fmt='-o', alpha=alpha_line, linewidth=3, markeredgewidth=0)

    elif error_type == 'scatter':
        # Plot mean lines first
        for group in groups:
            group_data = filtered_df[filtered_df[grouping_column] == group]
            mean_values = group_data.groupby(condition_column)['value'].mean().reindex(conditions)
            ax.plot(range(len(conditions)), mean_values, label=f'{grouping_column} = {group}',
                   color=group_colors[group], alpha=alpha_line, linewidth=3)

        # Scatter points with controlled offset and jitter
        offset_amount = 0.12  # ADJUST THIS to control spacing between groups
        jitter_amount = 0.05  # ADJUST THIS to control jitter within each group

        for i, group in enumerate(groups):
            group_data = filtered_df[filtered_df[grouping_column] == group]
            offset = (i - len(groups) / 2 + 0.5) * offset_amount

            for j, condition in enumerate(conditions):
                condition_data = group_data[group_data[condition_column] == condition]
                # Add random jitter to x-positions
                x_positions = j + offset + np.random.uniform(-jitter_amount, jitter_amount, len(condition_data))
                ax.scatter(x_positions, condition_data['value'],
                          facecolors=group_colors[group], alpha=alpha_marker, s=10, edgecolors='none')

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions)

    # Customize the plot
    ax.set_xlabel(condition_column.capitalize())
    ax.set_ylabel(selected_value_type)
    if show_legend:
        ax.legend(title=grouping_column)
    ax.grid(False)
    ax.set_ylim(y_min, y_max)

    return ax


def custom_bins(bin_size, step_size, window, causal=False):
    # Define the start and end of the full window range
    window_start, window_end = window

    # Initialize list to store [start, stop] pairs
    bins = []

    if step_size is None:
        step_size = bin_size

    if not causal:
        current_start = window_start - (0.5 * bin_size)
        stop_condition = 0.5 * bin_size
    else:
        # Start with the first bin at window_start - bin_size for full causality
        current_start = window_start - bin_size
        stop_condition = bin_size

    # Generate overlapping bins advancing by step_size (+1e9 to avoid float rounding errors)
    while current_start + stop_condition <= window_end + 1e-9:
        # Each bin goes from current_start to current_start + bin_size
        bin_start = current_start
        bin_end = current_start + bin_size
        bins.append([bin_start, bin_end])

        # Move the start of the window by step_size for the next bin
        current_start += step_size

    bins = np.array(bins)

    if causal:
        time = bins[:, 1]
    else:
        time = np.mean(bins, axis=1)

    bins = bins * 1e6

    # Convert bins to a numpy array
    return bins, time


def custom_histogram(data, bins):
    """
    Histgram function that allows for overlapping bins
    Args:
        data: binary events
        bins: list of start/stop times for bins

    Returns:
        histogram of event counts
    """
    # Initialize an empty list to store histogram counts for each bin
    hist = []
    # Iterate over each bin defined by start and stop values in the 2D array
    for start, stop in bins:
        # Count data points within each bin range [start, stop)
        count = np.sum((data >= start) & (data < stop))
        hist.append(count)
    return np.array(hist)


def custom_histogram_searchsort(data, bins):
    """
    Fast histogram with overlapping 2D bins using searchsorted.
    """
    data_sorted = np.sort(data)
    starts = bins[:, 0]
    stops = bins[:, 1]

    # Find insertion indices for all starts and stops at once
    start_indices = np.searchsorted(data_sorted, starts, side='left')
    stop_indices = np.searchsorted(data_sorted, stops, side='left')

    return stop_indices - start_indices


def get_motSVD(session_paths, timeline_file="realigned_timeline.npy", video_fixed_dir="Video_fixed",
               face_proc_fixed_filename=r"Face_re_encoded_proc.npy", face_proc_filename=r"Face_proc.npy"):

    if not isinstance(session_paths, list):
        session_paths = [session_paths]

    output_list = []

    for session in session_paths:
        print(str("\n") + session)
        # Skip session if timeline is not realigned
        timeline_path = os.path.join(session, timeline_file)
        if not os.path.isfile(timeline_path):
            print("Cannot find realigned timeline path, skipping session")
            continue

        # Find which file to load
        if os.path.isdir(os.path.join(session, video_fixed_dir)):
            data_path = os.path.join(session, video_fixed_dir, face_proc_fixed_filename)
        else:
            data_path = os.path.join(session, face_proc_filename)

        if not os.path.isfile(data_path):
            continue

        # Load data
        print("\nLoading data")
        face_data = safe_load_pickle_npy(data_path)
        timeline = np.load(timeline_path)

        # Do a check
        print("Extracting data")
        n_frames_svd = len(face_data['motSVD'][1])
        n_frames_timeline = len(timeline)

        print("Number of SVD frames: " + str(n_frames_svd))
        print("Number of timeline frames: " + str(n_frames_timeline))
        if n_frames_svd != n_frames_timeline:
            print("Number of SVD frames is not equal to number of timeline frames!")

        # Extract motSVD data
        motSVD_face = face_data['motSVD']

        # Add to list
        output_list.append({
            "session": session,
            "motSVD_face": motSVD_face,
            "timeline": timeline
        })

    return output_list

def safe_load_pickle_npy(path):
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == ():
            return arr.item()
        return arr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def get_motion(session_paths, timeline_file="realigned_timeline.npy", video_fixed_dir="Video_fixed",
               face_proc_fixed_filename=r"Face_re_encoded_proc.npy", face_proc_filename=r"Face_proc.npy"):

    output_list = []

    if not isinstance(session_paths, list):
        session_paths = [session_paths]

    for session in session_paths:
        print(str("\n") + session)
        # Skip session if timeline is not realigned
        timeline_path = os.path.join(session, timeline_file)
        if not os.path.isfile(timeline_path):
            print("Cannot find realigned timeline path, skipping session")
            continue

        # Find which file to load
        if os.path.isdir(os.path.join(session, video_fixed_dir)):
            data_path = os.path.join(session, video_fixed_dir, face_proc_fixed_filename)
        else:
            data_path = os.path.join(session, face_proc_filename)

        # Load data
        print("\nLoading data")
        face_data = np.load(data_path, allow_pickle=True).item()
        timeline = np.load(timeline_path)

        # Do a check
        print("Extracting data")
        n_frames_motion = len(face_data['motion'][1])
        n_frames_timeline = len(timeline)

        print("Number of motion frames: " + str(n_frames_motion))
        print("Number of timeline frames: " + str(n_frames_timeline))
        if n_frames_motion != n_frames_timeline:
            print("Number of motion frames is not equal to number of timeline frames!")

        # Extract motSVD data
        motion = face_data['motion']

        # Add to list
        output_list.append({
            "session": session,
            "motion": motion,
            "timeline": timeline
        })

    return output_list

def get_session_paths(base_path, starts_with="ALM_"):
    """
    Get the paths to all session folders under experiment directories (ALM_*).

    Parameters:
        base_path (str): The base directory containing the experiments.

    Returns:
        list: A list of paths to session folders.
    """
    session_paths = []

    # Iterate through the directories in the base_path
    for experiment in os.listdir(base_path):
        experiment_path = os.path.join(base_path, experiment)

        # Check if the folder matches the desired pattern (e.g., ALM_*)
        if os.path.isdir(experiment_path) and experiment.startswith(starts_with):
            # Iterate through subdirectories in the experiment folder
            for animal in os.listdir(experiment_path):
                animal_path = os.path.join(experiment_path, animal)

                if os.path.isdir(animal_path):
                    for session in os.listdir(animal_path):
                        session_path = os.path.join(animal_path, session)

                        if os.path.isdir(session_path):
                            session_paths.append(session_path)

    return session_paths


def load_TTL_matlab(folder):
    file = "TTL.mat"
    mat_contents = scipy.io.loadmat(os.path.join(folder, file))

    # Remove MATLAB's default metadata fields
    data_dict = {key: value for key, value in mat_contents.items() if not key.startswith('__')}

    # Extract the 'TTLs_struct' field
    ttls_struct = data_dict['TTLs_struct']

    # MATLAB structs are stored as NumPy structured arrays in Python
    # Extract the first (and usually only) element
    ttls_data = ttls_struct[0, 0]

    # Function to safely extract values from MATLAB structured arrays
    def extract_value(matlab_array):
        """Convert MATLAB's cell arrays or structured data to a native Python type."""
        if isinstance(matlab_array, np.ndarray):
            if matlab_array.size == 1:  # If it's a single element, extract the value
                return matlab_array.item()
            else:  # Otherwise, convert the whole array to a Python list
                return matlab_array.tolist()
        return matlab_array  # If it's already a scalar, return as is

    # Convert structured array to a Python dictionary
    python_dict = {
        'TTL': np.array(extract_value(ttls_data['TTL'])),
        'time': np.array(extract_value(ttls_data['time'])),
        'port': np.array(extract_value(ttls_data['port'])),
        'folder': str(ttls_data['folder'].item())  # Convert string safely
    }

    # Print extracted dictionary
    return python_dict


def gen_palette(n_colors):
    palette = glasbey.create_palette(palette_size=n_colors,
                                     colorblind_safe=False,
                                     cvd_severity=100,
                                     lightness_bounds=(15, 70))
    return palette

def get_baseline_psth(trialData, spikeData, bins=None, bin_size=None, window=(-0.5, -0.1)):
    """
    Used to get the baseline spike rate, useful for z-score calculations etc.

    Args:
        trialData: df containing trial information
        spikeData: df containing spiking information
        window: window used to calculate baseline

    Returns:
        psth: trial-by-trial psth per neuron
        mean: mean spike rate during baseline
        std: std of spike rate during baseline
    """
    std_all = []
    mean_all = []
    psth = []

    neuron_container = spikeData if isinstance(spikeData, pd.DataFrame) else pd.DataFrame([spikeData])

    for _, neuron in neuron_container.iterrows():
        ses = neuron['session_ID']
        trialData_neuron = trialData[trialData['session_ID'] == ses]

        _, per_trial = calc_psth_single(pd.DataFrame(neuron).transpose(),
                                        trialData_neuron,
                                        window=window,
                                        bin_size=bin_size,
                                        bins=bins, align_to='stimChange')

        std = np.std(per_trial)
        mean = np.mean(per_trial)

        std_all.append(std)
        mean_all.append(mean)
        psth.append(per_trial)

    return psth, std_all, mean_all


def z_score(data, data_window, baseline_window, baseline_mean=None, baseline_std=None):
    """
    Z-scores a PSTH trace based on a specified baseline window.

    Parameters:
    data (np.array): 1D array representing the PSTH data.
    data_window (tuple): (start_time, end_time) corresponding to the range of the PSTH.
    baseline_window (tuple): (baseline_start, baseline_end) defining the range for baseline calculation.

    Returns:
    np.array: Z-scored PSTH data.
    """
    if (baseline_mean is None) or (baseline_std is None):

        # Define time axis based on data window
        num_timepoints = len(data)
        time_axis = np.linspace(data_window[0], data_window[1], num_timepoints)

        # Get indices for baseline window
        baseline_indices = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])

        # Compute baseline mean and standard deviation
        if baseline_mean is None:
            baseline_mean = np.mean(data[baseline_indices])
        if baseline_std is None:
            baseline_std = np.std(data[baseline_indices])

    # Avoid division by zero
    if baseline_std == 0:
        baseline_std = 1

    # Compute z-scored data
    zscored_data = (data - baseline_mean) / baseline_std

    return zscored_data


def compute_auc_fast(spike_counts, labels, permutation_test=False, n_permutations=1000, return_permuted_aucs=False):
    """
    Computes the ROC AUC between two conditions and optionally performs permutation testing,
    using a rank-based approach for speed.

    Parameters:
    - spike_counts: array-like, neural activity data.
    - labels: array-like, binary condition labels (0 or 1).
    - permutation_test: bool, whether to perform permutation testing.
    - n_permutations: int, number of permutations for the test.

    Returns:
    - actual_auc: float, the computed AUC value.
    - p_value: float or None, the p-value from permutation testing (if enabled).
    """
    # Sanity checks
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Not enough distinct labels
        return np.nan, np.nan, None

    # Precompute ranks of spike_counts (1-based)
    # rankdata will give ranks in [1..N], average rank for ties if method='average'
    ranks = rankdata(spike_counts, method='average')

    # Number of positive and negative samples
    n1 = np.sum(labels == 1)
    n0 = len(labels) - n1

    # If there's no variation in spike_counts or no variation in labels
    if n1 == 0 or n0 == 0 or np.all(spike_counts == spike_counts[0]):
        return np.nan, np.nan, None

    # Sum of ranks for the actual positive labels
    sum_of_ranks_positives = np.sum(ranks[labels == 1])

    # Compute actual AUC using the rank-sum formula
    # AUC = (R_+ - n1(n1+1)/2) / (n1*n0)
    actual_auc = (sum_of_ranks_positives - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    if not permutation_test:
        return actual_auc, None, None

    # Permutation test
    permuted_aucs = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle labels
        perm_labels = np.random.permutation(labels)
        # Sum ranks for the "positive" in the permutation
        n1_perm = np.sum(perm_labels == 1)
        n0_perm = len(labels) - n1_perm

        # If a permutation by chance yields all 1s or all 0s,
        # skip or treat as degenerate:
        if n1_perm == 0 or n0_perm == 0:
            permuted_aucs[i] = 0.5  # or np.nan
            continue

        sum_ranks_positives_perm = np.sum(ranks[perm_labels == 1])
        perm_auc = (sum_ranks_positives_perm - n1_perm * (n1_perm + 1) / 2.0) / (n1_perm * n0_perm)
        permuted_aucs[i] = perm_auc

    # Two-sided p-value on |AUC - 0.5|, with Phipson-Smyth (+1/+1) correction
    # so p-values are conservative and never exactly zero.
    obs_dev = np.abs(actual_auc - 0.5)
    perm_dev = np.abs(permuted_aucs - 0.5)
    n_extreme = np.sum(perm_dev >= obs_dev)
    p_value = (n_extreme + 1) / (n_permutations + 1)

    if not return_permuted_aucs:
        permuted_aucs = None

    return actual_auc, p_value, permuted_aucs


def z_score2(data, data_window, baseline_window, baseline_mean=None, baseline_std=None):
    """
    Z-score a 1D or 2D PSTH array using a single baseline mean and std.

    If data is 2D, the shape is assumed to be (n_trials, n_timepoints), where
    the last dimension is time. A single baseline mean and std is computed from
    all values within the specified baseline window (flattened across trials).

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array of PSTH data. If 2D, shape should be (n_rows, n_timepoints).
    data_window : tuple (start_time, end_time)
        Time range covered by the PSTH data.
    baseline_window : tuple (baseline_start, baseline_end)
        Range of times used to calculate the baseline mean and std.
    baseline_mean : float, optional
        If provided, use this value as the baseline mean. Otherwise, compute it
        from the data.
    baseline_std : float, optional
        If provided, use this value as the baseline standard deviation.
        Otherwise, compute it from the data.

    Returns
    -------
    zscored_data : np.ndarray
        The z-scored PSTH, same shape as `data`.
    """

    if data.ndim > 2:
        raise ValueError("This function only supports 1D or 2D data.")
    if baseline_mean is None or baseline_std is None:
        # 1) Determine the length of the time dimension (assume it's the last axis).
        num_timepoints = data.shape[-1]

        # 2) Create a time axis over [start_time, end_time].
        #    For 1D data: shape = (num_timepoints,)
        #    For 2D data: shape = (1, num_timepoints) after broadcasting, but
        #    we'll only use it to build a boolean mask below.
        time_axis = np.linspace(data_window[0], data_window[1], num_timepoints)

        # 3) Identify which timepoints lie in the baseline window.
        baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])

        # 4) If baseline_mean or baseline_std is not provided, compute them from the data.

        # Extract baseline portion along the last dimension
        # If data is 2D, baseline_data.shape => (n_rows, n_baseline_pts)
        # If data is 1D, baseline_data.shape => (n_baseline_pts,)
        baseline_data = data[..., baseline_mask]

        # Flatten so we compute a single mean and std across *all* baseline points.
        baseline_data_flat = baseline_data.flatten()

        if baseline_mean is None:
            baseline_mean = np.mean(baseline_data_flat)
        if baseline_std is None:
            baseline_std = np.std(baseline_data_flat)

    # 5) Avoid division by zero
    if baseline_std == 0:
        baseline_std = 1.0

    # 6) Perform z-scoring
    zscored_data = (data - baseline_mean) / baseline_std

    return zscored_data

def psth_matrix(spikeData,
                trialData,
                window=(-0.5, 1.5),
                bin_size=0.005,
                smoothing_width=None,
                smooth_mode='regular',
                z_score_data=False,
                baseline_window=(-0.9, -0.1),
                bins=None,
                align_to='stimChange',
                z_score_mode=None,
                trialData_full=None):
    """
    Build a "psth" matrix shaped (number of time bins, number of trials, number of neurons)
    Args:
        spikeData:
        trialData:
        window:
        bin_size:

    Returns:

    """
    matrix = []

    for idx, spikeData_neuron in spikeData.iterrows():
        _, psth_per_trial = calc_psth_single(spikeData_neuron, trialData, window, bin_size, smoothing_width,
                                             bins=bins, align_to=align_to)

        if z_score_data:
            if z_score_mode == 'custom_baseline':
                _, baseline_std, baseline_mean = get_baseline_psth(trialData_full, spikeData_neuron, window=baseline_window,
                                                                   bins=bins)

            else:
                baseline_mean = None
                baseline_std = None

            psth_per_trial = z_score2(psth_per_trial, data_window=window, baseline_window=baseline_window,
                                      baseline_mean=baseline_mean, baseline_std=baseline_std)

        if smoothing_width is not None:
            psth_per_trial = smooth_psth_2(psth_per_trial, bin_size=bin_size, width=smoothing_width, mode=smooth_mode)

        matrix.append(psth_per_trial)

    matrix = np.stack(matrix, axis=0)

    return matrix


def colors(name):
    color = ""

    match name:
        case 'opto':
            color = '#0083edff'
        case 'lALM':
            color = '#C700B4'
        case 'rALM':
            color = '#00C713'

    return color


def smooth_psth_2(psth, bin_size, width, mode='regular'):
    """
    Apply Gaussian smoothing to the PSTH data, with an option for causal smoothing.

    Parameters:
    psth (np.array): The PSTH data (spike counts per bin). Can be 1D or 2D (trials x time).
    bin_size (float): The size of the bins for the histogram (in seconds).
    width (float): The width (standard deviation) of the Gaussian filter (in seconds).
    mode (str): 'regular' (default) for standard Gaussian smoothing,
                'causal' for causal Gaussian smoothing (half Gaussian filter).

    Returns:
    np.array: The smoothed PSTH data.
    """
    # Calculate sigma in bin units
    sigma = width / bin_size

    if mode == 'regular':
        # Apply standard Gaussian smoothing along the time axis (axis=-1 ensures 2D support)
        smoothed_psth = gaussian_filter1d(psth, sigma, axis=-1)
    elif mode == 'causal':
        # Create a half-Gaussian kernel
        filter_size = int(3 * sigma)  # Reasonable truncation at 3 sigma
        kernel = np.exp(-0.5 * (np.arange(filter_size + 1) / sigma) ** 2)
        kernel /= kernel.sum()  # Normalize

        # Apply causal convolution per trial
        smoothed_psth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='full')[:x.shape[-1]], axis=-1,
                                            arr=psth)
    else:
        raise ValueError("Invalid mode. Use 'regular' or 'causal'.")

    return smoothed_psth


def good_behavior_sessions(good_sessions_file=None):
    if good_sessions_file is None:
        good_sessions_file = config.resolve_pp("sessions_behavior_over_30.txt")
    good_sessions = []
    with open(good_sessions_file, "r") as file:
        for line in file:
            good_sessions.append(line.strip())  # strip() removes newline characters

    return good_sessions


def check_only_rALM(arr):
    """
    use this to check if sessionData is unilateral opto
    """
    strings_only = [x for x in arr if isinstance(x, str)]
    if len(strings_only) == 0:
        return True  # No strings, only empties
    return all(x == 'rALM' for x in strings_only)


def filter_trialData(trialData, conditions):
    filtered_df = trialData.copy()

    # Apply each filter condition (except decoder_variable)
    for key, value in conditions.items():
        if key in ['decoder_variable', 'align_to']:
            continue

        if isinstance(value, list):
            # If value is a list, filter rows where column is in that list
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        else:
            # Otherwise, filter for exact match
            filtered_df = filtered_df[filtered_df[key] == value]

    return filtered_df


def cluster_samples(trialData, stim_column, modality,
                    n_clusters=4, method='gmm', x_times=10, bin_size=10, verbose=False):
    """
    Clusters orientations (in degrees) into `n_clusters` using the specified method.

    Parameters:
        samples: pd.Series or np.ndarray of degrees [0-360)
        n_clusters: number of clusters
        method: 'gmm', 'kmeans', or 'quantile'

    Returns:
        labels: np.ndarray of cluster labels (0-based)
    """
    samples = trialData[stim_column]

    if modality in ["aud", "au", "auditory", "Y", "A"]:
        data_range = (13, 14)
        step_size = 1/360
        samples = (samples - 13) / (14 - 13) * 2 * np.pi
        X_circle = np.vstack((np.cos(samples), np.sin(samples))).T
    elif modality in ["vis", "visual", "X", "V"]:
        data_range = (0, 360)
        step_size = 1
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

    elif method == 'optimized_resample':
        if trialData.empty:
            ValueError("Method optimized_resample requires trialData (currently empty)")
        if verbose:
            print("Finding sessions that have overlapping stimulus distributions - This may take a while")
        qualifying_sessions_VL, qualifying_data_VL, trialData = \
            find_sessions_with_criteria(
                trialData, stim_column, n_stimuli=n_clusters, x_times=x_times, bin_size=bin_size, step_size=step_size,
                data_range=data_range, is_circular=True)

        if verbose:
            print(f"{stim_column} bin centers: {np.unique(qualifying_data_VL[stim_column + '_bin'])}")

        labels = trialData['stimulus_label']

    elif method == 'optimized_per_session':
        if trialData.empty:
            ValueError("Method optimized_per_session requires trialData (currently empty)")
        trialData_opt = find_two_stimuli_per_session(trialData, stim_column, x_times=x_times, bin_size=bin_size,
                                                     is_circular=True, step_size=step_size, data_range=data_range)
        labels = trialData_opt['stimulus_label']
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


def find_sessions_with_criteria(df: pd.DataFrame,
                                           column_name: str,
                                           n_stimuli: 2,
                                           x_times: int,
                                           bin_size: Union[int, float, None] = None,
                                           is_circular: bool = False,
                                           data_range: Union[Tuple[float, float], None] = None,
                                           step_size: Union[float, None] = None) -> Tuple[
    List[Union[str, int]], pd.DataFrame, pd.DataFrame]:
    """
    Vectorized version of the session filtering function for better performance.
    Finds sessions where at least n_stimuli of the SAME value each appear at least x_times.
    If no common stimuli exist across all sessions, finds the stimulus set that includes
    the maximum number of sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with session data
    column_name : str
        Column to analyze for stimuli
    n_stimuli : int
        Minimum number of unique stimuli required (should be 2 for this version)
    x_times : int
        Minimum occurrences required for each stimulus
    bin_size : Union[int, float, None], optional
        Size for binning values (default: None)
    is_circular : bool, optional
        Whether the data is circular (e.g., angles, time of day) (default: False)
    data_range : Union[Tuple[float, float], None], optional
        The full range of the data as (min, max) for circular calculations
        (e.g., (0, 360) for degrees, (13, 14) for a narrow range).
        Required if is_circular=True (default: None)
    step_size : Union[float, None], optional
        Step size for rotating bins to find optimal placement.
        If None, defaults to range_span/36 for circular or bin_size/10 for non-circular (default: None)

    Returns
    -------
    Tuple[List[Union[str, int]], pd.DataFrame, pd.DataFrame]
        - List of qualifying session IDs
        - DataFrame with only qualifying rows and stimulus labels
        - Original DataFrame with stimulus labels added (NaN for non-qualifying rows)
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
            import numpy as np

            min_val = working_df[column_name].min()
            max_val = working_df[column_name].max()

            if is_circular:
                if data_range is None:
                    raise ValueError("data_range must be provided when is_circular=True")

                range_min, range_max = data_range
                range_span = range_max - range_min

                # FIXED: Always create 2 bins separated by 1/4 of the range
                bin_separation = range_span / 4  # Fixed to 1/4 of range

                # Set default step size if not provided
                if step_size is None:
                    step_size = range_span / 36

                # Create rotation offsets to test
                num_steps = int(np.ceil(bin_separation / step_size))
                rotation_offsets = np.linspace(0, bin_separation, num_steps, endpoint=False)

                best_offset = 0
                max_rows = 0
                best_bin_centers = None

                # Try different rotation offsets
                for offset in rotation_offsets:
                    # Create 2 bin centers separated by 1/4 of the range
                    bin_centers = [
                        (range_min + offset - range_min) % range_span + range_min,
                        (range_min + offset + bin_separation - range_min) % range_span + range_min
                    ]

                    # Assign each value to nearest bin (considering circularity)
                    def assign_circular_bin(val):
                        # Normalize value to the range
                        normalized_val = (val - range_min) % range_span + range_min
                        min_dist = float('inf')
                        best_center = None
                        for center in bin_centers:
                            # Calculate circular distance
                            normalized_center = (center - range_min) % range_span + range_min
                            direct_dist = abs(normalized_val - normalized_center)
                            wrapped_dist = range_span - direct_dist
                            dist = min(direct_dist, wrapped_dist)
                            # Check if within bin_size/2 of center
                            if dist <= bin_size / 2 and dist < min_dist:
                                min_dist = dist
                                best_center = center
                        return best_center

                    temp_bins = working_df[column_name].apply(assign_circular_bin)
                    rows_in_bins = temp_bins.notna().sum()

                    if rows_in_bins > max_rows:
                        max_rows = rows_in_bins
                        best_offset = offset
                        best_bin_centers = bin_centers

                # Create final bins with best offset
                def assign_to_best_bin(val):
                    normalized_val = (val - range_min) % range_span + range_min
                    min_dist = float('inf')
                    best_center = None
                    for center in best_bin_centers:
                        normalized_center = (center - range_min) % range_span + range_min
                        direct_dist = abs(normalized_val - normalized_center)
                        wrapped_dist = range_span - direct_dist
                        dist = min(direct_dist, wrapped_dist)
                        if dist <= bin_size / 2 and dist < min_dist:
                            min_dist = dist
                            best_center = center
                    if best_center is not None:
                        return f"bin_center_{best_center:.1f}"
                    return None

                working_df[bin_column] = working_df[column_name].apply(assign_to_best_bin)

            else:
                # Non-circular data
                observed_range = max_val - min_val

                # FIXED: Always create 2 bins separated by 1/4 of the observed range
                bin_separation = observed_range / 4  # Fixed to 1/4 of range

                # Set default step size if not provided
                if step_size is None:
                    step_size = bin_size / 10

                # Create rotation offsets to test
                num_steps = int(np.ceil(bin_separation / step_size))
                rotation_offsets = np.linspace(min_val, min_val + bin_separation, num_steps, endpoint=False)

                best_offset = min_val
                max_rows = 0
                best_bin_centers = None

                # Try different offsets
                for offset in rotation_offsets:
                    # Create 2 bin centers separated by 1/4 of the range
                    bin_centers = [offset, offset + bin_separation]

                    # Assign each value to nearest bin
                    def assign_bin(val):
                        min_dist = float('inf')
                        best_center = None
                        for center in bin_centers:
                            dist = abs(val - center)
                            # Check if within bin_size/2 of center
                            if dist <= bin_size / 2 and dist < min_dist:
                                min_dist = dist
                                best_center = center
                        return best_center

                    temp_bins = working_df[column_name].apply(assign_bin)
                    rows_in_bins = temp_bins.notna().sum()

                    if rows_in_bins > max_rows:
                        max_rows = rows_in_bins
                        best_offset = offset
                        best_bin_centers = bin_centers

                # Create final bins with best offset
                def assign_to_best_bin(val):
                    min_dist = float('inf')
                    best_center = None
                    for center in best_bin_centers:
                        dist = abs(val - center)
                        if dist <= bin_size / 2 and dist < min_dist:
                            min_dist = dist
                            best_center = center
                    if best_center is not None:
                        return f"bin_center_{best_center:.1f}"
                    return None

                working_df[bin_column] = working_df[column_name].apply(assign_to_best_bin)

        else:
            # Categorical binning (unchanged)
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
        # Return empty results with original df (no labels)
        df_with_labels = df.copy()
        df_with_labels['stimulus_label'] = np.nan
        return [], pd.DataFrame(), df_with_labels

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
        from itertools import combinations
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
            df_with_labels = df.copy()
            df_with_labels['stimulus_label'] = np.nan
            return [], pd.DataFrame(), df_with_labels

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

    # Create original dataframe with labels
    df_with_labels = working_df.copy()
    df_with_labels['stimulus_label'] = np.nan

    # Assign labels to qualifying rows using the mask
    df_with_labels.loc[mask, 'stimulus_label'] = df_with_labels.loc[mask, analysis_column].map(stimulus_to_label)

    return final_qualifying_sessions, qualifying_data, df_with_labels


def find_two_stimuli_per_session(df: pd.DataFrame,
                                 column_name: str,
                                 x_times: int,
                                 bin_size: Union[int, float, None] = None,
                                 is_circular: bool = False,
                                 data_range: Union[Tuple[float, float], None] = None,
                                 step_size: Union[float, None] = None,
                                 verbose=False) -> pd.DataFrame:
    """
    Finds 2 maximally separated stimuli for each session independently.
    Each session gets its own optimized pair of stimuli.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with multiple sessions (must have 'session_ID' column)
    column_name : str
        Column to analyze for stimuli
    x_times : int
        Minimum occurrences required for each stimulus per session
    bin_size : Union[int, float, None], optional
        Size for binning values (default: None)
    is_circular : bool, optional
        Whether the data is circular (default: False)
    data_range : Union[Tuple[float, float], None], optional
        The full range of the data as (min, max) for circular calculations
    step_size : Union[float, None], optional
        Step size for rotating bins

    Returns
    -------
    pd.DataFrame
        Original dataframe with 'stimulus_label' column added (1, 2, or NaN)
    """
    import numpy as np
    from itertools import combinations

    if 'session_ID' not in df.columns:
        raise ValueError("DataFrame must contain 'session_ID' column")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Initialize output dataframe
    result_df = df.copy()
    result_df['stimulus_label'] = np.nan

    # Process each session independently
    for session_id in df['session_ID'].unique():
        session_mask = df['session_ID'] == session_id
        session_df = df[session_mask].copy()

        # Apply binning if specified
        if bin_size is not None and pd.api.types.is_numeric_dtype(session_df[column_name]):
            min_val = session_df[column_name].min()
            max_val = session_df[column_name].max()

            if is_circular:
                if data_range is None:
                    raise ValueError("data_range must be provided when is_circular=True")

                range_min, range_max = data_range
                range_span = range_max - range_min
                bin_separation = range_span / 4

                if step_size is None:
                    step_size = range_span / 36

                best_bin_centers = None
                max_rows = 0

                num_steps = int(np.ceil(bin_separation / step_size))
                rotation_offsets = np.linspace(0, bin_separation, num_steps, endpoint=False)

                for offset in rotation_offsets:
                    bin_centers = [
                        (range_min + offset - range_min) % range_span + range_min,
                        (range_min + offset + bin_separation - range_min) % range_span + range_min
                    ]

                    def assign_circular_bin(val):
                        normalized_val = (val - range_min) % range_span + range_min
                        min_dist = float('inf')
                        best_center = None
                        for center in bin_centers:
                            normalized_center = (center - range_min) % range_span + range_min
                            direct_dist = abs(normalized_val - normalized_center)
                            wrapped_dist = range_span - direct_dist
                            dist = min(direct_dist, wrapped_dist)
                            if dist <= bin_size / 2 and dist < min_dist:
                                min_dist = dist
                                best_center = center
                        return best_center

                    temp_bins = session_df[column_name].apply(assign_circular_bin)
                    rows_in_bins = temp_bins.notna().sum()

                    if rows_in_bins > max_rows:
                        max_rows = rows_in_bins
                        best_bin_centers = bin_centers

                    # Print bin centers and trial counts for this session


                def assign_to_best_bin(val):
                    normalized_val = (val - range_min) % range_span + range_min
                    min_dist = float('inf')
                    best_center = None
                    for center in best_bin_centers:
                        normalized_center = (center - range_min) % range_span + range_min
                        direct_dist = abs(normalized_val - normalized_center)
                        wrapped_dist = range_span - direct_dist
                        dist = min(direct_dist, wrapped_dist)
                        if dist <= bin_size / 2 and dist < min_dist:
                            min_dist = dist
                            best_center = center
                    return best_center if best_center is not None else None

                session_df['bin'] = session_df[column_name].apply(assign_to_best_bin)

                if best_bin_centers is not None:
                    session_df['bin'] = session_df[column_name].apply(assign_to_best_bin)
                    bin_counts = session_df['bin'].value_counts()
                    if verbose:
                        print(f"Session {session_id}:")
                        print(f"  Bin centers: {best_bin_centers}")
                        print(f"  Trials per bin: {bin_counts.to_dict()}")
                        print()

            else:  # Non-circular
                observed_range = max_val - min_val
                bin_separation = observed_range / 4

                if step_size is None:
                    step_size = bin_size / 10

                best_bin_centers = None
                max_rows = 0

                num_steps = int(np.ceil(bin_separation / step_size))
                rotation_offsets = np.linspace(min_val, min_val + bin_separation, num_steps, endpoint=False)

                for offset in rotation_offsets:
                    bin_centers = [offset, offset + bin_separation]

                    def assign_bin(val):
                        min_dist = float('inf')
                        best_center = None
                        for center in bin_centers:
                            dist = abs(val - center)
                            if dist <= bin_size / 2 and dist < min_dist:
                                min_dist = dist
                                best_center = center
                        return best_center

                    temp_bins = session_df[column_name].apply(assign_bin)
                    rows_in_bins = temp_bins.notna().sum()

                    if rows_in_bins > max_rows:
                        max_rows = rows_in_bins
                        best_bin_centers = bin_centers

                def assign_to_best_bin(val):
                    min_dist = float('inf')
                    best_center = None
                    for center in best_bin_centers:
                        dist = abs(val - center)
                        if dist <= bin_size / 2 and dist < min_dist:
                            min_dist = dist
                            best_center = center
                    return best_center if best_center is not None else None

                session_df['bin'] = session_df[column_name].apply(assign_to_best_bin)

            analysis_column = 'bin'
        else:
            analysis_column = column_name

        # Count occurrences of each stimulus in this session
        value_counts = session_df[analysis_column].value_counts()

        # Find stimuli that appear at least x_times
        qualifying_stimuli = value_counts[value_counts >= x_times].index.tolist()

        if len(qualifying_stimuli) < 2:
            continue  # Skip this session, leave labels as NaN

        # Find the pair of stimuli with maximum total trials
        best_pair = None
        max_total_trials = 0

        for stim1, stim2 in combinations(qualifying_stimuli, 2):
            total_trials = value_counts[stim1] + value_counts[stim2]
            if total_trials > max_total_trials:
                max_total_trials = total_trials
                best_pair = (stim1, stim2)

        selected_stimuli = set(best_pair)

        # Create labels for this session
        selected_stimuli_list = sorted(list(selected_stimuli))
        stimulus_to_label = {stimulus: i + 1 for i, stimulus in enumerate(selected_stimuli_list)}

        # Apply labels to the result dataframe
        for idx in session_df.index:
            stim_value = session_df.loc[idx, analysis_column]
            if stim_value in selected_stimuli:
                result_df.loc[idx, 'stimulus_label'] = stimulus_to_label[stim_value]

    return result_df


def calc_licks(trials, align_to='stimChange'):
    """
    Get normalized licktimes on a trial-by-trial basis for plotting into a rasterplot
    Args:
        trials:
        align_to:

    Returns:

    """

    licks_left = []
    licks_right = []

    for _, trial in trials.iterrows():
        # Get lick times for all trials
        lick_times = trial['lickTime'] - trial[align_to]
        lick_sides = trial['lickSide']

        lick_times = np.atleast_1d(lick_times)

        licks_left.append(lick_times[np.array(list(lick_sides)) == 'L'])
        licks_right.append(lick_times[np.array(list(lick_sides)) == 'R'])

    return licks_left, licks_right


@jit(nopython=True)
def align_and_filter_spikes(spiketimes, stim_times, window_start, window_end):
    """
    Numba-compiled spike alignment and filtering.
    Returns aligned spikes for all trials.
    """
    aligned = []
    for stim_time in stim_times:
        for spike in spiketimes:
            if window_start <= (spike - stim_time) <= window_end:
                aligned.append(spike - stim_time)
    return np.array(aligned)


def calculate_psth_modular_optimized(spike_data,
                                     trial_data,
                                     bin_size=0.01,
                                     window=(-0.5, 1.0),
                                     smoothing_width=None,
                                     z_score_psth=False,
                                     align_to='stimChange',
                                     trialData_full=None,
                                     calc_single_trial=False,
                                     step_size=None,
                                     use_causal_bins=False,
                                     z_score_mode=None,
                                     baseline_window=(-1, -0.1)):
    """
    Optimized PSTH calculation with the following improvements:
    1. Vectorized operations where possible
    2. Numba JIT compilation for inner loops
    3. Pre-filtering of trial data
    4. Reduced memory allocations
    5. Optimized histogram computation
    """

    # Pre-compute bins once
    if use_causal_bins or step_size is not None:
        bins, time = custom_bins(bin_size, step_size, window, causal=use_causal_bins)
    else:
        n_bins = int((window[1] - window[0]) / bin_size)
        bins = np.linspace(window[0], window[1], n_bins + 1) * 1e6
        time = bins[:-1] + 0.5 * bin_size * 1e6
        time = time / 1e6

    # Convert to numpy for faster indexing if needed
    bins_for_hist = bins if bins.ndim == 1 else bins[:, 1]

    # Pre-process trial data: group by session_ID for faster lookup
    trial_groups = trial_data.groupby('session_ID')

    # Pre-allocate lists with estimated capacity
    data_container = spike_data if isinstance(spike_data, pd.DataFrame) else pd.DataFrame([spike_data])
    n_neurons = len(data_container)

    psth_per_neuron = []
    neuron_ids = []
    hist_per_trial = [] if calc_single_trial else None

    window_start_us = window[0] * 1e6
    window_end_us = window[1] * 1e6

    if use_causal_bins:
        window_start_us -= (bin_size * 1e6)

    for index, neuron in data_container.iterrows():
        session_id = neuron['session_ID']
        spiketimes = np.atleast_1d(neuron['ts'])

        # Fast trial lookup using pre-grouped data
        try:
            trials = trial_groups.get_group(session_id)
        except KeyError:
            continue

        # Vectorized filtering of stim times
        stim_times = trials[align_to].values
        stim_times = stim_times[(stim_times != 0) & (~np.isnan(stim_times))]

        if len(stim_times) == 0:
            continue

        n_trials = len(stim_times)

        # Use compiled function for spike alignment
        aligned_spikes_neuron = align_and_filter_spikes(
            spiketimes, stim_times, window_start_us, window_end_us
        )

        # Single trial calculation if needed
        if calc_single_trial:
            trial_hists = []
            for stim_time in stim_times:
                mask = (spiketimes >= stim_time + window_start_us) & \
                       (spiketimes <= stim_time + window_end_us)
                trial_spikes = spiketimes[mask] - stim_time

                if bins.ndim == 2:
                    hist_single = custom_histogram(trial_spikes, bins=bins)
                else:
                    hist_single, _ = np.histogram(trial_spikes, bins_for_hist)
                trial_hists.append(hist_single / bin_size)
            hist_per_trial.extend(trial_hists)

        # Fast histogram computation
        if step_size is None and bins.ndim == 1:
            # Use optimized numpy histogram
            hist, _ = np.histogram(aligned_spikes_neuron, bins=bins_for_hist)
            psth = hist / (n_trials * bin_size)
        elif step_size is None:
            # Prepare bins for histogram
            bins_1d = bins[:, 1] if bins.ndim > 1 else bins
            bins_1d = np.concatenate([bins_1d, [bins_1d[-1] + bin_size]])

            if use_causal_bins:
                bins_1d = bins_1d - bin_size
            else:
                bins_1d = bins_1d - (bin_size / 2)

            hist, _ = np.histogram(aligned_spikes_neuron, bins=bins_1d)
            psth = hist / (n_trials * bin_size)
        else:
            hist = custom_histogram(aligned_spikes_neuron, bins=bins)
            psth = hist / (n_trials * bin_size)

        # Z-score PSTH
        if z_score_psth:
            if z_score_mode == 'custom_baseline':
                _, baseline_std, baseline_mean = get_baseline_psth(trial_data, spike_data,
                                                                   window=baseline_window, bins=bins)
            else:
                baseline_mean = None
                baseline_std = None

            psth = z_score2(psth, window, baseline_window=(-0.5, -0.1),
                           baseline_mean=baseline_mean, baseline_std=baseline_std)

        # Smooth PSTH
        if smoothing_width is not None:
            psth = smooth_psth(psth, bin_size, smoothing_width)

        psth_per_neuron.append(psth)
        neuron_ids.append(neuron['cluster_ID'])

    # Vectorized operations for final calculations
    all_psths_arr = np.vstack(psth_per_neuron)
    grand_average_psth = np.mean(all_psths_arr, axis=0)

    # Calculate SEM
    if len(data_container) == 1 and calc_single_trial:
        hist_per_trial = np.array(hist_per_trial)
        sem = np.std(hist_per_trial, axis=0) / np.sqrt(hist_per_trial.shape[0])
    else:
        sem = np.std(all_psths_arr, axis=0) / np.sqrt(all_psths_arr.shape[0])

    return grand_average_psth, all_psths_arr, sem, bins, neuron_ids, time


def plot_raster_and_licks_eventplot(ax, raster_data, raster_idx_tracker, color,
                                    alpha=1.0, linewidth=0.5,
                                    licks_left=None, licks_right=None,
                                    plot_licks=False, hide_ticks=True):
    """
    Ultra-fast raster and lick plotting using matplotlib's eventplot (80-100x faster).

    eventplot is specifically designed for plotting event trains (like spike rasters).
    This is the FASTEST method available.

    Parameters:
    -----------
    ax : matplotlib axis
    raster_data : list of tuples (trial_index, aligned_spikes)
    raster_idx_tracker : int, offset for trial indices
    color : color for spike lines
    alpha : float, transparency for spikes
    linewidth : float, width of spike lines
    licks_left : list of arrays, left lick times per trial
    licks_right : list of arrays, right lick times per trial
    plot_licks : bool or str, whether to plot licks

    Returns:
    --------
    int : number of trials plotted (to update raster_idx_tracker)
    """
    n_trials = len(raster_data)

    # Pre-allocate spike arrays
    spike_trains = [None] * n_trials
    spike_positions = np.zeros(n_trials)

    for i, (trial_index, aligned_spikes) in enumerate(raster_data):
        spike_trains[i] = aligned_spikes / 1e6
        spike_positions[i] = trial_index + raster_idx_tracker

    # Plot spikes
    if spike_trains:
        ax.eventplot(spike_trains,
                     lineoffsets=spike_positions,
                     colors=color,
                     linewidths=linewidth,
                     linelengths=1.0,  # Height of each tick (spans one trial)
                     alpha=alpha)

    # Plot licks if requested
    if plot_licks is not False and licks_left is not None and licks_right is not None:
        # Count non-empty lick trials for pre-allocation
        n_left = sum(1 for i, _ in raster_data
                     if i < len(licks_left) and len(licks_left[i]) > 0)
        n_right = sum(1 for i, _ in raster_data
                      if i < len(licks_right) and len(licks_right[i]) > 0)

        # Pre-allocate lick arrays
        if n_left > 0:
            left_trains = [None] * n_left
            left_positions = np.zeros(n_left)
            left_idx = 0

            for trial_index, _ in raster_data:
                if trial_index < len(licks_left) and len(licks_left[trial_index]) > 0:
                    left_trains[left_idx] = licks_left[trial_index] / 1e6
                    left_positions[left_idx] = trial_index + raster_idx_tracker
                    left_idx += 1

            ax.eventplot(left_trains,
                         lineoffsets=left_positions,
                         colors='green',
                         linewidths=0.35,
                         linelengths=1.0)

        if n_right > 0:
            right_trains = [None] * n_right
            right_positions = np.zeros(n_right)
            right_idx = 0

            for trial_index, _ in raster_data:
                if trial_index < len(licks_right) and len(licks_right[trial_index]) > 0:
                    right_trains[right_idx] = licks_right[trial_index] / 1e6
                    right_positions[right_idx] = trial_index + raster_idx_tracker
                    right_idx += 1

            ax.eventplot(right_trains,
                         lineoffsets=right_positions,
                         colors='green',
                         linewidths=0.35,
                         linelengths=1.0)
    if hide_ticks:
        ax.tick_params(axis='both', which='both',
                       left=False, right=False, top=False, bottom=False,
                       labelleft=False, labelright=False, labeltop=False, labelbottom=False)

    return n_trials


# Set the default fast plotting method (change this to switch methods)
plot_raster_and_licks_fast = plot_raster_and_licks_eventplot  # Fastest (80-100x)


# plot_raster_and_licks_fast = plot_raster_and_licks_linecollection  # Fast (50-60x)


def generate_psth_master_optimized(spike_data,
                                   trial_data_list,
                                   output_dir=config.FIGURE_DIR / "PSTH",
                                   bin_size=0.005,
                                   window=(-0.5, 1.0),
                                   smoothing_width=0.01,
                                   raster_plot=True,
                                   labels=None,
                                   svg=False,
                                   colors=None,
                                   return_figs=False,
                                   align_to='stimChange',
                                   z_score_psth=False,
                                   trialData_full=None,
                                   error_bars=True,
                                   calc_single_trial=False,
                                   linetypes=None,
                                   alphas=None,
                                   step_size=None,
                                   use_causal_bins=False,
                                   sort_order=None,
                                   plot_licks=False,
                                   opto_shading=False,
                                   plot_ax_labels=False,
                                   highlight_dict_conditions=None,
                                   highlight_colors=None,
                                   dpi=1000,
                                   highlight_plot_style='vspan',
                                   highlight_bar_height_data=None,
                                   figsize=(1.75, 1.35),
                                   height_ratios=(1, 2),
                                   y_lim=None,
                                   raster_dot_width=0.3,
                                   condition_separators=True):

    if spike_data is None:
        print("spike_data is None - skipping")
        return

    if spike_data.empty:
        print("The spike_data DataFrame is empty.")
        return

    required_spike_cols = {'session_ID', 'cluster_ID', 'ts'}
    if not required_spike_cols.issubset(spike_data.columns):
        print("The spike_data DataFrame is missing required columns.")
        return

    # ============================================================================
    # SETUP - DONE ONCE INSTEAD OF PER NEURON
    # ============================================================================

    # Create all necessary directories upfront
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)

    if svg:
        os.makedirs(os.path.join(output_dir, 'svg'), exist_ok=True)

    if raster_plot:
        os.makedirs(os.path.join(output_dir, 'raster'), exist_ok=True)

    # Set matplotlib settings ONCE (not per neuron)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    # Prepare trial_data_list
    if isinstance(trial_data_list, pd.DataFrame):
        trial_data_list = [trial_data_list]

    # Pre-group ALL trial data by session_ID for O(1) lookups
    trial_groups_list = [td.groupby('session_ID') for td in trial_data_list]
    n_conditions = len(trial_data_list)

    # Setup colors once
    if colors is None:
        cmap = colormaps['hsv']
        colors = [cmap(i / n_conditions) for i in range(n_conditions)]

    # Setup linetypes and alphas with defaults
    if linetypes is None:
        linetypes = ['-'] * n_conditions
    if alphas is None:
        alphas = [1] * n_conditions

    # Set default smoothing
    #if smoothing_width is None:
    #    if step_size is not None:
    #        smoothing_width = step_size
    #    else:
    #        smoothing_width = bin_size

    # Pre-compute highlight parameters if needed
    use_highlights = (highlight_dict_conditions is not None and
                      highlight_colors is not None)
    highlight_is_dict = (use_highlights and
                         isinstance(highlight_dict_conditions, dict))

    # ============================================================================
    # MAIN LOOP - OPTIMIZED
    # ============================================================================
    figures = []

    for index, neuron in spike_data.iterrows():
        session_id = neuron['session_ID']
        neuron_id = neuron['cluster_ID']

        # Create figure
        fig, ax = plt.subplots(2, 1, figsize=figsize,
                               gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
                               sharex=True, facecolor='none')

        ax[0].set_facecolor('none')
        ax[1].set_facecolor('none')

        raster_idx_tracker = 0

        # Loop through conditions
        for condition_idx in range(n_conditions):
            z_base = (n_conditions - condition_idx)  # condition 0 gets the highest

            # Fast trial lookup using pre-grouped data
            try:
                trials = trial_groups_list[condition_idx].get_group(session_id)
            except KeyError:
                continue

            if trials.empty:
                continue

            # Get line style parameters (cached)
            linetype = linetypes[condition_idx]
            alpha = alphas[condition_idx]
            color = colors[condition_idx]

            # Calculate PSTH
            psth, _, sem, bins, _, time = calculate_psth_modular_optimized(
                neuron, trials, bin_size, window, smoothing_width,
                align_to=align_to,
                z_score_psth=z_score_psth,
                trialData_full=trialData_full,
                calc_single_trial=calc_single_trial,
                step_size=step_size,
                use_causal_bins=use_causal_bins
            )

            # Calculate raster data
            raster_data, order = calc_raster(neuron, trials, window, align_to,
                                             sort_order=sort_order)

            # Handle licks if needed
            licks_left, licks_right = None, None
            if plot_licks is not False:
                if len(order) > 0:
                    licks_left, licks_right = calc_licks(trials.loc[order].copy(),
                                                         align_to=align_to)
                else:
                    licks_left, licks_right = calc_licks(trials, align_to=align_to)

                # Filter licks based on plot_licks setting
                if 'first_only' in plot_licks:
                    licks_left = [np.array([fl[fl >= 0][0]]) if (fl >= 0).any() else np.array([])
                                  for fl in licks_left]
                    licks_right = [np.array([fr[fr >= 0][0]]) if (fr >= 0).any() else np.array([])
                                   for fr in licks_right]

                if 'correct' in plot_licks:
                    left_correct = trials['leftCorrect'].values.astype(bool)
                    right_correct = trials['rightCorrect'].values.astype(bool)
                    licks_left = [lick if is_correct else np.array([])
                                  for lick, is_correct in zip(licks_left, left_correct)]
                    licks_right = [lick if is_correct else np.array([])
                                   for lick, is_correct in zip(licks_right, right_correct)]

            # Plot raster
            if raster_plot:
                ax_raster = ax[0]

                if condition_separators and raster_idx_tracker > 0:
                    ax_raster.axhline(raster_idx_tracker, color='black', linewidth=0.5, zorder=1, alpha=0.2)
                    raster_idx_tracker += 1  # reserve a blank row for the divider

                raster_idx_tracker += plot_raster_and_licks_fast(
                    ax_raster,
                    raster_data,
                    raster_idx_tracker,
                    color=color,
                    alpha=alpha,
                    linewidth=raster_dot_width,
                    licks_left=licks_left,
                    licks_right=licks_right,
                    plot_licks=plot_licks
                )

                # Configure raster axis (done once per condition)
                ax_raster.set_xlim(window[0], window[1])
                if plot_ax_labels:
                    ax_raster.set_ylabel('Trial', fontsize=6)

                ax_raster.tick_params(axis='y', which='major', labelsize=6)
                ax[0].tick_params(axis='x', which='both', bottom=False, top=False,
                                  labelbottom=False, labelrotation=60)
                ax[0].tick_params(axis='y', which='both', bottom=False, top=False,
                                  labelbottom=False, pad=0.5, labelrotation=60)
            else:
                # Still need to track raster index even if not plotting
                raster_idx_tracker += len(raster_data)

            # Plot PSTH
            ax_psth = ax[1]
            label = labels[condition_idx] if labels else None
            ax_psth.plot(time, psth, color=color, label=label,
                         linewidth=0.5, linestyle=linetype, alpha=alpha, zorder=z_base * 10 + 1, dash_capstyle='round')

            if y_lim is not None:
                ax_psth.set_ylim(y_lim)
            ax[1].tick_params(axis='both', pad=0.5)
            ax[1].tick_params(axis='y', labelrotation=60)

            for spine in ax_psth.spines.values():
                spine.set_zorder(100)

            # Error bars
            if error_bars:
                #bin_error = step_size if step_size is not None else bin_size
                bin_error = bin_size
                if smoothing_width is not None:
                    sem = smooth_psth(sem, bin_error, smoothing_width)
                ax_psth.fill_between(time, psth - sem, psth + sem,
                                     alpha=0.2, color=color, edgecolor='none', zorder=z_base * 10 + 1)

        # Configure PSTH axis labels
        if plot_ax_labels:
            ax_psth.set_xlabel('Time from stimulus (s)', fontsize=6)
            ylabel = 'Spike rate (Z-scores)' if z_score_psth else 'Spike rate (Hz)'
            ax_psth.set_ylabel(ylabel, fontsize=6)
        ax_psth.tick_params(axis='both', labelsize=6)

        # Dotted line at t=0
        ax_psth.axvline(0, linestyle='--', color='black', linewidth=0.5, alpha=0.5)
        ax_raster.axvline(0, linestyle='--', color='black', linewidth=0.5, alpha=0.5)

        # Opto shading
        if opto_shading:
            plot_utils.add_opto_shading(ax_psth)
            if raster_plot:
                plot_utils.add_opto_shading(ax[0])

        # Highlighting for e.g. statistical significance
        if use_highlights:
            _add_highlight_shading(ax_psth, neuron_id, time, bin_size, step_size,
                                   use_causal_bins, highlight_dict_conditions,
                                   highlight_colors, highlight_is_dict,
                                   highlight_plot_style=highlight_plot_style,
                                   highlight_bar_height_data=highlight_bar_height_data)

        # Set raster y-limits
        if raster_plot:
            ax[0].set_ylim(raster_idx_tracker, 0)

        lw = ax_psth.spines['top'].get_linewidth()

        # Save figure
        base_filename = f'{neuron_id}'
        png_path = os.path.join(output_dir, 'png', base_filename + '.png')
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=(lw/72)/2)

        # Plot svg version
        if svg:
            svg_path = os.path.join(output_dir, 'svg', base_filename + '.svg')
            plt.savefig(svg_path, bbox_inches='tight', transparent=True)

        if raster_plot:
            # Turn off all axis elements
            ax[0].set_frame_on(False)
            ax[0].axis('off')

            raster_png_path = os.path.join(output_dir, 'raster', base_filename + '_raster.png')
            extent = ax[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(raster_png_path, dpi=1000, bbox_inches=extent, transparent=True, pad_inches=0)

            # Save PSTH axis only as SVG (hide raster axis, then restore)
            ax[0].set_visible(False)
            psth_svg_path = os.path.join(output_dir, 'svg', base_filename + '_psth.svg')
            extent_psth = ax[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(psth_svg_path, bbox_inches=extent_psth, transparent=True, pad_inches=0)
            ax[0].set_visible(True)  # restore in case return_figs=True

        figures.append(fig)
        plt.close('all')

    if return_figs:
        return figures


def _pt_to_data_height(ax, height_pt):
    """Convert a height in points to a span in y data-coordinates."""
    fig = ax.figure
    # points -> display(pixels): pixels = pt * dpi / 72
    px = height_pt * fig.dpi / 72.0
    # display -> data: invert the transform at two points and take the y-delta
    inv = ax.transData.inverted()
    y0 = inv.transform((0, 0))[1]
    y1 = inv.transform((0, px))[1]
    return abs(y1 - y0)


def _compute_bar_geometry(ax_psth, dict_idx, highlight_plot_style,
                          highlight_bar_height_pt, highlight_bar_height_data,
                          bar_y_position):
    """
    Returns (bar_y_pos, bar_height) in data coordinates.
    bar_y_position: 'top' or 'bottom'
    Height priority: pt (if given) -> data (if given) -> 5% of y-range.
    """
    if highlight_plot_style != 'bar':
        # vspan ignores height/position
        return bar_y_position, highlight_bar_height_data

    ylim = ax_psth.get_ylim()
    yrange = ylim[1] - ylim[0]

    if highlight_bar_height_pt is not None:
        bar_height = _pt_to_data_height(ax_psth, highlight_bar_height_pt)
    elif highlight_bar_height_data is not None:
        bar_height = highlight_bar_height_data
    else:
        bar_height = 0.05 * yrange

    # stack successive conditions inward from the chosen edge
    if bar_y_position == 'top':
        bar_y_pos = ylim[1] - (dict_idx + 1) * bar_height
    else:  # 'bottom'
        bar_y_pos = ylim[0] + dict_idx * bar_height

    return bar_y_pos, bar_height


def _add_highlight_shading(ax_psth, neuron_id, time, bin_size, step_size,
                           use_causal_bins, highlight_dict_conditions,
                           highlight_colors, highlight_is_dict,
                           highlight_plot_style='vspan',
                           highlight_bar_height_data=None,
                           highlight_bar_height_pt=3,
                           bar_y_position='top'):
    """
    bar_y_position : 'top' or 'bottom' (only affects 'bar' style)
    highlight_bar_height_pt : bar thickness in points; overrides
        highlight_bar_height_data when set. Falls back to 5% of y-range.
    """
    if highlight_is_dict:
        iterator = enumerate(highlight_dict_conditions.items())
    else:
        iterator = enumerate(highlight_dict_conditions)

    for dict_idx, item in iterator:
        if highlight_is_dict:
            key_condition, highlight_dict_per_neuron = item
            if neuron_id not in highlight_dict_per_neuron:
                continue
            sig_array = np.array(highlight_dict_per_neuron[neuron_id])
        else:
            sig_array = np.array(item)

        bar_y_pos, bar_height = _compute_bar_geometry(
            ax_psth, dict_idx, highlight_plot_style,
            highlight_bar_height_pt, highlight_bar_height_data,
            bar_y_position
        )

        _shade_significant_regions(ax_psth, sig_array, time, bin_size,
                                   step_size, use_causal_bins,
                                   highlight_colors[dict_idx],
                                   plot_style=highlight_plot_style,
                                   bar_height_data=bar_height,
                                   bar_y_position=bar_y_pos)
def _shade_significant_regions(ax, sig_array, time, bin_size, step_size,
                               use_causal_bins, color, plot_style='vspan',
                               bar_height_data=None, bar_y_position='bottom'):
    """
    Helper to shade significant regions efficiently.
    Uses vectorized operations to find region boundaries.

    Parameters:
    -----------
    plot_style : str, default 'vspan'
        'vspan' - shade full height of plot (default)
        'bar' - plot as horizontal bars that can be stacked
    bar_height_data : float, default None
        Height of bars in data coordinates when plot_style='bar'
        If None, defaults to 5% of y-axis range
    bar_y_position : str or float, default 'bottom'
        'bottom' - bars at bottom of plot (in data coordinates)
        'top' - bars at top of plot
        float - specific y position in data coordinates
    """
    if len(sig_array) != len(time):
        print(f"Warning: sig_array length ({len(sig_array)}) doesn't match "
              f"number of bins ({len(time)})")
        return

    # Vectorized edge detection
    padded = np.concatenate(([False], sig_array, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Determine offset
    shade_offset = step_size if step_size is not None else bin_size

    # Vectorized shading calculation
    if use_causal_bins:
        x_starts = time[starts]
        x_ends = time[ends - 1]
    else:
        x_starts = time[starts] - shade_offset / 2
        x_ends = time[ends - 1] + shade_offset / 2

    # Plot based on style
    if plot_style == 'vspan':
        # Original behavior - full height shading
        for x_start, x_end in zip(x_starts, x_ends):
            ax.axvspan(x_start, x_end, color=color, alpha=0.3, linewidth=0)

    elif plot_style == 'bar':
        # Stacked bars at specific position in data coordinates
        from matplotlib.patches import Rectangle

        # Get current y-axis limits
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # Default bar height if not specified (5% of range)
        if bar_height_data is None:
            bar_height_data = 0.05 * y_range

        # Determine y position in data coordinates
        if bar_y_position == 'bottom':
            y_pos = ylim[0]
        elif bar_y_position == 'top':
            y_pos = ylim[1] - bar_height_data
        else:
            y_pos = float(bar_y_position)

        # Plot bars in data coordinates
        for x_start, x_end in zip(x_starts, x_ends):
            width = x_end - x_start
            rect = Rectangle((x_start, y_pos), width, bar_height_data,
                             facecolor=color, edgecolor='none',
                             alpha=0.6, clip_on=False)
            ax.add_patch(rect)


def balance_trials(trialData_1, trialData_2):
    """
    Balances number of trials (rows) in 2 dataframes by down-sampling to lowest trial count

    Args:
        trialData_1: df of trial data
        trialData_2: df of trial data

    Returns:
        balanced dataframes
    """
    n_min_trials = min(len(trialData_1), len(trialData_2))

    return (
        trialData_1.sample(n=n_min_trials, replace=False),
        trialData_2.sample(n=n_min_trials, replace=False)
    )


def psth_matrix_2(spikeData,
                trialData,
                window=(-0.5, 1.5),
                bin_size=0.005,
                smoothing_width=None,
                smooth_mode='regular',
                z_score_data=False,
                baseline_window=(-0.9, -0.1),
                bins=None,
                align_to='stimChange',
                z_score_mode=None):
    """
    Build a PSTH matrix shaped (neurons, trials, timepoints).

    Handles uneven trial counts across neurons by padding with zeros.
    Neurons from different sessions will only have data for their own session's trials,
    with zeros elsewhere.

    Args:
        spikeData: DataFrame where each row is a neuron
        trialData: DataFrame with trial information
        window: tuple of (start, end) times relative to alignment
        bin_size: size of time bins
        smoothing_width: width for smoothing (optional)
        smooth_mode: mode for smoothing
        z_score_data: whether to z-score the data
        baseline_window: window for baseline calculation
        bins: pre-computed time bins (optional)
        align_to: event to align to
        z_score_mode: mode for z-scoring

    Returns:
        np.ndarray: 3D array of shape (neurons, trials, timepoints)
    """
    # Collect PSTH data for each neuron
    psth_list = []
    max_trials = 0

    for idx, spikeData_neuron in spikeData.iterrows():
        _, psth_per_trial = calc_psth_single(spikeData_neuron, trialData, window, bin_size,
                                             smoothing_width, bins=bins, align_to=align_to)

        if z_score_data:
            if z_score_mode == 'custom_baseline':
                _, baseline_std, baseline_mean = get_baseline_psth(trialData, spikeData_neuron,
                                                                   window=baseline_window, bins=bins)
            else:
                baseline_mean = None
                baseline_std = None

            psth_per_trial = z_score2(psth_per_trial, data_window=window,
                                      baseline_window=baseline_window,
                                      baseline_mean=baseline_mean, baseline_std=baseline_std)

        if smoothing_width is not None:
            psth_per_trial = smooth_psth_2(psth_per_trial, bin_size=bin_size,
                                           width=smoothing_width, mode=smooth_mode)

        psth_list.append(psth_per_trial)
        max_trials = max(max_trials, psth_per_trial.shape[0])

    # Get number of timepoints (should be consistent across neurons)
    n_timepoints = psth_list[0].shape[1]
    n_neurons = len(psth_list)

    # Initialize matrix with zeros
    matrix = np.zeros((n_neurons, max_trials, n_timepoints))

    # Fill in the data for each neuron
    for neuron_idx, psth_per_trial in enumerate(psth_list):
        n_trials_this_neuron = psth_per_trial.shape[0]
        matrix[neuron_idx, :n_trials_this_neuron, :] = psth_per_trial

    return matrix


def psth_matrix_nan_full(spikeData,
                         trialData,
                         window=(-0.5, 1.5),
                         bin_size=0.005,
                         smoothing_width=None,
                         smooth_mode='regular',
                         z_score_data=False,
                         baseline_window=(-0.9, -0.1),
                         bins=None,
                         align_to='stimChange',
                         z_score_mode=None,
                         trialData_full=None):
    """
    Build a PSTH matrix shaped (neurons, trials, timepoints).
    """
    psth_list = []
    n_total_trials = len(trialData)

    # Create a mapping from original indices to positions in trialData
    index_to_position = {orig_idx: pos for pos, orig_idx in enumerate(trialData.index)}

    for idx, spikeData_neuron in spikeData.iterrows():
        _, psth_per_trial = calc_psth_single(spikeData_neuron, trialData, window, bin_size,
                                             smoothing_width, bins=bins, align_to=align_to)

        if z_score_data:
            if z_score_mode == 'custom_baseline':
                _, baseline_std, baseline_mean = get_baseline_psth(trialData_full, spikeData_neuron,
                                                                   window=baseline_window, bins=bins)
            else:
                baseline_mean = None
                baseline_std = None

            psth_per_trial = z_score2(psth_per_trial, data_window=window,
                                      baseline_window=baseline_window,
                                      baseline_mean=baseline_mean, baseline_std=baseline_std)

        if smoothing_width is not None:
            psth_per_trial = smooth_psth_2(psth_per_trial, bin_size=bin_size,
                                           width=smoothing_width, mode=smooth_mode)

        # Find which trials this neuron corresponds to
        neuron_session = spikeData_neuron['session_ID']
        trial_mask = trialData['session_ID'] == neuron_session

        # Get positions (not original indices)
        trial_positions = np.where(trial_mask)[0]

        # Create full trial array with NaNs
        psth_full = np.full((n_total_trials, psth_per_trial.shape[1]), np.nan)
        psth_full[trial_positions, :] = psth_per_trial

        psth_list.append(psth_full)

    matrix = np.stack(psth_list, axis=0)
    return matrix


def generate_grand_average_psth_optimized(spike_data_list,
                                          trial_data_list,
                                          labels=None,
                                          spike_labels=None,
                                          bin_size=0.005,
                                          window=(-0.5, 1.0),
                                          smoothing_width=0.01,
                                          output_dir=config.FIGURE_DIR / "PSTH",
                                          y_axis=None,
                                          align_to='stimChange',
                                          z_score_psth=False,
                                          trialData_full=None,
                                          error_bars=True,
                                          step_size=None,
                                          use_causal_bins=False,
                                          colors=None,
                                          linetypes=None,
                                          alphas=None,
                                          svg=False,
                                          return_fig=False,
                                          plot_ax_labels=True,
                                          opto_shading=False,
                                          dpi=300,
                                          base_file_name='grand_average_psth',
                                          log_scale=False):
    """
    Generate grand average PSTHs for all combinations of spike_data sets and trial sets.
    Plots all combinations in the same figure (n_spike_data × n_trial_data lines).

    Parameters:
    -----------
    spike_data_list : pd.DataFrame or list of pd.DataFrame
        DataFrame(s) containing spike times and metadata for each neuron.
        Required columns: 'session_ID', 'cluster_ID', 'ts'
    trial_data_list : list of pd.DataFrame or pd.DataFrame
        List of DataFrames containing different sets of trial information (conditions).
    labels : list of str, optional
        List of labels for each trial condition.
    spike_labels : list of str, optional
        List of labels for each spike_data set.
    bin_size : float
        Size of the bins for the histogram (in seconds).
    window : tuple
        Time window around the alignment event to consider (in seconds).
    smoothing_width : float, optional
        Width (std) of Gaussian filter for smoothing (in seconds).
    output_dir : str
        Directory where the output files will be saved.
    y_axis : tuple, optional
        Y-axis limits (min, max).
    align_to : str
        Column name in trial_data to align spikes to.
    z_score_psth : bool
        Whether to z-score the PSTH.
    trialData_full : pd.DataFrame, optional
        Full trial data for z-scoring baseline calculation.
    error_bars : bool
        Whether to plot SEM error bands.
    step_size : float, optional
        Step size for sliding window (if different from bin_size).
    use_causal_bins : bool
        Whether to use causal (backward-looking) bins.
    colors : list, optional
        List of colors for each condition.
    linetypes : list, optional
        List of line styles for each condition.
    alphas : list, optional
        List of alpha values for each condition.
    svg : bool
        Whether to also save as SVG.
    return_fig : bool
        Whether to return the figure object.
    plot_ax_labels : bool
        Whether to plot axis labels.
    opto_shading : bool
        Whether to add optogenetic stimulation shading.
    dpi : int
        DPI for saved figures.

    Returns:
    --------
    fig : matplotlib.figure.Figure (only if return_fig=True)
    """
    from datetime import datetime

    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================

    # Prepare spike_data_list (convert single DataFrame to list)
    if isinstance(spike_data_list, pd.DataFrame):
        spike_data_list = [spike_data_list]

    # Prepare trial_data_list
    if isinstance(trial_data_list, pd.DataFrame):
        trial_data_list = [trial_data_list]

    if not trial_data_list:
        print("The trial_data_list is empty.")
        return

    if not spike_data_list:
        print("The spike_data_list is empty.")
        return

    # Validate each spike_data
    required_spike_cols = {'session_ID', 'cluster_ID', 'ts'}
    for i, spike_data in enumerate(spike_data_list):
        if spike_data.empty:
            print(f"The spike_data DataFrame at index {i} is empty.")
            return
        if not required_spike_cols.issubset(spike_data.columns):
            print(f"The spike_data DataFrame at index {i} is missing required columns.")
            return

    # ============================================================================
    # SETUP
    # ============================================================================

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    grand_average_dir = os.path.join(output_dir, 'grand_average')
    os.makedirs(grand_average_dir, exist_ok=True)

    current_day = datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(grand_average_dir, current_day)
    os.makedirs(day_dir, exist_ok=True)

    # Set matplotlib settings
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    })

    # Pre-group trial data by session_ID for O(1) lookups
    trial_groups_list = [td.groupby('session_ID') for td in trial_data_list]

    n_spike_sets = len(spike_data_list)
    n_trial_sets = len(trial_data_list)
    n_combinations = n_spike_sets * n_trial_sets

    # Setup colors (one per combination)
    if colors is None:
        cmap = colormaps['hsv']
        colors = [cmap(i / n_combinations) for i in range(n_combinations)]

    # Setup linetypes and alphas with defaults
    if linetypes is None:
        linetypes = ['-'] * n_combinations
    if alphas is None:
        alphas = [1] * n_combinations

    # ============================================================================
    # CALCULATE PSTHs FOR ALL COMBINATIONS (spike_data × trial_data)
    # ============================================================================

    # Storage for all combinations
    all_neuron_psths = []  # List of lists, one per combination
    trial_counts = []  # Total trials per combination
    neuron_counts = []  # Number of neurons per combination
    combination_labels = []  # Labels for each combination
    time_bins = None

    combo_idx = 0
    for spike_idx, spike_data in enumerate(spike_data_list):
        for trial_idx, trial_data in enumerate(trial_data_list):
            trial_groups = trial_groups_list[trial_idx]

            combo_psths = []

            for index, neuron in spike_data.iterrows():
                session_id = neuron['session_ID']

                # Fast trial lookup using pre-grouped data
                try:
                    trials = trial_groups.get_group(session_id)
                except KeyError:
                    continue

                if trials.empty:
                    continue

                # Calculate PSTH for this neuron
                psth, _, _, bins, _, time = calculate_psth_modular_optimized(
                    neuron, trials, bin_size, window, smoothing_width,
                    align_to=align_to,
                    z_score_psth=z_score_psth,
                    trialData_full=trialData_full,
                    calc_single_trial=False,
                    step_size=step_size,
                    use_causal_bins=use_causal_bins
                )

                combo_psths.append(psth)

                # Store time bins (same for all)
                if time_bins is None:
                    time_bins = time

            all_neuron_psths.append(combo_psths)
            trial_counts.append(len(trial_data))  # Total trials in this trial_data set
            neuron_counts.append(len(combo_psths))

            # Build combination label
            spike_label = spike_labels[spike_idx] if spike_labels else f"Set{spike_idx + 1}"
            trial_label = labels[trial_idx] if labels else f"Cond{trial_idx + 1}"
            combination_labels.append(f"{spike_label} - {trial_label}")

            combo_idx += 1

    # ============================================================================
    # COMPUTE GRAND AVERAGES
    # ============================================================================

    grand_averages = []
    grand_sems = []

    for combo_idx in range(n_combinations):
        if len(all_neuron_psths[combo_idx]) == 0:
            print(f"No data for combination: {combination_labels[combo_idx]}")
            grand_averages.append(None)
            grand_sems.append(None)
            continue

        # Stack all neuron PSTHs and compute mean/SEM across neurons
        stacked_psths = np.vstack(all_neuron_psths[combo_idx])
        n_neurons = stacked_psths.shape[0]
        grand_avg = np.nanmean(stacked_psths, axis=0)
        grand_sem = np.nanstd(stacked_psths, axis=0) / np.sqrt(n_neurons)

        grand_averages.append(grand_avg)
        grand_sems.append(grand_sem)

    # ============================================================================
    # PLOTTING
    # ============================================================================

    # Calculate figure size to fit 1x1 inch subplot plus decorations
    left_margin = 0.6    # inches for y-label + tick labels
    right_margin = 0.05  # inches
    bottom_margin = 0.45 # inches for x-label + tick labels
    top_margin = 0.3     # inches for title

    subplot_width = 1.0  # inches
    subplot_height = 1.0 # inches

    fig_width = left_margin + subplot_width + right_margin
    fig_height = bottom_margin + subplot_height + top_margin

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Position axes in exact inches (converted to figure fraction)
    ax = fig.add_axes([
        left_margin / fig_width,
        bottom_margin / fig_height,
        subplot_width / fig_width,
        subplot_height / fig_height
    ])

    for combo_idx in range(n_combinations):
        if grand_averages[combo_idx] is None:
            continue

        psth = grand_averages[combo_idx]
        sem = grand_sems[combo_idx]
        color = colors[combo_idx] if combo_idx < len(colors) else colors[combo_idx % len(colors)]
        linetype = linetypes[combo_idx] if combo_idx < len(linetypes) else linetypes[combo_idx % len(linetypes)]
        alpha = alphas[combo_idx] if combo_idx < len(alphas) else alphas[combo_idx % len(alphas)]

        # Create label with trial count
        label = f"{combination_labels[combo_idx]} (n={trial_counts[combo_idx]})"

        # Plot mean PSTH
        ax.plot(time_bins, psth, color=color, label=label,
                linewidth=1.0, linestyle=linetype, alpha=alpha)

        # Plot error bars (SEM across neurons)
        if error_bars and sem is not None:
            ax.fill_between(time_bins, psth - sem, psth + sem,
                            alpha=0.2, color=color, edgecolor='none')

    # Stimulus onset line
    ax.axvline(0, linestyle='--', color='black', linewidth=0.5, alpha=0.5)

    # Title with neuron counts per combination
    if len(set(neuron_counts)) == 1:
        # All combinations have same neuron count
        title = f'Grand Average PSTH ({neuron_counts[0]} neurons)'
    else:
        # Different neuron counts per combination
        neuron_str = '/'.join(str(n) for n in neuron_counts)
        title = f'Grand Average PSTH ({neuron_str} neurons)'
    ax.set_title(title, fontsize=8)

    # Axis labels
    if plot_ax_labels:
        ax.set_xlabel('Time from stimulus (s)', fontsize=6)
        ylabel = 'Spike rate (Z-score)' if z_score_psth else 'Spike rate (Hz)'
        ax.set_ylabel(ylabel, fontsize=6)

    ax.tick_params(axis='both', labelsize=6, pad=0.5)
    ax.set_xlim(window[0], window[1])

    if y_axis is not None:
        ax.set_ylim(y_axis)

    ax.legend(fontsize=6, frameon=False)

    if log_scale:
        ax.set_yscale('log')

    # Opto shading
    if opto_shading:
        plot_utils.add_opto_shading(ax)

    # ============================================================================
    # SAVE FIGURE
    # ============================================================================

    # Generate unique filename
    counter = 0

    while True:
        file_name = base_file_name if counter == 0 else f"{base_file_name}_{counter}"
        png_path = os.path.join(day_dir, f"{file_name}.png")
        svg_path = os.path.join(day_dir, f"{file_name}.svg")

        if not os.path.exists(png_path) and not os.path.exists(svg_path):
            break
        counter += 1

    plt.savefig(png_path, dpi=dpi)

    if svg:
        plt.savefig(svg_path)

    print(f"Grand average PSTH saved to: {png_path}")

    if return_fig:
        return fig
    else:
        plt.close(fig)


def calculate_psth_modular_optimized_single_neuron(spike_data,
                                     trial_data,
                                     bin_size=0.01,
                                     window=(-0.5, 1.0),
                                     smoothing_width=None,
                                     z_score_psth=False,
                                     align_to='stimChange',
                                     trialData_full=None,
                                     step_size=None,
                                     use_causal_bins=False,
                                     bins=None,
                                     time=None):
    """
    Optimized PSTH calculation with the following improvements:
    1. Vectorized operations where possible
    2. Numba JIT compilation for inner loops
    3. Pre-filtering of trial data
    4. Reduced memory allocations
    5. Optimized histogram computation
    """
    if bins is None or time is None:
        # Pre-compute bins once
        if use_causal_bins or step_size is not None:
            bins, time = custom_bins(bin_size, step_size, window, causal=use_causal_bins)
        else:
            n_bins = int((window[1] - window[0]) / bin_size)
            bins = np.linspace(window[0], window[1], n_bins + 1) * 1e6
            time = bins[:-1] + 0.5 * bin_size * 1e6
            time = time / 1e6

    # Convert to numpy for faster indexing if needed
    bins_for_hist = bins if bins.ndim == 1 else bins[:, 1]

    neuron = spike_data if isinstance(spike_data, pd.DataFrame) else pd.DataFrame([spike_data])
    n_neurons = len(neuron)

    if n_neurons > 1:
        print("More than 1 neuron passed to single neuron function - Returning None")
        return None

    # Get trials only for this neuron
    session_id = neuron['session_ID'].item()
    trials = trial_data[trial_data['session_ID'] == session_id]

    psth_per_neuron = []
    hist_per_trial = []

    window_start_us = np.min(bins)
    window_end_us = np.max(bins)

    #if use_causal_bins:
    #    window_start_us -= (bin_size * 1e6)

    spiketimes = np.atleast_1d(neuron['ts'].item())

    # Vectorized filtering of stim times
    stim_times = trials[align_to].values
    stim_times = stim_times[(stim_times != 0) & (~np.isnan(stim_times))]

    n_trials = len(stim_times)

    # Use compiled function for spike alignment
    aligned_spikes_neuron = align_and_filter_spikes(
        spiketimes, stim_times, window_start_us, window_end_us
    )

    # Single trial calculation
    trial_hists = []
    for stim_time in stim_times:
        mask = (spiketimes >= stim_time + window_start_us) & \
               (spiketimes <= stim_time + window_end_us)
        trial_spikes = spiketimes[mask] - stim_time

        if bins.ndim == 2:
            hist_single = custom_histogram(trial_spikes, bins=bins)
        else:
            hist_single, _ = np.histogram(trial_spikes, bins_for_hist)
        trial_hists.append(hist_single)
    hist_per_trial.extend(trial_hists)

    # Fast histogram computation
    if step_size is None and bins.ndim == 1:
        # Use optimized numpy histogram
        hist, _ = np.histogram(aligned_spikes_neuron, bins=bins_for_hist)
        psth = hist / (n_trials * bin_size)
    elif step_size is None:
        # Prepare bins for histogram
        bins_1d = bins[:, 1] if bins.ndim > 1 else bins
        bins_1d = np.concatenate([bins_1d, [bins_1d[-1] + bin_size]])

        if use_causal_bins:
            bins_1d = bins_1d - bin_size
        else:
            bins_1d = bins_1d - (bin_size / 2)

        hist, _ = np.histogram(aligned_spikes_neuron, bins=bins_1d)
        psth = hist / (n_trials * bin_size)
    else:
        hist = custom_histogram(aligned_spikes_neuron, bins=bins)
        psth = hist / (n_trials * bin_size)

    # Z-score PSTH
    if z_score_psth:
        _, baseline_std, baseline_mean = get_baseline_psth(
            trialData_full, _, bin_size=bin_size, window=(-0.5, -0.1)
        )
        psth = z_score(psth, window, baseline_window=(-0.5, -0.1),
                       baseline_mean=baseline_mean, baseline_std=baseline_std)

    # Smooth PSTH
    if smoothing_width is not None:
        psth = smooth_psth(psth, bin_size, smoothing_width)

    psth_per_neuron.append(psth)
    neuron_id = neuron['cluster_ID'].item()

    # Calculate SEM
    hist_per_trial = np.array(hist_per_trial)
    sem = np.std(hist_per_trial, axis=0) / np.sqrt(hist_per_trial.shape[0])

    return psth, hist_per_trial, sem, bins, neuron_id, time


def filter_neurons_min_trials(spikeData, trialData_list, min_trials):
    """Keep only neurons that have >= min_trials in every trialData in the list.

    Parameters
    ----------
    spikeData : pd.DataFrame
        Must contain 'session_ID' and 'cluster_ID'.
    trialData_list : list of pd.DataFrame
        Each must contain 'session_ID'.
    min_trials : int
        Minimum number of trials required per session in every trialData.

    Returns
    -------
    pd.DataFrame
        Filtered spikeData.
    """
    # For each trialData, get the sessions that have enough trials
    valid_sessions_per_td = []
    for td in trialData_list:
        counts = td.groupby('session_ID').size()
        valid_sessions_per_td.append(set(counts[counts >= min_trials].index))

    # Sessions must meet the threshold in ALL trialData
    valid_sessions = set.intersection(*valid_sessions_per_td)

    return spikeData[spikeData['session_ID'].isin(valid_sessions)]