
import pandas as pd
import numpy as np
import os
from neural_analysis import utils
from matplotlib_venn import venn2, venn3
from matplotlib import colormaps
import math
import matplotlib.patches as patches
from mne.stats import permutation_cluster_test, combine_adjacency


def compute_neuron_overlap(conditions, labels):
    """
    Computes the overlap and unique neurons across multiple conditions.

    Args:
        conditions (list of lists): List of neuron lists for each condition.
        labels (list of str): List of labels for the conditions.

    Returns:
        dict: A dictionary containing the overlap and unique neurons for each condition.
    """
    if len(conditions) != len(labels):
        raise ValueError("The number of conditions and labels must match.")

    # Convert neuron lists to sets
    condition_sets = [set(neurons) for neurons in conditions]

    # Initialize the result dictionary
    overlap_info = {}

    # Compute unique neurons for each condition
    for i, label in enumerate(labels):
        unique_to_condition = condition_sets[i].copy()
        for j, other_set in enumerate(condition_sets):
            if i != j:
                unique_to_condition -= other_set
        overlap_info[f"unique_to_{label}"] = unique_to_condition

    # Compute pairwise overlaps and overall intersection
    overlap_info["overlap_all"] = set.intersection(*condition_sets)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i < j:
                overlap_info[f"overlap_{label1}_{label2}"] = condition_sets[i] & condition_sets[j]

    return overlap_info


def round_up_special(n):
    order_of_magnitude = 10 ** (math.floor(math.log10(n)))  # Determine the order of magnitude
    candidate = math.ceil(n / order_of_magnitude) * order_of_magnitude  # Round up to nearest multiple of order

    while (candidate // 4) % 10 not in {0, 5}:  # Check the last digit when divided by 4
        candidate += order_of_magnitude  # Increment in steps of the order of magnitude

    return candidate


def roc_load_results(folder, load_permuted_auc=False):
    # Define file names
    auc_file = os.path.join(folder, 'auc_scores.npy')
    neurons_file = os.path.join(folder, 'neurons.npy')
    params_file = os.path.join(folder, 'params.npz')
    filters_file = os.path.join(folder, 'filters.npz')
    n_trials_file = os.path.join(folder, 'n_trials.npy')
    p_values_file = os.path.join(folder, 'p_values.npy')
    permuted_auc_scores_file = os.path.join(folder, 'auc_scores_permuted.npy')

    # Load existing files
    auc_scores = np.load(auc_file, allow_pickle=True) if os.path.isfile(auc_file) else None
    neurons = np.load(neurons_file, allow_pickle=True) if os.path.isfile(neurons_file) else None
    p_values = np.load(p_values_file, allow_pickle=True) if os.path.isfile(p_values_file) else None
    permuted_auc_scores = np.load(permuted_auc_scores_file, allow_pickle=True) if\
        os.path.isfile(permuted_auc_scores_file) else None

    if os.path.isfile(auc_file):
        params = np.load(params_file, allow_pickle=True)
        params = params['arr_0'].item()
    else:
        params = None

    if os.path.isfile(filters_file):
        filters = np.load(filters_file, allow_pickle=True)
    else:
        filters = None

    n_trials = np.load(n_trials_file, allow_pickle=True) if os.path.isfile(n_trials_file) else None
    if not load_permuted_auc:
        return auc_scores, neurons, p_values, params, filters, n_trials
    else:
        return auc_scores, neurons, p_values, params, filters, n_trials, permuted_auc_scores


def format_ROC_plot_ax(ax, round_up=True):
    # Customize the plot
    ax.set_xlabel('Time from stimulus (s)', fontsize=6, fontname='Arial')
    ax.set_ylabel('Fraction significantly coding neurons', fontsize=6, fontname='Arial')
    ax.axvline(0, color='black', alpha=0.2, linestyle='--', label='Stimulus onset')
    ax.legend()
    ax.grid(True)

   # ax.set_ylim(bottom=0)

    # Customize xlabel and xtick sizes
    ax.tick_params(axis='x', labelsize=6, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=0, rotation=60)

    # Adjust y-tick label alignment
    for label in ax.get_yticklabels():
        label.set_ha("right")

    # Remove all figure backgrounds
    ax.patch.set_alpha(0)  # Remove background from axes
    ax.figure.patch.set_alpha(0)  # Remove figure background

    ax.grid(False)
    #lines = ax.get_lines()
    #lines[0].set_color('#0083edff')
    #lines[1].set_color('#646464ff')
    #lines[0].set_linewidth(1)
    #lines[1].set_linewidth(1)

    # Get current y-axis limits
    _, y_max = ax.get_ylim()

    # Set exactly 3 y-ticks spaced within this range
    if round_up:
        y_max = round_up_special(y_max)

    yticks = np.linspace(0, y_max, 3)
    ax.set_yticks(yticks)

    # Set x-ticks
    ax.set_xticks([-1, 0, 1])

    # Remove legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    return ax


def add_sig_lines_ax(ax, bool_arrs, colors, linewidth=5, alpha=0.5, timeline=None, outline=None):
    # If color for the significance line is missing
    if len(bool_arrs) - len(colors) == 1:
        colors.append('black')

    if timeline is not None:
        x_data = timeline
    else:
        line = ax.get_lines()[0]  # reference line for x-axis positions. this tends to fail, use timeline instead!
        x_data = line.get_xdata()

    y_range = np.ptp(ax.get_ylim())
    spacing = y_range * 0.05  # spacing set as 5% of y-axis range

    # First line at y=0, subsequent lines stacked above
    for idx, (bool_arr, color) in enumerate(zip(bool_arrs, colors)):
        clusters = []
        start = None

        # Identify clusters of True values
        for i, flag in enumerate(bool_arr):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                clusters.append((start, i - 1))
                start = None
        if start is not None:
            clusters.append((start, len(bool_arr) - 1))

        # Plot horizontal line for each cluster
        for start_idx, end_idx in clusters:
            x_start = x_data[start_idx]
            x_end = x_data[end_idx]

            ax.hlines(y=spacing * (idx + 1), xmin=x_start, xmax=x_end,
                      color=color, linewidth=linewidth, alpha=alpha)

    return ax


def add_opto_shading(ax):
    """
    Draws a semi-transparent rectangle spanning the entire height of the plot,
    from x=0 to x=1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the overlay to.
    """

    # Get current y-limits
    y_min, y_max = ax.get_ylim()

    # Draw the rectangle spanning entire height
    rect = patches.Rectangle(
        (0, y_min),  # Bottom-left corner (x=0, bottom of plot)
        1,  # Width (x=1 - x=0)
        y_max - y_min,  # Height (entire y-axis range)
        facecolor='#0083ed',
        alpha=0.1,  # Semi-transparent
        clip_on=False  # Ensures it's always visible
    )
    ax.add_patch(rect)


def find_neurons_with_low_p_values(p_values, neurons, auc_scores, min_consecutive_values=10, p_value_threshold=0.001):
    matching_neurons = {}

    if p_values is None:
        return {}

    for i, neuron_p_values in enumerate(p_values):
        # Check for min_consecutive_values consecutive p-values below p_value_threshold
        count = 0
        auc_scores_neuron = auc_scores[i, :]
        for idx, p in enumerate(neuron_p_values):
            if p < p_value_threshold:
                count += 1
                if count >= min_consecutive_values:
                    matching_neurons[str(neurons[i])] = neuron_p_values < p_value_threshold
                    break
            else:
                count = 0

    return matching_neurons


def build_jaccard_array(neurons, p_values, threshold=0.01):
    result = p_values.shape[1] * [None]

    for i in range(p_values.shape[1]):
        # Find indices where p-value < threshold for column i
        significant_mask = p_values[:, i] < threshold
        # Get corresponding neuron IDs
        result[i] = neurons[significant_mask]

    return result


def calc_time_resolved_jaccard(set1, set2):
    if len(set1) != len(set2):
        print(f"ERROR: set1 is of size {len(set1)}, set2 is of size {len(set2)}")
        return

    jaccard_array = np.zeros(len(set1))

    # Jaccard index: intersection / union
    for i, (set1_bin, set2_bin) in enumerate(zip(set1, set2)):
        if set1_bin is not None:
            set1_bin = set(set1_bin.flatten())
        if set2_bin is not None:
            set2_bin = set(set2_bin.flatten())

        if len(set1_bin.union(set2_bin)) > 0:
            jaccard_array[i] = len(set1_bin.intersection(set2_bin)) / len(set1_bin.union(set2_bin))
        else:
            jaccard_array[i] = 0

    return jaccard_array

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

