# Perform auROC analysis per neuron by calculating area under the curve (AUC) of receiver-operator characteristic for different trial contrasts

from neural_analysis import utils
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import shutil
import config

#warnings.filterwarnings("ignore")

def main():
    debug = False

    # "stimChange" or "firstLick"
    align_to = 'firstLick'

    if align_to == 'stimChange':
        alignment = 'stim_aligned'
    elif align_to == 'firstLick':
        alignment = 'lick_aligned'

    # Define folder to save results
    save_folder = config.DATA_DIR_PP_NEW / "ROC" / alignment

    os.makedirs(save_folder, exist_ok=True)
    perform_permutation_test = True

    # Parameters for data loading and neuron exclusion etc
    min_fire = 0.5
    quality = 'good'  # 'good', 'mua', None
    data_dir = config.DATA_DIR_EXP
    experiment = ["ALM_1", "ALM_2", "ALM_3"]
    animals = "all"
    z_score = True
    balance_trials = False  # Balance trials counts within contrast



    # Define window and bin size for PSTH
    window = [-1, 1.5]
    bin_size = 0.2
    step_size = 0.01
    bins, t_bins = utils.custom_bins(bin_size, step_size, window, causal=True)
    n_bins = len(bins)

    n_permutations = 1000

    params = {
        'window': window,
        'bin_size': bin_size,
        'step_size': step_size,
        'n_bins': n_bins,
        'min_fire': min_fire,
        'experiment': experiment,
        'z_scored': z_score,
        't_bins': t_bins
    }

    # Load behavioral and neuron data
    trialData_full, sessionData_full, spikeData_full = utils.load_data(data_dir, experiment, animals)

    # Identify unilateral opto sessions
    sessionData_full['unilat_opto'] = sessionData_full['optoarea'].apply(utils.check_only_rALM)

    # Cluster stimuli (orientation, pitch) for task-irrelevant stimulus information decoding
    stimulus_cluster_method = 'optimized_per_session'

    trialData_full.loc[:, 'visualOriPostChangeLeft_clust'] = \
        utils.cluster_samples(trialData_full, 'visualOriPostChangeLeft',
                              n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=15)

    trialData_full.loc[:, 'visualOriPostChangeRight_clust'] = \
        utils.cluster_samples(trialData_full, 'visualOriPostChangeRight',
                              n_clusters=2, method=stimulus_cluster_method, modality='vis', bin_size=15)

    trialData_full.loc[:, 'audioOctPostChange_clust'] = \
        utils.cluster_samples(trialData_full, 'audioOctPostChange',
                              n_clusters=2, method=stimulus_cluster_method, modality='aud', bin_size=1 / 24)

    areas = ['lALM', 'rALM']

    for opto_bool, opto_condition in [(True, 'unilat'), (True, 'bilat'), (False, 'opto_off')]:
        # If opto, select only unilateral opto sessions, and only lALM neurons
        if opto_bool:
            trialData = trialData_full[trialData_full['hasphotostim'] == True]
            trialData = trialData[trialData['optoLat'] == opto_condition]
            sessionData = sessionData_full[sessionData_full['session_ID'].isin(trialData['session_ID'].unique())]
            spikeData = spikeData_full[spikeData_full['session_ID'].isin(sessionData['session_ID'])]
        else:
            sessionData = sessionData_full
            trialData = trialData_full[trialData_full['hasphotostim'] == False]
            spikeData = spikeData_full

        for area in areas:
            spikeData_area = utils.exclude_neurons(spikeData, sessionData, min_fire, quality)
            spikeData_area = utils.filter_df(spikeData_area, area=area)
            spikeData_area.reset_index(drop=True, inplace=True)

            # Initialize arrays
            n_neurons = len(spikeData_area)

            # Initialize arrays to store AUCs and p-values
            auc_scores = np.empty((n_neurons, n_bins))
            auc_scores_permuted = np.full((n_neurons, n_bins, n_permutations), np.nan, dtype=np.float16) # Float16 to reduce memory size
            p_values = np.empty((n_neurons, n_bins)) if perform_permutation_test else None

            # Filters:
            contrast_filters_SA = {
                'visual_miss_cr': {
                    'filter_1':
                        {'trialType': 'X',
                         'noResponse': 1},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                },

                'auditory_miss_cr': {
                    'filter_1':
                        {'trialType': 'Y',
                         'noResponse': 1},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                },

                'left_lick_FA_CR': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'L'},
#
                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                },
#
                'right_lick_FA_CR': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'R'},
#
                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                    },
#
                'left_right_lick': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'L'},
#
                    'filter_2':
                        {'trialType': 'P',
                         'responseSide': 'R'},
                },
#
                'visual_hit_miss': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1},
#
                    'filter_2':
                        {'trialType': 'X',
                         'noResponse': 1},
                },
#
                'auditory_hit_miss': {
                    'filter_1':
                        {'trialType': 'Y',
                         'correctResponse': 1},
#
                    'filter_2':
                        {'trialType': 'Y',
                         'noResponse': 1},
                },

                'left_visual_miss_cr': {
                    'filter_1':
                        {'trialType': 'X',
                         'noResponse': 1,
                         'visualChangeSide': 1},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                },

                'right_visual_miss_cr': {
                    'filter_1':
                        {'trialType': 'X',
                         'noResponse': 1,
                         'visualChangeSide': 2},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1}
                },

                'left_visual_hit_miss': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1,
                         'visualChangeSide': 1},

                    'filter_2':
                        {'trialType': 'X',
                         'noResponse': 1,
                         'visualChangeSide': 1},
                },

                'right_visual_hit_miss': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1,
                         'visualChangeSide': 2},

                    'filter_2':
                        {'trialType': 'X',
                         'noResponse': 1,
                         'visualChangeSide': 2},
                },


                'visual_miss_auditory_miss': {
                    'filter_1':
                        {'trialType': 'X',
                         'noResponse': 1},
#
                    'filter_2':
                        {'trialType': 'Y',
                         'noResponse': 1},
                },
#
                'visual_hit_auditory_hit': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1},
#
                    'filter_2':
                        {'trialType': 'Y',
                         'correctResponse': 1},
                },
#
                #'visual_hit_FA': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'correctResponse': 1},
#
                #    'filter_2':
                #        {'trialType': 'P',
                #         'responseSide': 'L'},
                #},

                #'visual_left_hit_FA': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 1,
                #         'correctResponse': 1},
#
                #    'filter_2':
                #        {'trialType': 'P',
                #         'responseSide': 'L'},
                #},
#
                #'visual_right_hit_FA': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 2,
                #         'correctResponse': 1},
#
                #    'filter_2':
                #        {'trialType': 'P',
                #         'responseSide': 'L'},
                #},

                #'auditory_hit_FA': {
                #    'filter_1':
                #        {'trialType': 'Y',
                #         'correctResponse': 1},
##
                #    'filter_2':
                #        {'trialType': 'P',
                #         'responseSide': 'R'},
                #},

                #'visual_ori_left_miss': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'noResponse': 1,
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'noResponse': 1,
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 2}
                #},
#
                #'visual_ori_right_miss': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'noResponse': 1,
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'noResponse': 1,
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 2}
                #},

                #'visual_ori_left_hit': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'correctResponse': 1,
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'correctResponse': 1,
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 2}
                #},
#
                #'visual_ori_right_hit': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'correctResponse': 1,
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'correctResponse': 1,
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 2}
                #},

                #'visual_ori_left_all': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'visualChangeSide': 1,
                #         'visualOriPostChangeLeft_clust': 2}
                #},
#
                #'visual_ori_right_all': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'visualChangeSide': 2,
                #         'visualOriPostChangeLeft_clust': 2}
                #},
#
                #'visual_ori_left_all_max': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 1,
                #         'visualOriChange_norm': 2,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'visualChangeSide': 1,
                #         'visualOriChange_norm': 2,
                #         'visualOriPostChangeLeft_clust': 2}
                #},
#
                #'visual_ori_right_all_max': {
                #    'filter_1':
                #        {'trialType': 'X',
                #         'visualChangeSide': 2,
                #         'visualOriChange_norm': 2,
                #         'visualOriPostChangeLeft_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'X',
                #         'visualChangeSide': 2,
                #         'visualOriChange_norm': 2,
                #         'visualOriPostChangeLeft_clust': 2}
                #},

                #'auditory_pitch_miss': {
                #    'filter_1':
                #        {'trialType': 'Y',
                #         'noResponse': 1,
                #         'audioOctPostChange_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'Y',
                #         'noResponse': 1,
                #         'audioOctPostChange_clust': 2}
                #},
#
                #'auditory_pitch_hit': {
                #    'filter_1':
                #        {'trialType': 'Y',
                #         'correctResponse': 1,
                #         'audioOctPostChange_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'Y',
                #         'correctResponse': 1,
                #         'audioOctPostChange_clust': 2}
                #},

                #'auditory_pitch_all': {
                #    'filter_1':
                #        {'trialType': 'Y',
                #         'audioOctPostChange_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'Y',
                #         'audioOctPostChange_clust': 2}
                #},
#
                #'auditory_pitch_all_max': {
                #    'filter_1':
                #        {'trialType': 'Y',
                #         'audioOctChange_norm': 2,
                #         'audioOctPostChange_clust': 1},
#
                #    'filter_2':
                #        {'trialType': 'Y',
                #         'audioOctChange_norm': 2,
                #         'audioOctPostChange_clust': 2}
                #}

            }
            contrast_filters_LA = {
                'left_right_lick': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'L'},

                    'filter_2':
                        {'trialType': 'P',
                         'responseSide': 'R'},
                },

                'visual_hit_auditory_hit': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1},

                    'filter_2':
                        {'trialType': 'Y',
                         'correctResponse': 1},
                },

                'visual_hit_FA': {
                    'filter_1':
                        {'trialType': 'X',
                         'correctResponse': 1},

                    'filter_2':
                        {'trialType': 'P',
                         'responseSide': 'L'},
                },

                'auditory_hit_FA': {
                    'filter_1':
                        {'trialType': 'Y',
                         'correctResponse': 1},

                    'filter_2':
                        {'trialType': 'P',
                         'responseSide': 'R'}
                },

                'lick_no_lick': {
                    'filter_1':
                        {'trialType': 'P',
                         'noResponse': 0},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1},

                    'align_1': 'firstLick',
                    'align_2': 'stimChange',
                },

                'left_lick_no_lick': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'L'},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1},

                    'align_1': 'firstLick',
                    'align_2': 'stimChange',
                },

                'right_lick_no_lick': {
                    'filter_1':
                        {'trialType': 'P',
                         'responseSide': 'R'},

                    'filter_2':
                        {'trialType': 'P',
                         'noResponse': 1},

                    'align_1': 'firstLick',
                    'align_2': 'stimChange',
                },
            }

            if align_to == 'stimChange':
                contrast_filters = contrast_filters_SA
            elif align_to == 'firstLick':
                contrast_filters = contrast_filters_LA

            for contrast_name, filters in contrast_filters.items():
                filters_1 = filters['filter_1']
                filters_2 = filters['filter_2']

                # Per-condition alignment; fall back to the global align_to when not specified
                align_1 = filters.get('align_1', align_to)
                align_2 = filters.get('align_2', align_to)

                # Make / check output folders
                area_folder = os.path.join(save_folder, area)
                if not os.path.isdir(area_folder):
                    os.mkdir(area_folder)

                filter_folder = os.path.join(area_folder, opto_condition, contrast_name)

                if not os.path.isdir(filter_folder):
                    os.makedirs(filter_folder)
                else:
                    print("Folder for filter " + contrast_name +
                          " already exists, skipping. If you want to recalculate it, manually remove the folder")
                    continue

                n_trials = np.full((n_neurons, 2), np.nan)  # Initialize array to save number of trials per neuron
                neurons = [None] * n_neurons

                print(filters_1, filters_2)

                def process_neuron(neuron_index):
                    neuronData = spikeData_area.iloc[neuron_index:neuron_index + 1]
                    # Filter trials for the specific neuron
                    trialData_neuron = utils.filter_df_2(trialData, session_ID=neuronData['session_ID'].iloc[0])

                    # Filter trialData based on the filters
                    trialData_cond0 = utils.filter_df_2(trialData_neuron, **filters_1)
                    trialData_cond1 = utils.filter_df_2(trialData_neuron, **filters_2)

                    if balance_trials:
                        trialData_cond0, trialData_cond1 = utils.balance_trials(trialData_cond0, trialData_cond1)

                    n_trials_neuron = np.full((1, 2), np.nan)
                    n_trials_neuron[0, 0] = len(trialData_cond0)
                    n_trials_neuron[0, 1] = len(trialData_cond1)

                    # A single-class set makes AUC undefined; skip rather than feed garbage in
                    if len(trialData_cond0) == 0 or len(trialData_cond1) == 0:
                        print(f"Neuron {neuron_index}: one condition has 0 trials, skipping neuron")
                        return None

                    # Build labels in the same 0-then-1 order we will concatenate the PSTHs
                    trial_labels = np.concatenate([
                        np.zeros(len(trialData_cond0)),
                        np.ones(len(trialData_cond1)),
                    ])

                    # Calculate spike rates (PSTH) SEPARATELY per condition so each
                    # uses its own alignment event (e.g. firstLick vs stimChange).
                    _, psth_cond0 = utils.calc_psth_single(
                        neuronData, trialData_cond0, window=window,
                        bin_size=bin_size, align_to=align_1, bins=bins
                    )
                    _, psth_cond1 = utils.calc_psth_single(
                        neuronData, trialData_cond1, window=window,
                        bin_size=bin_size, align_to=align_2, bins=bins
                    )

                    # Rows = trials (cond0 first, then cond1), cols = bins
                    psth_per_trial = np.concatenate([psth_cond0, psth_cond1], axis=0)

                    if psth_per_trial.shape[0] != len(trial_labels):
                        print(f"PSTH rows != labels for neuron {neuron_index}, skipping neuron")
                        return None

                    if z_score:
                        # Baseline stats are per-neuron and applied uniformly, so
                        # z-scoring the concatenated array is equivalent to doing
                        # each condition separately.
                        _, baseline_std, baseline_mean = utils.get_baseline_psth(trialData=trialData_neuron,
                                                                                 spikeData=neuronData,
                                                                                 bin_size=bin_size)

                        psth_per_trial = utils.z_score2(data=psth_per_trial,
                                                        data_window=window,
                                                        baseline_window=None,
                                                        baseline_std=baseline_std,
                                                        baseline_mean=baseline_mean)

                    psth_per_trial = psth_per_trial.T  # Transpose to match dimensions (bins, trials)

                    auc_scores_row = np.empty(n_bins)
                    p_values_row = np.empty(n_bins) if perform_permutation_test else None
                    auc_permuted_row = np.empty((n_bins, n_permutations),
                                                dtype=np.float16) if perform_permutation_test else None

                    # Perform ROC analysis for each time bin
                    for bin_index in range(n_bins):
                        bin_spike_counts = psth_per_trial[bin_index, :]

                        if len(bin_spike_counts) != len(trial_labels):
                            print(f"Number of values is not equal to number of labels for neuron {neuron_index}")

                        auc, p_value, auc_permuted = utils.compute_auc_fast(
                            spike_counts=bin_spike_counts,
                            labels=trial_labels,
                            permutation_test=perform_permutation_test,
                            n_permutations=n_permutations,
                            return_permuted_aucs=False
                        )
                        auc_scores_row[bin_index] = auc
                        if perform_permutation_test:
                            p_values_row[bin_index] = p_value
                            auc_permuted_row[bin_index, :] = auc_permuted if auc_permuted is not None else np.nan

                    return neuron_index, auc_scores_row, p_values_row, neuronData['cluster_ID'].iloc[
                        0], n_trials_neuron, auc_permuted_row


                # Parallel processing of neurons
                print(f"Processing {n_neurons} neurons in parallel...")

                if debug:
                    results = []
                    for i in range(n_neurons):
                        result = process_neuron(i)
                        results.append(result)
                else:
                    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_neuron)(i) for i in range(n_neurons))

                # Collect the results
                for res in results:
                    if res is None:
                        continue
                    neuron_index, auc_scores_row, p_values_row, neuron_id, n_trials_neuron, auc_perm = res
                    auc_scores[neuron_index, :] = auc_scores_row
                    if perform_permutation_test:
                        p_values[neuron_index, :] = p_values_row
                        auc_scores_permuted[neuron_index, :, :] = auc_perm
                    neurons[neuron_index] = neuron_id
                    n_trials[neuron_index, 0] = n_trials_neuron[0][0]
                    n_trials[neuron_index, 1] = n_trials_neuron[0][1]

                # Save

                with open(os.path.join(filter_folder, 'auc_scores.npy'), 'wb') as f:
                    np.save(f, auc_scores)
                    f.flush()  # Force flush to disk

                with open(os.path.join(filter_folder, 'params.npz'), 'wb') as f:
                    np.savez(f, params)
                    f.flush()

                with open(os.path.join(filter_folder, 'neurons.npy'), 'wb') as f:
                    np.save(f, neurons)
                    f.flush()

                if perform_permutation_test:
                    with open(os.path.join(filter_folder, 'p_values.npy'), 'wb') as f:
                        np.save(f, p_values)
                        f.flush()

                    with open(os.path.join(filter_folder, 'auc_scores_permuted.npy'), 'wb') as f:
                        np.save(f, auc_scores_permuted)
                        f.flush()


                with open(os.path.join(filter_folder, 'filters.npz'), 'wb') as f:
                    np.savez(f, filters=[filters_1, filters_2])
                    f.flush()

                with open(os.path.join(filter_folder, 'n_trials.npy'), 'wb') as f:
                    np.save(f, n_trials)
                    f.flush()

                shutil.copy(__file__, os.path.join(filter_folder, 'script.py'))


print('ROC Analysis Complete')

if __name__ == "__main__":
    main()