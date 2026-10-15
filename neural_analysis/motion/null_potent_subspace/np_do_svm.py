# project neural data onto motion subspaces, then SVM-decode stimulus identity
# from null vs. potent subspace reconstructions

from neural_analysis import utils
from neural_analysis.SVM.utils_svm import sample_exhaustive
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import yaml

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import config

# ============================================================
# Reconstruction helpers (unchanged from your reconstruct script)
# ============================================================

def reconstruct_subspace(trialData, spikeData, Q, window, bin_size=None, smoothing_width=None,
                         smooth_mode='regular', z_score_spikes=False, bins=None):
    """
    Reconstruct trial-aligned spiking data restricted to the subspace defined by Q.

    Returns
    -------
    recon : ndarray of shape (n_bins, n_trials, n_neurons)
    """
    matrix = utils.psth_matrix_2(spikeData, trialData, window=window, bin_size=bin_size,
                                 smoothing_width=smoothing_width, z_score_data=z_score_spikes,
                                 bins=bins)
    matrix = matrix.transpose(2, 1, 0)  # -> (bins, trials, neurons)
    recon = matrix @ Q @ Q.T            # -> (bins, trials, neurons)
    return recon


def contains_rALM_not_lALM(entry):
    try:
        has_rALM, has_lALM = False, False
        for item in entry:
            if isinstance(item, str):
                if 'rALM' in item: has_rALM = True
                if 'lALM' in item: has_lALM = True
            elif hasattr(item, '__iter__'):
                for subitem in item:
                    if isinstance(subitem, str):
                        if 'rALM' in subitem: has_rALM = True
                        if 'lALM' in subitem: has_lALM = True
        return has_rALM and not has_lALM
    except:
        return False


# ============================================================
# SVM helpers, adapted from your megascript
# ============================================================

def sample_data_recon(data_contrast, lookup, label_col, n_trials_per_class,
                      recon_key, le, n_neurons=None, sample_mode="all"):
    """
    Adapted version of sample_data() from the megascript that pulls from
    pre-computed null/potent reconstructions instead of raw psth_ses.

    Parameters
    ----------
    data_contrast : DataFrame
        One row per session. Must have columns:
            session_ID, neuron_IDs, trial_labels_reset,
            null_recon, potent_recon  (each shape (n_neurons, n_trials, n_bins))
    recon_key : {'null', 'potent'}
        Which reconstruction to sample from.
    """
    recon_col = f'{recon_key}_recon'
    all_sampled_neurons = []
    matrix = []

    if sample_mode == "all":
        all_sampled_neurons = data_contrast['neuron_IDs'].explode().tolist()
    elif sample_mode == "per_session":
        for _, ses in data_contrast.iterrows():
            all_sampled_neurons.extend(sample_exhaustive(ses['neuron_IDs'], size=n_neurons))
    elif sample_mode == "total_n":
        all_neurons = data_contrast['neuron_IDs'].explode().tolist()
        all_sampled_neurons = sample_exhaustive(all_neurons, size=n_neurons)

    labels = []
    classes_unique = None  # set on first neuron, must be consistent

    for neuron in all_sampled_neurons:
        ses_neuron = lookup[neuron]
        spike_data_ses = data_contrast[data_contrast['session_ID'] == ses_neuron]

        neuron_ids = np.asarray(spike_data_ses['neuron_IDs'].iloc[0])
        neuron_pos = int(np.where(neuron_ids == neuron)[0][0])

        if classes_unique is None:
            classes_unique = list(spike_data_ses['trial_labels_reset'].iloc[0].unique())

        psth_neuron_cls = []
        labels_neuron = []
        for cls in classes_unique:
            bool_trials_cls = spike_data_ses['trial_labels_reset'].iloc[0] == cls
            idx_trials_cls = bool_trials_cls[bool_trials_cls].index
            sample_trials = sample_exhaustive(idx_trials_cls, n_trials_per_class)

            psth_cls = spike_data_ses[recon_col].iloc[0][neuron_pos, sample_trials, :]
            psth_neuron_cls.append(psth_cls)
            labels_neuron.extend([cls] * n_trials_per_class)

        psth_neuron = np.stack(psth_neuron_cls)  # shape (n_classes, n_trials_per_class, n_bins)
        psth_neuron = psth_neuron.reshape(-1, psth_neuron.shape[-1])  # (n_trials_total, n_bins)
        matrix.append(psth_neuron)
        labels = labels_neuron

    matrix = np.stack(matrix)               # (n_neurons, n_trials_total, n_bins)
    labels = le.fit_transform(np.array(labels))
    labels_shuffled = np.random.permutation(labels)
    return matrix, all_sampled_neurons, labels, labels_shuffled


def filter_trialData(trialData, conditions):
    filtered = trialData.copy()
    for key, value in conditions.items():
        if key in ['decoder_variable', 'align_to', 'k_folds']:
            continue
        if isinstance(value, list):
            filtered = filtered[filtered[key].isin(value)]
        else:
            filtered = filtered[filtered[key] == value]
    return filtered


# ============================================================
# Settings
# ============================================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Paths
data_dir = config.DATA_DIR_EXP
subdir_subspace = "lALMrALM"  # Subdirectory of previous run
subspace_dir = config.resolve_pp("null_potent_subspace/subspace_results", subdir_subspace)

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
output_dir_base = (
    config.DATA_DIR_PP_NEW / "null_potent_subspace" / "decode_from_subspace" / subdir_subspace
)
# Reconstruction parameters (match the subspace-estimation script)
window = (-1, 1.5)
bin_size = 0.2
step_size = 0.01
smoothing_width = None
areas = ['lALM', 'rALM']
z_score_spikes = True
unilat_opto_only = False
min_trials = 20  # min trials per class per session
min_fire = 0.5

bins, time = utils.custom_bins(bin_size, step_size, window, causal=True)
bins = bins[1:]
x_time = time[1:]

# SVM settings (mirroring the megascript)
np.random.seed(42)
le = LabelEncoder()
n_iterations = 1000
k_folds = 5
n_trials_per_class = min_trials
alpha_value = 0.01

cv = StratifiedKFold(n_splits=k_folds)
estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SGDClassifier(max_iter=10**6, tol=1e-4, penalty='l2', alpha=alpha_value)),
])

# Analyses to run — only the two miss-vs-CR contrasts, on opto-OFF trials,
# since subspaces were estimated from opto-OFF trials.
analyses = {
    'vis_presence_miss_cr': {
        'trialType': ['X', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'trialType',
    },
    'aud_presence_miss_cr': {
        'trialType': ['Y', 'P'],
        'noResponse': 1,
        'hasphotostim': False,
        'decoder_variable': 'trialType',
    },

    'vis_presence_miss_cr_opto': {
        'trialType': ['X', 'P'],
        'noResponse': 1,
        'hasphotostim': True,
        'decoder_variable': 'trialType',
    },
    'aud_presence_miss_cr_opto': {
        'trialType': ['Y', 'P'],
        'noResponse': 1,
        'hasphotostim': True,
        'decoder_variable': 'trialType',
    },
}

# ============================================================
# Load data
# ============================================================

trialData, sessionData, spikeData = utils.load_data(
    path_root=data_dir, experiment=['ALM_1', 'ALM_2', 'ALM_3'], animals='all'
)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire, 'good')
spikeData = spikeData[spikeData['area'].isin(areas)]

sessions_in_dir = [d.name for d in subspace_dir.iterdir() if d.is_dir()]

if unilat_opto_only:
    mask = sessionData['optoarea'].apply(contains_rALM_not_lALM)
    sessions_unilat = sessionData['session_ID'][mask].tolist()
    sessions_in_dir = [s for s in sessions_in_dir if s in sessions_unilat]

# Only opto-OFF trials (subspaces were estimated from these)
trialData = trialData[trialData['hasphotostim'] == False]


# ============================================================
# Build per-contrast data: for each session and each contrast,
# compute null and potent reconstructions on the relevant trials.
# ============================================================

output = {}
metadata = {}

for analysis_name, conditions in analyses.items():
    print(f"\n=== {analysis_name} ===")
    label_col = conditions['decoder_variable']
    trialData_contrast = filter_trialData(trialData, conditions).dropna(subset=[label_col])

    # Build the data_contrast DataFrame with reconstructions per session
    data_contrast = []

    for session in sessions_in_dir:
        results_dir = os.path.join(subspace_dir, session)
        if not os.path.isdir(results_dir):
            continue

        spikeData_ses = spikeData[spikeData['session_ID'] == session]
        trialData_ses = trialData_contrast[trialData_contrast['session_ID'] == session]

        # Trial counts per class
        class_counts = trialData_ses[label_col].value_counts()
        if len(class_counts) < 2 or class_counts.min() < min_trials:
            continue

        # Load subspaces for this session
        neurons = np.load(os.path.join(results_dir, 'neurons.npy'), allow_pickle=True)
        wm = np.load(os.path.join(results_dir, 'weight_matrices.npz'), allow_pickle=True)
        Qnull = wm['Qnull']
        Qpotent = wm['Qpotent']

        # Order spikeData_ses to match the neuron order used during subspace estimation
        spikeData_ses = spikeData_ses.set_index('cluster_ID').loc[neurons].reset_index()
        if len(spikeData_ses) != len(neurons):
            print(f"  {session}: neuron count mismatch, skipping")
            continue

        # Reconstructions: shape (bins, trials, neurons)
        null_recon = reconstruct_subspace(
            trialData_ses, spikeData_ses, Qnull, window=window,
            bin_size=bin_size, smoothing_width=smoothing_width,
            z_score_spikes=z_score_spikes, bins=bins,
        )
        potent_recon = reconstruct_subspace(
            trialData_ses, spikeData_ses, Qpotent, window=window,
            bin_size=bin_size, smoothing_width=smoothing_width,
            z_score_spikes=z_score_spikes, bins=bins,
        )

        # Reorder to (neurons, trials, bins) to match psth_matrix convention
        # used by sample_data in the megascript.
        null_recon = null_recon.transpose(2, 1, 0)
        potent_recon = potent_recon.transpose(2, 1, 0)

        data_contrast.append({
            'session_ID': session,
            'neuron_IDs': spikeData_ses['cluster_ID'].tolist(),
            'trial_labels': trialData_ses[label_col],
            'trial_labels_reset': trialData_ses[label_col].reset_index(drop=True),
            'null_recon': null_recon,
            'potent_recon': potent_recon,
        })

    if not data_contrast:
        print(f"  no qualifying sessions for {analysis_name}")
        continue

    data_contrast = pd.DataFrame(data_contrast)
    print(f"  {len(data_contrast)} sessions, "
          f"{sum(len(n) for n in data_contrast['neuron_IDs'])} neurons total")

    lookup = data_contrast.explode('neuron_IDs').set_index('neuron_IDs')['session_ID']

    metadata[analysis_name] = {
        'conditions': conditions,
        'n_sessions': len(data_contrast),
        'n_neurons_total': sum(len(n) for n in data_contrast['neuron_IDs']),
        'sessions': data_contrast['session_ID'].tolist(),
        'n_neurons_per_session': [len(n) for n in data_contrast['neuron_IDs']],
        'n_iterations': n_iterations,
        'k_folds': k_folds,
        'alpha': alpha_value,
        'window': window,
        'bin_size': bin_size,
        'step_size': step_size,
        'min_trials_per_class': min_trials,
    }

    # ============================================================
    # Pseudopopulation SVM, run separately on null and potent reconstructions
    # ============================================================
    for recon_key in ['null', 'potent']:
        print(f"  decoding from {recon_key} subspace ...")

        results_key = f'results_{analysis_name}_{recon_key}'
        results_shuf_key = f'results_shuffled_{analysis_name}_{recon_key}'
        output[results_key] = []
        output[results_shuf_key] = []

        for iteration in range(n_iterations):
            X, sampled_neurons, y, y_shuf = sample_data_recon(
                data_contrast, lookup, label_col,
                n_trials_per_class=n_trials_per_class,
                recon_key=recon_key, le=le, sample_mode='all',
            )
            # X shape: (n_neurons, n_trials, n_bins)
            for idx_bin, b in enumerate(bins):
                bin_matrix = X[:, :, idx_bin].T  # (n_trials, n_neurons)

                scores = cross_validate(estimator, bin_matrix, y, cv=cv,
                                        scoring='balanced_accuracy', n_jobs=-1)
                scores_shuf = cross_validate(estimator, bin_matrix, y_shuf, cv=cv,
                                             scoring='balanced_accuracy', n_jobs=-1)

                output[results_key].append({
                    'bin': b, 'iteration': iteration, 'scores': scores,
                })
                output[results_shuf_key].append({
                    'bin': b, 'iteration': iteration, 'scores': scores_shuf,
                })

# ============================================================
# Save
# ============================================================
output_dir_base.mkdir(parents=True, exist_ok=True)
np.save(output_dir_base / 'output.npy', output)
with open(output_dir_base / 'metadata.yaml', 'w') as f:
    yaml.dump(metadata, f)
shutil.copy(__file__, output_dir_base / 'script.py')
print(f"\nSaved to {output_dir_base}")