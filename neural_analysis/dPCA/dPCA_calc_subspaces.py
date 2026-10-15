import numpy as np
from neural_analysis import utils
import os
from datetime import datetime
from dPCA.dPCA import dPCA as dPCA_original
from dPCA_rebalanced import RebalancedDPCA
import argparse
import sys
import shutil
import yaml
from scipy.stats import kendalltau
import config
import subprocess

# ============================================================================
# MONKEY-PATCH: fix operator precedence bug in dPCA.train_test_split (line 680)
#
# The original code:
#   X * (N_samples/(N_samples-1))[(slices)]
# Python parses this as:
#   X * (N_samples / ((N_samples-1)[(slices)]))
# because [] binds tighter than /.
#
# (N_samples-1)[(slices)] adds a trailing dim: (n,s,d) -> (n,s,d,1)
# Then N_samples / (n,s,d,1) = (n,s,d) / (n,s,d,1) -> broadcast error
#
# The intended behavior:
#   X * ((N_samples/(N_samples-1))[(slices)])
# i.e. divide first, then add dim for broadcasting against time axis.
# ============================================================================

use_rebalanced_dpca = False

if use_rebalanced_dpca:
    dPCA = RebalancedDPCA
else:
    dPCA = dPCA_original

_original_train_test_split = dPCA.train_test_split

def _patched_train_test_split(self, X, trialX, N_samples=None, sample_ax=0):
    """Patched: fixes operator precedence on line 680."""
    import types

    def flat2d(A):
        return A.reshape((A.shape[0], -1))

    protect = self.protect
    n_unprotect = len(X.shape) - len(protect) if protect is not None else len(X.shape)
    n_protect = len(protect) if protect is not None else 0

    if sample_ax != 0:
        raise NotImplementedError('The sample axis needs to come first.')

    protected = self._check_protected(trialX, protect)

    if ~protected:
        axes = [self.labels.index(ax) + 2 for ax in protect]
        trialX = self._roll_back(trialX, axes)
        X = np.squeeze(self._roll_back(X[None, ...], axes))

    if N_samples is None:
        N_samples = self._get_n_samples(trialX, protect=self.protect)

    idx = (np.random.rand(*N_samples.shape) * N_samples).astype(int)

    blindX = np.empty(trialX.shape[1:])
    it = np.nditer(np.empty(N_samples.shape), flags=['multi_index'])
    while not it.finished:
        blindX[it.multi_index + (np.s_[:],) * n_protect] = \
            trialX[(idx[it.multi_index],) + it.multi_index + (np.s_[:],) * n_protect]
        it.iternext()

    # FIX: parenthesize divisions BEFORE indexing
    expand = (np.s_[:],) * n_unprotect + (None,) * n_protect
    trainX = X * (N_samples / (N_samples - 1))[expand] - blindX * (1.0 / (N_samples - 1))[expand]

    if ~protected:
        blindX = self._roll_back(blindX[..., None], axes, invert=True)[..., 0]
        trainX = self._roll_back(trainX[..., None], axes, invert=True)[..., 0]

    trainX -= np.mean(flat2d(trainX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]
    blindX -= np.mean(flat2d(blindX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]

    return trainX, blindX

dPCA.train_test_split = _patched_train_test_split
print("Applied monkey-patch for dPCA operator precedence bug (line 680).")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run dPCA analysis with specified brain area')
    parser.add_argument('--area', nargs='+', default=None,
                        help='Brain area(s) to analyze (e.g., lALM rALM)')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of dPCA components per marginalization')
    parser.add_argument('--regularizer', type=str, default='auto',
                        help='Regularization: "auto", or a float value')
    return parser.parse_args()


# ============================================================================
# CONFIGURATION
# ============================================================================

if len(sys.argv) > 1:
    args = parse_arguments()
    area = args.area if args.area is not None else ['lALM', 'rALM']
    n_components = args.n_components
    regularizer = args.regularizer
else:
    area = ['lALM', 'rALM']  # you can manually change this if you want to run the script instead of calling it from elsewhere
    n_components = 10
    regularizer = 'auto'

# Parse regularizer
if regularizer == 'auto':
    reg_value = 'auto'
else:
    reg_value = float(regularizer)

np.random.seed(42)

# Parameters
min_fire = 0.5
quality = 'good'
path_root = config.DATA_DIR_EXP
experiment = ["ALM_1", "ALM_2", "ALM_3"]
animals = "all"
script_name = os.path.splitext(os.path.basename(__file__))[0]

sessions_exclude = ['008202304121458', '006202306141358', '021202402021327',
                    '022202403051220', '026202403191125', '026202403201103']

# Time window and binning
window = (-1, 1.5)
bin_size = 0.2
step_size = 0.01
z_score_data = False
align_to = 'stimChange'
join_marginalizations = True
run_validation = False  # Run cross-validated significance analysis, reconstruction quality, and dPCA vs PCA comparison
n_shuffles = 100  # Number of label shuffles for significance (100 = p<0.01 threshold)
n_splits = 100  # Number of train/test splits per shuffle
z_score_mode = 'custom_baseline' # 'custom_baseline' to use global mean and std, None to use mean and std calculated over included trials
# dPCA labels: s = stimulus (trial type), d = decision (lick direction), t = time
dpca_labels = 'sdt'

# Output directory
output_dir = os.path.join(config.DATA_DIR_PP_NEW, "dPCA", "".join(area), "subspace_results")

print(f"Area: {area}")
print(f"n_components: {n_components}")
print(f"regularizer: {reg_value}")
print(f"Output: {output_dir}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
trialData, sessionData, spikeData = utils.load_data(path_root, experiment, animals)

# Filter sessions with hitrate < 0.3
(trialData, sessionData, spikeData, _) = utils.filter_sessions_hitrate(
    trialData, sessionData=sessionData, spikeData=spikeData, min_hitrate=0.3
)

# Manually exclude sessions
trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
spikeData = spikeData[~spikeData['session_ID'].isin(sessions_exclude)]
sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]

# Only good neurons from specified area(s)
spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=min_fire, quality=quality)
spikeData = spikeData[spikeData['area'].isin(area)]

# Fix empty entries for no response
trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)

# Control trials only (no photostim)
trialData = trialData[trialData['hasphotostim'] == False]

print(f"Neurons: {len(spikeData)}")
print(f"Trials: {len(trialData)}")
print(f"Sessions: {trialData['session_ID'].nunique()}")

# ============================================================================
# DEFINE CONDITIONS
# ============================================================================

# Stimulus dimension: trial types X (visual), Y (auditory), P (probe/catch)
stimulus_types = ['X', 'Y']
stimulus_labels = {s: i for i, s in enumerate(stimulus_types)}
n_stimuli = len(stimulus_types)

# Decision dimension: lick direction L (left), R (right)
# Exclude no-response trials for the dPCA (we need a choice on every trial)
decision_types = ['L', 'R']
decision_labels = {d: i for i, d in enumerate(decision_types)}
n_decisions = len(decision_types)

# Filter trials: must have a lick response (L or R) and be one of the stimulus types
trialData_dpca = trialData[
    (trialData['trialType'].isin(stimulus_types)) &
    (trialData['responseSide'].isin(decision_types))
].copy()

print(f"\nTrials after filtering for stimulus types & lick responses: {len(trialData_dpca)}")

# Show condition counts
print("\nCondition counts (stimulus × decision):")
condition_counts = trialData_dpca.groupby(['trialType', 'responseSide']).size().unstack(fill_value=0)
print(condition_counts)

# ============================================================================
# BUILD dPCA DATA MATRICES
# ============================================================================

print("\nBuilding dPCA matrices...")

# Compute time bins
bins, time_centers = utils.custom_bins(bin_size=bin_size, step_size=step_size,
                                        window=window, causal=True)
n_timebins = len(bins)

print(f"Time bins: {n_timebins}, window: {window}, bin_size: {bin_size}, step_size: {step_size}")

# Get all unique sessions that have neurons
sessions_with_neurons = spikeData['session_ID'].unique()

# For each neuron, compute spike rates per trial in each time bin
# Then organize into the n × s × d × t × k matrix
#
# dPCA expects:
#   X      shape: (n_neurons, n_stimuli, n_decisions, n_timebins)         — trial-averaged
#   trialX shape: (max_trials, n_neurons, n_stimuli, n_decisions, n_timebins) — trials FIRST
# Conditions with fewer trials are NaN-padded in the trailing trial slots.

all_neuron_data = []
neuron_ids_included = []

for ses in sessions_with_neurons:
    spikeData_ses = spikeData[spikeData['session_ID'] == ses]
    trialData_ses = trialData_dpca[trialData_dpca['session_ID'] == ses]

    if len(trialData_ses) == 0 or len(spikeData_ses) == 0:
        continue

    # Compute PSTH matrix: neurons × trials × timebins
    psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
                                  z_score_data=z_score_data, align_to=align_to,
                                  window=window, trialData_full=trialData, z_score_mode=z_score_mode)

    neuron_ids_ses = spikeData_ses['cluster_ID'].values
    trial_types_ses = trialData_ses['trialType'].values
    trial_decisions_ses = trialData_ses['responseSide'].values

    for neuron_idx, neuron_id in enumerate(neuron_ids_ses):
        neuron_trials = {}
        for s_type in stimulus_types:
            for d_type in decision_types:
                mask = (trial_types_ses == s_type) & (trial_decisions_ses == d_type)
                if np.sum(mask) > 0:
                    # psth_ses shape: (n_neurons, n_trials, n_timebins)
                    neuron_trials[(s_type, d_type)] = psth_ses[neuron_idx, mask, :]
                else:
                    neuron_trials[(s_type, d_type)] = None

        all_neuron_data.append({
            'neuron_id': neuron_id,
            'session_id': ses,
            'trials': neuron_trials
        })
        neuron_ids_included.append(neuron_id)

# Filter neurons: require at least 2 trials in EVERY s×d condition.
# This is necessary because regularizer='auto' uses leave-one-out CV,
# so each condition needs ≥2 trials (1 held out, ≥1 for training).
# Neurons with 0 or 1 trials in any condition cause divide-by-zero / NaN propagation.
n_before_filter = len(all_neuron_data)
filtered_neuron_data = []
filtered_neuron_ids = []

min_trials_for_cv = 2  # Minimum trials per condition (for leave-one-out to work)

for neuron_data, neuron_id in zip(all_neuron_data, neuron_ids_included):
    keep = True
    for (s_type, d_type), trials in neuron_data['trials'].items():
        if trials is None or trials.shape[0] < min_trials_for_cv:
            keep = False
            break
    if keep:
        filtered_neuron_data.append(neuron_data)
        filtered_neuron_ids.append(neuron_id)

all_neuron_data = filtered_neuron_data
neuron_ids_included = filtered_neuron_ids

n_neurons = len(all_neuron_data)
print(f"Neurons before condition filtering: {n_before_filter}")
print(f"Neurons after requiring ≥{min_trials_for_cv} trials in all {n_stimuli}×{n_decisions} conditions: {n_neurons}")
print(f"  ({n_before_filter - n_neurons} neurons removed)")

# Find max number of trials across all neurons and all conditions
max_trials = 0
for neuron_data in all_neuron_data:
    for (s_type, d_type), trials in neuron_data['trials'].items():
        if trials is not None:
            max_trials = max(max_trials, trials.shape[0])

print(f"Max trials per condition: {max_trials}")

# Build the trial-averaged matrix X: shape (n_neurons, n_stimuli, n_decisions, n_timebins)
X = np.zeros((n_neurons, n_stimuli, n_decisions, n_timebins))

# Build the single-trial matrix trialX: shape (max_trials, n_neurons, n_stimuli, n_decisions, n_timebins)
# NOTE: dPCA requires trials as the FIRST axis. Conditions with fewer trials are NaN-padded.
trialX = np.full((max_trials, n_neurons, n_stimuli, n_decisions, n_timebins), np.nan)

for n_idx, neuron_data in enumerate(all_neuron_data):
    for s_idx, s_type in enumerate(stimulus_types):
        for d_idx, d_type in enumerate(decision_types):
            trials = neuron_data['trials'][(s_type, d_type)]
            if trials is not None and trials.shape[0] > 0:
                n_trials_available = trials.shape[0]
                # Trial-averaged
                X[n_idx, s_idx, d_idx, :] = np.nanmean(trials, axis=0)
                # Single trials: (n_trials_available, n_timebins) -> fill first n_trials slots
                trialX[:n_trials_available, n_idx, s_idx, d_idx, :] = trials

print(f"\nX shape (n_neurons, n_stimuli, n_decisions, n_timebins): {X.shape}")
print(f"trialX shape (max_trials, n_neurons, n_stimuli, n_decisions, n_timebins): {trialX.shape}")

# Print per-condition fill statistics
print("\nNaN fill statistics per condition (fraction of filled trial slots):")
for s_idx, s_type in enumerate(stimulus_types):
    for d_idx, d_type in enumerate(decision_types):
        filled = np.sum(~np.isnan(trialX[:, :, s_idx, d_idx, 0]))
        total = trialX[:, :, s_idx, d_idx, 0].size
        print(f"  {s_type}-{d_type}: {filled}/{total} slots filled "
              f"({100*filled/total:.1f}%)")

# ============================================================================
# RUN dPCA
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING dPCA")
print("=" * 80)

if join_marginalizations:
    dpca = dPCA(labels=dpca_labels,
                join={'s': ['s', 'st'], 'd': ['d', 'dt'], 'sd': ['sd', 'sdt']},
                n_components=n_components,
                regularizer=reg_value,)
else:
    dpca = dPCA(labels=dpca_labels,
                n_components=n_components,
                regularizer=reg_value,)

dpca.protect = ['t']  # Time axis should not be shuffled during cross-validation

# Fit and transform
Z = dpca.fit_transform(X, trialX)
P_saved = {k: v.copy() for k, v in dpca.P.items()}
D_saved = {k: v.copy() for k, v in dpca.D.items()}

print("\nMarginalizations and components:")
for key in sorted(Z.keys()):
    print(f"  {key}: {Z[key].shape}")

# ============================================================================
# COMPUTE EXPLAINED VARIANCE
# ============================================================================

print("\nComputing explained variance...")
var_explained = dpca.explained_variance_ratio_

print("\nVariance explained by marginalization:")
for key in sorted(var_explained.keys()):
    print(f"  {key}: {np.sum(var_explained[key]):.2f}%")


# ============================================================================
# DEMIXING INDEX
# ============================================================================
# For each component, quantify how "pure" it is: what fraction of its variance
# comes from its own marginalization vs. leaking from others.
#
# Method: marginalize the centered X, project each marginalized dataset through
# the component's decoder vector, and measure the variance of each projection.
# The demixing index is the fraction from the component's own marginalization.

print("\nComputing demixing index...")

# Center X the same way dPCA does internally
X_centered_dpca = X - np.mean(X.reshape(n_neurons, -1), axis=1).reshape(n_neurons, *([1] * len(dpca_labels)))

# Get marginalized data
mXs = dpca._marginalize(X_centered_dpca)

# Print shapes
print("Marginalized data shapes:")
for key, mX in mXs.items():
    print(f"  {key}: {mX.shape}")

demixing_index = {}

for marg_key in sorted(Z.keys()):
    demixing_index[marg_key] = []

    for comp in range(Z[marg_key].shape[0]):
        decoder_vec = dpca.D[marg_key][:, comp]  # (n_neurons,)

        # Project each marginalized dataset through this component's decoder
        var_per_marg = {}
        total_component_var = 0

        for other_marg, mX in mXs.items():
            # mX shape: (n_neurons, ...) with some dims possibly collapsed
            # Flatten neuron axis vs everything else
            mX_flat = mX.reshape(n_neurons, -1)  # (n_neurons, n_something)

            # Project: decoder_vec (n_neurons,) @ mX_flat (n_neurons, n_something)
            projection = decoder_vec @ mX_flat  # (n_something,)
            v = np.sum(projection ** 2)
            var_per_marg[other_marg] = v
            total_component_var += v

        # Demixing index = fraction from own marginalization
        if total_component_var > 0:
            own_fraction = var_per_marg.get(marg_key, 0) / total_component_var
        else:
            own_fraction = 0

        demixing_index[marg_key].append({
            'demixing_index': own_fraction,
            'var_fractions': {k: v / total_component_var if total_component_var > 0 else 0
                              for k, v in var_per_marg.items()}
        })

# Print demixing summary
print("\nDemixing index per component (fraction of variance from own marginalization):")
print(f"{'Marg':<6} {'Comp':<6} {'Demix':>8} {'Var%':>8}  Breakdown")
print("-" * 80)
for marg_key in sorted(demixing_index.keys()):
    for comp_idx, comp_info in enumerate(demixing_index[marg_key][:3]):  # Top 3 components
        demix = comp_info['demixing_index']
        var_pct = var_explained[marg_key][comp_idx]

        # Show top contributors
        fracs = comp_info['var_fractions']
        top_contribs = sorted(fracs.items(), key=lambda x: -x[1])[:3]
        breakdown_str = ', '.join([f"{k}={v:.2f}" for k, v in top_contribs])

        print(f"{marg_key:<6} {comp_idx:<6} {demix:>8.3f} {var_pct:>7.2f}%  {breakdown_str}")
    print()

# ============================================================================
# AXIS ORTHOGONALITY & COMPONENT CORRELATIONS (cf. Kobak et al. Figure 3e)
# ============================================================================
# Upper triangle: dot products between encoder axes (f_i · f_j).
#   Encoder columns approximate correlations between how strongly each neuron
#   expresses each component. Non-orthogonal pairs (stars) mean neurons
#   expressing one component tend to also express another.
#   Significance: |f1·f2| > 3.3/sqrt(N) for p<0.001 (Gaussian approx),
#   AND Kendall robust correlation significant at p<0.001.
#
# Lower triangle: Pearson correlations between components (projected data).
#   Should be near zero if dPCA is working well.

print("\nComputing axis orthogonality and component correlations...")

# Collect encoder vectors and component time series for first n_axes components per marg
n_axes_per_marg = min(3, n_components)

# Use the actual marginalizations present in the fitted model
marg_order_axes = [k for k in ['t', 's', 'd', 'sd'] if k in dpca.P]
# If not using join, also check for the split marginalizations
for k in ['st', 'dt', 'sdt']:
    if k in dpca.P and k not in marg_order_axes:
        marg_order_axes.append(k)

encoder_vectors = []  # list of (n_neurons,) unit vectors
component_series = []  # list of flattened projected data
axis_labels_orth = []  # e.g. 's0', 's1', 'd0', etc.
axis_marg = []  # which marginalization each belongs to

for marg_key in marg_order_axes:
    P_marg = dpca.P[marg_key]  # (n_neurons, n_components)
    Z_marg = Z[marg_key]  # (n_components, n_stimuli, n_decisions, n_timebins)

    n_avail = min(n_axes_per_marg, P_marg.shape[1], Z_marg.shape[0])
    for comp in range(n_avail):
        enc_vec = P_marg[:, comp].copy()
        # Normalize to unit length for dot product interpretation
        enc_norm = np.linalg.norm(enc_vec)
        if enc_norm > 0:
            enc_vec = enc_vec / enc_norm

        encoder_vectors.append(enc_vec)
        component_series.append(Z_marg[comp].ravel())
        axis_labels_orth.append(f"{marg_key}{comp}")
        axis_marg.append(marg_key)

n_total_axes = len(encoder_vectors)
print(f"Analyzing {n_total_axes} axes across {len(marg_order_axes)} marginalizations")

# Significance threshold for dot products (p<0.001 under null of random unit vectors)
sig_threshold_dot = 3.3 / np.sqrt(n_neurons)
sig_alpha_kendall = 0.001

# Compute matrices
dot_products = np.full((n_total_axes, n_total_axes), np.nan)
correlations = np.full((n_total_axes, n_total_axes), np.nan)
significant_pairs = np.zeros((n_total_axes, n_total_axes), dtype=bool)

for i in range(n_total_axes):
    for j in range(n_total_axes):
        if i == j:
            dot_products[i, j] = 1.0
            correlations[i, j] = 1.0
            continue

        # Dot products between encoder axes
        dp = np.dot(encoder_vectors[i], encoder_vectors[j])
        dot_products[i, j] = dp

        # Correlations between components
        r = np.corrcoef(component_series[i], component_series[j])[0, 1]
        correlations[i, j] = r

        # Significance test (only for cross-marginalization pairs, upper triangle)
        if axis_marg[i] != axis_marg[j] and j > i:
            if abs(dp) > sig_threshold_dot:
                # Also check Kendall robust correlation on raw (unnormalized) encoder weights
                # Parse component index from label (characters after marg key)
                comp_i = int(axis_labels_orth[i][len(axis_marg[i]):])
                comp_j = int(axis_labels_orth[j][len(axis_marg[j]):])
                tau, p_kendall = kendalltau(
                    dpca.P[axis_marg[i]][:, comp_i],
                    dpca.P[axis_marg[j]][:, comp_j]
                )
                if p_kendall < sig_alpha_kendall:
                    significant_pairs[i, j] = True
                    significant_pairs[j, i] = True

# Print significant non-orthogonal pairs
n_cross_pairs = 0
n_sig_pairs = 0
for i in range(n_total_axes):
    for j in range(i + 1, n_total_axes):
        if axis_marg[i] != axis_marg[j]:
            n_cross_pairs += 1
            if significant_pairs[i, j]:
                n_sig_pairs += 1
                print(f"  Non-orthogonal: {axis_labels_orth[i]} vs {axis_labels_orth[j]}, "
                      f"dot={dot_products[i,j]:.3f}, corr={correlations[i,j]:.3f}")

print(f"\n{n_sig_pairs}/{n_cross_pairs} cross-marginalization pairs significantly non-orthogonal "
      f"(p<0.001, |dot|>{sig_threshold_dot:.3f})")

# ============================================================================
# VALIDATION (optional)
# ============================================================================
#
# 1. Projection verification: re-project the training data X through
#    dpca.transform() and compare to Z. If these don't match, something
#    went wrong upstream. (Sanity check.)
#
# 2. Significance analysis: cross-validated classification with label shuffling
#    (Kobak et al. 2016, Figure 3 shaded regions). For each component at each
#    timepoint, a held-out trial is classified by nearest class mean in the
#    projected 1D space. This is repeated n_splits times to get a mean accuracy.
#    Then labels are shuffled n_shuffles times to build a null distribution.
#    A timepoint is significant if the true accuracy exceeds the maximum across
#    all shuffles (i.e. p < 1/(n_shuffles+1)).
#    With join, tests the joined marginalizations (s, d, sd), not the split ones.
#
# 3. Reconstruction quality: measures ||X_marg - P @ D.T @ X||^2 for each
#    marginalization. Quantifies how much information the low-rank approximation
#    loses. Uses dpca.reconstruct() which is P @ D.T @ X_centered.
#
# 4. Cumulative variance: dPCA vs PCA comparison. dPCA components are not
#    orthogonal, so they explain less cumulative variance than PCA for the same
#    number of components. A large gap means the demixing constraint is costly.

if run_validation:
    from sklearn.decomposition import PCA

    print("\n" + "=" * 80)
    print("RUNNING VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Projection verification (sanity check)
    # ------------------------------------------------------------------
    print("\n--- Projection verification ---")

    Z_verify = dpca.transform(X)

    all_match = True
    for key in sorted(Z.keys()):
        max_diff = np.max(np.abs(Z[key] - Z_verify[key]))
        match_str = "OK" if max_diff < 1e-10 else f"MISMATCH (max diff = {max_diff:.2e})"
        print(f"  {key}: {match_str}")
        if max_diff >= 1e-10:
            all_match = False

    if all_match:
        print("  All marginalizations match: transform(X) == fit_transform(X)")
    else:
        print("  WARNING: Some marginalizations don't match!")

    # ------------------------------------------------------------------
    # 2. Significance analysis (cross-validated classification)
    # ------------------------------------------------------------------
    print(f"\n--- Significance analysis (n_shuffles={n_shuffles}, n_splits={n_splits}) ---")
    print("This may take a while...")

    # significance_analysis internally removes the last label (time) and tests
    # all remaining marginalizations. With join={'s':['s','st'], ...}, it tests
    # the joined keys: s, d, sd.
    # axis=True gives per-timepoint significance masks.
    # n_consecutive: require N consecutive significant timepoints to reduce noise.
    masks, true_scores, shuffled_scores = dpca.significance_analysis(
        X, trialX,
        n_shuffles=n_shuffles,
        n_splits=n_splits,
        n_consecutive=10,
        axis=True,
        full=True
    )

    print("\nSignificance masks (fraction of timepoints significant):")
    for key in sorted(masks.keys()):
        mask = masks[key]  # (n_components, n_timebins)
        for comp in range(min(3, mask.shape[0])):
            frac_sig = np.mean(mask[comp])
            print(f"  {key} comp {comp}: {frac_sig:.1%} of timepoints significant")

    # ------------------------------------------------------------------
    # 3. Reconstruction quality
    # ------------------------------------------------------------------
    print("\n--- Reconstruction quality ---")

    # Center X the same way dPCA does
    X_zero_mean = X - np.mean(X.reshape(n_neurons, -1), axis=1).reshape(
        n_neurons, *([1] * len(dpca_labels)))
    total_var_recon = np.sum(X_zero_mean ** 2)

    # Compute fresh marginalizations on centered X for the reconstruction target
    # (mXs from the demixing index section may have been computed on a differently
    # centered X, so recompute to be safe)
    mXs_recon = dpca._marginalize(X)

    recon_errors = {}
    for marg_key in sorted(Z.keys()):
        # Reconstruct: P @ D.T @ X_centered, reshaped to original shape
        X_recon = dpca.reconstruct(X, marg_key)
        X_recon_flat = X_recon.reshape(n_neurons, -1)

        # Target: the marginalized data for this key
        mX_target = mXs_recon[marg_key].reshape(n_neurons, -1)

        recon_error = np.sum((mX_target - X_recon_flat) ** 2) / total_var_recon * 100
        recon_errors[marg_key] = recon_error
        print(f"  {marg_key}: reconstruction error = {recon_error:.2f}% of total variance")

    # Also compute total reconstruction (sum of all marginalizations)
    X_recon_total = np.zeros_like(X_zero_mean)
    for marg_key in Z.keys():
        X_recon_total += dpca.reconstruct(X, marg_key)
    total_recon_error = np.sum((X_zero_mean - X_recon_total.reshape(n_neurons, -1).reshape(X_zero_mean.shape)) ** 2) / total_var_recon * 100
    print(f"  Total (sum of all margs vs centered X): {total_recon_error:.2f}% unexplained")

    # ------------------------------------------------------------------
    # 4. Cumulative variance: dPCA vs PCA
    # ------------------------------------------------------------------
    print("\n--- Cumulative variance: dPCA vs PCA ---")

    # dPCA cumulative variance: collect all components, sort by variance
    all_comp_vars = []
    for marg_key in sorted(Z.keys()):
        for comp_idx in range(len(var_explained[marg_key])):
            all_comp_vars.append(var_explained[marg_key][comp_idx])
    all_comp_vars_sorted = np.sort(all_comp_vars)[::-1]
    dpca_cumvar = np.cumsum(all_comp_vars_sorted)

    # PCA cumulative variance
    X_flat = X_zero_mean.reshape(n_neurons, -1)
    n_pca_components = min(len(all_comp_vars), X_flat.shape[0], X_flat.shape[1])
    pca = PCA(n_components=n_pca_components)
    pca.fit(X_flat.T)  # PCA expects (samples, features)
    pca_cumvar = np.cumsum(pca.explained_variance_ratio_ * 100)

    print(f"  Components  | dPCA cumul. | PCA cumul.")
    print(f"  -----------+-------------+-----------")
    for i in [0, 4, 9, min(19, len(dpca_cumvar) - 1)]:
        if i < len(dpca_cumvar) and i < len(pca_cumvar):
            print(f"  {i+1:>10d}  | {dpca_cumvar[i]:>10.1f}% | {pca_cumvar[i]:>9.1f}%")

    print("\nValidation complete.")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving results...")
os.makedirs(output_dir, exist_ok=True)

# Save dPCA results
save_dict = dict(
    X=X,
    trialX=trialX,
    Z={k: v for k, v in Z.items()},
    time_centers=time_centers,
    bins=bins,
    neuron_ids=np.array(neuron_ids_included),
    stimulus_types=np.array(stimulus_types),
    decision_types=np.array(decision_types),
    var_explained=var_explained,
    component_var=var_explained,
    demixing_index=demixing_index,
    dot_products=dot_products,
    correlations_between_components=correlations,
    significant_pairs=significant_pairs,
    axis_labels_orth=np.array(axis_labels_orth),
    axis_marg_orth=np.array(axis_marg),
    encoder_axes=np.array(encoder_vectors),  # (n_total_axes, n_neurons), same order as axis_labels_orth
)

if run_validation:
    save_dict['significance_masks'] = {k: v for k, v in masks.items()}
    save_dict['significance_true_scores'] = {k: v for k, v in true_scores.items()}
    save_dict['recon_errors'] = recon_errors
    save_dict['dpca_cumvar'] = dpca_cumvar
    save_dict['pca_cumvar'] = pca_cumvar

np.savez(os.path.join(output_dir, 'dpca_results.npz'),
         **save_dict,
         allow_pickle=True)

# Save encoder/decoder weights
np.savez(os.path.join(output_dir, 'dpca_weights.npz'),
         P=P_saved, D=D_saved, allow_pickle=True)

# Save metadata
metadata = {
    'area': area,
    'n_neurons': n_neurons,
    'n_stimuli': n_stimuli,
    'n_decisions': n_decisions,
    'n_timebins': n_timebins,
    'max_trials': int(max_trials),
    'stimulus_types': stimulus_types,
    'decision_types': decision_types,
    'window': list(window),
    'bin_size': bin_size,
    'step_size': step_size,
    'n_components': n_components,
    'regularizer': str(reg_value),
    'dpca_labels': dpca_labels,
    'min_fire': min_fire,
    'quality': quality,
    'sessions_exclude': sessions_exclude,
    'experiment': experiment,
    'z_score': z_score_data,
    'join_marginalizations': join_marginalizations,
    'run_validation': run_validation,
    'n_shuffles': n_shuffles if run_validation else None,
    'n_splits': n_splits if run_validation else None,
    'rebalanced': use_rebalanced_dpca
}

with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Save this script
shutil.copy(__file__, os.path.join(output_dir, 'script.py'))

print(f"\nData saved to: {output_dir}")

# ============================================================================
# CALL PLOTTING SCRIPT
# ============================================================================

print("\n" + "=" * 80)
print("CALLING PLOTTING SCRIPT")
print("=" * 80)

plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plotting/dPCA_plot.py')
subprocess.run([sys.executable, plot_script, '--input_dir', output_dir], check=True)

print(f"\nDone! Results saved to: {output_dir}")