"""
Rebalanced demixed Principal Component Analysis (dPCA) for unbalanced datasets.

Implements the rebalanced loss function from:
    Kobak et al. (2016) "Demixed principal component analysis of neural
    population data", eLife.

The key idea: when trial counts differ across conditions, the standard dPCA
solution is biased. The rebalanced formulation operates on PSTHs (treating all
parameter combinations equally) and adds an explicit noise covariance term
that is itself rebalanced (averaged across conditions rather than pooled).

The rebalanced loss is:
    L_f = ||X̃_f - F D X̃||² + SQT ||F D C̃_noise^{1/2}||² + λ||F D||²

Leading to the full-rank solution:
    A_RR = X̃_f X̃ᵀ (X̃ X̃ᵀ + SQT · C̃_noise + λI)⁻¹

And the reduced-rank solution via SVD of A_RR @ X̃.

Usage
-----
    from dPCA_rebalanced import RebalancedDPCA

    # X: (n_neurons, n_cond1, n_cond2, ..., n_time) - trial-averaged PSTHs
    # trialX: (n_trials, n_neurons, n_cond1, n_cond2, ..., n_time) - single trials
    #         pad with NaN where trial counts differ

    dpca = RebalancedDPCA(labels='st', n_components=10, regularizer='auto')
    Z = dpca.fit_transform(X, trialX)

    # Or with manual regularizer and explicit rebalancing:
    dpca = RebalancedDPCA(labels='st', n_components=10, regularizer=1e-3)
    Z = dpca.fit_transform(X, trialX)
"""

import numpy as np
from collections import OrderedDict
from scipy.linalg import pinv, sqrtm
from sklearn.utils.extmath import randomized_svd

def compute_rebalanced_noise_covariance(trialX):
    """
    Compute the rebalanced noise covariance matrix C̃_noise.

    From the paper:
        C̃_noise = (1/SQT) Σ_{s,d,t} C_noise(s,d,t)
                 = <C_noise(s,d,t)>_{s,d,t}

    where C_noise(s,d,t) is the noise covariance for a single condition,
    computed from the trial-to-trial variability. This treats all conditions
    equally regardless of trial count.

    Parameters
    ----------
    trialX : ndarray, shape (n_max_trials, n_neurons, n_cond1, n_cond2, ..., n_time)
        Single-trial data. Pad with NaN where trial counts differ across
        conditions.

    Returns
    -------
    C_noise : ndarray, shape (n_neurons, n_neurons)
        Rebalanced noise covariance matrix.

    n_conditions_total : int
        Total number of condition combinations (SQT in the paper, but
        here specifically the count of non-empty condition bins used in
        the averaging).
    """
    n_max_trials = trialX.shape[0]
    n_neurons = trialX.shape[1]

    # All axes after neurons and before time are condition axes.
    # The last axis is typically time, which is part of the "condition" tuple
    # in the paper's formulation (each s,d,t is a separate bin).
    # We flatten all condition dimensions (including time) into one.
    condition_shape = trialX.shape[2:]  # (n_cond1, n_cond2, ..., n_time)
    n_condition_bins = int(np.prod(condition_shape))

    # Reshape: (n_trials, n_neurons, n_condition_bins)
    trialX_flat = trialX.reshape(n_max_trials, n_neurons, n_condition_bins)

    C_noise_sum = np.zeros((n_neurons, n_neurons))
    n_valid_bins = 0

    for b in range(n_condition_bins):
        # Extract trials for this condition bin: (n_trials, n_neurons)
        trials = trialX_flat[:, :, b]  # (n_max_trials, n_neurons)

        # Find valid (non-NaN) trials
        valid_mask = ~np.any(np.isnan(trials), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < 2:
            # Need at least 2 trials to estimate covariance
            continue

        valid_trials = trials[valid_mask]  # (n_valid, n_neurons)

        # Center (subtract condition mean = PSTH for this bin)
        residuals = valid_trials - valid_trials.mean(axis=0, keepdims=True)

        # Covariance for this condition: C_noise(s,d,t)
        # Using unbiased estimator (n-1)
        C_bin = residuals.T @ residuals / (n_valid - 1)

        # Equal weight per condition (this IS the rebalancing)
        C_noise_sum += C_bin
        n_valid_bins += 1

    if n_valid_bins == 0:
        raise ValueError("No condition bins with >= 2 valid trials found.")

    C_noise = C_noise_sum / n_valid_bins

    return C_noise, n_valid_bins


def fit_rebalanced_dpca(X, trialX, marginalizations, labels, n_components=10,
                        regularizer=0, n_iter=0, marginalize_fn=None,
                        C_noise=None):
    """
    Fit the rebalanced dPCA model.

    Implements:
        A_RR = X̃_f X̃ᵀ (X̃ X̃ᵀ + SQT · C̃_noise + λI)⁻¹

    with reduced-rank approximation via SVD of A_RR @ X̃.

    Parameters
    ----------
    X : ndarray, shape (n_neurons, n_cond1, n_cond2, ..., n_time)
        Trial-averaged firing rates (PSTHs).

    trialX : ndarray, shape (n_max_trials, n_neurons, n_cond1, ...)
        Single-trial data, NaN-padded for unequal trial counts.

    marginalizations : OrderedDict
        Marginalization structure from dPCA._get_parameter_combinations().

    labels : str
        Parameter labels (e.g. 'st' for stimulus and time).

    n_components : int or dict
        Number of components per marginalization.

    regularizer : float
        Regularization strength (lambda). Applied as regularizer * var(X).

    n_iter : int
        Number of iterations for randomized SVD.

    marginalize_fn : callable, optional
        The marginalization function (dPCA._marginalize)

    C_noise : ndarray, shape (n_neurons, n_neurons), optional
        Precomputed rebalanced noise covariance. If None, it is computed
        from trialX. Pass this to avoid recomputing on every call (e.g.
        during cross-validation for regularizer optimization).

    Returns
    -------
    P : dict of ndarray
        Encoding matrices. P[key] has shape (n_neurons, n_components).

    D : dict of ndarray
        Decoding matrices. D[key] has shape (n_neurons, n_components).

    C_noise : ndarray
        The rebalanced noise covariance (for inspection/diagnostics).
    """
    n_neurons = X.shape[0]

    # --- Step 1: Center PSTHs ---
    X_centered = X - np.mean(X.reshape(n_neurons, -1), axis=1).reshape(
        (n_neurons,) + (1,) * (len(X.shape) - 1)
    )
    X_flat = X_centered.reshape(n_neurons, -1)  # (N, SQT)

    # --- Step 2: Compute marginalized PSTHs ---
    if marginalize_fn is not None:
        mXs = marginalize_fn(X_centered)
    else:
        raise ValueError("Must provide marginalize_fn (dPCA._marginalize)")

    # --- Step 3: Compute rebalanced noise covariance (or use precomputed) ---
    if C_noise is None:
        C_noise, n_valid_bins = compute_rebalanced_noise_covariance(trialX)

    # SQT = total number of condition bins (product of all condition dimensions)
    SQT = int(np.prod(X.shape[1:]))

    # --- Step 4: Build the regularized inverse ---
    # (X̃ X̃ᵀ + SQT · C̃_noise + λI)⁻¹
    XXT = X_flat @ X_flat.T  # (N, N)

    lam = regularizer * np.sum(X_centered ** 2) if regularizer > 0 else 0

    M = XXT + SQT * C_noise + lam * np.eye(n_neurons)
    M_inv = np.linalg.inv(M)

    # --- Step 5: For each marginalization, compute A_RR and do SVD ---
    P, D = {}, {}

    for key in marginalizations.keys():
        mX_flat = mXs[key].reshape(n_neurons, -1)  # X̃_f

        # Full-rank solution: A_RR = X̃_f X̃ᵀ M⁻¹
        A_RR = mX_flat @ X_flat.T @ M_inv

        # Reduced-rank: SVD of A_RR @ X̃
        n_comp = n_components[key] if isinstance(n_components, dict) else n_components

        U, s, Vt = randomized_svd(
            A_RR @ X_flat,
            n_components=n_comp,
            n_iter=n_iter,
            random_state=np.random.randint(int(10e5))
        )

        # F = U_q (encoding), D = U_qᵀ A_RR (decoding)
        P[key] = U                        # (N, q)
        D[key] = (U.T @ A_RR).T           # (N, q) — transposed to match dPCA convention

    return P, D, C_noise


# ---------------------------------------------------------------------------
# Drop-in subclass of dPCA
# ---------------------------------------------------------------------------

try:
    from dPCA.dPCA import dPCA

    class RebalancedDPCA(dPCA):
        """
        dPCA with rebalanced noise covariance for unbalanced datasets.

        Identical interface to dPCA, but _fit uses the rebalanced loss function
        when trialX is provided.

        Parameters
        ----------
        Same as dPCA. The key difference is that trialX is *required* for
        fitting (to estimate the noise covariance).

        Attributes
        ----------
        C_noise_ : ndarray, shape (n_neurons, n_neurons)
            The rebalanced noise covariance matrix estimated during fitting.
        """

        def __init__(self, labels=None, join=None, n_components=10,
                     regularizer=None, copy=True, n_iter=0):
            super().__init__(
                labels=labels, join=join, n_components=n_components,
                regularizer=regularizer, copy=copy, n_iter=n_iter
            )
            self.C_noise_ = None
            self._C_noise_cached = None
            self._trialX = None

        def _fit(self, X, trialX=None, mXs=None, center=True, SVD=None,
                 optimize=True):
            """
            Fit with rebalanced noise covariance if trialX is provided,
            otherwise fall back to standard dPCA.

            Note: trialX is cached as self._trialX on first call so that
            internal re-entries (e.g. from _optimize_regularization ->
            crossval_score -> _fit with optimize=False) still have access
            to the single-trial data for noise covariance estimation.

            C_noise is computed once from the full trialX and cached in
            self._C_noise_cached. It does not change between CV folds
            (it depends on the full single-trial data, not the train
            split), so recomputing it on every internal _fit call is
            wasteful.
            """

            # Cache trialX when provided; retrieve cache when not
            if trialX is not None:
                self._trialX = trialX
            else:
                trialX = getattr(self, '_trialX', None)

            if trialX is None:
                print("Warning: trialX not provided. Falling back to standard "
                      "dPCA (no noise covariance rebalancing).")
                return super()._fit(X, trialX=trialX, mXs=mXs, center=center,
                                    SVD=SVD, optimize=optimize)

            # Compute noise covariance once and cache it.
            if self._C_noise_cached is None:
                self._C_noise_cached, _ = compute_rebalanced_noise_covariance(trialX)

            # Optimize regularizer if requested
            if self.opt_regularizer_flag and optimize:
                if self.debug > 0:
                    print("Start optimizing regularization.")
                self._optimize_regularization(X, trialX)

            # Use the rebalanced fitting (pass cached C_noise)
            self.P, self.D, self.C_noise_ = fit_rebalanced_dpca(
                X=X,
                trialX=trialX,
                marginalizations=self.marginalizations,
                labels=self.labels,
                n_components=self.n_components,
                regularizer=self.regularizer,
                n_iter=self.n_iter,
                marginalize_fn=self._marginalize,
                C_noise=self._C_noise_cached,
            )

except ImportError:
    # dPCA package not installed
    pass