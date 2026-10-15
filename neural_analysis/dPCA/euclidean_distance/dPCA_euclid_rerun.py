"""
Loop runner: re-run incremental Euclidean analysis on multiple existing
output directories. Edit OUTPUT_DIRS below to point at the folders you want
to update, then run this script.

Each directory must already contain:
  - metadata.yaml (from the original euclidean run)
  - euclidean_<marg>.npz / opto_permutation_<marg>.npz files

The script reads each existing directory's metadata.yaml to recover the
analysis settings used originally (align_to, n_bootstrap, perm_window,
cluster settings, etc.) so the incremental computation is consistent with
what's already on disk. Then it calls run_euclidean_analysis with output_dir
set, which adds only the missing pairs / opto comparisons.
"""

import os
import sys
import yaml
import traceback

from neural_analysis.dPCA.euclidean_distance.dPCA_calc_euclid import (
    run_euclidean_analysis,
    DEFAULT_DPCA_DIR,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_CI,
    DEFAULT_ALIGN_TO,
    DEFAULT_INCLUDE_OPTO,
    DEFAULT_SUBSAMPLE_TRIALS,
    DEFAULT_OPTO_PERM_WINDOW,
    DEFAULT_CLUSTER_TIME_WINDOW,
    DEFAULT_CLUSTER_THRESHOLD_PCTILE,
    DEFAULT_P_ALPHA,
    DEFAULT_N_JOBS,
)

# ============================================================================
# EDIT THIS LIST
# ============================================================================
OUTPUT_DIRS = [
]

# Optional overrides — leave as None to recover from each dir's metadata.yaml.
# Set to a value here to override what's in metadata.yaml across all dirs.
OVERRIDE_DPCA_DIR = None          # e.g. r"C:\path\to\dpca_run"
OVERRIDE_N_JOBS = None            # e.g. -1, or 8
FORCE_STEMS = []                  # e.g. ['euclidean_s'] to rebuild from scratch
PRUNE_ORPHAN_PAIRS = False        # True drops cached pairs/comps not in current config


def _resolve_setting(meta, *keys, default):
    """Return first key found in meta, else default."""
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


def _to_tuple_window(v, default):
    """Coerce a window-like value (list/tuple/None) into a 2-tuple."""
    if v is None:
        return default
    try:
        return tuple(v)
    except TypeError:
        return default


def _load_settings_from_dir(output_dir):
    """Read settings from the dir's metadata.yaml. Falls back to defaults."""
    meta_path = os.path.join(output_dir, 'metadata.yaml')
    if not os.path.isfile(meta_path):
        print(f"  WARNING: no metadata.yaml in {output_dir}; using script defaults.")
        return {}
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f) or {}
    return meta


def _build_kwargs(output_dir):
    """Construct kwargs for run_euclidean_analysis based on the dir's metadata."""
    meta = _load_settings_from_dir(output_dir)

    # Run folder layout: .../dPCA/{area}/euclidean_distance/{run}  ->  subspace results at .../dPCA/{area}/subspace_results
    derived = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(output_dir))), 'subspace_results')
    if OVERRIDE_DPCA_DIR is not None:
        dpca_dir = OVERRIDE_DPCA_DIR
    elif os.path.isdir(derived):
        dpca_dir = derived
    else:
        dpca_dir = DEFAULT_DPCA_DIR  # None -> run_euclidean_analysis resolves the default

    n_jobs = OVERRIDE_N_JOBS if OVERRIDE_N_JOBS is not None \
        else _resolve_setting(meta, 'n_jobs', default=DEFAULT_N_JOBS)

    perm_window = _to_tuple_window(
        _resolve_setting(meta, 'opto_perm_window', 'perm_window', default=None),
        DEFAULT_OPTO_PERM_WINDOW)

    cluster_window_raw = meta.get('cluster_window', None)
    if cluster_window_raw is None:
        cluster_window = DEFAULT_CLUSTER_TIME_WINDOW
    else:
        # metadata stores it as [start, end] with None entries possible
        try:
            cw_start, cw_end = cluster_window_raw[0], cluster_window_raw[1]
            cluster_window = (cw_start, cw_end)
            if cw_start is None and cw_end is None:
                cluster_window = None
        except (TypeError, IndexError):
            cluster_window = DEFAULT_CLUSTER_TIME_WINDOW

    return dict(
        dpca_dir=dpca_dir,
        align_to=_resolve_setting(meta, 'align_to', default=DEFAULT_ALIGN_TO),
        n_bootstrap=int(_resolve_setting(meta, 'n_bootstrap', default=DEFAULT_N_BOOTSTRAP)),
        ci=_resolve_setting(meta, 'ci', default=DEFAULT_CI),
        include_opto=bool(_resolve_setting(meta, 'include_opto', default=DEFAULT_INCLUDE_OPTO)),
        subsample_trials=bool(_resolve_setting(meta, 'subsample_trials',
                                                default=DEFAULT_SUBSAMPLE_TRIALS)),
        OPTO_PERM_WINDOW=perm_window,
        CLUSTER_TIME_WINDOW=cluster_window,
        CLUSTER_THRESHOLD_PCTILE=float(_resolve_setting(meta, 'cluster_threshold_percentile',
                                                        default=DEFAULT_CLUSTER_THRESHOLD_PCTILE)),
        p_alpha=float(_resolve_setting(meta, 'p_alpha', default=DEFAULT_P_ALPHA)),
        n_jobs=n_jobs,
        output_dir=output_dir,
        force_stems=FORCE_STEMS,
        prune_orphan_pairs=PRUNE_ORPHAN_PAIRS,
    )


def main():
    # Deduplicate while preserving order, in case the list has accidental copies.
    seen = set()
    unique_dirs = []
    for d in OUTPUT_DIRS:
        if d in seen:
            print(f"NOTE: skipping duplicate entry: {d}")
            continue
        seen.add(d)
        unique_dirs.append(d)

    if not unique_dirs:
        print("ERROR: OUTPUT_DIRS is empty. Edit the script and add at least one path.")
        sys.exit(1)

    print(f"Will process {len(unique_dirs)} directory/directories:")
    for d in unique_dirs:
        print(f"  - {d}")
    print()

    failures = []
    for i, output_dir in enumerate(unique_dirs, 1):
        banner = "#" * 80
        print(f"\n{banner}\n# [{i}/{len(unique_dirs)}] {output_dir}\n{banner}")
        if not os.path.isdir(output_dir):
            print(f"  ERROR: directory does not exist; skipping.")
            failures.append((output_dir, "not a directory"))
            continue
        try:
            kwargs = _build_kwargs(output_dir)
            print("  Settings recovered from metadata.yaml (or defaults):")
            for k in ['dpca_dir', 'align_to', 'n_bootstrap', 'ci', 'include_opto',
                      'subsample_trials', 'OPTO_PERM_WINDOW', 'CLUSTER_TIME_WINDOW',
                      'CLUSTER_THRESHOLD_PCTILE', 'p_alpha', 'n_jobs']:
                print(f"    {k} = {kwargs[k]}")
            print()
            run_euclidean_analysis(**kwargs)
        except Exception as e:
            print(f"\n  ERROR processing {output_dir}: {e}")
            traceback.print_exc()
            failures.append((output_dir, str(e)))

    print("\n" + "=" * 80)
    print(f"Finished: {len(unique_dirs) - len(failures)}/{len(unique_dirs)} succeeded.")
    if failures:
        print("Failures:")
        for d, err in failures:
            print(f"  {d}\n    -> {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()