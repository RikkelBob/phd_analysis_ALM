# ============================================================================
# Resumable sweep driver for dPCA-space baseline deviation analysis.
#
# Loops over component sets x alignment events. Given a parent folder that
# contains previously-created run folders, it:
#   1. scans each subdir's metadata_deviation.yaml,
#   2. matches it to a (align_to, marg_components) combination in the sweep,
#   3. checks the present .npz files to see whether that run is complete,
#   4. reuses the existing folder (incrementally filling missing conditions)
#      or creates a new one for combinations never started.
#
# Per-combination resume (which individual conditions/pairs within a folder
# are already done) is delegated to run_deviation_analysis's own label-based
# incremental logic by passing output_dir=<existing folder>.
# ============================================================================

import os
import argparse
import yaml
import config

from dpca_deviation import (
    run_deviation_analysis,
    get_single_conditions,
    get_paired_conditions,
    DEFAULT_OPTO_LAT,
)


# ----------------------------------------------------------------------------
# Sweep settings
# ----------------------------------------------------------------------------
COMPONENT_SETS = [[0], [1], [2], [0, 1, 2]]
ALIGN_EVENTS   = ['firstLick']
MARG_KEYS = ['d']

# Baseline windows for stimulus- and lick-aligned data
PRE_WINDOW_BY_ALIGN = {
    'stimChange': (-1.0,  0.0),
    'firstLick':  (-2.0, -1.0),
}

def _comp_tag(comps):
    """Build a compact filename-safe tag from a component list."""
    return "comps" + "".join(str(c) for c in comps)


# ----------------------------------------------------------------------------
# Discover existing run folders under the parent dir and key them by
# (align_to, tuple(marg_components)).
# ----------------------------------------------------------------------------
def _scan_existing_runs(parent_dir):
    """Return {(align_to, comps_tuple): folder_path} for every subdir of
    parent_dir that contains a readable metadata_deviation.yaml."""
    found = {}
    if not parent_dir or not os.path.isdir(parent_dir):
        return found

    for name in sorted(os.listdir(parent_dir)):
        subdir = os.path.join(parent_dir, name)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, 'metadata_deviation.yaml')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = yaml.safe_load(f)
        except Exception as e:
            print(f"  WARNING: could not read {meta_path}: {e}")
            continue
        if not isinstance(meta, dict):
            continue

        align_to = meta.get('align_to')
        comps = meta.get('marg_components')
        if align_to is None or comps is None:
            print(f"  WARNING: {meta_path} missing align_to/marg_components, skipping.")
            continue

        key = (align_to, tuple(comps))
        if key in found:
            print(f"  WARNING: duplicate run folder for {key}: "
                  f"{found[key]} and {subdir}. Using the first.")
            continue
        found[key] = subdir
    return found


# ----------------------------------------------------------------------------
# Determine whether a given run folder is already complete: i.e. for each
# marginalization, the expected .npz exists and contains every label defined
# in the current condition definitions.
# ----------------------------------------------------------------------------
def _expected_labels(opto_lat, run_test1, run_test2):
    """Return {('test1'|'test2', marg_key): set(labels)} expected on disk."""
    expected = {}
    if run_test1:
        SC = get_single_conditions(opto_lat)
        for marg_key, conds in SC.items():
            if conds:
                expected[('test1', marg_key)] = {label for _, label in conds}
    if run_test2:
        PC = get_paired_conditions(opto_lat)
        for marg_key, pairs in PC.items():
            if pairs:
                expected[('test2', marg_key)] = {label for _, _, label in pairs}
    return expected


def _labels_in_npz(filepath, test):
    """Read the labels present in an deviation .npz without loading arrays
    into memory more than necessary."""
    import numpy as np
    if not os.path.isfile(filepath):
        return set()
    prefix = 'cond' if test == 'test1' else 'pair'
    labels = set()
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"  WARNING: could not read {filepath}: {e}")
        return set()
    i = 0
    while f"{prefix}{i}_label" in data:
        labels.add(str(data[f"{prefix}{i}_label"]))
        i += 1
    return labels


def _run_is_complete(folder, opto_lat, run_test1, run_test2):
    """True iff every expected (test, marg_key) npz exists and contains all
    expected labels."""
    expected = _expected_labels(opto_lat, run_test1, run_test2)
    for (test, marg_key), labels in expected.items():
        if test == 'test1':
            fp = os.path.join(folder, f'deviation_vs_baseline_{marg_key}.npz')
        else:
            fp = os.path.join(folder, f'deviation_change_{marg_key}.npz')
        present = _labels_in_npz(fp, test)
        if not labels.issubset(present):
            return False
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Script-level defaults (used when run with no CLI arguments)
# ----------------------------------------------------------------------------
PARENT_DIR       = config.DATA_DIR_PP_NEW / "dPCA" / "lALM" / "bsl_deviation"
OPTO_LAT         = DEFAULT_OPTO_LAT
RUN_TEST1        = True
RUN_TEST2        = True
FORCE_INCOMPLETE = False

def main():
    parser = argparse.ArgumentParser(
        description="Resumable sweep over component sets and alignment events "
                    "for dPCA-space deviation analysis.")
    parser.add_argument('--parent-dir', type=str, default=PARENT_DIR,
                        help="Folder containing previously-created run folders. "
                             "Existing folders matching a sweep combination are "
                             "reused and filled incrementally; combinations with "
                             "no folder get a fresh one created. "
                             f"Default: {PARENT_DIR}")
    parser.add_argument('--opto-lat', type=str, default=OPTO_LAT,
                        help="Must match the opto_lat used in the original runs.")
    parser.add_argument('--no-test1', dest='run_test1', action='store_false',
                        default=RUN_TEST1)
    parser.add_argument('--no-test2', dest='run_test2', action='store_false',
                        default=RUN_TEST2)
    parser.add_argument('--force-incomplete', action='store_true',
                        default=FORCE_INCOMPLETE,
                        help="Re-enter folders detected as complete to re-verify.")
    args = parser.parse_args()

    existing = _scan_existing_runs(args.parent_dir)
    if args.parent_dir:
        print(f"Scanned {args.parent_dir}: found {len(existing)} existing run folder(s).")
        for (al, cp), folder in existing.items():
            print(f"  {al:<12s} comps={str(list(cp)):<12s} -> {folder}")

    runs = []
    for align_to in ALIGN_EVENTS:
        for comps in COMPONENT_SETS:
            key = (align_to, tuple(comps))
            tag = _comp_tag(comps)
            existing_folder = existing.get(key)

            print("\n" + "#" * 80)
            print(f"# align_to = {align_to}   |   marg_components = {comps}")
            print("#" * 80)

            if existing_folder is not None:
                complete = _run_is_complete(
                    existing_folder, args.opto_lat,
                    args.run_test1, args.run_test2)
                if complete and not args.force_incomplete:
                    print(f"  COMPLETE — skipping. ({existing_folder})")
                    runs.append({
                        'align_to': align_to,
                        'marg_components': comps,
                        'save_dir': existing_folder,
                        'status': 'skipped_complete',
                    })
                    continue
                else:
                    status = 'resumed' if not complete else 'reverified'
                    print(f"  {'INCOMPLETE — resuming' if not complete else 'Re-verifying'}"
                          f" in existing folder: {existing_folder}")
            else:
                status = 'new'
                print(f"  No existing folder for this combination — creating new.")

            result = run_deviation_analysis(
                align_to=align_to,
                marg_components=comps,
                pre_window=PRE_WINDOW_BY_ALIGN[align_to],
                marg_keys=MARG_KEYS,
                output_tag=tag,
                output_dir=existing_folder,      # None -> fresh dir created
                run_test1=args.run_test1,
                run_test2=args.run_test2,
            )
            runs.append({
                'align_to': align_to,
                'marg_components': comps,
                'save_dir': result['save_dir'],
                'status': status,
            })

    # Summary
    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    for r in runs:
        print(f"  align={r['align_to']:<12s} comps={str(r['marg_components']):<12s} "
              f"[{r['status']:<17s}] -> {r['save_dir']}")


if __name__ == "__main__":
    main()
