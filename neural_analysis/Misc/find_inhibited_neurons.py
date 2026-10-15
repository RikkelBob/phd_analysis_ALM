from neural_analysis import utils
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# --- Params --- #
min_fire = 0.5
quality = 'good'
path_root = config.DATA_DIR_EXP
experiment = ["ALM_2", "ALM_3"]
animals = "all"
opto_lat = [['bilat'], ['unilat']]
bins = np.array([-1e6, 0, 1e6])
bin_time = [-0.5, 0.5]

bin_size = 0.050
step_size = 0.010

# Statistical test ('ttest', 'wilcoxon', 'zeta')
use_zeta = False # Whether to determine if a neuron was modulated with zetapy
stat_test = 'wilcoxon'
p_threshold = 0.05

save_psth = {
    'single_neuron': False,
    'grand_average': True
}

plot_combinations = {
    ('lALM', 'unilat'),
    ('rALM', 'unilat'),
    ('lALM', 'bilat'),
    ('rALM', 'bilat'),
}

areas = ['lALM', 'rALM', 'rV1']
output_dir = config.FIGURE_DIR / "inhibited_neurons"

# --- Load data --- #
trialData, sessionData, spikeData = utils.load_data(path_root, experiment, animals)

# Filter data based on behavioral performance
trialData, sessionData, spikeData, removed_sessions = utils.filter_sessions_hitrate(trialData, sessionData, spikeData,
                                                                                    min_hitrate=0.3)
# Only include good units with minimum firing rate
spikeData = spikeData[(spikeData['fr'] > min_fire) & (spikeData['group'] == 'good')]

print(f"Total number of neurons in lALM: {len(spikeData[spikeData['area'] == 'lALM'])}")
print(f"Total number of neurons in rALM: {len(spikeData[spikeData['area'] == 'rALM'])}")
print(f"Total number of neurons in V1: {len(spikeData[spikeData['area'] == 'rV1'])}")

results = {}

def calc_fraction_inhib_neurons(area, opto_cond):

    # --- Filter data --- #
    trialData_on = utils.filter_df(trialData, trialType='P', hasphotostim=True)
    trialData_on = trialData_on[trialData_on['optoLat'].isin(opto_cond)]

    trialData_off = utils.filter_df(trialData, trialType='P', hasphotostim=False)
    trialData_off = trialData_off[trialData_off['optoLat'].isin(opto_cond)]

    opto_sessions = trialData[trialData['optoLat'].isin(opto_cond)]['session_ID'].unique()

    spikeData_cond = spikeData[spikeData['session_ID'].isin(opto_sessions)]
    spikeData_cond = spikeData_cond[spikeData_cond['area'] == area]
    spikeData_cond = spikeData_cond[spikeData_cond['session_ID'].isin(trialData['session_ID'])]

    neuron_ids = list(spikeData_cond['cluster_ID'])

    # --- Calculate firing rate matrix and test for inhibition --- #
    fr_list = []
    fr_list_neuron_ids = []
    for idx, neuron in spikeData_cond.iterrows():
        _, fr_neuron, _, _, neuron_id, time = (
            utils.calculate_psth_modular_optimized_single_neuron(neuron, trialData_on,
                                                                 bins=bins, time=bin_time))
        fr_list.append(fr_neuron)
        fr_list_neuron_ids.append(neuron_id)

    if use_zeta:
        spikeData_zeta = utils.resp_units(spikeData_cond, trialData_on, n_resamples=1000)
        spikeData_unaff = spikeData_zeta[spikeData_zeta['zeta_p'] > p_threshold]
        unaff_neurons = spikeData_unaff['cluster_ID'].values

        mask_aff = ~np.isin(np.array(fr_list_neuron_ids), unaff_neurons)

        fr_list_zeta = [fr for fr, keep in zip(fr_list, mask_aff) if keep]
        fr_list_neuron_ids_zeta = [nid for nid, keep in zip(fr_list_neuron_ids, mask_aff) if keep]
        fr_list = fr_list_zeta
        fr_list_neuron_ids = fr_list_neuron_ids_zeta

        inh_neurons, exc_neurons, _ = utils.find_modulated_neurons(fr_list, neuron_ids=fr_list_neuron_ids,
                                                                               method=stat_test, p_threshold=p_threshold)
    else:
        inh_neurons, exc_neurons, unaff_neurons = utils.find_modulated_neurons(fr_list, neuron_ids=fr_list_neuron_ids,
                                                                               method=stat_test, p_threshold=p_threshold)

    spikeData_inh = spikeData_cond[spikeData_cond['cluster_ID'].isin(inh_neurons)]
    spikeData_exc = spikeData_cond[spikeData_cond['cluster_ID'].isin(exc_neurons)]
    spikeData_unaff = spikeData_cond[spikeData_cond['cluster_ID'].isin(unaff_neurons)]

    n_neurons = len(neuron_ids)
    n_inh_neurons = len(inh_neurons)
    n_exc_neurons = len(exc_neurons)
    n_unaff_neurons = len(unaff_neurons)

    print(area + " ")
    print(*opto_cond, sep=',')
    print(f"Total neurons: {n_neurons} \n",
          f"Inhibited neurons: {n_inh_neurons}, {(n_inh_neurons/n_neurons)*100:.2f}% \n",
          f"Excited neurons: {n_exc_neurons}, {(n_exc_neurons/n_neurons)*100:.2f}% \n",
          f"Unaffected neurons: {n_unaff_neurons}, {(n_unaff_neurons/n_neurons)*100:.2f}%")

    if any(save_psth.values()) and (area, opto_cond[0]) in plot_combinations:
        for modulation, spikeData_i in [('inhibited', spikeData_inh),
                                        ('excited', spikeData_exc),
                                        ('unaffected', spikeData_unaff)]:
            if save_psth['grand_average']:
                utils.generate_grand_average_psth_optimized(spikeData_i, [trialData_on],
                                                            output_dir=os.path.join(output_dir, area, opto_cond[0],
                                                                                    modulation),
                                                            labels=['Laser on'],
                                                            window=[-0.5, 1.5],
                                                            bin_size=bin_size,
                                                            step_size=step_size,
                                                            svg=True,
                                                            use_causal_bins=True,
                                                        log_scale=True)

            if save_psth['single_neuron']:
                utils.generate_psth_master_optimized(spikeData_i, [trialData_on],
                                                     output_dir=os.path.join(output_dir, area, opto_cond[0],
                                                                             modulation),
                                                     labels=['Laser on'],
                                                     window=[-0.5, 1.5],
                                                     bin_size=bin_size,
                                                     step_size=step_size,
                                                     svg=True,
                                                     use_causal_bins=True,
                                                        log_scale=True)
        # Plot all GAs in the same figure
        if save_psth['grand_average']:
            utils.generate_grand_average_psth_optimized([spikeData_inh, spikeData_exc, spikeData_unaff],
                                                        [trialData_on],
                                                        output_dir=os.path.join(output_dir, area, opto_cond[0],
                                                                                'combined_GA'),
                                                        labels=['Laser ON'],
                                                        spike_labels=['Inhibited', 'Excited', 'Unaffected'],
                                                        window=[-0.5, 1.5],
                                                        bin_size=bin_size,
                                                        step_size=step_size,
                                                        svg=True,
                                                        use_causal_bins=True,
                                                        log_scale=True,
                                                        colors=['#ef7e30ff', '#00d80dff', 'grey'])

    # --- Store results --- #
    cond_key = opto_cond[0]
    results[(area, cond_key)] = {
        'n_total': n_neurons,
        'n_inhibited': n_inh_neurons,
        'n_excited': n_exc_neurons,
        'n_unaffected': n_unaff_neurons,
        'inh_neuron_ids': list(inh_neurons),
        'exc_neuron_ids': list(exc_neurons),
        'unaff_neuron_ids': list(unaff_neurons),
    }


for area in areas:
    for cond in opto_lat:
        calc_fraction_inhib_neurons(area, cond)


# --- Helper to combine multiple result keys --- #
def combine_results(keys, label):
    """Sum counts across multiple (area, cond) keys and store under label."""
    combined = {'n_total': 0, 'n_inhibited': 0, 'n_excited': 0, 'n_unaffected': 0}
    for k in keys:
        if k not in results:
            print(f"Warning: {k} not found in results, skipping.")
            continue
        for field in combined:
            combined[field] += results[k][field]

    results[label] = combined

    total = combined['n_total']
    print(f"\n--- {label} ---")
    print(f"Total neurons: {total}")
    print(f"Inhibited: {combined['n_inhibited']}, {(combined['n_inhibited']/total)*100:.2f}%")
    print(f"Excited: {combined['n_excited']}, {(combined['n_excited']/total)*100:.2f}%")
    print(f"Unaffected: {combined['n_unaffected']}, {(combined['n_unaffected']/total)*100:.2f}%")


# --- Combine lALM + rALM per condition --- #
for cond_key in ['bilat', 'unilat']:
    combine_results(
        [('lALM', cond_key), ('rALM', cond_key)],
        ('ALM_combined', cond_key)
    )

# --- Combine ipsilateral conditions --- #
# Laser is on the right side, so ipsilateral = right hemisphere recordings
# lALM bilat:   left ALM recorded, laser hits both sides (so right side is stimulated)  -> ipsi to laser
# rALM unilat:  right ALM recorded, laser on right only                                 -> ipsi to laser
# rALM bilat:   right ALM recorded, laser hits both sides (so right side is stimulated)  -> ipsi to laser
combine_results(
    [('lALM', 'bilat'), ('rALM', 'unilat'), ('rALM', 'bilat')],
    ('ALM_ipsilateral', 'all')
)

# --- Contralateral for completeness (lALM unilat) --- #
combine_results(
    [('lALM', 'unilat')],
    ('ALM_contralateral', 'all')
)

# --- Convert to DataFrame --- #
rows = []
for k, v in results.items():
    rows.append({
        'area': k[0],
        'opto_condition': k[1],
        'n_total': v['n_total'],
        'n_inhibited': v['n_inhibited'],
        'n_excited': v['n_excited'],
        'n_unaffected': v['n_unaffected'],
    })

results_df = pd.DataFrame(rows)

results_df['pct_inhibited'] = (results_df['n_inhibited'] / results_df['n_total'] * 100).round(2)
results_df['pct_excited'] = (results_df['n_excited'] / results_df['n_total'] * 100).round(2)
results_df['pct_unaffected'] = (results_df['n_unaffected'] / results_df['n_total'] * 100).round(2)

print("\n--- Summary Table ---")
print(results_df.to_string(index=False))

# --- Pie charts for all conditions --- #
pie_colors = ['#ef7e30ff', '#00d80dff', 'grey']
pie_labels = ['Inhibited', 'Excited', 'Unaffected']

pie_conditions = [
    ('lALM', 'unilat'),
    ('lALM', 'bilat'),
    ('rALM', 'unilat'),
    ('rALM', 'bilat'),
    ('ALM_combined', 'unilat'),
    ('ALM_combined', 'bilat'),
]



# --- Pie charts for all conditions --- #
pie_colors = ['#ef7e30ff', '#00d80dff', 'grey']
pie_labels = ['Inhibited', 'Excited', 'Unaffected']

pie_conditions = [
    ('lALM', 'unilat'),
    ('lALM', 'bilat'),
    ('rALM', 'unilat'),
    ('rALM', 'bilat'),
    ('ALM_combined', 'unilat'),
    ('ALM_combined', 'bilat'),
]

# 14 mm diameter = 14 mm / 25.4 mm/inch ≈ 0.5512 inches
diameter_inches = 14 / 25.4

for area_key, cond_key in pie_conditions:
    df_cond = results_df[(results_df['area'] == area_key) & (results_df['opto_condition'] == cond_key)]
    if df_cond.empty:
        print(f"No results for {area_key} {cond_key}, skipping pie chart.")
        continue

    values = [
        int(df_cond['n_inhibited'].item()),
        int(df_cond['n_excited'].item()),
        int(df_cond['n_unaffected'].item()),
    ]

    fig, ax = plt.subplots(figsize=(diameter_inches, diameter_inches))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all padding
    ax.pie(values, labels=None, colors=pie_colors)  # Remove labels to fit tight
    ax.set_aspect('equal')

    fname = f"{area_key}_neurons_affected_by_opto_{cond_key}"
    fig.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    fig.savefig(os.path.join(output_dir, f"{fname}.svg"), dpi=300, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# --- Combined lALM + rALM grand average PSTH --- #
if save_psth['grand_average']:
    for cond in opto_lat:
        cond_key = cond[0]

        # Check that both areas have results
        if ('lALM', cond_key) not in results or ('rALM', cond_key) not in results:
            continue

        # Collect neuron IDs from both hemispheres
        inh_ids = results[('lALM', cond_key)]['inh_neuron_ids'] + results[('rALM', cond_key)]['inh_neuron_ids']
        exc_ids = results[('lALM', cond_key)]['exc_neuron_ids'] + results[('rALM', cond_key)]['exc_neuron_ids']
        unaff_ids = results[('lALM', cond_key)]['unaff_neuron_ids'] + results[('rALM', cond_key)]['unaff_neuron_ids']

        # Filter spikeData for neurons in both ALM areas from opto sessions
        opto_sessions = trialData[trialData['optoLat'].isin(cond)]['session_ID'].unique()
        spikeData_alm = spikeData[
            (spikeData['session_ID'].isin(opto_sessions)) &
            (spikeData['area'].isin(['lALM', 'rALM']))
        ]

        spikeData_inh_combined = spikeData_alm[spikeData_alm['cluster_ID'].isin(inh_ids)]
        spikeData_exc_combined = spikeData_alm[spikeData_alm['cluster_ID'].isin(exc_ids)]
        spikeData_unaff_combined = spikeData_alm[spikeData_alm['cluster_ID'].isin(unaff_ids)]

        trialData_on = utils.filter_df(trialData, trialType='P', hasphotostim=True)
        trialData_on = trialData_on[trialData_on['optoLat'].isin(cond)]

        utils.generate_grand_average_psth_optimized(
            [spikeData_inh_combined, spikeData_exc_combined, spikeData_unaff_combined],
            [trialData_on],
            output_dir=os.path.join(output_dir, 'ALM_combined', cond_key, 'combined_GA'),
            labels=['Laser ON'],
            spike_labels=['Inhibited', 'Excited', 'Unaffected'],
            window=[-0.5, 1.5],
            bin_size=bin_size,
            step_size=step_size,
            svg=True,
            use_causal_bins=True,
            log_scale=True,
            colors=['#ef7e30ff', '#00d80dff', 'grey'],
        )

# ============================================================ #
# --- Export all results to a single YAML file --- #
# ============================================================ #
import yaml
import datetime

yaml_output_path = os.path.join(output_dir, "opto_modulation_results.yaml")


def _to_native(obj):
    """Recursively convert numpy / pandas types to plain Python types
    so PyYAML can serialize them cleanly."""
    if isinstance(obj, dict):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.Index)):
        return _to_native(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _to_native(obj.to_dict(orient='records'))
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return str(obj)
    return obj


def _pct(part, total):
    return round((part / total) * 100, 2) if total else 0.0


# --- Metadata block (for reproducibility) --- #
results_export = {
    'metadata': {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'experiment': _to_native(experiment),
        'animals': animals,
        'quality': quality,
        'min_firing_rate': min_fire,
        'statistical_test': 'zeta' if use_zeta else stat_test,
        'use_zeta': use_zeta,
        'p_threshold': p_threshold,
        'bin_time': _to_native(bin_time),
        'bins': _to_native(bins),
        'removed_sessions': _to_native(removed_sessions),
        'neuron_counts_total': {
            'lALM': int(len(spikeData[spikeData['area'] == 'lALM'])),
            'rALM': int(len(spikeData[spikeData['area'] == 'rALM'])),
            'rV1':  int(len(spikeData[spikeData['area'] == 'rV1'])),
        },
    },
    'conditions': [],
}

# --- One entry per (area, condition) key in results --- #
for (area_key, cond_key), v in results.items():
    total = v['n_total']
    entry = {
        'area': area_key,
        'opto_condition': cond_key,
        'n_total': int(total),
        'n_inhibited': int(v['n_inhibited']),
        'n_excited': int(v['n_excited']),
        'n_unaffected': int(v['n_unaffected']),
        'pct_inhibited': _pct(v['n_inhibited'], total),
        'pct_excited': _pct(v['n_excited'], total),
        'pct_unaffected': _pct(v['n_unaffected'], total),
    }
    # Neuron ID lists only exist for the directly-computed conditions,
    # not the combined ones — add them only when present.
    for id_field in ('inh_neuron_ids', 'exc_neuron_ids', 'unaff_neuron_ids'):
        if id_field in v:
            entry[id_field] = _to_native(v[id_field])
    results_export['conditions'].append(entry)

# --- Write file --- #
os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True)
with open(yaml_output_path, 'w') as f:
    yaml.safe_dump(results_export, f, default_flow_style=False, sort_keys=False)

print(f"\nSaved all results to: {yaml_output_path}")