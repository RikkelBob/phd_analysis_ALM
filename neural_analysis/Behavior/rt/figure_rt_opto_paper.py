import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
from neural_analysis import utils
from pathlib import Path
import os
from neural_analysis import plot_utils
import numpy as np
import importlib
import config
importlib.reload(plot_utils)
global positions
positions = [1, 2, 3.5, 4.5, 6, 7, 8.5, 9.5]

matplotlib.rcParams['lines.linewidth'] = 0.5        # Line plots
matplotlib.rcParams['axes.linewidth'] = 0.5         # Axes spines
matplotlib.rcParams['xtick.major.width'] = 0.5      # X tick marks
matplotlib.rcParams['ytick.major.width'] = 0.5      # Y tick marks
matplotlib.rcParams['xtick.minor.width'] = 0.5      # Minor ticks
matplotlib.rcParams['ytick.minor.width'] = 0.5
matplotlib.rcParams['grid.linewidth'] = 0.5         # Grid lines
matplotlib.rcParams['patch.linewidth'] = 0.5        # Box/violin outlines

# Set global font properties
matplotlib.rcParams['font.family'] = 'sans-serif'           # Font family
matplotlib.rcParams['font.sans-serif'] = ['Arial']          # Specific font (or 'Helvetica', 'DejaVu Sans', etc.)
matplotlib.rcParams['font.size'] = 6                        # Base font size
matplotlib.rcParams['axes.titlesize'] = 8                   # Title font size
matplotlib.rcParams['axes.labelsize'] = 6                   # Axis label font size
matplotlib.rcParams['xtick.labelsize'] = 6                  # X tick label size
matplotlib.rcParams['ytick.labelsize'] = 6                  # Y tick label size
matplotlib.rcParams['legend.fontsize'] = 6                  # Legend font size
matplotlib.rcParams['figure.titlesize'] = 8                 # Figure title size


def plot_rt(data, condition_labels, color_pattern, plot_type='violin', positions=positions):
    fig, ax = plt.subplots(figsize=(1.35, 1.15), constrained_layout=True)
    ax.spines[['right', 'top']].set_visible(False)

    positions = positions[:len(data)]

    # Iterate over each box and set properties
    if plot_type == 'box':
        box = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, showfliers=False)
        for i in range(len(data)):
            color = color_pattern[i]  # Assign color based on the defined pattern

            # Set face color and outline properties for the boxes
            box['boxes'][i].set_facecolor(color)
            box['boxes'][i].set_linewidth(0)  # Set outline to zero
            box['boxes'][i].set_edgecolor('none')

            # Apply color to whiskers and caps
            box['whiskers'][2 * i].set_color(color)
            box['whiskers'][2 * i + 1].set_color(color)
            box['caps'][2 * i].set_color(color)
            box['caps'][2 * i + 1].set_color(color)

            # Customize medians
            box['medians'][i].set_color('black')
            box['medians'][i].set_linewidth(1)


    elif plot_type == 'violin':
        violin = ax.violinplot(data, positions=positions, widths=0.75,
                               showmeans=True, showmedians=True, showextrema=True,
                               bw_method='scott')

        # Customize violin colors
        for i, pc in enumerate(violin['bodies']):
            pc.set_facecolor(color_pattern[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')

        # Style the median line - make it a short black stripe
        violin['cmedians'].set_edgecolor('black')
        violin['cmedians'].set_linewidth(1)

        # Style the mean line
        violin['cmeans'].set_edgecolor('grey')  # Different color to distinguish from median
        violin['cmeans'].set_linewidth(1)

        # Style the extrema (min/max bars)
        violin['cmaxes'].set_edgecolor('black')  # or any color you want
        violin['cmaxes'].set_linewidth(0.5)
        violin['cmins'].set_edgecolor('black')
        violin['cmins'].set_linewidth(0.5)
        violin['cbars'].set_edgecolor('black')  # the vertical line connecting min to max
        violin['cbars'].set_linewidth(0.5)

    # Add title and labels
    ax.set_title('Reaction times between sensory modalities across 4 groups', fontsize=6)
    ax.set_xlabel('Conditions')
    ax.set_ylabel('Reaction time (ms)')
    ax.set_xticks(positions)
    ax.set_xticklabels(condition_labels, rotation=15, ha='right', fontsize=6)

    # add a grace period line
    ax.axhline(y=100, color='grey', linestyle='--', linewidth=0.5)

    ax.tick_params(axis='y', which='major', labelsize=6)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labelrotation=60)
    ax.tick_params(axis='y', bottom=False, top=False, labelbottom=False, pad=0.5, labelrotation=60)

    ax.xaxis.label.set_size(6)  # X-axis label
    ax.yaxis.label.set_size(6)  # Y-axis label
    ax.title.set_size(8)  # Title (if you have one)

    ax.set_ylim(0, 1050)  # Adjust these values based on your data range

    return fig, ax

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'

path_root = config.DATA_DIR_EXP
experiment = ["ALM_2", "ALM_3"]
path_figures_rt = config.FIGURE_DIR / "behavior/rt"
if not os.path.isdir(path_figures_rt):
    os.mkdir(path_figures_rt)

path_significance_txt = os.path.join(path_figures_rt, "significance_results_opto_graceperiod.txt")
grace_period = 0.1

combined_data_vis = [None, None, None,
                     None, None, None]
combined_labels_vis = ['Control', 'Unilateral', 'Bilateral',
                       'Control', 'Unilateral', 'Bilateral']

combined_data_aud = [None, None, None,
                     None, None, None]
combined_labels_aud = ['Control', 'Unilateral', 'Bilateral',
                       'Control', 'Unilateral', 'Bilateral']

# Load full dataset
trialData_full, sessionData_full, spikeData_full = utils.load_data(path_root, experiment,
                                                    data_types=['trialData', 'sessionData'],
                                                    exclude_grace_violations=True)
per_animal = False

(trialData_full,
 _,
 _,
 sessions_exclude) = utils.filter_sessions_hitrate(trialData_full,
                                                   sessionData=None,
                                                   spikeData=None,
                                                   min_hitrate=0.3)

if per_animal:
    animals = trialData_full['mousename'].unique()
    animal_dir = 'per_animal'
else:
    animals = ['']
    animal_dir = ""

for animal in animals:
    for opto_lat in [['unilat'], ['bilat'], ['unilat', 'bilat']]:
        trialData = trialData_full[trialData_full['optoLat'].isin(opto_lat)]

        if per_animal:
            trialData = trialData[trialData['mousename'] == animal]

        trialData_vis_thr = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 1)
        ]

        trialData_vis_max = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 2)
        ]

        trialData_vis_thr_opto = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 1)
        ]

        trialData_vis_max_opto = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 2)
        ]

        trialData_aud_thr = trialData[
            (trialData['trialType'] == 'Y') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['audioOctChange_norm'] == 1)
        ]

        trialData_aud_max = trialData[
            (trialData['trialType'] == 'Y') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['audioOctChange_norm'] == 2)
        ]

        trialData_aud_thr_opto = trialData[
            (trialData['trialType'] == 'Y') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['audioOctChange_norm'] == 1)
        ]

        trialData_aud_max_opto = trialData[
            (trialData['trialType'] == 'Y') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['audioOctChange_norm'] == 2)
        ]

        ## Visual left
        trialData_vis_thr_left = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 1) &
            (trialData['visualChangeSide'] == 1)
        ]

        trialData_vis_max_left = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 2) &
            (trialData['visualChangeSide'] == 1)
        ]

        trialData_vis_thr_opto_left = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 1) &
            (trialData['visualChangeSide'] == 1)
        ]

        trialData_vis_max_opto_left = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 2) &
            (trialData['visualChangeSide'] == 1)
        ]

        ## Visual right
        trialData_vis_thr_right = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 1) &
            (trialData['visualChangeSide'] == 2)
        ]

        trialData_vis_max_right = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 0) &
            (trialData['visualOriChange_norm'] == 2) &
            (trialData['visualChangeSide'] == 2)
        ]

        trialData_vis_thr_opto_right = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 1) &
            (trialData['visualChangeSide'] == 2)
        ]

        trialData_vis_max_opto_right = trialData[
            (trialData['trialType'] == 'X') &
            (trialData['correctResponse'] == 1) &
            (trialData['hasphotostim'] == 1) &
            (trialData['visualOriChange_norm'] == 2) &
            (trialData['visualChangeSide'] == 2)
        ]
        #trialData_long_rt = utils.filter_df_adv(trialData, reactionTime=('>', 1))

        condition_labels = ['Visual thr',
                            'Visual max',
                            'Visual thr, opto',
                            'Visual max, opto',
                            'Auditory thr',
                            'Auditory max',
                            'Auditory thr, opto',
                            'Auditory max, opto']

        condition_labels_left_right = ['Visual left thr',
                                       'Visual right thr',
                                       'Visual left thr, opto',
                                       'Visual right thr, opto',
                                       'Visual left max',
                                       'Visual right max',
                                       'Visual left max, opto',
                                       'Visual right max, opto']

        # Ensure there are no NaN values and convert from us to ms
        vis_rt_thr = trialData_vis_thr['reactionTime'].dropna() / 1000
        vis_rt_max = trialData_vis_max['reactionTime'].dropna() / 1000

        vis_rt_thr_opto = trialData_vis_thr_opto['reactionTime'].dropna() / 1000
        vis_rt_max_opto = trialData_vis_max_opto['reactionTime'].dropna() / 1000

        aud_rt_thr = trialData_aud_thr['reactionTime'].dropna() / 1000
        aud_rt_max = trialData_aud_max['reactionTime'].dropna() / 1000

        aud_rt_thr_opto = trialData_aud_thr_opto['reactionTime'].dropna() / 1000
        aud_rt_max_opto = trialData_aud_max_opto['reactionTime'].dropna() / 1000

        vis_rt_thr_left = trialData_vis_thr_left['reactionTime'].dropna() / 1000
        vis_rt_max_left = trialData_vis_max_left['reactionTime'].dropna() / 1000

        vis_rt_thr_opto_left = trialData_vis_thr_opto_left['reactionTime'].dropna() / 1000
        vis_rt_max_opto_left = trialData_vis_max_opto_left['reactionTime'].dropna() / 1000

        vis_rt_thr_right = trialData_vis_thr_right['reactionTime'].dropna() / 1000
        vis_rt_max_right = trialData_vis_max_right['reactionTime'].dropna() / 1000

        vis_rt_thr_opto_right = trialData_vis_thr_opto_right['reactionTime'].dropna() / 1000
        vis_rt_max_opto_right = trialData_vis_max_opto_right['reactionTime'].dropna() / 1000

        print(f"\n{'=' * 60}")
        print(f"Animal: {animal if animal else 'ALL'}, opto_lat: {opto_lat}")
        print(f"{'=' * 60}")
        print(f"  Vis thr:        mean={vis_rt_thr.mean():.1f}, median={vis_rt_thr.median():.1f}, n={len(vis_rt_thr)}")
        print(f"  Vis max:        mean={vis_rt_max.mean():.1f}, median={vis_rt_max.median():.1f}, n={len(vis_rt_max)}")
        print(
            f"  Vis thr opto:   mean={vis_rt_thr_opto.mean():.1f}, median={vis_rt_thr_opto.median():.1f}, n={len(vis_rt_thr_opto)}")
        print(
            f"  Vis max opto:   mean={vis_rt_max_opto.mean():.1f}, median={vis_rt_max_opto.median():.1f}, n={len(vis_rt_max_opto)}")
        print(f"  Aud thr:        mean={aud_rt_thr.mean():.1f}, median={aud_rt_thr.median():.1f}, n={len(aud_rt_thr)}")
        print(f"  Aud max:        mean={aud_rt_max.mean():.1f}, median={aud_rt_max.median():.1f}, n={len(aud_rt_max)}")
        print(
            f"  Aud thr opto:   mean={aud_rt_thr_opto.mean():.1f}, median={aud_rt_thr_opto.median():.1f}, n={len(aud_rt_thr_opto)}")
        print(
            f"  Aud max opto:   mean={aud_rt_max_opto.mean():.1f}, median={aud_rt_max_opto.median():.1f}, n={len(aud_rt_max_opto)}")

        # Add data for combined plot
        if opto_lat == ['unilat']:
            combined_data_aud[1] = aud_rt_thr_opto
            combined_data_aud[4] = aud_rt_max_opto

            combined_data_vis[1] = vis_rt_thr_opto
            combined_data_vis[4] = vis_rt_max_opto

        if opto_lat == ['bilat']:
            combined_data_aud[2] = aud_rt_thr_opto
            combined_data_aud[5] = aud_rt_max_opto

            combined_data_vis[2] = vis_rt_thr_opto
            combined_data_vis[5] = vis_rt_max_opto

        if opto_lat == ['unilat', 'bilat']:
            combined_data_aud[0] = aud_rt_thr
            combined_data_aud[3] = aud_rt_max

            combined_data_vis[0] = vis_rt_thr
            combined_data_vis[3] = vis_rt_max

        # Define data to be plotted
        data = [vis_rt_thr, vis_rt_max, vis_rt_thr_opto, vis_rt_max_opto,
                aud_rt_thr, aud_rt_max, aud_rt_thr_opto, aud_rt_max_opto]

        data_vis_left_right = [vis_rt_thr_left, vis_rt_thr_right, vis_rt_thr_opto_left, vis_rt_thr_opto_right,
                               vis_rt_max_left, vis_rt_max_right, vis_rt_max_opto_left, vis_rt_max_opto_right]

        if any(arr.size == 0 for arr in data):
            print(f"Missing data for {animal}, skipping")
            continue

        # Colors for the main opto plot
        blues = plot_utils.get_blues()
        reds = plot_utils.get_reds()
        colors = [blues[2], blues[4], reds[2], reds[4]]

        # Define the exact color pattern for the boxes
        color_pattern = [colors[0], colors[1], colors[0], colors[1], colors[2], colors[3], colors[2], colors[3]]

        fig, ax = plot_rt(data, condition_labels, color_pattern)

        ax.set_xticks([])
        ax.set_position([0.1, 0.1, 0.8, 0.8])

        #x_ticklabels = [
        #    'Thr', 'Max', 'Thr', 'Sup', 'Max',
        #    'Min', 'Sub', 'Thr', 'Sup', 'Max'
        #]
    #
        #ax.set_xticklabels(x_ticklabels, rotation=45, ha='right')

        color_pattern_LR = [blues[2], blues[2], blues[4], blues[4], blues[2], blues[2], blues[4], blues[4]]

        for i, label in enumerate(condition_labels_left_right):
            condition_labels_left_right[i] = condition_labels_left_right[i] + " " + str(len(data_vis_left_right[i]))

        fig_left_right, ax_left_right = plot_rt(data_vis_left_right, condition_labels_left_right, color_pattern_LR)
        ax_left_right.set_position([0.1, 0.1, 0.8, 0.8])

        if not os.path.exists(path_figures_rt):
            os.makedirs(path_figures_rt)

        if per_animal:
            if not os.path.isdir(os.path.join(path_figures_rt, animal_dir, animal)):
                os.makedirs(os.path.join(path_figures_rt, animal_dir, animal))

        fig.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_opto' + ''.join(opto_lat) + '.png'), format='png', dpi=1000)
        fig.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_opto' + ''.join(opto_lat) + '.svg'), format="svg", dpi=1000)

        fig_left_right.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_opto_left_right' + ''.join(opto_lat) + '.png'), format='png', dpi=1000)
        fig_left_right.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_opto_left_right' + ''.join(opto_lat) + '.svg'), format="svg", dpi=1000)


    if any(arr.size == 0 for arr in combined_data_aud):
        print(f"Missing data for {animal}, skipping")
        continue

    if any(arr.size == 0 for arr in combined_data_vis):
        print(f"Missing data for {animal}, skipping")
        continue

    positions_combined = [1, 2, 3, 4.5, 5.5, 6.5]
    color_pattern_vis = [colors[0], colors[0], colors[0],
                         colors[1], colors[1], colors[1]]
    color_pattern_aud = [colors[2], colors[2], colors[2],
                         colors[3], colors[3], colors[3]]

    fig_combined_aud, ax_combined_aud = plot_rt(combined_data_aud, combined_labels_aud, color_pattern_aud,
                                                positions=positions_combined)
    ax_combined_aud.set_position([0.1, 0.1, 0.8, 0.8])

    fig_combined_vis, ax_combined_vis = plot_rt(combined_data_vis, combined_labels_vis, color_pattern_vis,
                                                positions=positions_combined)
    ax_combined_vis.set_position([0.1, 0.1, 0.8, 0.8])

    fig_combined_aud.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_aud_opto_combined' + '.png'), format='png', dpi=1000)
    fig_combined_aud.savefig(Path(path_figures_rt, animal_dir, animal,  'rt_aud_opto_combined' + '.svg'), format="svg", dpi=1000)

    fig_combined_vis.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_vis_opto_combined' + '.png'), format='png', dpi=1000)
    fig_combined_vis.savefig(Path(path_figures_rt, animal_dir, animal, 'rt_vis_opto_combined' + '.svg'), format="svg", dpi=1000)