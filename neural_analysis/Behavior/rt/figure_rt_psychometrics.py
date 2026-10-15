import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from neural_analysis import utils, plot_utils
from pathlib import Path
import os
from neural_analysis.Misc.psych_id import good_psych_sessions
import config

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
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.size'] = 6                        # Base font size
matplotlib.rcParams['axes.titlesize'] = 8                   # Title font size
matplotlib.rcParams['axes.labelsize'] = 6                   # Axis label font size
matplotlib.rcParams['xtick.labelsize'] = 6                  # X tick label size
matplotlib.rcParams['ytick.labelsize'] = 6                  # Y tick label size
matplotlib.rcParams['legend.fontsize'] = 6                  # Legend font size
matplotlib.rcParams['figure.titlesize'] = 8                 # Figure title size

min_fire = 2
quality = None  # 'good', 'mua', None
path_root = config.DATA_DIR_PSYCH
path_figures_rt = config.FIGURE_DIR / "behavior/rt"
path_significance_txt = os.path.join(path_figures_rt, "significance_results_psychometrics.txt")
sessions_exclude = ['2063202206291107', '2063202207041053', '2063202207051016', '2054202203251409']
bool_ses, good_sessions = good_psych_sessions()

# Load full dataset
trialData, sessionData, spikeData = utils.load_data(path_root,
                                                    [""],
                                                    data_types=['trialData', 'sessionData'],
                                                    exclude_grace_violations=True)

# Exclude sessions
trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]

# Select only sessions with proper psychometrics
trialData = trialData[trialData['session_ID'].isin(good_sessions)]
sessionData = sessionData[sessionData['session_ID'].isin(good_sessions)]

trialData.reset_index()
sessionData.reset_index()

filter_grace_period = False
if filter_grace_period:
    trialData = utils.filter_df_adv(trialData, reactionTime=('>', 0.1))

trialData_vis_min = utils.filter_df_2(trialData, trialType='X', correctResponse=True,
                                      visualOriChange=[-2, 2])
trialData_vis_sub = utils.filter_df_2(trialData, trialType='X', correctResponse=True,
                                      visualOriChange=[-5, 5])
trialData_vis_thr = utils.filter_df_2(trialData, trialType='X', correctResponse=True,
                                      visualOriChange=[-10, 10])
trialData_vis_sup = utils.filter_df_2(trialData, trialType='X', correctResponse=True,
                                      visualOriChange=[-30, 30])
trialData_vis_max = utils.filter_df_2(trialData, trialType='X', correctResponse=True,
                                      visualOriChange=[-90, 90])

trialData_aud_min = utils.filter_df_2(trialData, trialType='Y', correctResponse=True,
                                      audioOctChange=[-0.0078125, 0.0078125,
                                                      -0.00390625, 0.00390625])
trialData_aud_sub = utils.filter_df_2(trialData, trialType='Y', correctResponse=True,
                                      audioOctChange=[-0.03125, 0.03125,
                                                      -0.015625, 0.015625])
trialData_aud_thr = utils.filter_df_2(trialData, trialType='Y', correctResponse=True,
                                      audioOctChange=[-0.0625, 0.0625])
trialData_aud_sup = utils.filter_df_2(trialData, trialType='Y', correctResponse=True,
                                      audioOctChange=[-0.125, 0.125])
trialData_aud_max = utils.filter_df_2(trialData, trialType='Y', correctResponse=True,
                                      audioOctChange=[-0.5, 0.5])

trialData_long_rt = utils.filter_df_adv(trialData, reactionTime=('>',1))

condition_labels = ['Visual', 'Visual, Opto', 'Auditory', 'Auditory, Opto']

# Ensure there are no NaN values
vis_rt_min = trialData_vis_min['reactionTime'].dropna() * 1000
vis_rt_sub = trialData_vis_sub['reactionTime'].dropna() * 1000
vis_rt_thr = trialData_vis_thr['reactionTime'].dropna() * 1000
vis_rt_sup = trialData_vis_sup['reactionTime'].dropna() * 1000
vis_rt_max = trialData_vis_max['reactionTime'].dropna() * 1000

aud_rt_min = trialData_aud_min['reactionTime'].dropna() * 1000
aud_rt_sub = trialData_aud_sub['reactionTime'].dropna() * 1000
aud_rt_thr = trialData_aud_thr['reactionTime'].dropna() * 1000
aud_rt_sup = trialData_aud_sup['reactionTime'].dropna() * 1000
aud_rt_max = trialData_aud_max['reactionTime'].dropna() * 1000

x_ticks = [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6]

# Combine data and positions, filtering out empty datasets
all_data = [
    (vis_rt_min, x_ticks[0]), (vis_rt_sub, x_ticks[1]), (vis_rt_thr, x_ticks[2]), (vis_rt_sup, x_ticks[3]), (vis_rt_max, x_ticks[4]),
    (aud_rt_min, x_ticks[5]), (aud_rt_sub, x_ticks[6]), (aud_rt_thr, x_ticks[7]), (aud_rt_sup, x_ticks[8]), (aud_rt_max, x_ticks[9])
]

filtered_data = [(d, p) for d, p in all_data if len(d) > 0]
data, positions = zip(*filtered_data)

# Perform Mann-Whitney U tests within and between groups
p_values_within = []
p_values_between = []

# Within group comparisons
within_groups = [
    (vis_rt_min, vis_rt_sub),
    (vis_rt_sub, vis_rt_thr),
    (vis_rt_thr, vis_rt_sup),
    (vis_rt_sup, vis_rt_max),
    (aud_rt_min, aud_rt_sub),
    (aud_rt_sub, aud_rt_thr),
    (aud_rt_thr, aud_rt_sup),
    (aud_rt_sup, aud_rt_max)
]

within_labels = [
    "Visual Min vs. Visual Sub",
    "Visual Sub vs. Visual Threshold",
    "Visual Threshold vs. Visual Super",
    "Visual Super vs. Visual Maximum",
    "Auditory Min vs. Auditory Sub",
    "Auditory Sub vs. Auditory Threshold",
    "Auditory Threshold vs. Auditory Super",
    "Auditory Super vs. Auditory Maximum"
]

for (rt1, rt2) in within_groups:
    if len(rt1) > 0 and len(rt2) > 0:
        _, p_value = mannwhitneyu(rt1, rt2)
        p_values_within.append(p_value)
    else:
        p_values_within.append(None)

# Between group comparisons
between_groups = [
    (vis_rt_min, aud_rt_min),
    (vis_rt_sub, aud_rt_sub),
    (vis_rt_thr, aud_rt_thr),
    (vis_rt_sup, aud_rt_sup),
    (vis_rt_max, aud_rt_max)
]

between_labels = [
    "Visual Min vs. Auditory Min",
    "Visual Sub vs. Auditory Sub",
    "Visual Threshold vs. Auditory Threshold",
    "Visual Super vs. Auditory Super",
    "Visual Maximum vs. Auditory Maximum"
]

for (rt1, rt2) in between_groups:
    if len(rt1) > 0 and len(rt2) > 0:
        _, p_value = mannwhitneyu(rt1, rt2)
        p_values_between.append(p_value)
    else:
        p_values_between.append(None)


# Determine significance levels
def get_significance(p_value):
    if p_value is None:
        return 'N/A'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


significances_within = [get_significance(p) for p in p_values_within]
significances_between = [get_significance(p) for p in p_values_between]

# Creating the boxplot without showing outliers and with custom colors
fig, ax = plt.subplots(figsize=(2.5, 1.15), dpi=300)
ax.spines[['right', 'top']].set_visible(False)
# Colors for the plots
blues = plot_utils.get_blues()
reds = plot_utils.get_reds()

colors = blues + reds

# Apply colors to each component of the plot per condition
plot_type = 'violin'

if plot_type == 'box':
    box = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, showfliers=False)

    for i in range(len(data)):
        if i < 5:  # First 5 are visual conditions
            color = blues[i]
        else:  # Last 5 are auditory conditions
            color = reds[i - 5]

        box['boxes'][i].set_facecolor(color)
        box['whiskers'][2 * i].set_color(color)
        box['whiskers'][2 * i + 1].set_color(color)
        box['caps'][2 * i].set_color(color)
        box['caps'][2 * i + 1].set_color(color)
        box['medians'][i].set_color('black')
        box['medians'][i].set_linewidth(1.5)

elif plot_type == 'violin':
    parts = ax.violinplot(data, positions=positions, widths=0.4, showmeans=False, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        if i < 5:  # First 5 are visual conditions
            color = blues[i]
        else:  # Last 5 are auditory conditions
            color = reds[i - 5]

        pc.set_facecolor(color)
        pc.set_alpha(0.8)
        pc.set_edgecolor('none')
        #pc.set_edgecolor('black')
        pc.set_linewidth(0.5)

    # Color the violin plot components (whiskers, caps, medians)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1 if partname == 'cmedians' else 0.5)


elif plot_type == 'violin':
    violin = ax.violinplot(data, positions=positions, widths=0.4)

# Add title and labels
ax.set_title('Reaction times between sensory modalities across multiple conditions')
ax.set_xlabel('Conditions')
ax.set_ylabel('Reaction time (ms)')

# Define custom x-tick labels

x_ticklabels = [
    'Min', 'Sub', 'Thr', 'Sup', 'Max',
    'Min', 'Sub', 'Thr', 'Sup', 'Max'
]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels, rotation=45, ha='right')

legend_elements = [
    plt.Line2D([0], [0], color=colors[0], lw=2, label='Visual Min'),
    plt.Line2D([0], [0], color=colors[1], lw=2, label='Visual Sub'),
    plt.Line2D([0], [0], color=colors[2], lw=2, label='Visual Thr'),
    plt.Line2D([0], [0], color=colors[3], lw=2, label='Visual Sup'),
    plt.Line2D([0], [0], color=colors[4], lw=2, label='Visual Max'),
    plt.Line2D([0], [0], color=colors[5], lw=2, label='Auditory Min'),
    plt.Line2D([0], [0], color=colors[6], lw=2, label='Auditory Sub'),
    plt.Line2D([0], [0], color=colors[7], lw=2, label='Auditory Thr'),
    plt.Line2D([0], [0], color=colors[8], lw=2, label='Auditory Sup'),
    plt.Line2D([0], [0], color=colors[9], lw=2, label='Auditory Max')
]

# Adjusting y-axis limits
#ax.set_ylim(0, max([rt.max() for rt in data if len(rt) > 0]) + 0.1)  # Adjust these values based on your data range

ax.set_ylim(0, 1050)

# Add line at grace period
ax.axhline(y=100, color='grey', linestyle='--', linewidth=0.5)

ax.tick_params(labelsize=6)  # Tick labels
ax.xaxis.label.set_size(6)  # X-axis label
ax.yaxis.label.set_size(6)  # Y-axis label
ax.title.set_size(8)  # Title (if you have one)
ax.set_position([0.1, 0.1, 0.8, 0.8])


#ax.legend(handles=legend_elements, loc='upper right')

# Save the figure
if not os.path.exists(path_figures_rt):
    os.makedirs(path_figures_rt)

#fig.tight_layout()
plt.savefig(os.path.join(path_figures_rt, 'rt_grouped_psychometrics.png'), format="png", dpi=300)
plt.savefig(os.path.join(path_figures_rt, 'rt_grouped_psychometrics.svg'), format="svg", dpi=300)

# Write significance results to a text file
with open(path_significance_txt, 'w') as f:
    f.write("Within Group Comparisons:\n")
    for label, p_value, significance in zip(within_labels, p_values_within, significances_within):
        if p_value is not None:
            f.write(f"{label}: p-value = {p_value:.4f}, significance = {significance}\n")
        else:
            f.write(f"{label}: p-value = N/A, significance = {significance}\n")
    f.write("\nBetween Group Comparisons:\n")
    for label, p_value, significance in zip(between_labels, p_values_between, significances_between):
        if p_value is not None:
            f.write(f"{label}: p-value = {p_value:.4f}, significance = {significance}\n")
        else:
            f.write(f"{label}: p-value = N/A, significance = {significance}\n")

print('End')
