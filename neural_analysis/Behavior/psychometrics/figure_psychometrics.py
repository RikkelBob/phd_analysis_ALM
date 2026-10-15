import pandas as pd
import matplotlib.pyplot as plt
from neural_analysis.utils import load_matlab_struct
from neural_analysis.Misc.psych_id import good_psych_sessions
import numpy as np
from matplotlib.patches import Circle
import matplotlib.lines as mlines
import matplotlib as mpl
import os
import config

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pp_data_dir = config.DATA_DIR_PP
data_file = pp_data_dir / "psych_beh_pp.mat"
output_dir = config.FIGURE_DIR / "behavior"

# Load processed psychometrics data
data = load_matlab_struct(data_file)
df_full = pd.DataFrame(data['output'])
bool_sessions, ses = good_psych_sessions()
df = df_full[df_full['session'].isin(ses)]


def ax_to_fig_coords(ax, ax_coords):
    bbox = ax.get_position()
    x_fig = bbox.x0 + ax_coords[0] * (bbox.x1 - bbox.x0)
    y_fig = bbox.y0 + ax_coords[1] * (bbox.y1 - bbox.y0)
    return (x_fig, y_fig)

def data_to_axes_coords(ax, data_y):
    ylim = ax.get_ylim()
    ax_y = (np.log10(data_y) - np.log10(ylim[0])) / (np.log10(ylim[1]) - np.log10(ylim[0])) if ax.get_yscale() == 'log' else (data_y - ylim[0]) / (ylim[1] - ylim[0])
    return ax_y

def data_to_axes_x(ax, data_x):
    xlim = ax.get_xlim()
    if ax.get_xscale() == 'log':
        return (np.log10(data_x) - np.log10(xlim[0])) / (np.log10(xlim[1]) - np.log10(xlim[0]))
    else:
        return (data_x - xlim[0]) / (xlim[1] - xlim[0])

def aggregate_lists(series):
    arrays = np.array(series.tolist())
    return np.nanmean(arrays, axis=0)


# Aggregate the data
grouped = df.groupby('animal').agg({
    'audio_hitrate': aggregate_lists,
    'visual_hitrate': aggregate_lists,
    'audio_side_error': aggregate_lists,
    'visual_side_error': aggregate_lists,
    'probe_FA_right': aggregate_lists,
    'probe_FA_left': aggregate_lists,
    'probe_CR': aggregate_lists
})

# Initialize plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(55/25.4, 35/25.4), dpi=300)
fig.subplots_adjust(wspace=0.05)

for animal in grouped.index:
    xdata_au_list = df[df['animal'] == animal]['xdata_au']
    xdata_vis_list = df[df['animal'] == animal]['xdata_vis']
    xdata_au = xdata_au_list.iloc[0] if len(xdata_au_list) > 0 else []
    xdata_vis = xdata_vis_list.iloc[0] if len(xdata_vis_list) > 0 else []

    visual_hitrate = grouped.loc[animal, 'visual_hitrate']
    visual_error = grouped.loc[animal, 'visual_side_error']
    auditory_hitrate = grouped.loc[animal, 'audio_hitrate']
    auditory_error = grouped.loc[animal, 'audio_side_error']
    probe_FA_left = grouped.loc[animal, 'probe_FA_left']
    probe_FA_right = grouped.loc[animal, 'probe_FA_right']

    ax1.plot(xdata_vis, visual_hitrate, color='blue', linewidth=1, zorder=3)
    ax1.plot(xdata_vis, visual_error, linestyle='dashed', color='red', linewidth=1)
    ax1.invert_xaxis()

    ax2.plot(xdata_au, auditory_hitrate, color='red', linewidth=1, zorder=3)
    ax2.plot(xdata_au, auditory_error, linestyle='dashed', color='blue', linewidth=1)

# Customize axes
ax1.set_xlabel('Orientation change')
ax1.set_ylabel('Hit Rate / Error Rate')
ax1.set_title('Visual Psychometric Curves')
ax1.set_xscale('log')

ax2.set_xlabel('Octave change')
ax2.get_yaxis().set_visible(False)
ax2.set_title('Auditory Psychometric Curves')
ax2.set_xscale('log')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Force draw to finalize limits and positions
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)

fig.canvas.draw()

mid_x = (ax_to_fig_coords(ax1, (1.0, 0))[0] + ax_to_fig_coords(ax2, (0.0, 0))[0]) / 2

# Add probe dots and connecting lines
for animal in grouped.index:
    xdata_au_list = df[df['animal'] == animal]['xdata_au']
    xdata_vis_list = df[df['animal'] == animal]['xdata_vis']
    xdata_au = xdata_au_list.iloc[0] if len(xdata_au_list) > 0 else []
    xdata_vis = xdata_vis_list.iloc[0] if len(xdata_vis_list) > 0 else []

    visual_hitrate = grouped.loc[animal, 'visual_hitrate']
    visual_error = grouped.loc[animal, 'visual_side_error']
    auditory_hitrate = grouped.loc[animal, 'audio_hitrate']
    auditory_error = grouped.loc[animal, 'audio_side_error']
    probe_FA_right = grouped.loc[animal, 'probe_FA_left']
    probe_FA_left = grouped.loc[animal, 'probe_FA_right']

    # First visual point (rightmost after invert = smallest change = index 0)
    vis_last_x = xdata_vis[0]
    vis_hit_last = visual_hitrate[0]
    vis_err_last = visual_error[0]

    # First auditory point (leftmost = smallest change = index 0)
    au_first_x = xdata_au[0]
    au_hit_first = auditory_hitrate[0]
    au_err_first = auditory_error[0]

    # Convert to figure coords
    # Visual last point (right edge of ax1 after invert)
    vis_hit_fig = ax_to_fig_coords(ax1, (data_to_axes_x(ax1, vis_last_x), data_to_axes_coords(ax1, vis_hit_last)))
    vis_err_fig = ax_to_fig_coords(ax1, (data_to_axes_x(ax1, vis_last_x), data_to_axes_coords(ax1, vis_err_last)))

    # Auditory first point (left edge of ax2)
    au_hit_fig = ax_to_fig_coords(ax2, (data_to_axes_x(ax2, au_first_x), data_to_axes_coords(ax2, au_hit_first)))
    au_err_fig = ax_to_fig_coords(ax2, (data_to_axes_x(ax2, au_first_x), data_to_axes_coords(ax2, au_err_first)))

    # Probe dots
    fa_left_fig_y = ax_to_fig_coords(ax1, (0, data_to_axes_coords(ax1, probe_FA_left)))[1]
    fa_right_fig_y = ax_to_fig_coords(ax2, (0, data_to_axes_coords(ax2, probe_FA_right)))[1]

    # Left lick line (blue): visual hitrate → probe_FA_left → auditory error
    fig.lines.append(mlines.Line2D([vis_hit_fig[0], mid_x], [vis_hit_fig[1], fa_left_fig_y],
                                   transform=fig.transFigure, color='blue', linewidth=1, clip_on=False, zorder=3))
    fig.lines.append(mlines.Line2D([mid_x, au_err_fig[0]], [fa_left_fig_y, au_err_fig[1]],
                                   transform=fig.transFigure, color='blue', linewidth=1, linestyle='dashed', clip_on=False, zorder=2))

    # Right lick line (red): visual error → probe_FA_right → auditory hitrate
    fig.lines.append(mlines.Line2D([vis_err_fig[0], mid_x], [vis_err_fig[1], fa_right_fig_y],
                                   transform=fig.transFigure, color='red', linewidth=1, linestyle='dashed', clip_on=False, zorder=2))
    fig.lines.append(mlines.Line2D([mid_x, au_hit_fig[0]], [fa_right_fig_y, au_hit_fig[1]],
                                   transform=fig.transFigure, color='red', linewidth=1, clip_on=False, zorder=3))

    # Optional: draw probe dots
    plot_probe_dots = True
    if plot_probe_dots:
        circle_left = Circle((mid_x, fa_left_fig_y), 0.007, transform=fig.transFigure, color='blue', clip_on=False, zorder=5)
        fig.patches.append(circle_left)
        circle_right = Circle((mid_x, fa_right_fig_y), 0.007, transform=fig.transFigure, color='red', clip_on=False, zorder=5)
        fig.patches.append(circle_right)

plt.savefig(os.path.join(output_dir, 'psychometric_rr.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'psychometric_rr.svg'), dpi=300)