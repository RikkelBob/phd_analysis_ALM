import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.patches import Circle
import pandas as pd
from neural_analysis import utils
import os
from neural_analysis.Misc.psych_id import good_psych_sessions
import numpy as np
import config

# Font and style settings
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

psych_data_dir = config.DATA_DIR_PSYCH
print(psych_data_dir)
output_dir = config.FIGURE_DIR / "behavior"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
sessions_exclude = ['2063202206291107', '2063202207041053', '2063202207051016', '2054202203251409']
bool_ses, good_sessions = good_psych_sessions()

# Load full dataset
trialData, sessionData, spikeData = utils.load_data(psych_data_dir,
                                                    [""],
                                                    data_types=['trialData', 'sessionData'])

trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]
trialData = trialData[trialData['session_ID'].isin(good_sessions)]
sessionData = sessionData[sessionData['session_ID'].isin(good_sessions)]
trialData.reset_index()
sessionData.reset_index()
trialData = utils.filter_df_adv(trialData, reactionTime=('>', 0.1))
grouped = sessionData.groupby(['session_ID', 'mousename'])

psych_data = {'mouse': [], 'session': [],
              'psychometrics_vis': [], 'psychometrics_aud': [],
              'psychometrics_vis_left': [], 'psychometrics_vis_right': []}

for (session_ID, mousename), group in grouped:
    trialData_mouse = trialData.loc[(trialData['session_ID'] == session_ID)]
    print(session_ID + " has " + str(trialData_mouse.shape[0]) + " trials")

    if trialData_mouse['visualChangeSide'].isin([1, 2]).all():
        trials_vis_left = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, visualOriChange=[-90, 90])
        ]
        trials_vis_right = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, visualOriChange=[-90, 90])
        ]
        hits_vis_left = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, correctResponse=1, visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, correctResponse=1, visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, correctResponse=1, visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, correctResponse=1, visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, correctResponse=1, visualOriChange=[-90, 90])
        ]
        hits_vis_right = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, correctResponse=1, visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, correctResponse=1, visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, correctResponse=1, visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, correctResponse=1, visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, correctResponse=1, visualOriChange=[-90, 90])
        ]
        err_vis_left = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, responseSide='R', visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, responseSide='R', visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, responseSide='R', visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, responseSide='R', visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=1, responseSide='R', visualOriChange=[-90, 90])
        ]
        err_vis_right = [
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, responseSide='R', visualOriChange=[-2, 2]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, responseSide='R', visualOriChange=[-5, 5]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, responseSide='R', visualOriChange=[-10, 10]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, responseSide='R', visualOriChange=[-30, 30]),
            utils.filter_df_2(trialData_mouse, trialType='X', visualChangeSide=2, responseSide='R', visualOriChange=[-90, 90])
        ]

    trials_vis = [
        utils.filter_df_2(trialData_mouse, trialType='X', visualOriChange=[-2, 2]),
        utils.filter_df_2(trialData_mouse, trialType='X', visualOriChange=[-5, 5]),
        utils.filter_df_2(trialData_mouse, trialType='X', visualOriChange=[-10, 10]),
        utils.filter_df_2(trialData_mouse, trialType='X', visualOriChange=[-30, 30]),
        utils.filter_df_2(trialData_mouse, trialType='X', visualOriChange=[-90, 90])
    ]
    trials_aud = [
        utils.filter_df_2(trialData_mouse, trialType='Y', audioOctChange=[-0.0078125, 0.0078125, -0.00390625, 0.00390625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', audioOctChange=[-0.03125, 0.03125, -0.015625, 0.015625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', audioOctChange=[-0.0625, 0.0625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', audioOctChange=[-0.125, 0.125]),
        utils.filter_df_2(trialData_mouse, trialType='Y', audioOctChange=[-0.5, 0.5])
    ]
    hits_vis = [
        utils.filter_df_2(trialData_mouse, trialType='X', correctResponse=1, visualOriChange=[-2, 2]),
        utils.filter_df_2(trialData_mouse, trialType='X', correctResponse=1, visualOriChange=[-5, 5]),
        utils.filter_df_2(trialData_mouse, trialType='X', correctResponse=1, visualOriChange=[-10, 10]),
        utils.filter_df_2(trialData_mouse, trialType='X', correctResponse=1, visualOriChange=[-30, 30]),
        utils.filter_df_2(trialData_mouse, trialType='X', correctResponse=1, visualOriChange=[-90, 90])
    ]
    hits_aud = [
        utils.filter_df_2(trialData_mouse, trialType='Y', correctResponse=1, audioOctChange=[-0.0078125, 0.0078125, -0.00390625, 0.00390625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', correctResponse=1, audioOctChange=[-0.03125, 0.03125, -0.015625, 0.015625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', correctResponse=1, audioOctChange=[-0.0625, 0.0625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', correctResponse=1, audioOctChange=[-0.125, 0.125]),
        utils.filter_df_2(trialData_mouse, trialType='Y', correctResponse=1, audioOctChange=[-0.5, 0.5])
    ]
    err_vis = [
        utils.filter_df_2(trialData_mouse, trialType='X', responseSide='R', visualOriChange=[-2, 2]),
        utils.filter_df_2(trialData_mouse, trialType='X', responseSide='R', visualOriChange=[-5, 5]),
        utils.filter_df_2(trialData_mouse, trialType='X', responseSide='R', visualOriChange=[-10, 10]),
        utils.filter_df_2(trialData_mouse, trialType='X', responseSide='R', visualOriChange=[-30, 30]),
        utils.filter_df_2(trialData_mouse, trialType='X', responseSide='R', visualOriChange=[-90, 90])
    ]
    err_aud = [
        utils.filter_df_2(trialData_mouse, trialType='Y', responseSide='L', audioOctChange=[-0.0078125, 0.0078125, -0.00390625, 0.00390625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', responseSide='L', audioOctChange=[-0.03125, 0.03125, -0.015625, 0.015625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', responseSide='L', audioOctChange=[-0.0625, 0.0625]),
        utils.filter_df_2(trialData_mouse, trialType='Y', responseSide='L', audioOctChange=[-0.125, 0.125]),
        utils.filter_df_2(trialData_mouse, trialType='Y', responseSide='L', audioOctChange=[-0.5, 0.5])
    ]
    probes = [utils.filter_df_2(trialData_mouse, trialType='P')]
    probes_FA = [
        utils.filter_df_2(trialData_mouse, trialType='P', responseSide='L'),
        utils.filter_df_2(trialData_mouse, trialType='P', responseSide='R'),
        utils.filter_df_2(trialData_mouse, trialType='P', correctResponse=0)
    ]

    n_probes = probes[0].shape[0]

    # Hit rates and error rates ordered max→sup→thr→sub→min
    vis_hitrate = np.array([hits_vis[i].shape[0] / trials_vis[i].shape[0] for i in [4, 3, 2, 1, 0]])
    vis_error = np.array([err_vis[i].shape[0] / trials_vis[i].shape[0] for i in [4, 3, 2, 1, 0]])
    aud_hitrate = np.array([hits_aud[i].shape[0] / trials_aud[i].shape[0] for i in [4, 3, 2, 1, 0]])
    aud_error = np.array([err_aud[i].shape[0] / trials_aud[i].shape[0] for i in [4, 3, 2, 1, 0]])

    probe_FA_left = probes_FA[0].shape[0] / n_probes
    probe_FA_right = probes_FA[1].shape[0] / n_probes

    # 11-element arrays: [max,sup,thr,sub,min, probe, min,sub,thr,sup,max]
    # vis line: visual hits (left lick) → probe FA left → auditory errors (left lick)
    # aud line: auditory hits (right lick) → probe FA right → visual errors (right lick)
    psych_data['psychometrics_vis'].append(
        np.concatenate([vis_hitrate, [probe_FA_left], aud_error[::-1]]))
    psych_data['psychometrics_aud'].append(
        np.concatenate([aud_hitrate, [probe_FA_right], vis_error[::-1]]))

    psych_data['mouse'].append(mousename)
    psych_data['session'].append(session_ID)

    if trialData_mouse['visualChangeSide'].isin([1, 2]).all():
        vis_left_hitrate = np.array([hits_vis_left[i].shape[0] / trials_vis_left[i].shape[0] for i in [4, 3, 2, 1, 0]])
        vis_right_hitrate = np.array([hits_vis_right[i].shape[0] / trials_vis_right[i].shape[0] for i in [4, 3, 2, 1, 0]])
        psych_data['psychometrics_vis_left'].append(
            np.concatenate([vis_left_hitrate, [probe_FA_left], aud_error[::-1]]))
        psych_data['psychometrics_vis_right'].append(
            np.concatenate([vis_right_hitrate, [probe_FA_left], aud_error[::-1]]))
    else:
        psych_data['psychometrics_vis_left'].append(np.full(11, np.nan))
        psych_data['psychometrics_vis_right'].append(np.full(11, np.nan))

psych_df = pd.DataFrame(psych_data)


def average_lists(lists):
    return np.nanmean(lists, axis=0)

def sem_lists(lists):
    arr = np.array(lists)
    n = np.sum(~np.isnan(arr), axis=0)
    return np.nanstd(arr, axis=0) / np.sqrt(n)


# ============================================================
# Plotting function
# ============================================================
def plot_psychometric_lr(vis_left_mean, vis_left_sem, vis_right_mean, vis_right_sem,
                         aud_mean, aud_sem, save_name, plot_probe_dots=True):

    xdata = np.arange(5)
    x_labels = ['Min', 'Sub', 'Thr', 'Sup', 'Max']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(55 / 25.4, 35 / 25.4), dpi=300)
    fig.subplots_adjust(wspace=0.05)

    # Visual panel
    ax1.errorbar(xdata, vis_left_mean[:5][::-1], yerr=vis_left_sem[:5][::-1],
                 color='lightblue', linewidth=1.5, capsize=0, capthick=1, zorder=3)
    ax1.errorbar(xdata, vis_right_mean[:5][::-1], yerr=vis_right_sem[:5][::-1],
                 color='blue', linewidth=1.5, capsize=0, capthick=1, zorder=3)
    ax1.errorbar(xdata, aud_mean[6:], yerr=aud_sem[6:],
                 color='red', linewidth=1.5, linestyle='dashed', capsize=0, capthick=1, zorder=2)
    ax1.set_xticks(xdata)
    ax1.set_xticklabels(x_labels)
    ax1.invert_xaxis()
    ax1.set_xlabel('Visual change')
    ax1.set_ylabel('Response rate')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Auditory panel
    ax2.errorbar(xdata, aud_mean[:5][::-1], yerr=aud_sem[:5][::-1],
                 color='red', linewidth=1.5, capsize=0, capthick=1, zorder=3)
    ax2.errorbar(xdata, vis_left_mean[6:], yerr=vis_left_sem[6:],
                 color='lightblue', linewidth=1.5, linestyle='dashed', capsize=0, capthick=1, zorder=2)
    ax2.errorbar(xdata, vis_right_mean[6:], yerr=vis_right_sem[6:],
                 color='blue', linewidth=1.5, linestyle='dashed', capsize=0, capthick=1, zorder=2)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.set_xticks(xdata)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel('Auditory change')
    ax2.get_yaxis().set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.canvas.draw()

    def ax_to_fig(ax, ac):
        bbox = ax.get_position()
        return (bbox.x0 + ac[0] * (bbox.x1 - bbox.x0),
                bbox.y0 + ac[1] * (bbox.y1 - bbox.y0))

    def d2ax_y(ax, y):
        ylim = ax.get_ylim()
        return (y - ylim[0]) / (ylim[1] - ylim[0])

    def d2ax_x(ax, x):
        xlim = ax.get_xlim()
        return (x - xlim[0]) / (xlim[1] - xlim[0])

    mid_x = (ax_to_fig(ax1, (1.0, 0))[0] + ax_to_fig(ax2, (0.0, 0))[0]) / 2

    # Connect each vis line through its probe to the aud panel
    for vis_mean, vis_sem, vis_color in [(vis_left_mean, vis_left_sem, 'lightblue'),
                                         (vis_right_mean, vis_right_sem, 'blue')]:
        probe_vis = vis_mean[5]
        probe_vis_sem = vis_sem[5]
        vis_end = vis_mean[4]       # min saliency
        vis_err_end = vis_mean[6]   # min error on aud side

        vis_hit_fig = ax_to_fig(ax1, (d2ax_x(ax1, 0), d2ax_y(ax1, vis_end)))
        vis_err_fig = ax_to_fig(ax2, (d2ax_x(ax2, 0), d2ax_y(ax2, vis_err_end)))
        probe_fig_y = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_vis)))[1]

        fig.lines.append(mlines.Line2D(
            [vis_hit_fig[0], mid_x], [vis_hit_fig[1], probe_fig_y],
            transform=fig.transFigure, color=vis_color, linewidth=1.5, clip_on=False))
        fig.lines.append(mlines.Line2D(
            [mid_x, vis_err_fig[0]], [probe_fig_y, vis_err_fig[1]],
            transform=fig.transFigure, color=vis_color, linewidth=1.5, linestyle='dashed', clip_on=False))

        if plot_probe_dots:
            fig.patches.append(Circle((mid_x, probe_fig_y), 0.005,
                                      transform=fig.transFigure, color=vis_color, clip_on=False, zorder=5))
            sem_top = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_vis + probe_vis_sem)))[1]
            sem_bot = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_vis - probe_vis_sem)))[1]
            fig.lines.append(mlines.Line2D([mid_x, mid_x], [sem_bot, sem_top],
                                           transform=fig.transFigure, color=vis_color, linewidth=1.5, clip_on=False,
                                           zorder=5))

    # Auditory line through probe
    probe_aud = aud_mean[5]
    aud_end = aud_mean[4]
    aud_err_end = aud_mean[6]

    aud_err_fig = ax_to_fig(ax1, (d2ax_x(ax1, 0), d2ax_y(ax1, aud_err_end)))
    aud_hit_fig = ax_to_fig(ax2, (d2ax_x(ax2, 0), d2ax_y(ax2, aud_end)))
    probe_aud_fig_y = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_aud)))[1]

    fig.lines.append(mlines.Line2D(
        [aud_err_fig[0], mid_x], [aud_err_fig[1], probe_aud_fig_y],
        transform=fig.transFigure, color='red', linewidth=1.5, linestyle='dashed', clip_on=False))
    fig.lines.append(mlines.Line2D(
        [mid_x, aud_hit_fig[0]], [probe_aud_fig_y, aud_hit_fig[1]],
        transform=fig.transFigure, color='red', linewidth=1.5, clip_on=False))

    if plot_probe_dots:
        fig.patches.append(Circle((mid_x, probe_aud_fig_y), 0.005,
                                  transform=fig.transFigure, color='red', clip_on=False, zorder=5))
        probe_aud_sem = aud_sem[5]
        sem_top = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_aud + probe_aud_sem)))[1]
        sem_bot = ax_to_fig(ax1, (0, d2ax_y(ax1, probe_aud - probe_aud_sem)))[1]
        fig.lines.append(mlines.Line2D([mid_x, mid_x], [sem_bot, sem_top],
                                       transform=fig.transFigure, color='red', linewidth=1.5, clip_on=False, zorder=5))

    plt.savefig(os.path.join(output_dir, save_name + '.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, save_name + '.svg'), dpi=300)
def plot_psychometric(vis_mean, vis_sem, aud_mean, aud_sem,
                      save_name, vis_color='blue', aud_color='red',
                      plot_probe_dots=True):
    """
    vis_mean/aud_mean: length-11 arrays
      [0-4]: hit rates (max→min)
      [5]:   probe FA
      [6-10]: error rates on opposite modality (min→max)
    """

    xdata = np.arange(5)
    x_labels_vis = ['max', 'sup', 'thr', 'sub', 'min']
    x_labels_aud = ['min', 'sub', 'thr', 'sup', 'max']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(55 / 25.4, 45 / 25.4), dpi=300)
    fig.subplots_adjust(wspace=0.05)

    # Visual panel (left): vis hits + aud errors
    ax1.errorbar(xdata, vis_mean[:5][::-1], yerr=vis_sem[:5][::-1],
                 color=vis_color, linewidth=1.5, capsize=0, capthick=1, zorder=3)
    ax1.errorbar(xdata, aud_mean[6:], yerr=aud_sem[6:],
                 color=aud_color, linewidth=1.5, linestyle='dashed', capsize=0, capthick=1, zorder=2)
    ax1.set_xticks(xdata)
    ax1.set_xticklabels(['min', 'sub', 'thr', 'sup', 'max'])
    ax1.invert_xaxis()
    ax1.set_xlabel('Visual change')
    ax1.set_ylabel('Response rate')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Auditory panel (right): aud hits + vis errors
    ax2.errorbar(xdata, aud_mean[:5][::-1], yerr=aud_sem[:5][::-1],
                 color=aud_color, linewidth=1.5, capsize=0, capthick=1, zorder=3)
    ax2.errorbar(xdata, vis_mean[6:], yerr=vis_sem[6:],
                 color=vis_color, linewidth=1.5, linestyle='dashed', capsize=0, capthick=1, zorder=2)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.set_xticks(xdata)
    ax2.set_xticklabels(['min', 'sub', 'thr', 'sup', 'max'])
    ax2.set_xlabel('Auditory change')
    ax2.get_yaxis().set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.canvas.draw()

    # --- Connecting lines and probe dots ---
    def ax_to_fig(ax, ax_coords):
        bbox = ax.get_position()
        return (bbox.x0 + ax_coords[0] * (bbox.x1 - bbox.x0),
                bbox.y0 + ax_coords[1] * (bbox.y1 - bbox.y0))

    def data_to_ax_y(ax, y):
        ylim = ax.get_ylim()
        return (y - ylim[0]) / (ylim[1] - ylim[0])

    def data_to_ax_x(ax, x):
        xlim = ax.get_xlim()
        return (x - xlim[0]) / (xlim[1] - xlim[0])

    mid_x = (ax_to_fig(ax1, (1.0, 0))[0] + ax_to_fig(ax2, (0.0, 0))[0]) / 2

    # Probe values
    probe_vis = vis_mean[5]
    probe_aud = aud_mean[5]

    # Endpoints closest to middle: index 4 = min saliency
    # On ax1 (inverted), x=4 is the rightmost point
    # On ax2, x=0 is the leftmost point
    vis_hit_end = vis_mean[4]  # min saliency hit
    aud_err_on_vis = aud_mean[6]  # min aud error on vis panel
    aud_hit_end = aud_mean[4]  # min saliency hit
    vis_err_on_aud = vis_mean[6]  # min vis error on aud panel

    vis_hit_fig = ax_to_fig(ax1, (data_to_ax_x(ax1, 0), data_to_ax_y(ax1, vis_hit_end)))
    aud_err_vis_fig = ax_to_fig(ax1, (data_to_ax_x(ax1, 0), data_to_ax_y(ax1, aud_err_on_vis)))
    aud_hit_fig = ax_to_fig(ax2, (data_to_ax_x(ax2, 0), data_to_ax_y(ax2, aud_hit_end)))
    vis_err_aud_fig = ax_to_fig(ax2, (data_to_ax_x(ax2, 0), data_to_ax_y(ax2, vis_err_on_aud)))

    # Probe y in figure coords (use ax1 for consistent y mapping)
    probe_vis_fig_y = ax_to_fig(ax1, (0, data_to_ax_y(ax1, probe_vis)))[1]
    probe_aud_fig_y = ax_to_fig(ax1, (0, data_to_ax_y(ax1, probe_aud)))[1]

    # Vis color: vis hit → probe_vis → vis error on aud panel
    fig.lines.append(mlines.Line2D(
        [vis_hit_fig[0], mid_x], [vis_hit_fig[1], probe_vis_fig_y],
        transform=fig.transFigure, color=vis_color, linewidth=1.5, clip_on=False))
    fig.lines.append(mlines.Line2D(
        [mid_x, vis_err_aud_fig[0]], [probe_vis_fig_y, vis_err_aud_fig[1]],
        transform=fig.transFigure, color=vis_color, linewidth=1.5, linestyle='dashed', clip_on=False))

    # Aud color: aud error on vis panel → probe_aud → aud hit
    fig.lines.append(mlines.Line2D(
        [aud_err_vis_fig[0], mid_x], [aud_err_vis_fig[1], probe_aud_fig_y],
        transform=fig.transFigure, color=aud_color, linewidth=1.5, linestyle='dashed', clip_on=False))
    fig.lines.append(mlines.Line2D(
        [mid_x, aud_hit_fig[0]], [probe_aud_fig_y, aud_hit_fig[1]],
        transform=fig.transFigure, color=aud_color, linewidth=1.5, clip_on=False))

    if plot_probe_dots:
        fig.patches.append(Circle((mid_x, probe_vis_fig_y), 0.007,
                                  transform=fig.transFigure, color=vis_color, clip_on=False, zorder=5))
        fig.patches.append(Circle((mid_x, probe_aud_fig_y), 0.007,
                                  transform=fig.transFigure, color=aud_color, clip_on=False, zorder=5))

    plt.savefig(os.path.join(output_dir, save_name + '.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, save_name + '.svg'), dpi=300)


# ============================================================
# Grand average
# ============================================================
vis_all = np.array(psych_df['psychometrics_vis'].tolist())
aud_all = np.array(psych_df['psychometrics_aud'].tolist())

plot_psychometric(
    vis_mean=np.nanmean(vis_all, axis=0),
    vis_sem=sem_lists(vis_all),
    aud_mean=np.nanmean(aud_all, axis=0),
    aud_sem=sem_lists(aud_all),
    save_name='psychometric_grandavg'
)

# ============================================================
# Left/right visual split
# ============================================================
vis_left_all = np.array(psych_df['psychometrics_vis_left'].tolist())
vis_right_all = np.array(psych_df['psychometrics_vis_right'].tolist())

plot_psychometric(
    vis_mean=np.nanmean(vis_left_all, axis=0),
    vis_sem=sem_lists(vis_left_all),
    aud_mean=np.nanmean(aud_all, axis=0),
    aud_sem=sem_lists(aud_all),
    save_name='psychometric_vis_left',
    vis_color='lightblue'
)

plot_psychometric(
    vis_mean=np.nanmean(vis_right_all, axis=0),
    vis_sem=sem_lists(vis_right_all),
    aud_mean=np.nanmean(aud_all, axis=0),
    aud_sem=sem_lists(aud_all),
    save_name='psychometric_vis_right',
    vis_color='blue'
)

# ============================================================
# Combined visual (left + right averaged together)
# ============================================================
plot_psychometric(
    vis_mean=np.nanmean(vis_all, axis=0),
    vis_sem=sem_lists(vis_all),
    aud_mean=np.nanmean(aud_all, axis=0),
    aud_sem=sem_lists(aud_all),
    save_name='psychometric_grandavg'
)

vis_left_all = np.array(psych_df['psychometrics_vis_left'].tolist())
vis_right_all = np.array(psych_df['psychometrics_vis_right'].tolist())
aud_all = np.array(psych_df['psychometrics_aud'].tolist())

plot_psychometric_lr(
    vis_left_mean=np.nanmean(vis_left_all, axis=0),
    vis_left_sem=sem_lists(vis_left_all),
    vis_right_mean=np.nanmean(vis_right_all, axis=0),
    vis_right_sem=sem_lists(vis_right_all),
    aud_mean=np.nanmean(aud_all, axis=0),
    aud_sem=sem_lists(aud_all),
    save_name='psychometric_lr_split'
)

print('end')