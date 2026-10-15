import numpy as np
import matplotlib.patches as patches

def get_blues():
    blues = ["#e5e7ffff", "#b3b7ffff", "#8087ffff", "#4c56ffff", "#000effff"]
    return blues

def get_reds():
    reds = ["#ffe5e5ff", "#ffb3b3ff", "#ff8080ff", "#ff0000ff", "#ff0000ff"]
    return reds



def thr_max_rt():
    visual_blue = ["#6699ffff", "#b3ccffff"]
    auditory_red = ["#ff6464ff", "#ffb3b3ff"]

    return visual_blue, auditory_red

def add_sig_lines_ax(ax, bool_arrs, colors=None, linewidth=5, alpha=0.5, timeline=None):

    if colors is None:
        data_lines = ax.get_lines()
        colors = [data_line.get_color() for data_line in data_lines]

    if timeline is not None:
        x_data = timeline
    else:
        line = ax.get_lines()[0]  # reference line for x-axis positions. this tends to fail, use timeline instead!
        x_data = line.get_xdata()

    y_range = np.ptp(ax.get_ylim())
    spacing = y_range * 0.05  # spacing set as 5% of y-axis range
    y_min = ax.get_ylim()[0]

    # First line at y_min, subsequent lines stacked above
    for idx, (bool_arr, color) in enumerate(zip(bool_arrs, colors)):
        clusters = []
        start = None

        # Identify clusters of True values
        for i, flag in enumerate(bool_arr):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                clusters.append((start, i - 1))
                start = None
        if start is not None:
            clusters.append((start, len(bool_arr) - 1))

        # Plot horizontal line for each cluster
        for start_idx, end_idx in clusters:
            x_start = x_data[start_idx]
            x_end = x_data[end_idx]

            ax.hlines(y=(spacing * (idx + 1))+y_min, xmin=x_start, xmax=x_end,
                      color=color, linewidth=linewidth, alpha=alpha)

    return ax

def add_sig_lines_ax_editable(ax, bool_arrs, colors=None, linewidth=5, alpha=0.5, timeline=None):

    if colors is None:
        data_lines = ax.get_lines()
        colors = [data_line.get_color() for data_line in data_lines]

    if timeline is not None:
        x_data = timeline
    else:
        line = ax.get_lines()[0]  # reference line for x-axis positions. this tends to fail, use timeline instead!
        x_data = line.get_xdata()

    y_range = np.ptp(ax.get_ylim())
    spacing = y_range * 0.05  # spacing set as 5% of y-axis range
    y_min = ax.get_ylim()[0]

    # First line at y_min, subsequent lines stacked above
    for idx, (bool_arr, color) in enumerate(zip(bool_arrs, colors)):
        clusters = []
        start = None

        # Identify clusters of True values
        for i, flag in enumerate(bool_arr):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                clusters.append((start, i - 1))
                start = None
        if start is not None:
            clusters.append((start, len(bool_arr) - 1))

        y_val = (spacing * (idx + 1)) + y_min

        # Plot horizontal line for each cluster as an individual Line2D
        for start_idx, end_idx in clusters:
            x_start = x_data[start_idx]
            x_end = x_data[end_idx]

            ax.plot([x_start, x_end], [y_val, y_val],
                    color=color, linewidth=linewidth, alpha=alpha,
                    solid_capstyle='butt', label='_sig_line')

    return ax

def paper_colors(name):
    match name:
        case "probe":
            return "#888888ff"
        case "vis_thr":
            return "#6699ffcc"
        case "vis_max":
            return "#000effff"
        case "aud_thr":
            return "#ff9393ff"
        case "aud_max":
            return "#ff0000ff"
        case "left_ALM":
            return "#fa0effff"
        case "right_ALM":
            return "#1bff46ff"
        case "shuffle":
            return "grey"
        case "opto":
            return "#5dbaffff"
        case "opto_bi":
            return "#ff00eaff"
        case "opto_uni":
            return "#0dff1aff"
        case "vis_side":
            return "#9f00ffff"
        case "vis_ori":
            return "#9f00ffff"
        case "aud_pitch":
            return "#ffac1eff"
        case "left_lick":
            return "#00178aff"
        case "right_lick":
            return "#8b0100ff"
        case "vis_hit":
            return "#000effff"
        case "aud_hit":
            return "#ff0000ff"
        case "aud_miss":
            return "#ff9393ff"
        case "vis_miss":
            return "#6699ffcc"
        case "opto_purple":
            return "#c85dffe5"


    return None


def add_opto_shading(ax, window=None):
    """
    Draws a semi-transparent rectangle spanning the entire height of the plot,
    from x=window[0] to x=window[1].

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the overlay to.
    """

    if window is None:
        window = [0, 1]

    # Get current y-limits
    y_min, y_max = ax.get_ylim()

    # Draw the rectangle spanning entire height
    rect = patches.Rectangle(
        (window[0], y_min),  # Bottom-left corner (x=0, bottom of plot)
        window[1],  # Width (x=1 - x=0)
        y_max - y_min,  # Height (entire y-axis range)
        facecolor='#0083ed',
        alpha=0.2,  # Semi-transparent
        clip_on=False  # Ensures it's always visible
    )
    ax.add_patch(rect)


def add_t0_line(ax):
    ax.axvline(x=0, color='grey', lw=1, ls='--')

    return ax
