import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from neural_analysis import utils
from datetime import datetime
import matplotlib as mpl
import yaml
import argparse
import sys
from scipy.ndimage import label as ndi_label
from joblib import Parallel, delayed
import config

mpl.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 6,
    'figure.titlesize': 10, 'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
DEFAULT_DPCA_AREA = "lALMrALM"
DEFAULT_DPCA_DIR = config.resolve_pp("dPCA", DEFAULT_DPCA_AREA, "subspace_results")
DEFAULT_PATH_ROOT = config.DATA_DIR_EXP
DEFAULT_EXPERIMENT = ["ALM_1", "ALM_2", "ALM_3"]
DEFAULT_ANIMALS = "all"
DEFAULT_SESSIONS_EXCLUDE = ['008202304121458', '006202306141358', '021202402021327',
                            '022202403051220', '026202403191125', '026202403201103']
DEFAULT_OPTO_LAT = 'unilat'
DEFAULT_N_BOOTSTRAP = 2 #10000
DEFAULT_CI = 95
DEFAULT_ALIGN_TO = 'stimChange'
DEFAULT_INCLUDE_OPTO = False
DEFAULT_SUBSAMPLE_TRIALS = False
DEFAULT_OPTO_PERM_WINDOW = (0, 1)
DEFAULT_CLUSTER_TIME_WINDOW = None
DEFAULT_CLUSTER_THRESHOLD_PCTILE = 95.0
DEFAULT_P_ALPHA = 0.01
DEFAULT_N_JOBS = -1
DEFAULT_PARALLEL_BACKEND = 'loky'

# Marginalization components used for projection
MARG_COMPONENTS = {'d': [0, 1, 2], 's': [0, 1, 2], 'sd': [0, 1, 2]}

# ============================================================================
# CONDITION PAIR DEFINITIONS
# ============================================================================
def get_condition_pairs(opto_lat, include_opto=True):
    CONDITION_PAIRS = {
        'd': [
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'Visual L vs R (Control)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Auditory L vs R (Control)'),
            ({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
             'Visual hit vs Auditory hit (Ctrl)'),
            ({'trialType': 'X', 'noResponse': 1, 'visualOriChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'X', 'noResponse': 1, 'visualOriChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (visual misses)'),
            ({'trialType': 'Y', 'noResponse': 1, 'audioOctChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'noResponse': 1, 'audioOctChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (Auditory misses)'),
            ({'trialType': 'X', 'correctResponse': 1, 'visualOriChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'X', 'correctResponse': 1, 'visualOriChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (visual hits)'),
            ({'trialType': 'Y', 'correctResponse': 1, 'audioOctChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'correctResponse': 1, 'audioOctChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (Auditory hits)'),
            ({'trialType': 'X', 'responseSide': 'R', 'visualOriChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': 'R', 'visualOriChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (visual hits)'),
            ({'trialType': 'Y', 'responseSide': 'L', 'audioOctChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'L', 'audioOctChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (Auditory hits)'),
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Visual vs Auditory L'),
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Visual R vs Auditory R'),

            *([
                ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual L vs R (Opto)'),
                ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Auditory L vs R (Opto)'),
                ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
                 {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual Left: Ctrl vs Opto'),
                ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
                 {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Auditory Right: Ctrl vs Opto'),
                ({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                 {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual hit vs Auditory hit (Opto)'),
                ({'responseSide': 'L', 'hasphotostim': False},
                 {'responseSide': 'R', 'hasphotostim': False},
                 'All trials L vs R (Control)'),
                ({'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 {'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'All trials L vs R (Opto)'),
                ({'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': False},
                 {'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 'All left licks: Ctrl vs Opto'),
                ({'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': False},
                 {'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'All right licks: Ctrl vs Opto'),
            ] if include_opto else []),
        ],
        's': [
            # --- Hit vs error within modality ---
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'Visual hit vs error'),
            ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Auditory hit vs error'),

            # --- Visual L vs R in stimulus subspace ---
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'Visual L vs R (Control) [s]'),
            *([
                ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual L vs R (Opto) [s]'),
            ] if include_opto else []),

            # --- Auditory L vs R in stimulus subspace ---
            ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Auditory L vs R (Control) [s]'),
            *([
                ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Auditory L vs R (Opto) [s]'),
            ] if include_opto else []),

            # --- Vis vs Aud separation ---
            ({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
             'Vis vs Aud Hits (Control)'),
            *([({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud Hits (Opto)'),
            ] if include_opto else []),
            ({'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
             'Vis vs Aud Misses (Control)'),
            *([({'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud Misses (Opto)'),
               ({'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
                {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis miss ctrl vs opto)'),
               ({'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
                {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Aud miss ctrl vs opto)'),
            ] if include_opto else []),
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Vis vs Aud Errors (Control)'),
            *([({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud Errors (Opto)'),
            ] if include_opto else []),
            ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'Vis vs Aud Left licks (Control)'),
            *([({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud Left licks (Opto)'),
            ] if include_opto else []),
            ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'Vis vs Aud Right licks (Control)'),
            *([({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud Right licks (Opto)'),
            ] if include_opto else []),

            # --- Individual trajectory shifts: ctrl vs opto ---
            *([
                ({'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
                 {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual Left: Ctrl vs Opto'),
                ({'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
                 {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                 'Auditory Right: Ctrl vs Opto'),
            ] if include_opto else []),

            # --- All visual responded trials: ctrl vs opto ---
            *([
                ({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
                 {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
                 'Visual all responded: Ctrl vs Opto'),
                ({'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
                 {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
                 'Auditory all responded: Ctrl vs Opto'),
            ] if include_opto else []),

            # --- Saliency scaling ---
            ({'trialType': 'X', 'noResponse': 1, 'visualOriChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'X', 'noResponse': 1, 'visualOriChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (visual misses)'),
            ({'trialType': 'Y', 'noResponse': 1, 'audioOctChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'noResponse': 1, 'audioOctChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (Auditory misses)'),
            ({'trialType': 'X', 'correctResponse': 1, 'visualOriChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'X', 'correctResponse': 1, 'visualOriChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (visual hits)'),
            ({'trialType': 'Y', 'correctResponse': 1, 'audioOctChange_norm': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'correctResponse': 1, 'audioOctChange_norm': 2, 'hasphotostim': False},
             'Saliency scaling (Auditory hits)'),

            ({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'Vis vs Aud all responded (Control)'),
            *([({'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
                'Vis vs Aud all responded (Opto)'),
               ] if include_opto else []),
        ],
        'sd': [
            ({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
             {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'Visual Hit vs Error (Control)'),
            *([({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True, 'optoLat': opto_lat},
                'Visual Hit vs Error (Opto)'),
            ] if include_opto else []),
            ({'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
             {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'Auditory Hit vs Error (Control)'),
            *([({'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True, 'optoLat': opto_lat},
                'Auditory Hit vs Error (Opto)'),
               ({'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
                {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Visual Hit: Ctrl vs Opto'),
               ({'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
                {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
                'Auditory Hit: Ctrl vs Opto'),
               ({'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
                {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
                'Visual Error: Ctrl vs Opto'),
               ({'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
                {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
                'Auditory Error: Ctrl vs Opto'),
               ] if include_opto else []),
        ],
    }
    return CONDITION_PAIRS


def get_opto_comparisons(opto_lat):
    OC = {
        's': [
            {'type': 'separation_change',
             'label': 'Opto effect on Vis vs Aud hits separation',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Vis vs Aud separation (all trials)',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'shift_difference',
             'label': 'Opto shift: Visual hits vs Auditory hits (interaction)',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Vis vs Aud misses separation',
             'filter_a_ctrl': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'shift_difference',
             'label': 'Opto shift: Visual misses vs Auditory misses (interaction)',
             'filter_a_ctrl': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_ctrl': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': False},
             'filter_b_opto': {'trialType': 'Y', 'noResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Vis vs Aud left-lick separation',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Vis vs Aud right-lick separation',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'shift_difference',
             'label': 'Opto shift: Visual vs Auditory all responded (interaction)',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': False},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': ['L', 'R'], 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Visual L vs R separation [s]',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change',
             'label': 'Opto effect on Auditory L vs R separation [s]',
             'filter_a_ctrl': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},

            {'type': 'separation_change_difference',
             'label': 'Opto effect on Vis vs Aud separation: Left vs Right licks (interaction)',
             # A = left licks: Vis-L vs Aud-L
             'filter_a1_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a2_ctrl': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a1_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_a2_opto': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             # B = right licks: Vis-R vs Aud-R
             'filter_b1_ctrl': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b2_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b1_opto': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b2_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
        ],
        'd': [
            {'type': 'separation_change',
             'label': 'Opto effect on All trials L vs R separation',
             'filter_a_ctrl': {'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
            {'type': 'separation_change',
             'label': 'Opto effect on hit trials L vs R separation',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True,
                               'optoLat': opto_lat}},
            {'type': 'separation_change_difference',
             'label': 'Opto effect on L vs R: Visual vs Auditory (interaction)',
             'filter_a1_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a2_ctrl': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a1_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_a2_opto': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b1_ctrl': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b2_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b1_opto': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b2_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
            {'type': 'separation_change',
             'label': 'Opto effect on Visual L vs R separation',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'X', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
            {'type': 'separation_change',
             'label': 'Opto effect on Auditory L vs R separation',
             'filter_a_ctrl': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'Y', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
            {'type': 'separation_change',
             'label': 'Opto effect all trials L vs R separation',
             'filter_a_ctrl': {'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': False},
             'filter_a_opto': {'trialType': ['X', 'Y'], 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': ['X', 'Y'], 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}},
            {'type': 'shift_difference',
             'label': 'Opto shift: Visual hits vs Auditory hits (interaction)',
             'filter_a_ctrl': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'responseSide': 'L', 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_ctrl': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': False},
             'filter_b_opto': {'trialType': 'Y', 'responseSide': 'R', 'hasphotostim': True, 'optoLat': opto_lat}}
        ],
        'sd': [
            # 1. Opto effect on visual hit vs error separation
            {'type': 'separation_change',
             'label': 'Opto effect on Visual Hit vs Error separation',
             'filter_a_ctrl': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True,
                               'optoLat': opto_lat}},

            # 2. Opto effect on auditory hit vs error separation
            {'type': 'separation_change',
             'label': 'Opto effect on Auditory Hit vs Error separation',
             'filter_a_ctrl': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
             'filter_b_ctrl': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'filter_a_opto': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b_opto': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True,
                               'optoLat': opto_lat}},

            # 3. Does opto change hit/error separation differently for visual vs auditory? (interaction)
            {'type': 'separation_change_difference',
             'label': 'Opto effect on Hit vs Error separation: Visual vs Auditory (interaction)',
             'filter_a1_ctrl': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': False},
             'filter_a2_ctrl': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'filter_a1_opto': {'trialType': 'X', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_a2_opto': {'trialType': 'X', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True,
                                'optoLat': opto_lat},
             'filter_b1_ctrl': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': False},
             'filter_b2_ctrl': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': False},
             'filter_b1_opto': {'trialType': 'Y', 'correctResponse': 1, 'hasphotostim': True, 'optoLat': opto_lat},
             'filter_b2_opto': {'trialType': 'Y', 'correctResponse': 0, 'noResponse': 0, 'hasphotostim': True,
                                'optoLat': opto_lat}},
        ],
    }
    return OC


# ============================================================================
# CLUSTER-BASED PERMUTATION TEST (Maris & Oostenveld 2007)
# ============================================================================
def _build_cluster_time_mask(time_centers, cluster_window):
    if cluster_window is None: return np.ones(len(time_centers), dtype=bool)
    start, end = cluster_window
    mask = np.ones(len(time_centers), dtype=bool)
    if start is not None: mask &= (time_centers >= start)
    if end is not None: mask &= (time_centers <= end)
    return mask

def _find_clusters_1d(trace_windowed, threshold_windowed):
    above = trace_windowed > threshold_windowed
    if not np.any(above): return []
    labeled, n_clusters = ndi_label(above)
    clusters = []
    for k in range(1, n_clusters + 1):
        mask = labeled == k; idx = np.where(mask)[0]
        mass = float(np.sum(trace_windowed[mask] - threshold_windowed[mask]))
        clusters.append((int(idx[0]), int(idx[-1]), mass))
    return clusters

def cluster_permutation_test(observed, null_distribution, time_centers,
                              cluster_window=None,
                              threshold_percentile=95.0,
                              p_alpha=0.01):
    n_perms, n_timebins = null_distribution.shape
    win_mask = _build_cluster_time_mask(time_centers, cluster_window)
    win_idx = np.where(win_mask)[0]
    threshold_full = np.full(n_timebins, np.nan)
    threshold_windowed = np.nanpercentile(null_distribution[:, win_mask], threshold_percentile, axis=0)
    threshold_full[win_mask] = threshold_windowed
    obs_clusters_win = _find_clusters_1d(observed[win_mask], threshold_windowed)
    max_null_masses = np.zeros(n_perms)
    for p in range(n_perms):
        clusters_p = _find_clusters_1d(null_distribution[p, win_mask], threshold_windowed)
        if clusters_p: max_null_masses[p] = max(c[2] for c in clusters_p)
    observed_clusters, cluster_mask, p_map = [], np.zeros(n_timebins, dtype=bool), np.full(n_timebins, np.nan)
    for (s_win, e_win, mass) in obs_clusters_win:
        p = (np.sum(max_null_masses >= mass) + 1) / (n_perms + 1)
        s_full, e_full = int(win_idx[s_win]), int(win_idx[e_win])
        observed_clusters.append({'start_idx': s_full, 'end_idx': e_full,
            't_start': float(time_centers[s_full]), 't_end': float(time_centers[e_full]),
            'mass': float(mass), 'p_value': float(p)})
        if p < p_alpha: cluster_mask[s_full:e_full+1] = True; p_map[s_full:e_full+1] = p
    return {'observed_clusters': observed_clusters, 'max_null_masses': max_null_masses,
            'threshold_full': threshold_full, 'cluster_mask': cluster_mask, 'p_map': p_map,
            'cluster_window': cluster_window if cluster_window is not None else (None, None),
            'threshold_percentile': float(threshold_percentile)}

def _print_cluster_summary(cluster_result, prefix="    ", p_alpha=0.01):
    clusters = cluster_result['observed_clusters']
    if not clusters: print(f"{prefix}Cluster test: no clusters above threshold."); return
    n_sig = sum(1 for c in clusters if c['p_value'] < p_alpha)
    print(f"{prefix}Cluster test: {len(clusters)} cluster(s), {n_sig} significant (p<{p_alpha})")
    for i, c in enumerate(clusters):
        sig_flag = " *" if c['p_value'] < p_alpha else ""
        print(f"{prefix}  Cluster {i}: t=[{c['t_start']:+.3f}, {c['t_end']:+.3f}]s, "
              f"mass={c['mass']:.3f}, p={c['p_value']:.4f}{sig_flag}")

def _cluster_fields_to_save_dict(cluster_result, prefix):
    out = {}
    out[f"{prefix}_cluster_max_null_masses"] = cluster_result['max_null_masses']
    out[f"{prefix}_cluster_threshold_full"] = cluster_result['threshold_full']
    out[f"{prefix}_cluster_mask"] = cluster_result['cluster_mask']
    out[f"{prefix}_cluster_p_map"] = cluster_result['p_map']
    out[f"{prefix}_cluster_threshold_percentile"] = cluster_result['threshold_percentile']
    cw = cluster_result['cluster_window']
    out[f"{prefix}_cluster_window"] = np.array([np.nan if cw[0] is None else cw[0],
                                                 np.nan if cw[1] is None else cw[1]], dtype=float)
    for field, dtype in [('start_idx',int),('end_idx',int),('t_start',float),
                         ('t_end',float),('mass',float),('p_value',float)]:
        out[f"{prefix}_cluster_{field}"] = np.array(
            [c[field] for c in cluster_result['observed_clusters']], dtype=dtype)
    return out

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _make_filter_key(f):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in f.items()))

def filter_trials(trialData, filter_dict):
    df = trialData.copy()
    for col, val in filter_dict.items():
        if isinstance(val, list): df = df[df[col].isin(val)]
        elif isinstance(val, bool): df = df[df[col] == val]
        else: df = df[df[col] == val]
    return df

def precompute_single_trial_psths(spikeData_proj, trialData_subset, neuron_sessions,
                                   bins, align_to, window, z_score_data, z_score_mode, trialData_full):
    if align_to == 'firstLick':
        trialData_subset = trialData_subset[trialData_subset['responseSide'].isin(['L', 'R'])]
        trialData_subset = trialData_subset[trialData_subset['reactionTime'] > (0.1 * 1e6)]
    n_trials = len(trialData_subset)
    if n_trials == 0: return {}, 0
    session_psths = {}
    for ses in np.unique(neuron_sessions):
        neuron_mask = neuron_sessions == ses; neuron_indices = np.where(neuron_mask)[0]
        if len(neuron_indices) == 0: continue
        trialData_ses = trialData_subset[trialData_subset['session_ID'] == ses]
        if len(trialData_ses) == 0: continue
        spikeData_ses = spikeData_proj.iloc[neuron_indices]
        psth_ses = utils.psth_matrix(spikeData_ses, trialData_ses, bins=bins,
            z_score_data=z_score_data, align_to=align_to, window=window,
            trialData_full=trialData_full, z_score_mode=z_score_mode)
        session_psths[ses] = (psth_ses, neuron_indices)
    return session_psths, n_trials

def ensure_in_cache(psth_cache, filter_dict, spikeData_proj, trialData_full,
                    neuron_sessions, bins, align_to, window, z_score_data, z_score_mode):
    key = _make_filter_key(filter_dict)
    if key not in psth_cache:
        trials = filter_trials(trialData_full, filter_dict)
        session_psths, n_trials = precompute_single_trial_psths(
            spikeData_proj, trials, neuron_sessions, bins, align_to, window,
            z_score_data, z_score_mode, trialData_full)
        psth_cache[key] = (session_psths, n_trials)
        print(f"      Precomputed {n_trials} trials for new condition")
    return psth_cache

# ============================================================================
# VECTORIZED CONDITION AVERAGING
# ============================================================================
def condition_average_from_precomputed(session_psths, n_neurons, n_timebins):
    X_cond = np.full((n_neurons, n_timebins), np.nan)
    for ses, (psth_ses, neuron_indices) in session_psths.items():
        if psth_ses.ndim != 3 or psth_ses.shape[1] == 0:
            continue
        ses_mean = np.nanmean(psth_ses, axis=1)
        X_cond[neuron_indices, :] = ses_mean
    return X_cond

def bootstrap_condition_average_from_precomputed(session_psths, n_neurons, n_timebins, rng):
    X_cond = np.full((n_neurons, n_timebins), np.nan)
    for ses, (psth_ses, neuron_indices) in session_psths.items():
        if psth_ses.ndim != 3:
            continue
        n_trials_ses = psth_ses.shape[1]
        if n_trials_ses == 0:
            continue
        boot_idx = rng.choice(n_trials_ses, size=n_trials_ses, replace=True)
        resampled = psth_ses[:, boot_idx, :]
        ses_mean = np.nanmean(resampled, axis=1)
        X_cond[neuron_indices, :] = ses_mean
    return X_cond

def project_condition_average(X_cond, decoder_matrix, neuron_means, components):
    D_use = decoder_matrix[:, components]
    X_centered = X_cond - neuron_means[:, np.newaxis]
    nan_mask = np.isnan(X_centered[:, 0]); X_centered[nan_mask, :] = 0.0
    return D_use.T @ X_centered

def compute_projected_distance(session_psths_a, session_psths_b, n_neurons, n_timebins,
                               decoder_matrix, neuron_means, components):
    X_a = condition_average_from_precomputed(session_psths_a, n_neurons, n_timebins)
    X_b = condition_average_from_precomputed(session_psths_b, n_neurons, n_timebins)
    proj_a = project_condition_average(X_a, decoder_matrix, neuron_means, components)
    proj_b = project_condition_average(X_b, decoder_matrix, neuron_means, components)
    return np.sqrt(np.sum((proj_a - proj_b) ** 2, axis=0))

# ============================================================================
# PERMUTATION HELPERS
# ============================================================================
def _pool_ctrl_opto_sessions(session_psths_ctrl, session_psths_opto):
    pooled = {}
    for ses in set(session_psths_ctrl.keys()) | set(session_psths_opto.keys()):
        psth_ctrl, idx_ctrl = session_psths_ctrl.get(ses, (np.empty((0,0,0)), np.array([])))
        psth_opto, idx_opto = session_psths_opto.get(ses, (np.empty((0,0,0)), np.array([])))
        neuron_indices = idx_ctrl if len(idx_ctrl) > 0 else idx_opto
        n_ctrl = psth_ctrl.shape[1] if psth_ctrl.ndim == 3 else 0
        n_opto = psth_opto.shape[1] if psth_opto.ndim == 3 else 0
        if n_ctrl + n_opto == 0: continue
        trials = np.concatenate([x for x in [psth_ctrl, psth_opto] if x.ndim == 3 and x.shape[1] > 0], axis=1)
        pooled[ses] = (trials, neuron_indices, n_ctrl)
    return pooled

def _pool_ab_sessions(session_psths_a, session_psths_b):
    pooled = {}
    for ses in set(session_psths_a.keys()) | set(session_psths_b.keys()):
        psth_a, idx_a = session_psths_a.get(ses, (np.empty((0,0,0)), np.array([])))
        psth_b, idx_b = session_psths_b.get(ses, (np.empty((0,0,0)), np.array([])))
        neuron_indices = idx_a if len(idx_a) > 0 else idx_b
        n_a = psth_a.shape[1] if psth_a.ndim == 3 else 0
        n_b = psth_b.shape[1] if psth_b.ndim == 3 else 0
        if n_a + n_b == 0: continue
        trials = np.concatenate([x for x in [psth_a, psth_b] if x.ndim == 3 and x.shape[1] > 0], axis=1)
        pooled[ses] = (trials, neuron_indices, n_a)
    return pooled

def _shuffle_pooled(pooled, rng):
    g1, g2 = {}, {}
    for ses, (trials, neuron_indices, n_first) in pooled.items():
        perm = rng.permutation(trials.shape[1])
        g1[ses] = (trials[:, perm[:n_first], :], neuron_indices)
        g2[ses] = (trials[:, perm[n_first:], :], neuron_indices)
    return g1, g2

def _shuffle_pooled_balanced(pooled, rng):
    g1, g2 = {}, {}
    for ses, (trials, neuron_indices, n_first) in pooled.items():
        n_second = trials.shape[1] - n_first
        if n_first == 0 or n_second == 0:
            g1[ses] = (trials, neuron_indices); g2[ses] = (trials[:,:0,:], neuron_indices); continue
        n_bal = min(n_first, n_second)
        idx_first = rng.choice(n_first, size=n_bal, replace=False)
        idx_second = rng.choice(n_second, size=n_bal, replace=False) + n_first
        balanced = trials[:, np.concatenate([idx_first, idx_second]), :]
        perm = rng.permutation(2 * n_bal)
        g1[ses] = (balanced[:, perm[:n_bal], :], neuron_indices)
        g2[ses] = (balanced[:, perm[n_bal:], :], neuron_indices)
    return g1, g2

# ============================================================================
# PARALLEL WORKER FUNCTIONS (module-level so loky can pickle them)
# ============================================================================
def _boot_worker(seed, spa, spb, n_neurons, n_timebins,
                 decoder_matrix, neuron_means, components):
    rng = np.random.default_rng(seed)
    Xa = bootstrap_condition_average_from_precomputed(spa, n_neurons, n_timebins, rng)
    Xb = bootstrap_condition_average_from_precomputed(spb, n_neurons, n_timebins, rng)
    pa = project_condition_average(Xa, decoder_matrix, neuron_means, components)
    pb = project_condition_average(Xb, decoder_matrix, neuron_means, components)
    return np.sqrt(np.sum((pa - pb) ** 2, axis=0))

def _perm_distance_worker(seed, pooled, n_neurons, n_timebins,
                          decoder_matrix, neuron_means, components, subsample):
    rng = np.random.default_rng(seed)
    sfn = _shuffle_pooled_balanced if subsample else _shuffle_pooled
    sa, sb = sfn(pooled, rng)
    return compute_projected_distance(sa, sb, n_neurons, n_timebins,
                                      decoder_matrix, neuron_means, components)

def _perm_difference_worker(seed, pa, pb, n_neurons, n_timebins,
                            decoder_matrix, neuron_means, components, subsample):
    rng = np.random.default_rng(seed)
    sfn = _shuffle_pooled_balanced if subsample else _shuffle_pooled
    sac, sao = sfn(pa, rng); sbc, sbo = sfn(pb, rng)
    dc = compute_projected_distance(sac, sbc, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components)
    do = compute_projected_distance(sao, sbo, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components)
    return dc - do

def _perm_shift_difference_worker(seed, pa, pb, n_neurons, n_timebins,
                                  decoder_matrix, neuron_means, components, subsample):
    rng = np.random.default_rng(seed)
    sfn = _shuffle_pooled_balanced if subsample else _shuffle_pooled
    sac, sao = sfn(pa, rng); sbc, sbo = sfn(pb, rng)
    da = compute_projected_distance(sac, sao, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components)
    db = compute_projected_distance(sbc, sbo, n_neurons, n_timebins,
                                    decoder_matrix, neuron_means, components)
    return da - db

def _perm_sep_change_diff_worker(seed, pa1, pa2, pb1, pb2, n_neurons, n_timebins,
                                  decoder_matrix, neuron_means, components, subsample):
    rng = np.random.default_rng(seed)
    sfn = _shuffle_pooled_balanced if subsample else _shuffle_pooled
    sa1c, sa1o = sfn(pa1, rng); sa2c, sa2o = sfn(pa2, rng)
    sb1c, sb1o = sfn(pb1, rng); sb2c, sb2o = sfn(pb2, rng)
    dac = compute_projected_distance(sa1c, sa2c, n_neurons, n_timebins, decoder_matrix, neuron_means, components)
    dao = compute_projected_distance(sa1o, sa2o, n_neurons, n_timebins, decoder_matrix, neuron_means, components)
    dbc = compute_projected_distance(sb1c, sb2c, n_neurons, n_timebins, decoder_matrix, neuron_means, components)
    dbo = compute_projected_distance(sb1o, sb2o, n_neurons, n_timebins, decoder_matrix, neuron_means, components)
    return (dac - dao) - (dbc - dbo)

def _spawn_seeds(n, base_seed=42):
    return np.random.SeedSequence(base_seed).spawn(n)

# ============================================================================
# PARALLELIZED PERMUTATION TESTS
# ============================================================================
def permutation_null_distance(session_psths_a, session_psths_b, n_neurons, n_timebins,
                              decoder_matrix, neuron_means, components,
                              n_permutations=1000, base_seed=42, subsample=False,
                              n_jobs=-1, backend='loky'):
    pooled = _pool_ab_sessions(session_psths_a, session_psths_b)
    seeds = _spawn_seeds(n_permutations, base_seed)
    rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(_perm_distance_worker)(s, pooled, n_neurons, n_timebins,
                                        decoder_matrix, neuron_means, components, subsample)
        for s in seeds)
    return np.array(rows)

def permutation_null_difference(sa_ctrl, sb_ctrl, sa_opto, sb_opto, n_neurons, n_timebins,
                                decoder_matrix, neuron_means, components,
                                n_permutations=1000, base_seed=42, subsample=False,
                                n_jobs=-1, backend='loky'):
    pa = _pool_ctrl_opto_sessions(sa_ctrl, sa_opto)
    pb = _pool_ctrl_opto_sessions(sb_ctrl, sb_opto)
    seeds = _spawn_seeds(n_permutations, base_seed)
    rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(_perm_difference_worker)(s, pa, pb, n_neurons, n_timebins,
                                          decoder_matrix, neuron_means, components, subsample)
        for s in seeds)
    return np.array(rows)

def permutation_null_shift_difference(sa_ctrl, sa_opto, sb_ctrl, sb_opto, n_neurons, n_timebins,
                                       decoder_matrix, neuron_means, components,
                                       n_permutations=1000, base_seed=42, subsample=False,
                                       n_jobs=-1, backend='loky'):
    pa = _pool_ctrl_opto_sessions(sa_ctrl, sa_opto)
    pb = _pool_ctrl_opto_sessions(sb_ctrl, sb_opto)
    seeds = _spawn_seeds(n_permutations, base_seed)
    rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(_perm_shift_difference_worker)(s, pa, pb, n_neurons, n_timebins,
                                                decoder_matrix, neuron_means, components, subsample)
        for s in seeds)
    return np.array(rows)

def permutation_null_separation_change_difference(
        a1c, a2c, a1o, a2o, b1c, b2c, b1o, b2o, n_neurons, n_timebins,
        decoder_matrix, neuron_means, components, n_permutations=1000,
        base_seed=42, subsample=False, n_jobs=-1, backend='loky'):
    pa1 = _pool_ctrl_opto_sessions(a1c, a1o); pa2 = _pool_ctrl_opto_sessions(a2c, a2o)
    pb1 = _pool_ctrl_opto_sessions(b1c, b1o); pb2 = _pool_ctrl_opto_sessions(b2c, b2o)
    seeds = _spawn_seeds(n_permutations, base_seed)
    rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(_perm_sep_change_diff_worker)(s, pa1, pa2, pb1, pb2, n_neurons, n_timebins,
                                               decoder_matrix, neuron_means, components, subsample)
        for s in seeds)
    return np.array(rows)

# ============================================================================
# LOADING SAVED RESULTS
# ============================================================================
def _extract_pair_from_npz(data, prefix):
    """Pull a single pair's results out of an open npz file (by prefix)."""
    res = {'label': str(data[f"{prefix}_label"]),
           'distance': data[f"{prefix}_distance"],
           'ci_low': data[f"{prefix}_ci_low"],
           'ci_high': data[f"{prefix}_ci_high"],
           'boot_distribution': data[f"{prefix}_boot_distribution"],
           'n_trials_A': int(data[f"{prefix}_n_trials_A"]),
           'n_trials_B': int(data[f"{prefix}_n_trials_B"])}
    if f"{prefix}_null_distribution" in data:
        res['null_distribution'] = data[f"{prefix}_null_distribution"]
    if f"{prefix}_p_values" in data:
        res['p_values'] = data[f"{prefix}_p_values"]
    if f"{prefix}_observed_window_avg" in data:
        res['observed_window_avg'] = float(data[f"{prefix}_observed_window_avg"])
        res['null_window_avg'] = data[f"{prefix}_null_window_avg"]
        res['p_value_window_avg'] = float(data[f"{prefix}_p_value_window_avg"])
        res['window'] = tuple(data[f"{prefix}_window"])
    if f"{prefix}_skipped" in data:
        res['skipped'] = bool(data[f"{prefix}_skipped"])
    if f"{prefix}_cluster_mask" in data:
        for fld in ['cluster_mask','cluster_p_map','cluster_threshold_full','cluster_max_null_masses',
                    'cluster_start_idx','cluster_end_idx','cluster_t_start','cluster_t_end',
                    'cluster_mass','cluster_p_value']:
            res[fld] = data[f"{prefix}_{fld}"]
        res['cluster_window'] = tuple(data[f"{prefix}_cluster_window"])
        res['cluster_threshold_percentile'] = float(data[f"{prefix}_cluster_threshold_percentile"])
    return res


def load_euclidean_results(filepath):
    """Load euclidean results as a list, preserving original pair order."""
    data = np.load(filepath, allow_pickle=True); results = []; i = 0
    while f"pair{i}_label" in data:
        results.append(_extract_pair_from_npz(data, f"pair{i}"))
        i += 1
    return results


def load_existing_pairs_by_label(filepath):
    """Return dict {label: result_dict} from an existing euclidean file.

    Used for incremental matching: pairs are identified by label string.
    Returns ({}, {}) if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        return {}, {}
    data = np.load(filepath, allow_pickle=True)
    pairs_by_label = {}
    i = 0
    while f"pair{i}_label" in data:
        res = _extract_pair_from_npz(data, f"pair{i}")
        pairs_by_label[res['label']] = res
        i += 1
    file_meta = {}
    for k in ['time_centers', 'components', 'marg_key', 'n_bootstrap', 'ci',
              'perm_window', 'subsample_trials', 'cluster_window_start',
              'cluster_window_end', 'cluster_threshold_percentile']:
        if k in data:
            file_meta[k] = data[k]
    return pairs_by_label, file_meta


def load_existing_opto_comparisons_by_label(filepath):
    """Return dict {(label, type): result_dict} from an existing opto file.

    (label, type) is used because the same label could in principle apply to
    different test types; this matches how the original on-disk format
    serializes both fields per comparison.
    """
    if not os.path.exists(filepath):
        return {}, {}
    data = np.load(filepath, allow_pickle=True)
    comps_by_key = {}
    i = 0
    while f"comp{i}_label" in data:
        prefix = f"comp{i}"
        res = {'label': str(data[f"{prefix}_label"]),
               'type': str(data[f"{prefix}_type"]),
               'observed': data[f"{prefix}_observed"],
               'null_distribution': data[f"{prefix}_null_distribution"],
               'p_values': data[f"{prefix}_p_values"],
               'observed_window_avg': float(data[f"{prefix}_observed_window_avg"]),
               'null_window_avg': data[f"{prefix}_null_window_avg"],
               'p_value_window_avg': float(data[f"{prefix}_p_value_window_avg"]),
               'window': tuple(data[f"{prefix}_window"])}
        if f"{prefix}_cluster_mask" in data:
            for fld in ['cluster_mask','cluster_p_map','cluster_threshold_full',
                        'cluster_max_null_masses','cluster_start_idx','cluster_end_idx',
                        'cluster_t_start','cluster_t_end','cluster_mass','cluster_p_value']:
                res[fld] = data[f"{prefix}_{fld}"]
            res['cluster_window'] = tuple(data[f"{prefix}_cluster_window"])
            res['cluster_threshold_percentile'] = float(data[f"{prefix}_cluster_threshold_percentile"])
        comps_by_key[(res['label'], res['type'])] = res
        i += 1
    file_meta = {}
    for k in ['time_centers', 'cluster_window_start', 'cluster_window_end',
              'cluster_threshold_percentile', 'subsample_trials']:
        if k in data:
            file_meta[k] = data[k]
    return comps_by_key, file_meta


def build_psth_cache(filter_dicts, spikeData_proj, trialData_full, neuron_sessions,
                     bins, align_to, window, z_score_data, z_score_mode):
    psth_cache = {}; unique_keys = {}
    for f in filter_dicts:
        key = _make_filter_key(f)
        if key not in unique_keys: unique_keys[key] = f
    print(f"  Precomputing single-trial PSTHs for {len(unique_keys)} unique conditions...")
    for key, fd in unique_keys.items():
        trials = filter_trials(trialData_full, fd)
        sp, nt = precompute_single_trial_psths(spikeData_proj, trials, neuron_sessions, bins, align_to, window, z_score_data, z_score_mode, trialData_full)
        psth_cache[key] = (sp, nt); print(f"    {nt} trials precomputed")
    print(f"  Precomputation done.\n"); return psth_cache

# ============================================================================
# COMPATIBILITY CHECK
# ============================================================================
def _check_compatibility(file_meta, current_meta, filename):
    """Warn (don't abort) on mismatched settings between cached file and current run."""
    if not file_meta:
        return True
    issues = []
    for key, label in [('n_bootstrap', 'n_bootstrap'),
                       ('ci', 'ci'),
                       ('subsample_trials', 'subsample_trials'),
                       ('cluster_threshold_percentile', 'cluster_threshold_percentile')]:
        if key in file_meta and key in current_meta:
            v_old = file_meta[key]
            v_new = current_meta[key]
            try:
                v_old_s = v_old.item() if hasattr(v_old, 'item') and v_old.ndim == 0 else v_old
            except Exception:
                v_old_s = v_old
            if v_old_s != v_new:
                issues.append(f"    {label}: file={v_old_s} vs current={v_new}")
    if 'perm_window' in file_meta:
        v_old = tuple(file_meta['perm_window'].tolist()) if hasattr(file_meta['perm_window'], 'tolist') else tuple(file_meta['perm_window'])
        v_new = tuple(current_meta.get('perm_window', ()))
        if v_old != v_new:
            issues.append(f"    perm_window: file={v_old} vs current={v_new}")
    if 'cluster_window_start' in file_meta and 'cluster_window' in current_meta:
        cw_new = current_meta['cluster_window']
        f_start = file_meta['cluster_window_start']
        f_end = file_meta.get('cluster_window_end', np.nan)
        try:
            f_start_s = float(f_start) if not (hasattr(f_start, 'item') and np.isnan(f_start)) else None
        except (TypeError, ValueError):
            f_start_s = None
        try:
            f_end_s = float(f_end) if not (hasattr(f_end, 'item') and np.isnan(f_end)) else None
        except (TypeError, ValueError):
            f_end_s = None
        n_start, n_end = (None, None) if cw_new is None else cw_new
        def _eq(a, b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return a == b
        if not (_eq(f_start_s, n_start) and _eq(f_end_s, n_end)):
            issues.append(f"    cluster_window: file=({f_start_s}, {f_end_s}) vs current=({n_start}, {n_end})")
    if 'time_centers' in file_meta and 'time_centers' in current_meta:
        tc_old = np.asarray(file_meta['time_centers'])
        tc_new = np.asarray(current_meta['time_centers'])
        if tc_old.shape != tc_new.shape or not np.allclose(tc_old, tc_new):
            issues.append(f"    time_centers: shape or values differ "
                          f"(file={tc_old.shape}, current={tc_new.shape})")
    if issues:
        print(f"  WARNING: {filename} was generated with different settings:")
        for line in issues:
            print(line)
        print(f"  Merged results may be inconsistent. Consider --force {os.path.splitext(filename)[0]}.")
        return False
    return True

# ============================================================================
# MAIN DISTANCE FUNCTION
# ============================================================================
def dpca_euclidean_distance(spikeData_proj, trialData_full, neuron_sessions,
                            decoder_matrix, neuron_means, condition_pairs, bins, time_centers,
                            align_to='stimChange', window=(-1,1.5), z_score_data=False, z_score_mode=None,
                            components=None, n_bootstrap=1000, ci=95, perm_window=(0,1),
                            cluster_window=None, cluster_threshold_pctile=95.0,
                            subsample=False, p_alpha=0.01, n_jobs=-1, backend='loky',
                            psth_cache=None, pair_seed_offsets=None):
    """
    Compute Euclidean distances for `condition_pairs`.

    Parameters
    ----------
    psth_cache : dict or None
        Pre-built PSTH cache. If None, one is built here.
    pair_seed_offsets : list[int] or None
        Per-pair seed offset (one per pair). Defaults to range(len(condition_pairs)).
        Pass explicit offsets when running incrementally so existing pairs
        could in principle keep stable seeds (not required for correctness).
    """
    n_timebins = len(bins); n_neurons = len(spikeData_proj)
    if components is None: components = list(range(decoder_matrix.shape[1]))
    win_mask = (time_centers >= perm_window[0]) & (time_centers <= perm_window[1])
    if psth_cache is None:
        all_filters = []
        for fa, fb, label in condition_pairs: all_filters.extend([fa, fb])
        psth_cache = build_psth_cache(all_filters, spikeData_proj, trialData_full,
                                       neuron_sessions, bins, align_to, window,
                                       z_score_data, z_score_mode)
    if pair_seed_offsets is None:
        pair_seed_offsets = list(range(len(condition_pairs)))
    results = []
    for pair_idx, ((fa, fb, label), seed_offset) in enumerate(zip(condition_pairs, pair_seed_offsets)):
        print(f"  {label}")
        ka, kb = _make_filter_key(fa), _make_filter_key(fb)
        psth_cache = ensure_in_cache(psth_cache, fa, spikeData_proj, trialData_full,
                                      neuron_sessions, bins, align_to, window,
                                      z_score_data, z_score_mode)
        psth_cache = ensure_in_cache(psth_cache, fb, spikeData_proj, trialData_full,
                                      neuron_sessions, bins, align_to, window,
                                      z_score_data, z_score_mode)
        spa, na = psth_cache[ka]; spb, nb = psth_cache[kb]
        if na < 2 or nb < 2:
            print(f"    WARNING: Not enough trials ({na} vs {nb}), skipping.")
            results.append({'label': label, 'distance': np.full(n_timebins, np.nan),
                'ci_low': np.full(n_timebins, np.nan), 'ci_high': np.full(n_timebins, np.nan),
                'boot_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'null_distribution': np.full((n_bootstrap, n_timebins), np.nan),
                'p_values': np.full(n_timebins, np.nan), 'observed_window_avg': np.nan,
                'null_window_avg': np.full(n_bootstrap, np.nan), 'p_value_window_avg': np.nan,
                'window': perm_window, 'n_trials_A': na, 'n_trials_B': nb, 'skipped': True}); continue

        dist = compute_projected_distance(spa, spb, n_neurons, n_timebins,
                                          decoder_matrix, neuron_means, components)

        boot_base_seed = 42 + seed_offset * 1000
        print(f"    Bootstrapping ({n_bootstrap} iter, n_jobs={n_jobs})...")
        boot_seeds = _spawn_seeds(n_bootstrap, boot_base_seed)
        boot_rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_boot_worker)(s, spa, spb, n_neurons, n_timebins,
                                   decoder_matrix, neuron_means, components)
            for s in boot_seeds)
        boot = np.array(boot_rows)

        alpha = (100-ci)/2
        ci_low = np.nanpercentile(boot, alpha, axis=0); ci_high = np.nanpercentile(boot, 100-alpha, axis=0)
        sub_str = " (balanced null)" if subsample else ""
        print(f"    {na} vs {nb} trials{sub_str}, mean dist = {np.nanmean(dist):.3f}")

        print(f"    Permuting ({n_bootstrap} iter, n_jobs={n_jobs})...")
        null_dist = permutation_null_distance(
            spa, spb, n_neurons, n_timebins, decoder_matrix, neuron_means, components,
            n_permutations=n_bootstrap, base_seed=boot_base_seed + 500,
            subsample=subsample, n_jobs=n_jobs, backend=backend)

        n_perms = null_dist.shape[0]
        pvals = (np.sum(null_dist >= dist[np.newaxis,:], axis=0)+1)/(n_perms+1)
        obs_win = np.nanmean(dist[win_mask]); null_win = np.nanmean(null_dist[:,win_mask], axis=1)
        p_win = (np.sum(null_win >= obs_win)+1)/(n_perms+1)
        cr = cluster_permutation_test(dist, null_dist, time_centers,
                                       cluster_window=cluster_window,
                                       threshold_percentile=cluster_threshold_pctile,
                                       p_alpha=p_alpha)
        print(f"    Sig. timepoints (p<{p_alpha}): {np.sum(pvals < p_alpha)}/{n_timebins}")
        print(f"    Window-avg ({perm_window[0]}-{perm_window[1]}s): obs = {obs_win:.3f}, p = {p_win:.4e}")
        _print_cluster_summary(cr, p_alpha=p_alpha)
        results.append({'label': label, 'distance': dist, 'ci_low': ci_low, 'ci_high': ci_high,
            'boot_distribution': boot, 'null_distribution': null_dist, 'p_values': pvals,
            'observed_window_avg': obs_win, 'null_window_avg': null_win, 'p_value_window_avg': p_win,
            'window': perm_window, 'n_trials_A': na, 'n_trials_B': nb, 'cluster_result': cr})
    return results, psth_cache


def file_needs_computing(filepath, force_stems):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem in force_stems: print(f"  [FORCE] Will recompute: {os.path.basename(filepath)}"); return True
    if not os.path.exists(filepath): print(f"  [MISSING] Will compute: {os.path.basename(filepath)}"); return True
    print(f"  [EXISTS] Skipping: {os.path.basename(filepath)}"); return False


def _build_pair_save_dict(results, time_centers, components, marg_key, n_bootstrap, ci,
                          perm_window, subsample_trials, cluster_window, cluster_threshold_pctile):
    """Serialize pair results into the original on-disk format (no filter metadata)."""
    save_dict = {}
    for i, res in enumerate(results):
        pf = f"pair{i}"
        save_dict[f"{pf}_label"] = res['label']
        for fld in ['distance','ci_low','ci_high','boot_distribution']:
            save_dict[f"{pf}_{fld}"] = res[fld]
        save_dict[f"{pf}_n_trials_A"] = res['n_trials_A']
        save_dict[f"{pf}_n_trials_B"] = res['n_trials_B']
        save_dict[f"{pf}_skipped"] = bool(res.get('skipped', False))
        if 'null_distribution' in res:
            save_dict[f"{pf}_null_distribution"] = res['null_distribution']
        if 'p_values' in res:
            save_dict[f"{pf}_p_values"] = res['p_values']
        if 'observed_window_avg' in res:
            save_dict[f"{pf}_observed_window_avg"] = res['observed_window_avg']
            save_dict[f"{pf}_null_window_avg"] = res['null_window_avg']
            save_dict[f"{pf}_p_value_window_avg"] = res['p_value_window_avg']
            save_dict[f"{pf}_window"] = np.array(res['window'])
        if 'cluster_result' in res and res['cluster_result'] is not None:
            save_dict.update(_cluster_fields_to_save_dict(res['cluster_result'], pf))
        elif 'cluster_mask' in res:
            # round-tripped from disk
            for fld in ['cluster_mask','cluster_p_map','cluster_threshold_full',
                        'cluster_max_null_masses','cluster_start_idx','cluster_end_idx',
                        'cluster_t_start','cluster_t_end','cluster_mass','cluster_p_value']:
                if fld in res:
                    save_dict[f"{pf}_{fld}"] = res[fld]
            if 'cluster_window' in res:
                cw = res['cluster_window']
                save_dict[f"{pf}_cluster_window"] = np.array(
                    [np.nan if cw[0] is None else cw[0],
                     np.nan if cw[1] is None else cw[1]], dtype=float)
            if 'cluster_threshold_percentile' in res:
                save_dict[f"{pf}_cluster_threshold_percentile"] = res['cluster_threshold_percentile']

    save_dict.update({
        'time_centers': time_centers,
        'components': np.array(components),
        'marg_key': marg_key,
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'perm_window': np.array(perm_window),
        'subsample_trials': subsample_trials,
        'cluster_window_start': np.nan if cluster_window is None or cluster_window[0] is None else cluster_window[0],
        'cluster_window_end': np.nan if cluster_window is None or cluster_window[1] is None else cluster_window[1],
        'cluster_threshold_percentile': cluster_threshold_pctile,
    })
    return save_dict


def _build_opto_save_dict(opto_perm_results, time_centers, cluster_window,
                          cluster_threshold_pctile, subsample_trials):
    sd = {'time_centers': time_centers}
    for i, res in enumerate(opto_perm_results):
        pf = f"comp{i}"
        for fld in ['label','type','observed','null_distribution','p_values',
                    'observed_window_avg','null_window_avg','p_value_window_avg']:
            sd[f"{pf}_{fld}"] = res[fld]
        sd[f"{pf}_window"] = np.array(res['window'])
        if 'cluster_result' in res and res['cluster_result'] is not None:
            sd.update(_cluster_fields_to_save_dict(res['cluster_result'], pf))
        elif 'cluster_mask' in res:
            for fld in ['cluster_mask','cluster_p_map','cluster_threshold_full',
                        'cluster_max_null_masses','cluster_start_idx','cluster_end_idx',
                        'cluster_t_start','cluster_t_end','cluster_mass','cluster_p_value']:
                if fld in res: sd[f"{pf}_{fld}"] = res[fld]
            if 'cluster_window' in res:
                cw = res['cluster_window']
                sd[f"{pf}_cluster_window"] = np.array(
                    [np.nan if cw[0] is None else cw[0],
                     np.nan if cw[1] is None else cw[1]], dtype=float)
            if 'cluster_threshold_percentile' in res:
                sd[f"{pf}_cluster_threshold_percentile"] = res['cluster_threshold_percentile']
    sd.update({
        'cluster_window_start': np.nan if cluster_window is None or cluster_window[0] is None else cluster_window[0],
        'cluster_window_end': np.nan if cluster_window is None or cluster_window[1] is None else cluster_window[1],
        'cluster_threshold_percentile': cluster_threshold_pctile,
        'subsample_trials': subsample_trials,
    })
    return sd


def _print_results_summary(results, time_centers):
    print(f"\n  {'Label':<45s} {'Trials':>12s} {'Mean':>8s} {'Pre':>8s} {'Post':>8s} {'Peak':>8s} {'Peak t':>7s}")
    print(f"  {'-'*45} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    pre_mask = time_centers < 0; post_mask = time_centers > 0
    for res in results:
        d = res['distance']; na, nb = res['n_trials_A'], res['n_trials_B']
        if res.get('skipped', False) or np.all(np.isnan(d)):
            print(f"  {res['label']:<45s} {f'{na} vs {nb}':>12s} {'[skipped]'}"); continue
        print(f"  {res['label']:<45s} {f'{na} vs {nb}':>12s} {np.nanmean(d):>8.3f} "
              f"{np.nanmean(d[pre_mask]):>8.3f} {np.nanmean(d[post_mask]):>8.3f} "
              f"{np.nanmax(d):>8.3f} {time_centers[np.nanargmax(d)]:>7.3f}")
    print()


# ============================================================================
# PUBLIC ENTRY POINT
# ----------------------------------------------------------------------------
# Incremental behavior
# --------------------
# When `output_dir` points to an existing folder, the script:
#   - Loads each existing euclidean_<marg>.npz / opto_permutation_<marg>.npz.
#   - Identifies which pairs / opto comparisons in the current config are
#     already present, MATCHED BY LABEL (pair) or (LABEL, TYPE) (opto).
#   - Computes only the missing ones.
#   - Re-saves the file with merged results in the SAME on-disk schema as
#     the original script (no extra filter metadata is written).
#
# IMPORTANT: matching is purely by label string. Renaming a label in
# get_condition_pairs() will be treated as adding a new pair (and the
# old one becomes an orphan). If you change a pair's filters but keep the
# label the same, the cached (now stale) version is reused — use --force
# to rebuild from scratch.
# ============================================================================
def run_euclidean_analysis(
        dpca_dir=DEFAULT_DPCA_DIR,
        path_root=DEFAULT_PATH_ROOT,
        experiment=DEFAULT_EXPERIMENT,
        animals=DEFAULT_ANIMALS,
        sessions_exclude=DEFAULT_SESSIONS_EXCLUDE,
        opto_lat=DEFAULT_OPTO_LAT,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
        ci=DEFAULT_CI,
        align_to=DEFAULT_ALIGN_TO,
        include_opto=DEFAULT_INCLUDE_OPTO,
        subsample_trials=DEFAULT_SUBSAMPLE_TRIALS,
        OPTO_PERM_WINDOW=DEFAULT_OPTO_PERM_WINDOW,
        CLUSTER_TIME_WINDOW=DEFAULT_CLUSTER_TIME_WINDOW,
        CLUSTER_THRESHOLD_PCTILE=DEFAULT_CLUSTER_THRESHOLD_PCTILE,
        p_alpha=DEFAULT_P_ALPHA,
        n_jobs=DEFAULT_N_JOBS,
        parallel_backend=DEFAULT_PARALLEL_BACKEND,
        output_dir=None,
        force_stems=None,
        output_tag=None,
        prune_orphan_pairs=False):
    """Run the dPCA-space Euclidean distance analysis (with incremental support)."""
    force_stems = set(force_stems) if force_stems is not None else set()

    # --------------------------------------------------------------------
    # LOAD dPCA RESULTS
    # --------------------------------------------------------------------
    print(f"Using N_JOBS = {n_jobs}")
    if dpca_dir is None:
        dpca_dir = config.resolve_pp("dPCA", DEFAULT_DPCA_AREA, "subspace_results")
    print(f"Loading dPCA results from: {dpca_dir}")
    with open(os.path.join(dpca_dir, 'metadata.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)
    area = metadata['area']; window_dpca = tuple(metadata['window'])
    bin_size = metadata['bin_size']; step_size = metadata['step_size']
    if 'z_score' in metadata:
        z_score_data = metadata['z_score']
    else:
        print("WARNING: No 'z_score' in metadata. Defaulting to False.")
        z_score_data = False
    z_score_mode = 'custom_baseline' if z_score_data else None
    weights_npz = np.load(os.path.join(dpca_dir, 'dpca_weights.npz'), allow_pickle=True)
    D = weights_npz['D'].item(); P = weights_npz['P'].item()
    results_npz = np.load(os.path.join(dpca_dir, 'dpca_results.npz'), allow_pickle=True)
    neuron_ids_dpca = results_npz['neuron_ids']; X_dpca = results_npz['X']
    neuron_means = X_dpca.reshape(X_dpca.shape[0], -1).mean(axis=1); n_neurons = len(neuron_ids_dpca)
    print(f"Area: {area}\nNeurons: {n_neurons}\nMarginalizations: {sorted(D.keys())}")

    # --------------------------------------------------------------------
    # LOAD EXPERIMENTAL DATA
    # --------------------------------------------------------------------
    print("\nLoading experimental data...")
    trialData, sessionData, spikeData = utils.load_data(path_root, experiment, animals)
    (trialData, sessionData, spikeData, _) = utils.filter_sessions_hitrate(
        trialData, sessionData=sessionData, spikeData=spikeData, min_hitrate=0.3)
    trialData = trialData[~trialData['session_ID'].isin(sessions_exclude)]
    spikeData = spikeData[~spikeData['session_ID'].isin(sessions_exclude)]
    sessionData = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]
    spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=0.5, quality='good')
    spikeData = spikeData[spikeData['area'].isin(area)]
    trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)
    trialData_full = trialData.copy()
    print(f"Loaded {len(spikeData)} neurons, {len(trialData_full)} trials")
    print(f"Align to: {align_to}")
    print(f"Subsample trials for balanced null: {subsample_trials}")

    spikeData_proj = spikeData[spikeData['cluster_ID'].isin(neuron_ids_dpca)].copy()
    neuron_order_map = {nid: i for i, nid in enumerate(neuron_ids_dpca)}
    spikeData_proj['_dpca_order'] = spikeData_proj['cluster_ID'].map(neuron_order_map)
    spikeData_proj = spikeData_proj.sort_values('_dpca_order').drop('_dpca_order', axis=1)
    assert np.array_equal(spikeData_proj['cluster_ID'].values, neuron_ids_dpca), "Neuron order mismatch!"
    neuron_sessions = spikeData_proj['session_ID'].values
    print(f"Matched {len(spikeData_proj)} neurons to dPCA model")
    bins, time_centers = utils.custom_bins(bin_size=bin_size, step_size=step_size,
                                            window=window_dpca, causal=True)

    # --------------------------------------------------------------------
    # RESOLVE OUTPUT DIRECTORY
    # --------------------------------------------------------------------
    if output_dir is not None:
        save_dir = output_dir
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"output_dir does not exist: {save_dir}")
        print(f"\nReusing existing output directory: {save_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_parts = [f"euclidean_{align_to}"]
        if subsample_trials:
            tag_parts.append("sub")
        if output_tag:
            tag_parts.append(str(output_tag))
        tag_parts.append(timestamp)
        dir_name = "_".join(tag_parts)
        save_dir = os.path.join(config.DATA_DIR_PP_NEW, "dPCA", "".join(area), "euclidean_distance", dir_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCreated new output directory: {save_dir}")

    # --------------------------------------------------------------------
    # RUN EUCLIDEAN DISTANCE
    # --------------------------------------------------------------------
    print("\n" + "="*80 + "\nCOMPUTING EUCLIDEAN DISTANCES IN dPCA SPACE\n" + "="*80)
    all_results = {}; all_psth_caches = {}
    CONDITION_PAIRS = get_condition_pairs(opto_lat, include_opto)
    OPTO_COMPARISONS = get_opto_comparisons(opto_lat) if include_opto else {}

    current_meta = {
        'time_centers': time_centers,
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'subsample_trials': subsample_trials,
        'perm_window': OPTO_PERM_WINDOW,
        'cluster_window': CLUSTER_TIME_WINDOW,
        'cluster_threshold_percentile': CLUSTER_THRESHOLD_PCTILE,
    }

    for marg_key, condition_pairs in CONDITION_PAIRS.items():
        if marg_key not in D:
            print(f"\nWARNING: marginalization '{marg_key}' not in model. Skipping."); continue
        components = MARG_COMPONENTS.get(marg_key, [0,1,2])
        eucl_path = os.path.join(save_dir, f'euclidean_{marg_key}.npz')
        diff_path = os.path.join(save_dir, f'euclidean_diff_{marg_key}.npz')
        opto_path = os.path.join(save_dir, f'opto_permutation_{marg_key}.npz')
        eucl_stem = f"euclidean_{marg_key}"
        diff_stem = f"euclidean_diff_{marg_key}"
        opto_stem = f"opto_permutation_{marg_key}"

        print(f"\n{'='*60}\nMarginalization: {marg_key} — components {components}\n{'='*60}")

        # ----------------------------------------------------------------
        # INCREMENTAL EUCLIDEAN (label-based matching)
        # ----------------------------------------------------------------
        force_eucl = eucl_stem in force_stems
        existing_pairs, eucl_file_meta = ({}, {})
        if not force_eucl:
            existing_pairs, eucl_file_meta = load_existing_pairs_by_label(eucl_path)
            if existing_pairs:
                _check_compatibility(eucl_file_meta, current_meta, os.path.basename(eucl_path))

        # Walk current condition_pairs in order, deduplicating on label
        # (consistent with on-disk uniqueness key).
        current_pair_entries = []  # list of (label, fa, fb)
        seen_labels = set()
        for fa, fb, label in condition_pairs:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            current_pair_entries.append((label, fa, fb))

        if force_eucl:
            missing_pairs = [(fa, fb, label) for (label, fa, fb) in current_pair_entries]
            print(f"  [FORCE] Recomputing all {len(missing_pairs)} pairs.")
        else:
            missing_pairs = [(fa, fb, label) for (label, fa, fb) in current_pair_entries
                             if label not in existing_pairs]
            if existing_pairs and not missing_pairs:
                print(f"  All {len(current_pair_entries)} pairs already computed "
                      f"({len(existing_pairs)} cached). Skipping euclidean computation.")
            elif existing_pairs:
                print(f"  {len(existing_pairs)} pairs cached (matched by label), "
                      f"{len(missing_pairs)} new pair(s) to compute.")
            else:
                print(f"  No cached pairs. Computing all {len(missing_pairs)}.")

        psth_cache = None

        # Compute missing pairs (if any)
        new_results_by_label = {}
        if missing_pairs:
            # Stable seed offsets based on position in current config.
            label_to_offset = {label: idx for idx, (label, _, _) in enumerate(current_pair_entries)}
            seed_offsets = [label_to_offset[label] for (_, _, label) in missing_pairs]
            new_filters = []
            for fa, fb, _ in missing_pairs:
                new_filters.extend([fa, fb])
            psth_cache = build_psth_cache(new_filters, spikeData_proj, trialData_full,
                                           neuron_sessions, bins, align_to, window_dpca,
                                           z_score_data, z_score_mode)
            new_results, psth_cache = dpca_euclidean_distance(
                spikeData_proj, trialData_full, neuron_sessions,
                decoder_matrix=D[marg_key], neuron_means=neuron_means,
                condition_pairs=missing_pairs, bins=bins, time_centers=time_centers,
                align_to=align_to, window=window_dpca,
                z_score_data=z_score_data, z_score_mode=z_score_mode,
                components=components, n_bootstrap=n_bootstrap, ci=ci,
                perm_window=OPTO_PERM_WINDOW,
                cluster_window=CLUSTER_TIME_WINDOW,
                cluster_threshold_pctile=CLUSTER_THRESHOLD_PCTILE,
                subsample=subsample_trials, p_alpha=p_alpha,
                n_jobs=n_jobs, backend=parallel_backend,
                psth_cache=psth_cache, pair_seed_offsets=seed_offsets)
            for res in new_results:
                new_results_by_label[res['label']] = res

        # Merge: walk current config order; take new if available, else cached
        merged_results = []
        consumed_labels = set()
        for label, fa, fb in current_pair_entries:
            if label in new_results_by_label:
                merged_results.append(new_results_by_label[label])
            elif label in existing_pairs:
                merged_results.append(existing_pairs[label])
                consumed_labels.add(label)

        # Append orphan pairs (cached pairs whose labels aren't in current config)
        if not force_eucl and not prune_orphan_pairs:
            orphans = [res for label, res in existing_pairs.items()
                       if label not in {l for l, _, _ in current_pair_entries}]
            if orphans:
                print(f"  Keeping {len(orphans)} orphan pair(s) from existing file "
                      f"(labels not in current config). Use prune_orphan_pairs=True to drop.")
                merged_results.extend(orphans)

        all_results[marg_key] = merged_results
        if psth_cache is not None:
            all_psth_caches[marg_key] = psth_cache

        wrote_eucl = False
        if missing_pairs or force_eucl:
            _print_results_summary(merged_results, time_centers)
            save_dict = _build_pair_save_dict(
                merged_results, time_centers, components, marg_key,
                n_bootstrap, ci, OPTO_PERM_WINDOW, subsample_trials,
                CLUSTER_TIME_WINDOW, CLUSTER_THRESHOLD_PCTILE)
            np.savez(eucl_path, **save_dict, allow_pickle=True)
            print(f"  Saved: {os.path.basename(eucl_path)}")
            wrote_eucl = True

        # ----------------------------------------------------------------
        # DIFFERENCE ANALYSIS (regenerated when underlying eucl changes
        # or when diff file is missing/forced; cheap)
        # ----------------------------------------------------------------
        force_diff = diff_stem in force_stems
        diff_missing = not os.path.exists(diff_path)
        if include_opto and (force_diff or diff_missing or wrote_eucl):
            print(f"\n  --- Difference Analysis (Control - Opto) for {marg_key} ---")
            results = merged_results
            alpha_ci = (100 - ci) / 2
            label_map = {res['label']: i for i, res in enumerate(results)}
            diff_pairs = []
            for cl in [l for l in label_map if '(Control)' in l]:
                ol = cl.replace('(Control)', '(Opto)')
                if ol in label_map:
                    diff_pairs.append((label_map[cl], label_map[ol], cl.replace(' (Control)', '')))
            if not diff_pairs:
                print("    No matched Control/Opto pairs found.")
            else:
                diff_results = []
                for ic, io, dl in diff_pairs:
                    rc, ro = results[ic], results[io]
                    if rc.get('skipped') or ro.get('skipped') \
                            or np.all(np.isnan(rc['distance'])) or np.all(np.isnan(ro['distance'])):
                        print(f"    {dl}: [skipped]")
                        continue
                    dd = rc['distance'] - ro['distance']
                    bd = rc['boot_distribution'] - ro['boot_distribution']
                    dcl = np.nanpercentile(bd, alpha_ci, axis=0)
                    dch = np.nanpercentile(bd, 100 - alpha_ci, axis=0)
                    fp = np.nanmean(bd > 0, axis=0)
                    diff_results.append({'label': dl, 'diff_distance': dd, 'diff_ci_low': dcl,
                                         'diff_ci_high': dch, 'boot_diff': bd, 'frac_positive': fp})
                    pm = time_centers > 0
                    print(f"    {dl}: mean diff post={np.nanmean(dd[pm]):+.3f}, "
                          f"peak |diff|={np.nanmax(np.abs(dd)):.3f}")
                ds = {}
                for i, dr in enumerate(diff_results):
                    pf = f"diff{i}"
                    for fld in ['label', 'diff_distance', 'diff_ci_low',
                                'diff_ci_high', 'boot_diff', 'frac_positive']:
                        ds[f"{pf}_{fld}"] = dr[fld]
                ds['time_centers'] = time_centers
                np.savez(diff_path, **ds, allow_pickle=True)
                print(f"    Saved: {os.path.basename(diff_path)}")

        # ----------------------------------------------------------------
        # INCREMENTAL OPTO PERMUTATION TESTS (matched by (label, type))
        # ----------------------------------------------------------------
        if include_opto and marg_key in OPTO_COMPARISONS:
            comparisons = OPTO_COMPARISONS[marg_key]
            if not comparisons:
                continue

            force_opto = opto_stem in force_stems
            existing_opto, opto_file_meta = ({}, {})
            if not force_opto:
                existing_opto, opto_file_meta = load_existing_opto_comparisons_by_label(opto_path)
                if existing_opto:
                    _check_compatibility(opto_file_meta, current_meta, os.path.basename(opto_path))

            current_opto_entries = []  # list of ((label, type), comp)
            seen_opto_keys = set()
            for comp in comparisons:
                key = (comp['label'], comp['type'])
                if key in seen_opto_keys:
                    continue
                seen_opto_keys.add(key)
                current_opto_entries.append((key, comp))

            if force_opto:
                missing_comps = [comp for _, comp in current_opto_entries]
                print(f"\n  --- Opto Permutation Tests for {marg_key} ---")
                print(f"  [FORCE] Recomputing all {len(missing_comps)} comparisons.")
            else:
                missing_comps = [comp for key, comp in current_opto_entries if key not in existing_opto]
                if existing_opto and not missing_comps:
                    print(f"\n  --- Opto Permutation Tests for {marg_key} ---")
                    print(f"  All {len(current_opto_entries)} comparison(s) already computed. Skipping.")
                elif existing_opto:
                    print(f"\n  --- Opto Permutation Tests for {marg_key} ---")
                    print(f"  {len(existing_opto)} cached (matched by label+type), "
                          f"{len(missing_comps)} new comparison(s) to compute.")
                else:
                    print(f"\n  --- Opto Permutation Tests for {marg_key} ---")
                    print(f"  No cached comparisons. Computing all {len(missing_comps)}.")

            new_opto_by_key = {}
            if missing_comps:
                win_mask = (time_centers >= OPTO_PERM_WINDOW[0]) & (time_centers <= OPTO_PERM_WINDOW[1])
                print(f"  Window: {OPTO_PERM_WINDOW[0]}–{OPTO_PERM_WINDOW[1]}s "
                      f"({np.sum(win_mask)} bins)")
                if subsample_trials:
                    print(f"  Null balancing: ON")

                if psth_cache is None:
                    psth_cache = all_psth_caches.get(marg_key, {})

                opto_filters = []
                for comp in missing_comps:
                    for k in comp:
                        if k.startswith('filter_'):
                            opto_filters.append(comp[k])
                if not psth_cache:
                    psth_cache = build_psth_cache(
                        opto_filters, spikeData_proj, trialData_full, neuron_sessions,
                        bins, align_to, window_dpca, z_score_data, z_score_mode)
                else:
                    for fd in opto_filters:
                        psth_cache = ensure_in_cache(
                            psth_cache, fd, spikeData_proj, trialData_full, neuron_sessions,
                            bins, align_to, window_dpca, z_score_data, z_score_mode)

                all_psth_caches[marg_key] = psth_cache
                n_timebins = len(time_centers)

                opto_key_to_idx = {key: idx for idx, (key, _) in enumerate(current_opto_entries)}

                for comp in missing_comps:
                    comp_idx_stable = opto_key_to_idx[(comp['label'], comp['type'])]
                    label, test_type = comp['label'], comp['type']
                    print(f"\n  {label} ({test_type})")

                    if test_type in ('separation_change', 'shift_difference'):
                        fkeys = ['filter_a_ctrl', 'filter_b_ctrl',
                                 'filter_a_opto', 'filter_b_opto']
                    elif test_type == 'separation_change_difference':
                        fkeys = ['filter_a1_ctrl', 'filter_a2_ctrl',
                                 'filter_a1_opto', 'filter_a2_opto',
                                 'filter_b1_ctrl', 'filter_b2_ctrl',
                                 'filter_b1_opto', 'filter_b2_opto']
                    else:
                        print(f"    Unknown test type: {test_type}")
                        continue

                    all_present = True
                    for fk in fkeys:
                        psth_cache = ensure_in_cache(
                            psth_cache, comp[fk], spikeData_proj, trialData_full,
                            neuron_sessions, bins, align_to, window_dpca,
                            z_score_data, z_score_mode)
                        if psth_cache[_make_filter_key(comp[fk])][1] < 2:
                            print(f"    WARNING: <2 trials for {fk}. Skipping.")
                            all_present = False
                            break
                    if not all_present:
                        continue

                    comp_base_seed = 10000 + comp_idx_stable * 1000

                    if test_type == 'separation_change':
                        pac = psth_cache[_make_filter_key(comp['filter_a_ctrl'])][0]
                        pbc = psth_cache[_make_filter_key(comp['filter_b_ctrl'])][0]
                        pao = psth_cache[_make_filter_key(comp['filter_a_opto'])][0]
                        pbo = psth_cache[_make_filter_key(comp['filter_b_opto'])][0]
                        dc = compute_projected_distance(pac, pbc, n_neurons, n_timebins,
                                                         D[marg_key], neuron_means, components)
                        do = compute_projected_distance(pao, pbo, n_neurons, n_timebins,
                                                         D[marg_key], neuron_means, components)
                        observed = dc - do
                        print(f"    Permuting ({n_bootstrap} iter, n_jobs={n_jobs})...")
                        null = permutation_null_difference(
                            pac, pbc, pao, pbo, n_neurons, n_timebins,
                            D[marg_key], neuron_means, components,
                            n_permutations=n_bootstrap, base_seed=comp_base_seed,
                            subsample=subsample_trials, n_jobs=n_jobs, backend=parallel_backend)
                    elif test_type == 'shift_difference':
                        pac = psth_cache[_make_filter_key(comp['filter_a_ctrl'])][0]
                        pao = psth_cache[_make_filter_key(comp['filter_a_opto'])][0]
                        pbc = psth_cache[_make_filter_key(comp['filter_b_ctrl'])][0]
                        pbo = psth_cache[_make_filter_key(comp['filter_b_opto'])][0]
                        da = compute_projected_distance(pac, pao, n_neurons, n_timebins,
                                                         D[marg_key], neuron_means, components)
                        db = compute_projected_distance(pbc, pbo, n_neurons, n_timebins,
                                                         D[marg_key], neuron_means, components)
                        observed = da - db
                        print(f"    Permuting ({n_bootstrap} iter, n_jobs={n_jobs})...")
                        null = permutation_null_shift_difference(
                            pac, pao, pbc, pbo, n_neurons, n_timebins,
                            D[marg_key], neuron_means, components,
                            n_permutations=n_bootstrap, base_seed=comp_base_seed,
                            subsample=subsample_trials, n_jobs=n_jobs, backend=parallel_backend)
                    elif test_type == 'separation_change_difference':
                        psths = {k: psth_cache[_make_filter_key(comp[k])][0] for k in fkeys}
                        dac = compute_projected_distance(psths['filter_a1_ctrl'],
                                                          psths['filter_a2_ctrl'],
                                                          n_neurons, n_timebins,
                                                          D[marg_key], neuron_means, components)
                        dao = compute_projected_distance(psths['filter_a1_opto'],
                                                          psths['filter_a2_opto'],
                                                          n_neurons, n_timebins,
                                                          D[marg_key], neuron_means, components)
                        dbc = compute_projected_distance(psths['filter_b1_ctrl'],
                                                          psths['filter_b2_ctrl'],
                                                          n_neurons, n_timebins,
                                                          D[marg_key], neuron_means, components)
                        dbo = compute_projected_distance(psths['filter_b1_opto'],
                                                          psths['filter_b2_opto'],
                                                          n_neurons, n_timebins,
                                                          D[marg_key], neuron_means, components)
                        observed = (dac - dao) - (dbc - dbo)
                        print(f"    Permuting ({n_bootstrap} iter, n_jobs={n_jobs})...")
                        null = permutation_null_separation_change_difference(
                            psths['filter_a1_ctrl'], psths['filter_a2_ctrl'],
                            psths['filter_a1_opto'], psths['filter_a2_opto'],
                            psths['filter_b1_ctrl'], psths['filter_b2_ctrl'],
                            psths['filter_b1_opto'], psths['filter_b2_opto'],
                            n_neurons, n_timebins, D[marg_key], neuron_means, components,
                            n_permutations=n_bootstrap, base_seed=comp_base_seed,
                            subsample=subsample_trials, n_jobs=n_jobs, backend=parallel_backend)

                    n_perms = null.shape[0]
                    pvals = (np.sum(null >= observed[np.newaxis, :], axis=0) + 1) / (n_perms + 1)
                    obs_win = np.mean(observed[win_mask])
                    null_win = np.mean(null[:, win_mask], axis=1)
                    p_win = (np.sum(null_win >= obs_win) + 1) / (n_perms + 1)
                    cr = cluster_permutation_test(observed, null, time_centers,
                                                   cluster_window=CLUSTER_TIME_WINDOW,
                                                   threshold_percentile=CLUSTER_THRESHOLD_PCTILE,
                                                   p_alpha=p_alpha)
                    pm = time_centers > 0
                    pi = np.argmax(np.abs(observed))
                    print(f"    Time-resolved: {np.sum(pvals[pm] < p_alpha)}/{np.sum(pm)} sig. (p<{p_alpha})")
                    print(f"    Peak diff: {observed[pi]:+.3f} at t={time_centers[pi]:.3f}s "
                          f"(p={pvals[pi]:.4e})")
                    print(f"    Window-averaged: mean diff = {obs_win:+.3f}, p = {p_win:.4e}")
                    _print_cluster_summary(cr, p_alpha=p_alpha)

                    new_opto_by_key[(label, test_type)] = {
                        'label': label, 'type': test_type, 'observed': observed,
                        'null_distribution': null, 'p_values': pvals,
                        'observed_window_avg': obs_win, 'null_window_avg': null_win,
                        'p_value_window_avg': p_win, 'window': OPTO_PERM_WINDOW,
                        'cluster_result': cr,
                    }

            # Merge new + existing opto comparisons
            merged_opto = []
            for key, comp in current_opto_entries:
                if key in new_opto_by_key:
                    merged_opto.append(new_opto_by_key[key])
                elif key in existing_opto:
                    merged_opto.append(existing_opto[key])

            if not force_opto and not prune_orphan_pairs:
                current_opto_keys = {k for k, _ in current_opto_entries}
                opto_orphans = [res for k, res in existing_opto.items()
                                if k not in current_opto_keys]
                if opto_orphans:
                    print(f"  Keeping {len(opto_orphans)} orphan opto comparison(s) "
                          f"((label, type) not in current config). "
                          f"Use prune_orphan_pairs=True to drop.")
                    merged_opto.extend(opto_orphans)

            if missing_comps or force_opto:
                if merged_opto:
                    sd = _build_opto_save_dict(merged_opto, time_centers,
                                                CLUSTER_TIME_WINDOW,
                                                CLUSTER_THRESHOLD_PCTILE,
                                                subsample_trials)
                    np.savez(opto_path, **sd, allow_pickle=True)
                    print(f"\n    Saved: {os.path.basename(opto_path)}")

    # --------------------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------------------
    run_metadata = {
        'area': area, 'n_neurons': n_neurons, 'align_to': align_to,
        'window': list(window_dpca), 'bin_size': bin_size, 'step_size': step_size,
        'n_bootstrap': n_bootstrap, 'ci': ci, 'z_score_data': z_score_data, 'metric': 'euclidean',
        'marg_components': {k: v for k, v in MARG_COMPONENTS.items()},
        'include_opto': include_opto, 'subsample_trials': subsample_trials,
        'opto_perm_window': list(OPTO_PERM_WINDOW), 'perm_window': list(OPTO_PERM_WINDOW),
        'cluster_window': [None if (CLUSTER_TIME_WINDOW is None or CLUSTER_TIME_WINDOW[0] is None) else CLUSTER_TIME_WINDOW[0],
                           None if (CLUSTER_TIME_WINDOW is None or CLUSTER_TIME_WINDOW[1] is None) else CLUSTER_TIME_WINDOW[1]],
        'cluster_threshold_percentile': CLUSTER_THRESHOLD_PCTILE,
        'p_alpha': p_alpha,
        'n_jobs': n_jobs,
        'last_run': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(save_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(run_metadata, f, default_flow_style=False)
    print(f"\nDone! Results in: {save_dir}")

    return {
        'save_dir': save_dir,
        'all_results': all_results,
        'all_psth_caches': all_psth_caches,
        'run_metadata': run_metadata,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def _parse_window(s):
    if s is None: return None
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 2: raise argparse.ArgumentTypeError("Window must be 'start,end'")
    def _p(x):
        if x.lower() in ('none', 'null', ''): return None
        return float(x)
    return (_p(parts[0]), _p(parts[1]))


def _main_cli():
    parser = argparse.ArgumentParser(
        description="Compute Euclidean distances in dPCA space. "
                    "Re-run with --output-dir to incrementally compute only "
                    "newly-added pairs/comparisons (matched by label).")
    parser.add_argument('--dpca-dir', type=str, default=DEFAULT_DPCA_DIR)
    parser.add_argument('--align-to', type=str, default=DEFAULT_ALIGN_TO)
    parser.add_argument('--subsample-trials', action='store_true',
                        default=DEFAULT_SUBSAMPLE_TRIALS)
    parser.add_argument('--no-include-opto', dest='include_opto', action='store_false',
                        default=DEFAULT_INCLUDE_OPTO)
    parser.add_argument('--n-bootstrap', type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument('--opto-perm-window', type=_parse_window,
                        default=DEFAULT_OPTO_PERM_WINDOW,
                        help="'start,end' in seconds (e.g. '0,1')")
    parser.add_argument('--cluster-window', type=_parse_window,
                        default=DEFAULT_CLUSTER_TIME_WINDOW,
                        help="'start,end' in seconds, or 'None,None'")
    parser.add_argument('--cluster-threshold-pctile', type=float,
                        default=DEFAULT_CLUSTER_THRESHOLD_PCTILE)
    parser.add_argument('--p-alpha', type=float, default=DEFAULT_P_ALPHA)
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Path to an existing results directory "
                             "(triggers incremental computation).")
    parser.add_argument('--output-tag', type=str, default=None,
                        help="Extra tag for auto-generated output directory name.")
    parser.add_argument('--force', nargs='*', default=[],
                        help="Force full recompute of specific file stems "
                             "(e.g. 'euclidean_s opto_permutation_s'). "
                             "Bypasses incremental logic for those files.")
    parser.add_argument('--prune-orphan-pairs', action='store_true',
                        help="Drop pairs/comparisons present on disk but no "
                             "longer in the current config. Default: keep them.")
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS,
                        help="Number of parallel workers (-1 = all cores).")
    args = parser.parse_args()

    try:
        run_euclidean_analysis(
            dpca_dir=args.dpca_dir,
            align_to=args.align_to,
            subsample_trials=args.subsample_trials,
            include_opto=args.include_opto,
            n_bootstrap=args.n_bootstrap,
            OPTO_PERM_WINDOW=args.opto_perm_window,
            CLUSTER_TIME_WINDOW=args.cluster_window,
            CLUSTER_THRESHOLD_PCTILE=args.cluster_threshold_pctile,
            p_alpha=args.p_alpha,
            output_dir=args.output_dir,
            output_tag=args.output_tag,
            force_stems=args.force,
            prune_orphan_pairs=args.prune_orphan_pairs,
            n_jobs=args.n_jobs,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    _main_cli()