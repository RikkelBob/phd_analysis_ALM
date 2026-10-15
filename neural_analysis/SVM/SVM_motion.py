import numpy as np
import os
import scipy.ndimage
from neural_analysis import utils
import pandas as pd
import yaml
from neural_analysis.motion.tools.utils_motion import extract_frames
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from joblib import Parallel, delayed
import shutil
from neural_analysis.motion.tools import utils_motion

import config


# ============================================================
# Global RNG / decoder settings — change here
# ============================================================
RNG_SEED = 42
N_SHUFFLES = 1000
N_REAL_REPEATS = 20         # with 5 folds → 25 real-score estimates averaged
MIN_TRIALS_PER_LABEL = 20


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def filter_session_dict_trials_per_condition(data_dict, min_count_per_label):
    """
    Filter sessions based on minimum count per unique label across all conditions.

    Args:
        data_dict: Dictionary with conditions as keys, each containing 'session_ID', 'labels', and 'data'
        min_count_per_label: Minimum number of entries required for each unique label

    Returns:
        Filtered dictionary with same structure, containing only sessions that meet the criteria
    """
    filtered_dict = {}

    # Loop over each condition
    for condition_name, condition_data in data_dict.items():
        print(f"\nProcessing condition: {condition_name}")

        filtered_dict[condition_name] = {
            'session_ID': [],
            'labels': [],
            'data': []
        }

        # Loop over sessions within this condition
        for i, session_id in enumerate(condition_data['session_ID']):
            labels = condition_data['labels'][i]

            # Get unique labels and their counts
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Check if all unique labels have at least min_count_per_label entries
            if np.all(counts >= min_count_per_label) and len(unique_labels) > 1:
                # Keep this session
                filtered_dict[condition_name]['session_ID'].append(session_id)
                filtered_dict[condition_name]['labels'].append(labels)
                filtered_dict[condition_name]['data'].append(condition_data['data'][i])
            else:
                # Remove this session and print the ID
                print(f"Removing session: {session_id} (condition: {condition_name})")

    return filtered_dict


def get_sessions_with_sufficient_trials(trialData, config, k_folds):
    """
    Get session IDs for sessions that have at least k_folds trials in each condition.
    Uses only trial data - much more efficient than loading SVD data first.

    Parameters:
    -----------
    trialData : pd.DataFrame
        Trial data containing session_ID and all trial parameters
    config : MotionAnalysisConfig
        Configuration object containing condition definitions
    k_folds : int
        Minimum number of trials required for each label in each condition

    Returns:
    --------
    list
        List of session_IDs that meet the criteria
    """
    valid_session_ids = []

    # Get unique session IDs
    session_ids = trialData['session_ID'].unique()

    for session_ID in session_ids:
        trialData_ses = trialData[trialData['session_ID'] == session_ID]

        session_valid = True

        # Check each condition
        for condition_name, condition_params in config.conditions.items():
            # Start with all trials for this session
            trialData_cond = trialData_ses.copy()

            # Apply filters for each parameter (except special keys)
            special_keys = {'labels', 'legend'}

            for param_name, param_value in condition_params.items():
                if param_name not in special_keys:
                    if isinstance(param_value, list):
                        # Handle multiple values (e.g., 'trialType': ['X', 'Y'])
                        trialData_cond = trialData_cond[trialData_cond[param_name].isin(param_value)]
                    else:
                        # Handle single values (e.g., 'noResponse': True)
                        trialData_cond = trialData_cond[trialData_cond[param_name] == param_value]

            if trialData_cond.empty:
                session_valid = False
                break

            # Create labels based on condition definitions
            labels = np.zeros(len(trialData_cond))
            for label_value, label_func in condition_params['labels'].items():
                mask = label_func(trialData_cond)
                labels[mask] = label_value

            # Only keep trials with valid labels (exclude label 0)
            valid_trials = labels > 0
            if not np.any(valid_trials):
                session_valid = False
                break

            labels = labels[valid_trials]

            # Check if each label has at least k_folds trials
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if np.sum(labels == label) < k_folds:
                    session_valid = False
                    break

            if not session_valid:
                break

        if session_valid:
            valid_session_ids.append(session_ID)

    return valid_session_ids


def process_session_data(session, sessionData, trialData, config, SVD_dict_per_session, rng=None):
    """Process motion data for a single session.

    Note: trial balancing via subsampling has been removed — class imbalance is now
    handled by `class_weight='balanced'` inside the classifier.
    """
    session_folder = os.path.basename(session['session'])
    sessionData_ses = sessionData[sessionData['folderName'] == session_folder]
    if sessionData_ses.empty:
        return None
    sampling_interval = 1 / 25 if (sessionData_ses['experiment'] == 'ALM_1').iloc[0] else 1 / 80
    session_ID = sessionData_ses['session_ID'].item()
    trialData_ses = trialData[trialData['session_ID'] == session_ID]

    # Temporal smoothing kernel
    smoothing_kernel = utils_motion.causal_gaussian_kernel(0.1, 1 / sampling_interval)

    # Processing loop per condition
    for condition_name, condition_params in config.conditions.items():
        # Start with all trials for this session
        trialData_cond = trialData_ses.copy()

        # Apply filters for each parameter (except special keys)
        special_keys = {'labels', 'legend'}
        for param_name, param_value in condition_params.items():
            if param_name not in special_keys:
                if isinstance(param_value, list):
                    trialData_cond = trialData_cond[trialData_cond[param_name].isin(param_value)]
                else:
                    trialData_cond = trialData_cond[trialData_cond[param_name] == param_value]

        if trialData_cond.empty:
            continue

        motion_data = session['motSVD_face'][1]
        kernel_origin = -((len(smoothing_kernel) - 1) // 2)
        motion_data = scipy.ndimage.convolve1d(motion_data, smoothing_kernel, axis=0, origin=kernel_origin)

        # Extract motion data
        trial_aligned_motion, good_trials = extract_frames(
            session['timeline'],
            motion_data,
            trialData_cond['stimChange'],
            config.window,
            config.n_components,
            sampling_interval
        )

        # Filter trial data for good trials
        trialData_cond = trialData_cond[good_trials]

        # Create labels based on condition definitions
        labels = np.zeros(len(trialData_cond))
        for label_value, label_func in condition_params['labels'].items():
            mask = label_func(trialData_cond)
            labels[mask] = label_value

        # Only keep trials with valid labels (exclude label 0)
        valid_trials = labels > 0
        if not np.any(valid_trials):
            print(f"Warning: No valid trials for {condition_name} in session {session_ID}")
            continue

        trial_aligned_motion = trial_aligned_motion[valid_trials]
        labels = labels[valid_trials]

        # Initialize condition dict if it doesn't exist
        if condition_name not in SVD_dict_per_session:
            SVD_dict_per_session[condition_name] = {
                'session_ID': [],
                'labels': [],
                'data': []
            }

        # Append session data
        SVD_dict_per_session[condition_name]['session_ID'].append(str(session_ID))
        SVD_dict_per_session[condition_name]['labels'].append(labels)
        SVD_dict_per_session[condition_name]['data'].append(trial_aligned_motion)

    return session_ID


class MotionAnalysisConfig:
    """Configuration class for motion analysis parameters."""

    def __init__(self):
        # Directories
        self.root_path = config.DATA_DIR_PP / "motion"
        self.data_path = config.DATA_DIR_EXP
        self.output_dir = config.DATA_DIR_PP_NEW / "SVM" / "decode_from_motion"

        # Experiment parameters
        self.experiments = ['ALM_2', 'ALM_3']
        self.window = (-1, 1.5)
        self.baseline_window = [-1, -0.1]
        self.n_components = 500

        # Condition definitions
        self.conditions = self.define_conditions_limited()

    def define_conditions(self):
        """
        Define analysis conditions. Modify this method to add/change conditions.
        Each condition should specify:
        - Any DataFrame column name as a key with the desired values to filter on
        - labels: dictionary mapping label names to trial selection criteria
        - legend: list of legend labels for plotting
        - label_ignore: list of labels to ignore (optional)
        """
        return {
            'vis_presence': {
                'trialType': ['X', 'P'],
                'noResponse': True,
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'X',
                    2: lambda trial_data: trial_data['trialType'] == 'P'
                },
                'legend': ['Visual', 'Probe'],
            },

            'au_presence': {
                'trialType': ['Y', 'P'],
                'noResponse': True,
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'Y',
                    2: lambda trial_data: trial_data['trialType'] == 'P'
                },
                'legend': ['Auditory', 'Probe'],
            },

            'response_presence': {
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['correctResponse'] == 1,
                    2: lambda trial_data: trial_data['noResponse'] == 1
                },
                'legend': ['Correct', 'None'],
            },

            'response_side': {
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['responseSide'] == 'L',
                    2: lambda trial_data: trial_data['responseSide'] == 'R',
                    3: lambda trial_data: trial_data['responseSide'] == 'N'
                },
                'legend': ['Left', 'Right', 'None'],
            },

            'vis_side_miss': {
                'trialType': 'X',
                'hasphotostim': False,
                'noResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['visualChangeSide'] == 1,
                    2: lambda trial_data: trial_data['visualChangeSide'] == 2
                },
                'legend': ['Left', 'Right'],
            },

            'vis_side_hit': {
                'trialType': 'X',
                'hasphotostim': False,
                'correctResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['visualChangeSide'] == 1,
                    2: lambda trial_data: trial_data['visualChangeSide'] == 2
                },
                'legend': ['Left', 'Right'],
            },

            'vis_side': {
                'trialType': 'X',
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['visualChangeSide'] == 1,
                    2: lambda trial_data: trial_data['visualChangeSide'] == 2
                },
                'legend': ['Left', 'Right'],
            },

            'trial_type': {
                'hasphotostim': False,
                'noResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'X',
                    2: lambda trial_data: trial_data['trialType'] == 'Y',
                    3: lambda trial_data: trial_data['trialType'] == 'P'
                },
                'legend': ['Visual', 'Auditory', 'Probe'],
            },

            'au_pitch_all': {
                'trialType': 'Y',
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
                    2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
                },
            },

            'vis_ori_left_all': {
                'trialType': 'X',
                'hasphotostim': False,
                'visualChangeSide': 1,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 2
                }
            },

            'vis_ori_right_all': {
                'trialType': 'X',
                'hasphotostim': False,
                'visualChangeSide': 2,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 2
                },
            },

            'au_pitch_miss': {
                'trialType': 'Y',
                'hasphotostim': False,
                'noResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
                    2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
                },
            },

            'vis_ori_left_miss': {
                'trialType': 'X',
                'hasphotostim': False,
                'noResponse': True,
                'visualChangeSide': 1,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 2
                }
            },

            'vis_ori_right_miss': {
                'trialType': 'X',
                'hasphotostim': False,
                'noResponse': True,
                'visualChangeSide': 2,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 2
                },
            },

            'au_pitch_hit': {
                'trialType': 'Y',
                'hasphotostim': False,
                'correctResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
                    2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
                }
            },

            'vis_ori_left_hit': {
                'trialType': 'X',
                'hasphotostim': False,
                'correctResponse': True,
                'visualChangeSide': 1,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 2
                }
            },

            'vis_ori_right_hit': {
                'trialType': 'X',
                'hasphotostim': False,
                'correctResponse': True,
                'visualChangeSide': 2,
                'labels': {
                    1: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 1,
                    2: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 2
                }
            },

            'modality': {
                'trialType': ['X', 'Y'],
                'hasphotostim': False,
                'noResponse': True,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'X',
                    2: lambda trial_data: trial_data['trialType'] == 'Y'
                }
            }
        }

    def define_conditions_limited(self):
        """
        Define analysis conditions. Modify this method to add/change conditions.
        Each condition should specify:
        - Any DataFrame column name as a key with the desired values to filter on
        - labels: dictionary mapping label names to trial selection criteria
        - legend: list of legend labels for plotting
        """
        return {
            'vis_presence': {
                'trialType': ['X', 'P'],
                'noResponse': True,
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'X',
                    2: lambda trial_data: trial_data['trialType'] == 'P'
                },
                'legend': ['Visual', 'Probe'],
            },

            'au_presence': {
                'trialType': ['Y', 'P'],
                'noResponse': True,
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: trial_data['trialType'] == 'Y',
                    2: lambda trial_data: trial_data['trialType'] == 'P'
                },
                'legend': ['Auditory', 'Probe'],
            },

            'vis_hit_aud_hit': {
                'trialType': ['X', 'Y'],
                'noResponse': False,
                'hasphotostim': False,
                'labels': {
                    1: lambda trial_data: (trial_data['trialType'] == 'X') & (trial_data['correctResponse'] == 1),
                    2: lambda trial_data: (trial_data['trialType'] == 'Y') & (trial_data['correctResponse'] == 1)
                },
                'legend': ['Visual', 'Probe'],
            },

            #'vis_hit_miss': {
            #    'trialType': ['X'],
            #    'hasphotostim': False,
            #    'labels': {
            #        1: lambda trial_data: trial_data['correctResponse'] == 1,
            #        2: lambda trial_data: trial_data['noResponse'] == 1
            #    },
            #    'legend': ['Visual', 'Probe'],
            #},

            #'aud_hit_miss': {
            #    'trialType': ['Y'],
            #    'hasphotostim': False,
            #    'labels': {
            #        1: lambda trial_data: trial_data['correctResponse'] == 1,
            #        2: lambda trial_data: trial_data['noResponse'] == 1
            #    },
            #    'legend': ['Visual', 'Probe'],
            #},

            #'vis_side_miss': {
            #    'trialType': 'X',
            #    'hasphotostim': False,
            #    'noResponse': True,
            #    'labels': {
            #        1: lambda trial_data: trial_data['visualChangeSide'] == 1,
            #        2: lambda trial_data: trial_data['visualChangeSide'] == 2
            #    },
            #    'legend': ['Left', 'Right'],
            #},

            #'vis_side_opto': {
            #    'trialType': 'X',
            #    'hasphotostim': True,
            #    'optoLat': 'unilat',
            #    'labels': {
            #        1: lambda trial_data: trial_data['visualChangeSide'] == 1,
            #        2: lambda trial_data: trial_data['visualChangeSide'] == 2
            #    },
            #    'legend': ['Left', 'Right'],
            #},

            #'vis_side_opto_control': {
            #    'trialType': 'X',
            #    'hasphotostim': False,
            #    'optoLat': 'unilat',
            #    'labels': {
            #        1: lambda trial_data: trial_data['visualChangeSide'] == 1,
            #        2: lambda trial_data: trial_data['visualChangeSide'] == 2
            #    },
            #    'legend': ['Left', 'Right'],
            #},

            #'au_pitch_all': {
            #    'trialType': 'Y',
            #    'hasphotostim': False,
            #    'labels': {
            #        1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
            #        2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
            #    },
            #},

            #'au_pitch_all_opto': {
            #    'trialType': 'Y',
            #    'hasphotostim': True,
            #    'optoLat': 'unilat',
            #    'labels': {
            #        1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
            #        2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
            #    },
            #},

            #'au_pitch_all_opto_control': {
            #    'trialType': 'Y',
            #    'hasphotostim': False,
            #    'optoLat': 'unilat',
            #    'labels': {
            #        1: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 1,
            #        2: lambda trial_data: trial_data['audioOctPostChange_clustered'] == 2
            #    },
            #},

            #'vis_ori_left_all': {
            #    'trialType': 'X',
            #    'hasphotostim': False,
            #    'visualChangeSide': 1,
            #    'labels': {
            #        1: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 1,
            #        2: lambda trial_data: trial_data['visualOriPostChangeLeft_clustered'] == 2
            #    }
            #},

            #'vis_ori_right_all': {
            #    'trialType': 'X',
            #    'hasphotostim': False,
            #    'visualChangeSide': 2,
            #    'labels': {
            #        1: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 1,
            #        2: lambda trial_data: trial_data['visualOriPostChangeRight_clustered'] == 2
            #    },
            #},
        }


def main():
    """Main analysis pipeline."""

    # Initialize configuration
    motion_config = MotionAnalysisConfig()
    load_preprocessed_data = False
    save_preprocessed_data = True

    # Create output directories
    output_dir = motion_config.output_dir
    ensure_dir(output_dir)
    load_dir = config.resolve_pp("SVM", "decode_from_motion") if load_preprocessed_data else output_dir

    # ---------------------------------------------------------
    # Build decoder
    # ---------------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)

    k_folds = 5
    cv_real = RepeatedStratifiedKFold(
        n_splits=k_folds, n_repeats=N_REAL_REPEATS, random_state=RNG_SEED
    )
    cv_shuffle = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RNG_SEED)

    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(class_weight='balanced', max_iter=10**5,
                          random_state=RNG_SEED, dual='auto'))
    ])

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    print("Loading data...")
    trialData, sessionData, _ = utils.load_data(motion_config.data_path, motion_config.experiments)
    session_paths = [s for s in utils.get_session_paths(motion_config.root_path)
                     if any(sub in s for sub in motion_config.experiments)]

    # DEBUG
    # session_paths = session_paths[0:2]

    # Only include sessions with kfolds trials for every condition
    ses_exclude = utils_motion.bad_motion_sessions()
    valid_session_IDs = sessionData[~sessionData['session_ID'].isin(ses_exclude)]['session_ID']
    valid_session_folders = sessionData[sessionData['session_ID'].isin(valid_session_IDs)]['folderName']

    valid_session_paths = [full_path for full_path in session_paths
                           if any(valid_session_IDs in full_path for valid_session_IDs in valid_session_folders)]

    session_paths = valid_session_paths

    # Fix empty entries for no response
    trialData['responseSide'] = trialData['responseSide'].apply(lambda x: 'N' if len(x) == 0 else x)

    # Add stimulus orientation / pitch clustering for decoding
    for _, session in sessionData.iterrows():
        trialData_ses_mask = trialData['session_ID'] == session['session_ID']

        stimulus_cluster_method = 'optimized_per_session'
        trialData.loc[trialData_ses_mask, 'visualOriPostChangeLeft_clustered'] = \
            utils.cluster_samples(trialData[trialData_ses_mask], 'visualOriPostChangeLeft',
                                  n_clusters=4, method=stimulus_cluster_method, modality='vis', bin_size=45)

        trialData.loc[trialData_ses_mask, 'visualOriPostChangeRight_clustered'] = \
            utils.cluster_samples(trialData[trialData_ses_mask], 'visualOriPostChangeRight',
                                  n_clusters=4, method=stimulus_cluster_method, modality='vis', bin_size=45)

        trialData.loc[trialData_ses_mask, 'audioOctPostChange_clustered'] = \
            utils.cluster_samples(trialData[trialData_ses_mask], 'audioOctPostChange',
                                  n_clusters=4, method=stimulus_cluster_method, modality='aud', bin_size=1 / 8)

    if load_preprocessed_data:
        SVD_dict_per_session = np.load(os.path.join(load_dir, 'SVD_dict.npy'), allow_pickle=True).item()
    else:
        # Load motion data
        print("Loading motion data...")
        SVD_motion = utils.get_motSVD(session_paths, timeline_file="realigned_timeline_nlx_ts.npy")
        SVD_motion = pd.DataFrame(SVD_motion)

        # Initialize storage for all sessions
        SVD_dict_per_session = {
            key: {
                'session_ID': [],
                'labels': [],
                'data': []
            } for key in motion_config.conditions.keys()
        }

        # Process each session
        print("Processing sessions...")
        for idx_ses, session in SVD_motion.iterrows():
            print(f"Processing session {idx_ses + 1}/{len(SVD_motion)}")
            process_session_data(session, sessionData, trialData, motion_config, SVD_dict_per_session, rng=rng)

        if save_preprocessed_data:
            np.save(os.path.join(output_dir, 'SVD_dict.npy'), SVD_dict_per_session, allow_pickle=True)

    SVD_dict_per_session = filter_session_dict_trials_per_condition(SVD_dict_per_session, min_count_per_label=MIN_TRIALS_PER_LABEL)

    # ---------------------------------------------------------
    # Generate metadata file (YAML)
    # ---------------------------------------------------------
    print("Generating metadata...")
    metadata = {}

    for condition in SVD_dict_per_session.keys():
        session_ids = SVD_dict_per_session[condition]['session_ID']
        labels_list = SVD_dict_per_session[condition]['labels']

        mice = sessionData[sessionData['session_ID'].astype(str).isin(
            [str(s) for s in session_ids])]['mousename'].unique()

        trials_per_label = {}
        trials_per_label_per_session = {}
        for ses_idx, session_id in enumerate(session_ids):
            labels = labels_list[ses_idx]
            unique_labels, counts = np.unique(labels, return_counts=True)
            trials_per_label_per_session[str(session_id)] = {
                int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)
            }
            for lbl, cnt in zip(unique_labels, counts):
                trials_per_label[int(lbl)] = trials_per_label.get(int(lbl), 0) + int(cnt)

        legend = motion_config.conditions[condition].get('legend', None)

        metadata[condition] = {
            'session_IDs': [str(s) for s in session_ids],
            'n_sessions': len(session_ids),
            'animals': sorted([str(m) for m in mice]),
            'n_animals': len(mice),
            'total_trials': int(sum(trials_per_label.values())),
            'trials_per_label': trials_per_label,
            'trials_per_label_per_session': trials_per_label_per_session,
            'legend': list(legend) if legend is not None else None,
        }

    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Metadata saved to {metadata_path}")


    # ---------------------------------------------------------
    # Create output dict
    # ---------------------------------------------------------
    output = {}
    for name in motion_config.conditions:
        output['results_' + name] = []
        output['results_shuffled_' + name] = []

    # ---------------------------------------------------------
    # Perform decoding for each condition and each session separately
    # ---------------------------------------------------------
    for condition in SVD_dict_per_session.keys():

        if len(SVD_dict_per_session[condition]['session_ID']) == 0:
            print(f" {condition} has no valid sessions - Skipping")
            continue

        print(f"Starting SVM for {condition}")

        for ses_idx, session_id in enumerate(SVD_dict_per_session[condition]['session_ID']):
            labels = SVD_dict_per_session[condition]['labels'][ses_idx]
            n_bins = SVD_dict_per_session[condition]['data'][ses_idx].shape[1]

            for idx_bin in range(n_bins):
                bin_matrix = SVD_dict_per_session[condition]['data'][ses_idx][:, idx_bin, :].copy()

                # Real score, averaged over repeated folds (parallelized across folds)
                results_bin = cross_validate(
                    estimator=estimator,
                    X=bin_matrix,
                    y=labels,
                    cv=cv_real,
                    scoring="balanced_accuracy",
                    n_jobs=-1
                )

                # Shuffled null distribution: 1000 permutations, parallelized across shuffles.
                # Pre-draw all permutations with the seeded RNG so the parallel workers are
                # deterministic and reproducible (independent of n_jobs).
                shuffled_labels_all = [rng.permutation(labels) for _ in range(N_SHUFFLES)]

                def _shuffle_score(y_shuf):
                    scores_shuf = cross_validate(
                        estimator=estimator,
                        X=bin_matrix,
                        y=y_shuf,
                        cv=cv_shuffle,
                        scoring="balanced_accuracy",
                        n_jobs=1
                    )
                    return scores_shuf['test_score'].mean()

                shuffled_scores = np.array(
                    Parallel(n_jobs=-1)(
                        delayed(_shuffle_score)(y_shuf) for y_shuf in shuffled_labels_all
                    )
                )

                output['results_' + condition].append({
                    'bin': (idx_bin, idx_bin + 1),
                    'scores': results_bin,
                    'session_ID': session_id
                })

                output['results_shuffled_' + condition].append({
                    'bin': (idx_bin, idx_bin + 1),
                    'shuffled_scores': shuffled_scores,  # shape (N_SHUFFLES,)
                    'session_ID': session_id
                })

    save_dir = os.path.join(output_dir)
    ensure_dir(save_dir)
    np.save(os.path.join(save_dir, 'output.npy'), output)
    shutil.copy(__file__, os.path.join(save_dir, 'script.py'))


if __name__ == "__main__":
    main()