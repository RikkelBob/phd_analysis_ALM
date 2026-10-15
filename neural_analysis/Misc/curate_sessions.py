from neural_analysis import utils
import numpy as np
import config


def find_sessions_insufficient_stimuli(trialData):
    for session in trialData['session_ID'].unique():
        trialData_ses = trialData[trialData['session_ID'] == session]
        if len(np.abs(trialData_ses['visualOriChange']).unique()) < 3:
            print(f"Session {session} has fewer than 2 visual stimulus conditions")

        if len(np.abs(trialData_ses['audioOctChange']).unique()) < 3:
            print(f"Session {session} has fewer than 2 auditory stimuli conditions")


output_file = config.DATA_DIR_PP_NEW / "sessions_behavior_over_30.txt"

# - Load raw data - #
trialData, sessionData, spikeData = utils.load_data(
    path_root=config.DATA_DIR_EXP,
    experiment=['ALM_3', 'ALM_2', 'ALM_1'])

spikeData = utils.exclude_neurons(spikeData, sessionData, min_fire=0, quality='good')

# Filter sessions with hit-rate < 0.3 for either modality
(trialData_filtered,
 sessionData_filtered,
 spikeData_filtered,
 sessions_exclude) = utils.filter_sessions_hitrate(trialData=trialData,
                                                   sessionData=sessionData,
                                                   spikeData=spikeData,
                                                   min_hitrate=0.3)

find_sessions_insufficient_stimuli(trialData_filtered)

sessions_include = sessionData[~sessionData['session_ID'].isin(sessions_exclude)]['session_ID']

unique_animals = sessionData_filtered['mousename'].unique()

print(f"Total number of sessions - {len(sessionData_filtered['session_ID'].unique())}")

for animal in sessionData['mousename'].unique():
    print(f"Animal {animal} has {len(sessionData_filtered[sessionData_filtered['mousename'] == animal])} sessions")

with open(output_file, "w") as file:
    file.write("\n".join(sessions_include))