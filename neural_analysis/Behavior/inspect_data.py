from neural_analysis import utils
import numpy as np
import config

# - Load raw data - #
trialData, sessionData, spikeData = utils.load_data(
    path_root=config.DATA_DIR_EXP,
    experiment=['ALM_3', 'ALM_2', 'ALM_1'],
    animals=None,
    data_types=['trialData', 'sessionData', 'spikeData'])

print(f"Total animals: {len(sessionData['mousename'].unique())}")
print(f"Total sessions: {len(sessionData)}")

trialData, sessionData, spikeData, removed_sessions = utils.filter_sessions_hitrate(trialData, sessionData, spikeData,
                                                                                    min_hitrate=0.3)
print(f"Total sessions after filtering: {len(sessionData)}")

spikeData = spikeData[(spikeData['group'] == 'good') & (spikeData['fr'] > 0.5)]

session_dict = {
    'session_ID': [],
    'n_trials': [],
    'animal': [],
    'n_neurons_lALM': [],
    'n_neurons_rALM': [],
    'n_neurons_V1': [],
    'experiment': [],
    'opto_type': []
}

for animal in sessionData['mousename'].unique():
    print(animal)

    animal_sessions = sessionData[sessionData['mousename'] == animal]['session_ID']

    for session in animal_sessions:
        n_trials_ses = len(trialData[trialData['session_ID'] == session])
        n_neurons_lALM_ses = len(spikeData[(spikeData['session_ID'] == session) & (spikeData['area'] == 'lALM')])
        n_neurons_rALM_ses = len(spikeData[(spikeData['session_ID'] == session) & (spikeData['area'] == 'rALM')])
        n_neurons_V1_ses = len(spikeData[(spikeData['session_ID'] == session) & (spikeData['area'] == 'rV1')])
        experiment = sessionData[sessionData['session_ID'] == session]['experiment'].iloc[0]
        opto_type = str(trialData[trialData['session_ID'] == session]['optoLat'].unique()[0])

        print(f"{session} - n_trials = {n_trials_ses}"
              f" - n_neurons_lALM = {n_neurons_lALM_ses}"
              f" - n_neurons_rALM = {n_neurons_rALM_ses}"
              f" - n_neurons_V1 = {n_neurons_V1_ses}\n")

        session_dict['session_ID'].append(session)
        session_dict['n_trials'].append(n_trials_ses)
        session_dict['n_neurons_lALM'].append(n_neurons_lALM_ses)
        session_dict['n_neurons_rALM'].append(n_neurons_rALM_ses)
        session_dict['n_neurons_V1'].append(n_neurons_V1_ses)
        session_dict['animal'].append(animal)
        session_dict['experiment'].append(experiment)
        session_dict['opto_type'].append(opto_type)

for key in session_dict:
    session_dict[key] = np.array(session_dict[key])

total_neurons_lALM = np.sum(session_dict['n_neurons_lALM'])
total_neurons_rALM = np.sum(session_dict['n_neurons_rALM'])
total_neurons_ALM = np.sum((session_dict['n_neurons_rALM'] + session_dict['n_neurons_lALM']))
total_neurons_V1 = np.sum(session_dict['n_neurons_V1'])

mean_neurons_lALM = np.mean(session_dict['n_neurons_lALM'][session_dict['n_neurons_lALM'] > 0])
mean_neurons_rALM = np.mean(session_dict['n_neurons_rALM'][session_dict['n_neurons_rALM'] > 0])
mean_neurons_ALM = np.mean((session_dict['n_neurons_rALM'] + session_dict['n_neurons_lALM'])[(session_dict['n_neurons_rALM'] + session_dict['n_neurons_lALM']) > 0])
mean_neurons_V1 = np.mean(session_dict['n_neurons_rALM'][session_dict['n_neurons_V1'] > 0])

std_neurons_lALM = np.std(session_dict['n_neurons_lALM'][session_dict['n_neurons_lALM'] > 0])
std_neurons_rALM = np.std(session_dict['n_neurons_rALM'][session_dict['n_neurons_rALM'] > 0])
std_neurons_ALM = np.std((session_dict['n_neurons_rALM'] + session_dict['n_neurons_lALM']))
std_neurons_V1 = np.std(session_dict['n_neurons_rALM'][session_dict['n_neurons_V1'] > 0])

print(f"Number of lALM neurons - Total: {total_neurons_lALM} - Mean per session: {mean_neurons_lALM} +- {std_neurons_lALM} (std)")
print(f"Number of rALM neurons - Total: {total_neurons_rALM} - Mean per session:  {mean_neurons_rALM} +- {std_neurons_rALM} (std)")
print(f"Number of combined ALM neurons - Total: {total_neurons_ALM} - Mean per session:  {mean_neurons_ALM} +- {std_neurons_ALM} (std)")
print(f"Number of V1 neurons - Total: {total_neurons_V1} - Mean per session:  {mean_neurons_V1} +- {std_neurons_V1} (std)\n")

n_sessions_lALM = len(session_dict['n_neurons_lALM'][session_dict['n_neurons_lALM'] > 0])
n_sessions_rALM = len(session_dict['n_neurons_rALM'][session_dict['n_neurons_rALM'] > 0])
n_sessions_V1 = len(session_dict['n_neurons_V1'][session_dict['n_neurons_V1'] > 0])

n_sessions_lALM_valid_opto = len(session_dict['n_neurons_lALM'][(session_dict['n_neurons_lALM'] > 0) & (np.isin(session_dict['experiment'], ['ALM_2', 'ALM_3']))])
n_sessions_rALM_valid_opto = len(session_dict['n_neurons_rALM'][(session_dict['n_neurons_rALM'] > 0) & (np.isin(session_dict['experiment'], ['ALM_2', 'ALM_3']))])
n_sessions_V1_valid_opto = len(session_dict['n_neurons_V1'][(session_dict['n_neurons_V1'] > 0) & (np.isin(session_dict['experiment'], ['ALM_2', 'ALM_3']))])
n_sessions_ALM_bilateral = len(session_dict['n_neurons_lALM'][(session_dict['n_neurons_lALM'] > 0) & (session_dict['n_neurons_rALM'] > 0)])

print(f"Number of lALM sessions: {n_sessions_lALM}")
print(f"Number of rALM sessions: {n_sessions_rALM}")
print(f"Number of V1 sessions: {n_sessions_V1}")
print(f"Number of bilateral ALM sessions: {n_sessions_ALM_bilateral}")

print(f"Number of lALM sessions with valid opto: {n_sessions_lALM_valid_opto}")
print(f"Number of rALM sessions with valid opto: {n_sessions_rALM_valid_opto}")
print(f"Number of V1 sessions with valid opto: {n_sessions_V1_valid_opto}")

sessions_unilat_bool = session_dict['opto_type'] == 'unilat'
sessions_bilat_bool = session_dict['opto_type'] == 'bilat'

sessions_unilat_valid_opto_bool = (np.isin(session_dict['experiment'], ['ALM_2', 'ALM_3'])) & sessions_unilat_bool
sessions_bilat_valid_opto_bool = (np.isin(session_dict['experiment'], ['ALM_2', 'ALM_3'])) & sessions_bilat_bool

print(f"Number of unilateral opto sessions: {np.sum(sessions_unilat_bool)}")
print(f"Number of bilateral opto sessions: {np.sum(sessions_bilat_bool)}")

print(f"Number of unilateral sessions with valid opto: {np.sum(sessions_unilat_valid_opto_bool)}- "
      f"N_neurons_lALM: {np.sum(session_dict['n_neurons_lALM'][sessions_unilat_valid_opto_bool])}- "
      f"N_neurons_rALM: {np.sum(session_dict['n_neurons_rALM'][sessions_unilat_valid_opto_bool])}")
print(f"Number of bilateral sessions with valid opto: {np.sum(sessions_bilat_valid_opto_bool)}- "
      f"N_neurons_lALM: {np.sum(session_dict['n_neurons_lALM'][sessions_bilat_valid_opto_bool])}- "
      f"N_neurons_rALM: {np.sum(session_dict['n_neurons_rALM'][sessions_bilat_valid_opto_bool])}")

print("End of script")