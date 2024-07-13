### a general framework to implement and evaluate existing methodologies for detecting mTBI
### Method A: McNerney (Select Symptoms + Band Powers)
### Method C: Lewine et al 2019 (Band Power Ratios + Connectivity),
### Method B: Thanjavur 2021 (Raw LSTM), 


import numpy as np
import json
import pandas as pd
import sklearn
import mne
import mne_connectivity
import os
import pickle

import mtbi_detection.data.load_dataset as ld
import mtbi_detection.data.data_utils as du
import mtbi_detection.features.feature_utils as fu
import mtbi_detection.data.load_symptoms as ls
import mtbi_detection.modeling.model_utils as mu

DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')
TDPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'psd_transform')
OTHERRESULTS_SAVEPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'results', 'other_methods')

def load_data(internal_folder='data/internal/', methodology='METHODA'):
    """
    This will load the data for the given methodology
    Inputs:
        internal_folder: str, the folder where the split information is stored
        methodology: str, the methodology to use
    Outputs:
        dev_datapath_dict:
            {subj: [list of mne fif filepaths]}
        ival_datapath_dict:
            {subj: [list of mne fif filepaths]}
        holdout_datapath_dict:
            {subj: [list of mne fif filepaths]}
    """
    split_subjs = ld.load_splits(base_folder=internal_folder)

def get_methodology(methodology='lewine'):
    """
    Returns the classifier pipeline for the given methodology
    Inputs:
        methodology: str, the methodology to use
    Outputs:
        methdology_classifier: a tuple that contains a (data preprocessing block, classifier)
        The preprocessing block must take in the dictionary of data and return something that the classifier can use
    """
    preprocessor = None
    classifier = None
    return (preprocessor, classifier)

def get_cv_results(data, methodology_classifier, cv=5):
    """
    Get the cross validation results for the given data and methodology
    Inputs:
        data: dict, {subj: [list of mne fif filepaths]}
        methodology_classifier: a tuple that contains a (data preprocessing block, classifier)
        cv: int or sklearn cross validation object
    Outputs:
        cv_results: dict, {metric: [list of scores]}
    """
    if isinstance(cv, int):
        cv = sklearn.model_selection.StratifiedKFold(n_splits=cv)
    cv_results = sklearn.model_selection.cross_validate(methodology_classifier, data, cv=cv) ## this would be nice...
    # return cv_results
    pass

def get_fitted_methodology(data, methodology_classifier):
    """
    Fit the methodology to the data
    Inputs:
        data: dict, {subj: [list of mne fif filepaths]}
        methodology_classifier: a tuple that contains a (data preprocessing block, classifier)
    Outputs:
        fitted_methodology: the fitted methodology
    """
    # fitted_methodology = methodology_classifier.fit(data)
    # return fitted_methodology
    pass

### 
### METHOD A
def eval_on_unseen(model, dev_X, dev_y, ival_X, ival_y, holdout_X, holdout_y, savepath=FEATUREPATH, name='symptoms', model_name='gradient_boosting_classifier'):
    
    model.fit(dev_X, dev_y)
    ival_score = mu.score_binary_model(model, ival_X, ival_y)
    holdout_score = mu.score_binary_model(model, holdout_X, holdout_y)
    print(f'ival_score:')
    mu.print_binary_scores(ival_score)
    print(f'holdout_score:')
    mu.print_binary_scores(holdout_score)
    with open(os.path.join(savepath, f'{model_name}_{name}_ival_score.json'), 'w') as f:
        json.dump(du.make_dict_saveable(ival_score), f)

    with open(os.path.join(savepath, f'{model_name}_{name}_holdout_score.json'), 'w') as f:
        json.dump(du.make_dict_saveable(holdout_score), f)

    with open(os.path.join(savepath, f'{model_name}_{name}_ival_score.pkl'), 'wb') as f:
        pickle.dump(ival_score, f)

    with open(os.path.join(savepath, f'{model_name}_{name}_holdout_score.pkl'), 'wb') as f:
        pickle.dump(holdout_score, f)

    unseen_X = pd.concat([ival_X, holdout_X], axis=0)
    unseen_y = fu.get_y_from_df(unseen_X)
    
    unseen_score = mu.score_binary_model(model, unseen_X, unseen_y)
    print(f'unseen_score:')
    mu.print_binary_scores(unseen_score)
    with open(os.path.join(savepath, f'{model_name}_{name}_unseen_score.json'), 'w') as f:
        json.dump(du.make_dict_saveable(mu.score_binary_model(model, unseen_X, unseen_y)), f)

    with open(os.path.join(savepath, f'{model_name}_{name}_unseen_score.pkl'), 'wb') as f:
        pickle.dump(mu.score_binary_model(model, unseen_X, unseen_y), f)

    return ival_score, holdout_score, unseen_score

def cv_score_and_save(X,y,groups, basesavepath=OTHERRESULTS_SAVEPATH, name='symptoms'):
    savepath = os.path.join(basesavepath, 'methoda')
    multi_model_methoda = mu.quick_classify_df(X, y, groups=groups,cv=5, random_state=42)
    with open(os.path.join(savepath, f'multi_model_methoda_{name}_cv.json'), 'w') as f:
        json.dump(du.make_dict_saveable(multi_model_methoda), f)
    with open(os.path.join(savepath, f'multi_model_methoda_{name}_cv.pkl'), 'wb') as f:
        pickle.dump(multi_model_methoda, f)

    total_boost_approx =  sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=1, loss='exponential')
    total_boost_cv =  mu.quick_classify_df(X, y, groups=groups,cv=5, random_state=42, models=[total_boost_approx], model_names=['GradientBoosting'])
    with open(os.path.join(savepath, f'gradient_boosting_{name}_cv.json'), 'w') as f:
        json.dump(du.make_dict_saveable(total_boost_cv), f)
    with open(os.path.join(savepath, f'gradient_boosting_{name}_cv.pkl'), 'wb') as f:
        pickle.dump(total_boost_cv, f)
    return multi_model_methoda, total_boost_cv

def methoda_fused_process_data(datapath_dict):
    """
    Given a data dictionary with format {subj: {'open': [fif_path], 'closed': [fif_path]}
    Return the Method A processed data matrices
        X, y, groups
    """
    subjs = np.array(list(datapath_dict.keys())).astype(float)
    eeg_X, _, _ = methoda_process_eeg(datapath_dict)
    symp_X, _ , _ = methoda_process_symptoms(subjs)
    scaled_eeg_features_df = eeg_X.groupby(eeg_X.index).mean()
    scaled_eeg_features_df.index = scaled_eeg_features_df.index.astype(int)
    X = pd.concat([scaled_eeg_features_df, symp_X], axis=1).dropna()
    y = fu.get_y_from_df(X)
    groups = X.index
    return X, y, groups

def methoda_process_symptoms(subjs):
    """
    Process the symptoms for the Method A dataset
    """
    all_mcnern_symp = methoda_load_symptoms()
    assert type(subjs[0]) == type(all_mcnern_symp.index[0]), f"Subjs must be of type {type(all_mcnern_symp.index[0])}, not {type(subjs[0])}"
    X = all_mcnern_symp.loc[[s for s in all_mcnern_symp.index if s in subjs]]
    y = fu.get_y_from_df(X)
    groups = X.index
    return X, y, groups

def methoda_process_eeg(datapath_dict):
    """
    Given a data dictionary with format {subj: {'open': [fif_path], 'closed': [fif_path]}
    Return the Method A processed data matrices
        X, y, groups
    """
    subjs = list(datapath_dict.keys())
    channels = ['F7', 'F8']
    closed_out_dict = _methoda_process_datapathdict(datapath_dict, state='closed', smooth=False, channels=channels)
    all_psds = closed_out_dict['psds_stack']
    psd_freqs = closed_out_dict['psd_freqs']
    psd_subjs = closed_out_dict['psds_groups']
    eeg_features_df = methoda_extract_eeg_features(all_psds, channels, psd_subjs, psd_freqs)


    X = eeg_features_df
    y = fu.get_y_from_df(X)
    groups = X.index
    return X, y, groups

def methoda_extract_eeg_features(psd_array, channels, groups, psd_freqs):
    freq_bins = {
        'delta': [1, 3],
        'theta': [3, 7],
        'alpha': [8, 13],
        'beta': [13, 30],
        'sigma': [12, 15],
        'gamma': [25, 40],
    }

    n_channels = len(channels)
    n_freqs = len(freq_bins)
    n_groups = len(groups)
    unique_groups = np.unique(groups)
    assert psd_array.shape == (n_groups, n_channels, len(psd_freqs)), f"psd_array shape: {psd_array.shape}, expected: {(n_groups, n_channels, len(psd_freqs))}"
    eeg_features = np.empty((n_groups, n_channels*n_freqs))
    for group_idx, group in enumerate(unique_groups):
        group_idx = np.where(groups  == group)[0]
        group_psd = psd_array[group_idx]
        for ch_idx in range(n_channels):
            for freq_idx, (low, high) in enumerate(freq_bins.values()):
                freq_range = (psd_freqs >= low) & (psd_freqs < high)
                freq_psd = group_psd[:, ch_idx, freq_range]
                eeg_features[group_idx, ch_idx*n_freqs + freq_idx] = np.sum(freq_psd, axis=1)

    eeg_features_df = pd.DataFrame(eeg_features, columns=[f'{ch}_{band}' for ch in channels for band in freq_bins.keys()], index=groups)
    return eeg_features_df

def methoda_load_symptoms():
    symptoms = ls.load_symptoms()
    symp_cols = ['InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal', 'Rivermead.Questionnaire.RPQHeadachesScale', 'Rivermead.Questionnaire.RPQNauseaScale',  'Rivermead.Questionnaire.RPQLightSensScale', 'Rivermead.Questionnaire.RPQNoiseSensScale', 'Rivermead.Questionnaire.RPQLongToThinkScale', 'MACE_FITBIR.Scores.MACEImmdtMemScore']

    selected_symp = symptoms[symp_cols]
    # make the memory column = 15-orignal
    selected_symp.loc[:, 'MACE_FITBIR.Scores.MACEImmdtMemScore'] = 15 - selected_symp['MACE_FITBIR.Scores.MACEImmdtMemScore']

    binary_selected_symp = selected_symp.copy()
    # now turn all non 0 values to 1
    binary_selected_symp[binary_selected_symp != 0] = 1

    kmeans_mem_df = du.sorted_cluster_labeling(selected_symp, 'MACE_FITBIR.Scores.MACEImmdtMemScore', n_clusters=7)
    kmeans_loc_df = du.sorted_cluster_labeling(selected_symp, 'InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal', n_clusters=7)

    selected_symp_scores = selected_symp.copy()
    selected_symp_scores['MACE_FITBIR.Scores.MACEImmdtMemScore'] = kmeans_mem_df['Cluster']
    selected_symp_scores['InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal'] = kmeans_loc_df['Cluster']

    new_col_names = ['LOC', 'Headache', 'Nausea', 'LightSens', 'NoiseSens', 'LongToThink', 'Memory']
    selected_symp_scores.columns = new_col_names
    binary_selected_symp.columns = new_col_names

    avg_selected_symp = selected_symp_scores.mean(axis=1)
    methoda_symptom_df = binary_selected_symp.copy()
    methoda_symptom_df['Average_Score'] = avg_selected_symp

    return methoda_symptom_df

def _methoda_process_datapathdict(datapathdict, fs_resample=256, ref_channels='Fz', channels=['F3', 'F4', 'F7', 'F8'], nfft=256, nperseg=256, noverlap=128, fmin=0.3, fmax=128, state='closed', smooth=False):
    # authors use 256, AF7, AF8 ref to Fpz, I switch it up
    raw_dict = {subj: [mne.io.read_raw_fif(path, preload=False, verbose=0) for path in paths[state]] for subj, paths in datapathdict.items()}

    processed_raws = {**raw_dict}

    # resample the raws to 256
    for subj, raws in processed_raws.items():
        for raw in raws:
            raw.resample(fs_resample)

    for subj, raws in processed_raws.items():
        for raw in raws:
            raw.set_eeg_reference(ref_channels=[ref_channels], verbose=0).pick(channels)

        
    psd_dict = {subj: [raw.copy().pick(channels).compute_psd(method='welch', n_fft=nfft, n_per_seg=nperseg, n_overlap=noverlap, fmin=fmin, fmax=fmax, verbose=0) for raw in raws] for subj, raws in processed_raws.items()}

    epochs_dict = {subj: [mne.make_fixed_length_epochs(raw, duration=1, overlap=0.5, verbose=0) for raw in raws] for subj, raws in processed_raws.items()}

    epochs_stack, epochs_groups = stack_epochs(epochs_dict, channels=channels)
    psds_stack, psds_groups = stack_psds(psd_dict, channels)
    psd_freqs = psd_dict[str(list(psd_dict.keys())[0])][0].freqs
    assert np.all([np.all(psd_freqs == psd_dict[str(subj)][0].freqs) for subj in psd_dict.keys()])


    out_dict = {'epochs_stack': epochs_stack, 'epochs_groups': epochs_groups, 'psds_stack': psds_stack, 'psds_groups': psds_groups, 'psd_freqs': psd_freqs, 'channels': channels}
    return out_dict

def stack_epochs(epochs_dict,channels):
    stack_epochs0 = []
    stack_epochs1 = []
    stack_groups0 = []
    stack_groups1 = []
    for subj in epochs_dict.keys():
        epochs = epochs_dict[str(subj)]
        if len(epochs) == 2:
            stack_epochs0.append(epochs[0])
            stack_epochs1.append(epochs[1])
            stack_groups0.extend([subj]*(epochs[0].get_data(picks=channels).shape[0]))
            stack_groups1.extend([subj]*(epochs[1].get_data(picks=channels).shape[0]))
        elif len(epochs) == 1:
            stack_epochs0.append(epochs[0])
            stack_groups0.extend([subj]*(epochs[0].get_data(picks=channels).shape[0]))
        else:
            print(f"Subject {subj} has {len(epochs)} epochs")
    stack_epochs0 = mne.concatenate_epochs(stack_epochs0)
    stack_epochs1 = mne.concatenate_epochs(stack_epochs1)
    stack_groups0 = np.array(stack_groups0)
    stack_groups1 = np.array(stack_groups1)

    print(f"Shapes of stack_epochs0: {stack_epochs0.get_data().shape}, stack_epochs1: {stack_epochs1.get_data().shape}, stack_groups0: {stack_groups0.shape}, stack_groups1: {stack_groups1.shape}")

    all_epochs = mne.concatenate_epochs([stack_epochs0, stack_epochs1])
    all_groups = np.concatenate([stack_groups0, stack_groups1])
    print(f"Shapes of all_epochs: {all_epochs.get_data().shape}, all_groups: {all_groups.shape}")
    return all_epochs, all_groups

def stack_psds(psd_dict,channels):
    # stack the psds
    all_psds = []
    all_subjs = []
    for subj, psds in psd_dict.items():
        for psd in psds:
            all_psds.append(psd.get_data(picks=channels))
            all_subjs.append(subj)

    all_psds = np.stack(all_psds)
    all_subjs = np.array(all_subjs)
    print(f"Shapes of all_psds: {all_psds.shape}, all_subjs: {all_subjs.shape}")
    return all_psds, all_subjs

### Method C: Lewine 2019 and Thatcher 1987
def pass_good_epochs(epochs):
    return epochs


def methodc_process_data(datapath_dict):
    """
    Given a data dictionary with format {subj: [fif_path]}
    Return the MethodC processed data matrices
        X, y, groups
    """

    # Define frequency bands
    freq_bands = {"delta": [1.0, 3.5],
                "theta": [4.0, 7.5],
                "alpha": [8, 12],
                "beta": [12.5, 25],
                "high beta": [25.5, 30],
                "gamma": [30, 50]}

    # Process each subject's data
    processed_raws =  {subj: [mne.io.read_raw_fif(path, preload=False, verbose=0) for path in paths['closed']] for subj, paths in datapath_dict.items()}
    channels = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3','T4', 'T5', 'T6', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    raw_features = {subj: {} for subj in processed_raws.keys()}
    for sdx, (subj, raws) in enumerate(processed_raws.items()):

        # join the raws
        epochs_list = []
        print(f"Processing subject {subj} ({sdx+1}/{len(processed_raws)})")
        for raw in raws:
            # raw = mne.concatenate_raws(raws)

            # Apply artifact rejection
            ecg_channel = ld.get_ecg_channel_locations(subjs=[subj], base_folder='../data/tables/')[0]
            raw.set_channel_types({ecg_channel: 'ecg'})
            # detect ecg using find_ecg_events
            ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_channel, verbose=False)
            # Epoch the data into 2-second-long segments
            epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.5, verbose=False) # "75 % sliding window"
            epochs = pass_good_epochs(epochs)
            epochs_list.append(epochs)
        epochs = mne.concatenate_epochs(epochs_list, verbose=False)

        # Perform FFT and compute power spectral density (PSD)
        n_fft = 512
        n_overlap = int(0.75*n_fft) # "75 % sliding window"
        psds, freqs = epochs.compute_psd(method='welch', picks=channels, fmin=0.5, fmax=55.0, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_fft, n_jobs=1, verbose=False).get_data(picks=channels, return_freqs=True, fmin=0.5, fmax=55.0)

        psds = psds.mean(axis=0)
        # print(f"PSDS shape: {psds.shape}, freqs shape: {freqs.shape}")
        # Log-transform the PSD data
        psds = 10 * np.log10(psds*1e12) + 30 # convert microvolts to picovolts and take the log and add offset
        # print(f"Max PSD: {np.max(psds)}, Min PSD: {np.min(psds)}, 25th percentile: {np.percentile(psds, 25)}, 75th percentile: {np.percentile(psds, 75)}")
        # Compute power in each frequency band at each electrode
        power = {band: np.mean(psds[:, (freqs >= fmin) & (freqs <= fmax)], axis=-1)
                for band, (fmin, fmax) in freq_bands.items()}
        total_power = np.sum(psds, axis=-1)
        relative_power = {band: power[band] / total_power for band in freq_bands.keys()}
        
        # Compute measures of functional connectivity
        coh_connectivity = methodc_coherence_connectivity(epochs, channels, freq_bands)

        power_asymmetry = methodc_power_asymmetry(psds, freqs, channels, freq_bands)

        phase_connectivity = methodc_phase_connectivity(epochs, channels, freq_bands)
  
        global_abs_power = {band: np.mean(power[band]) for band in freq_bands.keys()}
        global_rel_power = {band: np.mean(relative_power[band]) for band in freq_bands.keys()}
        global_coh = {}
        for band in freq_bands.keys():
            band_coh = coh_connectivity[band]
            # average across homotopic derivations
            coh_channels =coh_connectivity['channels']
            channel_pairs = [('Fp1', 'Fp2'), ('F7', 'F8'), ('F3', 'F4'), ('T3', 'T4'), ('C3', 'C4'), ('T5', 'T6'), ('P3','P4'), ('O1','O2')]
            channel_cohs = []
            for ch1, ch2 in channel_pairs:
                ch1dx = coh_channels.index(ch1)
                ch2dx = coh_channels.index(ch2)
                channel_cohs.append(band_coh[ch1dx, ch2dx])
            # print(f"Band: {band}, Channel coherences: {channel_cohs}")
            global_coh[band] = np.mean(channel_cohs)

        # Store the results back into the dictionary
        raw_features[subj] = {"epochs": epochs, "abs_power": power, "rel_power": relative_power, 
                                "coherence": coh_connectivity, "power_asymmetry": power_asymmetry,
                                "phase_connectivity": phase_connectivity, "global_rel_power": global_rel_power,
                                "global_abs_power": global_abs_power, "global_coh": global_coh}

    # Create the X, y, groups matrices
    all_subjs = list(processed_raws.keys())
    flattened_coherence = np.stack([np.concatenate([np.ravel(raw_features[subj]["coherence"][band], order='C') for band in freq_bands.keys()]) for subj in all_subjs])
    flattened_power_asymmetry = np.stack([np.concatenate([np.ravel(raw_features[subj]["power_asymmetry"][band], order='C') for band in freq_bands.keys()]) for subj in all_subjs])
    flattened_phase_connectivity = np.stack([np.concatenate([np.ravel(raw_features[subj]["phase_connectivity"][band], order='C') for band in freq_bands.keys()]) for subj in all_subjs])

    flattened_abs_power = np.stack([np.concatenate([raw_features[subj]["abs_power"][band] for band in freq_bands.keys()]) for subj in all_subjs])
    flattened_rel_power = np.stack([np.concatenate([raw_features[subj]["rel_power"][band] for band in freq_bands.keys()]) for subj in all_subjs])
    global_feats = np.stack([[raw_features[subj]["global_rel_power"][band] for band in freq_bands.keys()] + [raw_features[subj]["global_abs_power"][band] for band in freq_bands.keys()]+[raw_features[subj]["global_coh"][band] for band in freq_bands.keys()] for subj in all_subjs])
    print(f"Flattened coherence shape: {flattened_coherence.shape}, flattened power asymmetry shape: {flattened_power_asymmetry.shape}, flattened phase connectivity shape: {flattened_phase_connectivity.shape}")

    coh_channels = raw_features[subj]["coherence"]['channels']
    pow_channels = raw_features[subj]["power_asymmetry"]['channels']
    phase_channels = raw_features[subj]["phase_connectivity"]['channels']

    coherence_cols = [f"Coherence {band}_{ch1}_{ch2}" for band in freq_bands.keys() for ch1 in coh_channels for ch2 in coh_channels]
    power_asymmetry_cols = [f"PowAsym {band}_{ch1}_{ch2}" for band in freq_bands.keys() for ch1 in pow_channels for ch2 in pow_channels]
    phase_connectivity_cols = [f"PhaseDiff {band}_{ch1}_{ch2}" for band in freq_bands.keys() for ch1 in phase_channels for ch2 in phase_channels]
    abs_power_cols = [f"AbsPow {band}_{ch}" for band in freq_bands.keys() for ch in pow_channels]
    rel_power_cols = [f"RelPow {band}_{ch}" for band in freq_bands.keys() for ch in pow_channels]
    global_cols = [f"GlobalRelPow {band}" for band in freq_bands.keys()] + [f"GlobalAbsPow {band}" for band in freq_bands.keys()] + [f"GlobalCoherence {band}" for band in freq_bands.keys()] 

    power_asymmetry_df = pd.DataFrame(flattened_power_asymmetry, columns=power_asymmetry_cols, index=all_subjs)
    coherence_df = pd.DataFrame(flattened_coherence, columns=coherence_cols, index=all_subjs)
    phase_connectivity_df = pd.DataFrame(flattened_phase_connectivity, columns=phase_connectivity_cols, index=all_subjs)

    abs_power_df = pd.DataFrame(flattened_abs_power, columns=abs_power_cols, index=all_subjs)
    rel_power_df = pd.DataFrame(flattened_rel_power, columns=rel_power_cols, index=all_subjs)

    global_df = pd.DataFrame(global_feats, columns=global_cols, index=all_subjs)

    X = pd.concat([power_asymmetry_df, coherence_df, phase_connectivity_df, abs_power_df, rel_power_df, global_df], axis=1)
    y = fu.get_y_from_df(X)
    groups = X.index
    return X, y, groups, raw_features

def methodc_power_asymmetry(psd, freqs, channels, freq_bands):
    """
    Given a psd of shape (n_channels, n_freqs), a list of channels, and some frequency bands, 
    return the power asymmetry matrix as defined in Thatcher EEG Discrimination 1989

    Inputs:
        psd: np.array, shape (n_channels, n_freqs) ideally log-transformed
        channels: list of strings (no midline channels)
        freq_bands: dict of frequency bands e.g.
        {"delta": [1.0, 3.5],
                "theta": [4.0, 7.5],
                "alpha": [8, 12],
                "beta": [12.5, 25],
                "high beta": [25.5, 30],
                "gamma": [30, 50]}
    '( left - right/left + right) for the inter-hemisphere comparisons 
      and (anterior derivation- posterior derivation/anterior + posterior derivation)'
    From Thatcher 1987: EEG amplitude was computed as the square root of power.

    Output:
        power_asymmetry: dict of np.array, shape (n_channels, n_channels) for each frequency band
    """
    assert all([ch[-1].isdigit() for ch in channels]), "Channels must end in a number"
    assert psd.shape[0] == len(channels), f"PSD must have the same number of channels as the channels list, {psd.shape[0]} != {len(channels)}"
    assert np.all(psd >= 0), f"PSD must be log-transformed and non-negative, min: {np.min(psd)}"
    assert freq_bands.keys() == set(freq_bands.keys()), "Frequency bands must be a dict"
    power_asymmetry = {}
    # print(len(channels), channels)
    for band, (fmin, fmax) in freq_bands.items():
        power_asymmetry[band] = np.zeros((len(channels), len(channels)))
        for ch1dx, ch1 in enumerate(channels):
            for c2, ch2 in enumerate(channels[ch1dx+1:]):
                ch2dx = c2 + ch1dx + 1
                freqdx = (freqs >= fmin) & (freqs <= fmax)
                band_power_ch1 = np.sqrt(np.mean(psd[ch1dx, freqdx]))
                band_power_ch2 = np.sqrt(np.mean(psd[ch2dx, freqdx]))
                ch_pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
                ch1_pos = ch_pos[ch1]
                ch2_pos = ch_pos[ch2]
                ch1_num = int(ch1[-1])
                ch2_num = int(ch2[-1])
                if (ch1_num+ch2_num) % 2 == 0:
                    anterior_chdx = ch1dx if ch1_pos[1] > ch2_pos[1] else ch2dx
                    posterior_chdx = ch2dx if ch1_pos[1] > ch2_pos[1] else ch1dx
                    anterior_power = band_power_ch1 if ch1_pos[1] > ch2_pos[1] else band_power_ch2
                    posterior_power = band_power_ch2 if ch1_pos[1] > ch2_pos[1] else band_power_ch1
                    band_ratio = (anterior_power - posterior_power) / (anterior_power + posterior_power)
                    power_asymmetry[band][anterior_chdx, posterior_chdx] = band_ratio
                    power_asymmetry[band][posterior_chdx, anterior_chdx] = band_ratio
                elif (ch1_num+ch2_num) % 2 == 1:
                    left_power = band_power_ch1 if ch1_num % 2 == 1 else band_power_ch2
                    right_power = band_power_ch1 if ch1_num % 2 == 0 else band_power_ch2
                    band_ratio = (left_power - right_power) / (left_power + right_power)
                    power_asymmetry[band][ch1dx, ch2dx] = band_ratio
                    power_asymmetry[band][ch2dx, ch1dx] = band_ratio
                else:
                    raise ValueError(f"Something went wrong with {ch1} and {ch2}: {ch1_num} and {ch2_num}")

                # print(f"Band: {band}, chdx1: {ch1dx}, chdx2: {ch2dx} Ch1: {ch1}, Ch2: {ch2}, Ratio: {band_ratio}, Ch1 Power: {band_power_ch1}, Ch2 Power: {band_power_ch2}")
        # assert that the matrix is symmetric
        assert np.allclose(power_asymmetry[band], power_asymmetry[band].T), f"Matrix for {band} is not symmetric"
        assert np.diag(power_asymmetry[band]).all() == 0, f"Diagonal of matrix for {band} is not zero"

    power_asymmetry['channels'] = channels
    return power_asymmetry


def methodc_phase_connectivity(epochs, channels, freq_bands):
    """
    Given an mne epochs object and a dictionary of frequency bands, compute the phase connectivity matrix
    as defined in Thatcher EEG Discrimination 1989
    Inputs:
        epochs: mne epochs object
        freq_bands: dict of frequency bands e.g.
        {"delta": [1.0, 3.5],
                "theta": [4.0, 7.5],
                "alpha": [8, 12],
                "beta": [12.5, 25],
                "high beta": [25.5, 30],
                "gamma": [30, 50]}
    Outputs:
        phase_connectivity: dict of np.array, shape (n_channels, n_channels) for each frequency band
    
    The phase angle Qxy between two channels is the ratio of
    the quadspectrum to the cospectrum or Qxy arctan qxy/rxy
    which was computed in radians and transformed to degrees
    (Bendat and Piersol, 1980; Otnes and Enochson, 1972). The
    absolute phase delay in degrees was computed by squaring
    R.W. Thatcher et al. / Clinical Neurophysiology 116 (2005) 2129–2141 2131
    and then taking the square root of the phase angle or Q2xy.
    """
    assert freq_bands.keys() == set(freq_bands.keys()), "Frequency bands must be a dict"
    assert type(epochs) == mne.epochs.Epochs or type(epochs)==mne.epochs.EpochsArray, f"Epochs must be an mne epochs object, not {type(epochs)}"

    epochs_data = epochs.get_data(picks=channels)

    phase_connectivity = {}
    for band, (fmin, fmax) in freq_bands.items():
        phase_coh = np.zeros((len(channels), len(channels)))
        cohy_array = mne_connectivity.spectral_connectivity_epochs(epochs_data, method='cohy', mode='fourier', sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax, faverage=True, verbose=False).get_data('dense')
        for ch1dx, ch1 in enumerate(channels):
            for ch2dx, ch2 in enumerate(channels):
                # cohy_array = cohy.xarray.data.reshape((len(channels), len(channels)))
                if ch1dx == ch2dx:
                    continue
                subband_phase =  np.angle(cohy_array[ch1dx, ch2dx])
                # mean_phase = np.angle(np.sum(np.exp(1j*subband_phase), axis=2))
                # norm_phase = np.degrees(mean_phase)/(fmax-fmin)
                norm_phase = np.degrees(subband_phase) / (fmax-fmin)
                phase_coh[ch1dx, ch2dx] = norm_phase
        phase_connectivity[band] = phase_coh


    phase_connectivity['channels'] = channels
                
    return phase_connectivity
                

def methodc_coherence_connectivity(epochs, channels, freq_bands):
    """
    Given an mne epochs object and a dictionary of frequency bands, compute the phase connectivity matrix
    as defined in Thatcher EEG Discrimination 1989
    Inputs:
        epochs: mne epochs object
        freq_bands: dict of frequency bands e.g.
        {"delta": [1.0, 3.5],
                "theta": [4.0, 7.5],
                "alpha": [8, 12],
                "beta": [12.5, 25],
                "high beta": [25.5, 30],
                "gamma": [30, 50]}
    Outputs:
        phase_connectivity: dict of np.array, shape (n_channels, n_channels) for each frequency band
    
    The phase angle Qxy between two channels is the ratio of
    the quadspectrum to the cospectrum or Qxy arctan qxy/rxy
    which was computed in radians and transformed to degrees
    (Bendat and Piersol, 1980; Otnes and Enochson, 1972). The
    absolute phase delay in degrees was computed by squaring
    R.W. Thatcher et al. / Clinical Neurophysiology 116 (2005) 2129–2141 2131
    and then taking the square root of the phase angle or Q2xy.
    """
    assert freq_bands.keys() == set(freq_bands.keys()), "Frequency bands must be a dict"
    assert type(epochs) == mne.epochs.Epochs or type(epochs)==mne.epochs.EpochsArray, f"Epochs must be an mne epochs object, not {type(epochs)}"

    epochs_data = epochs.get_data(picks=channels)

    coh_connectivity = {}
    for band, (fmin, fmax) in freq_bands.items():
        band_coh = np.zeros((len(channels), len(channels)))
        coh_array = mne_connectivity.spectral_connectivity_epochs(epochs_data, method='coh', mode='fourier', sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax, faverage=True, verbose=False).get_data('dense')
        # print(coh.coords)
        # print(coh.xarray.data.shape)
        # print(np.tril(coh.xarray.data))
        # coh_array = coh.xarray.data.reshape((len(channels), len(channels)))
        # print(f"Band: {band}, Coh shape: {coh_array.shape}, coh_array: {coh_array}")
        for ch1dx, ch1 in enumerate(channels):
            for ch2dx, ch2 in enumerate(channels):
                if ch1dx == ch2dx:
                    continue
                subband_coh =  np.abs(coh_array[ch1dx, ch2dx])
                # print(f"Band: {band}, chdx1: {ch1dx}, chdx2: {ch2dx} Ch1: {ch1}, Ch2: {ch2}, Coherence: {subband_coh}")
                band_coh[ch1dx, ch2dx] = subband_coh

        coh_connectivity[band] = band_coh

    coh_connectivity['channels'] = channels
                
    return coh_connectivity

def main(methodology='methodc', savepath=OTHERRESULTS_SAVEPATH):

    # load up development dataset and unseen dataset
    dev_data, ival_data, holdout_data = load_data(methodology=methodology)

    # get their class which must be essentially a sklearn pipeline
    methodology_classifier = get_methodology(methodology)

    # get the performance on the development data
    dev_cv_results = get_cv_results(dev_data, methodology_classifier)

    print_and_save_cv_results(dev_cv_results, savepath=savepath, results_name=f'{methodology}_dev_cv_results.json')

    fitted_methodology = get_fitted_methodology(dev_data, methodology_classifier)

    # get the performance on the ival data
    ival_cv_results = get_test_score(ival_data, fitted_methodology)

    print_and_save_test_results(ival_cv_results, savepath=savepath, results_name=f'{methodology}_ival_cv_results.json')

    # get the performance on the holdout data
    holdout_cv_results = get_test_score(holdout_data, fitted_methodology)
                                        
    print_and_save_test_results(holdout_cv_results, savepath=savepath, results_name=f'{methodology}_holdout_cv_results.json')

    unseen_data = combine_data(ival_data, holdout_data)

    # get the performance on the unseen data
    unseen_cv_results = get_test_score(unseen_data, fitted_methodology)

    print_and_save_test_results(unseen_cv_results, savepath=savepath, results_name=f'{methodology}_unseen_cv_results.json')

    # get the perturbation scores
    perturbation_scores = get_perturbation_scores(dev_data, unseen_data, fitted_methodology)

    print_and_save_perturbation_scores(perturbation_scores, savepath=savepath, results_name=f'{methodology}_perturbation_scores.json')


    out_dict = {'dev_cv_results': dev_cv_results,
                'ival_cv_results': ival_cv_results,
                'holdout_cv_results': holdout_cv_results,
                'unseen_cv_results': unseen_cv_results,
                'perturbation_scores': perturbation_scores}

    return out_dict


