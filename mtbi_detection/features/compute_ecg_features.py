import pandas as pd
import os
import mtbi_detection.features.feature_utils as fu
ECG_FEATURE_PATH = '/scratch/ap60/mtbi_detection/ecg_features/'

def load_ecg_features(ecgfeaturepath=ECG_FEATURE_PATH, choose_subjs='train', internal_folder='data/internal/'):
    """
    Load ECG features from the ECG_FEATURE_PATH
    """

    five_min_ecg_features = pd.read_csv(os.path.join(ECG_FEATURE_PATH, 'five_min_ecg_features.csv'), index_col=0)
    open_ecg_features = pd.read_csv(os.path.join(ECG_FEATURE_PATH, 'eyes_open_ecg_features.csv'), index_col=0)
    closed_ecg_features = pd.read_csv(os.path.join(ECG_FEATURE_PATH, 'eyes_closed_ecg_features.csv'), index_col=0)
    open_ecg_features.columns = ['ECGopen_' + col for col in open_ecg_features.columns]
    closed_ecg_features.columns = ['ECGclosed_' + col for col in closed_ecg_features.columns]
    five_min_ecg_features.columns = ['ECGfive_' + col for col in five_min_ecg_features.columns]
    all_ecg_features = [open_ecg_features, closed_ecg_features] #, five_min_ecg_features]
    full_ecg_features = pd.concat(all_ecg_features, axis=1, join='inner')
    selected_full_ecg_features = fu.select_subjects_from_dataframe(full_ecg_features, choose_subjs, internal_folder=internal_folder)
    assert set(selected_full_ecg_features.index) == set(fu.select_subjects_from_dataframe(selected_full_ecg_features, choose_subjs, internal_folder=internal_folder).index)
    return selected_full_ecg_features