### call the features I want to compute
import argparse
import time
import numpy as np
import pandas as pd
import os
import json

import mtbi_detection.data.data_utils as du
import mtbi_detection.data.load_dataset as ld
import mtbi_detection.data.load_open_closed_data as locd
import mtbi_detection.data.transform_data as td
import mtbi_detection.features.feature_utils as fu

import mtbi_detection.features.compute_psd_features as cpf
import mtbi_detection.features.compute_maximal_power_features as cmpf
import mtbi_detection.features.compute_spectral_edge_frequencies as csef
import mtbi_detection.features.compute_complexity_features as ccf
import mtbi_detection.features.compute_psd_complexity_features as cpcf
import mtbi_detection.features.compute_network_features as cnf
# import src.features.parameterize_spectra as psf
# import src.models.estimate_probabilities_from_symptoms as epfs


CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')
TDPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'psd_transform')

roi_dict = {
    'Temporal': ['F7', 'F8', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',  'Fz'],
    'Anterior': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'T1', 'T2'],
    'Temporocentral': ['C3', 'C4', 'Cz', 'T3', 'T4'],
    'Occiparietal': ['P3', 'P4', 'Pz', 'O1', 'O2'],
    'Parietal': ['P3', 'P4', 'Pz'],
    'Posterior': ['P3', 'P4', 'Pz', 'O1', 'O2', 'T5', 'T6'],
    'Centroposterior': ['P3', 'P4', 'Pz', 'O1', 'O2', 'T5', 'T6', 'C3', 'C4', 'Cz'],
    'Central': ['C3', 'C4', 'Cz'],
    'Midline': ['Fz', 'Cz', 'Pz'],
    'Left': ['Fp1', 'F3', 'F7', 'T1', 'T3', 'C3', 'T5', 'P3', 'O1'],
    'Right': ['Fp2', 'F4', 'F8', 'T2', 'T4', 'T6', 'C4', 'T6', 'P4', 'O2'],
    'LTemporocentral': ['F7', 'T1', 'T3', 'T5'],
    'RTemporocentral': ['F8', 'T2', 'T4', 'T6'],
    'LAnterior': ['Fp1', 'F3', 'F7', 'T1'],
    'RAnterior': ['Fp2', 'F4', 'F8', 'T2'],
    'LCentroposterior': ['T3', 'T5', 'C3', 'P3', 'O1'],
    'RCentroposterior': ['T4', 'T6', 'C4', 'P4', 'O2'],
    'LPosterior': ['T5', 'P3', 'O1'],
    'RPosterior': ['T6', 'P4', 'O2'],
    'Full': CHANNELS
}

def main(open_closed_path=LOCD_DATAPATH, tables_folder='data/tables/', internal_folder='data/internal/', use_regional=False, return_separate=False, remove_noise=True, choose_subjs='train', **kwargs):
    
    if open_closed_path[-1] == '/':
        base_savepath = open_closed_path[:-1]
    else:
        base_savepath = open_closed_path
    savepath = os.path.dirname(os.path.dirname(base_savepath))
    savepath = os.path.join(savepath, 'features')

    starttime = time.time()
    locd_params = extract_locd_params(**kwargs)


    # load the data
    print("Loading data")
    loadtime = time.time()
    open_closed_dict = locd.load_open_closed_pathdict(**locd_params, savepath=open_closed_path)
    
    # transform the data
    td_params = extract_transform_params(**kwargs)
    transform_data_dict = td.main(locd_params=locd_params, locd_savepath=open_closed_path, n_jobs=kwargs['n_jobs'], as_paths=True, **td_params)
    
    subjs = [str(s) for s in transform_data_dict['subj_data'] if str(s).isnumeric()]
    select_subjs = ld.load_splits(internal_folder)[choose_subjs]
    subjs = [s for s in subjs if int(s) in select_subjs]
    channels = transform_data_dict['channels']

    print(f"My channels: {channels}")
    print(f"Time to load data: {time.time()-loadtime} seconds")
    du.clean_params_path(savepath)
    featurepath, _ = du.check_and_make_params_folder(savepath, {**locd_params, **td_params}, make_new_paramdir=True, save_early=True)
    
    # psd features
    print("Computing PSD powers and ratios")
    psdtime = time.time()
    psd_params = extract_power_params(**kwargs)
    band_powers = cpf.compute_psd_features(transform_data_dict, choose_subjs=choose_subjs, ratio=False, channels=channels, state='all', save=True, featurepath=featurepath, **psd_params)
    band_ratios = cpf.compute_psd_features(transform_data_dict, choose_subjs=choose_subjs, ratio=True, channels=channels, state='all', save=True, featurepath=featurepath, **psd_params)
    print(f"Finished computing PSD features in {time.time()-psdtime} seconds, band_powers shape: {band_powers.shape}, band_ratios shape: {band_ratios.shape}")

    # maximal power features
    print("Computing Maximal Power PSD features")
    maximtime = time.time()
    maximal_power_params = {'power_increment': kwargs['power_increment'], 'num_powers': kwargs['num_powers'], 'percentile_edge_method': kwargs['percentile_edge_method'], 'choose_subjs': choose_subjs}
    maximal_power_df = cmpf.main(transform_data_dict=transform_data_dict, save=True, featurepath=featurepath, internal_folder=internal_folder, **maximal_power_params)
    print(f"Finished computing maximal power features in {time.time()-maximtime} seconds, feature array shape: {maximal_power_df.shape}")

    # sef features
    print("Computing spectral edge features")
    seftime = time.time()
    sef_params = {'edge_increment': kwargs['edge_increment'], 'num_edges': kwargs['num_edges'], 'log_edges':kwargs['log_edges'], 'reverse_log': kwargs['reverse_log'], 'spectral_edge_method': kwargs['spectral_edge_method'], 'choose_subjs': choose_subjs}
    sef_df = csef.main(transform_data_dict, channels=channels, save=True, featurepath=featurepath, internal_folder=internal_folder, **sef_params)
    print(f"Finished computing spectral edge features in {time.time()-seftime} seconds, feature array shape: {sef_df.shape}")

    # complexity features
    print("Computing complexity features")
    complexityt = time.time()
    complexity_params = {'window_len': kwargs['window_len'], 'overlap': kwargs['overlap'], 'choose_subjs': choose_subjs}
    complexity_feature_df = ccf.main(open_closed_params=locd_params, channels=channels, save=True, featurepath=featurepath, verbosity=kwargs['verbosity'], internal_folder=internal_folder, **complexity_params)
    print(f"Finished computing complexity features in {time.time()-complexityt} seconds, feature array shape: {complexity_feature_df.shape}")

    # psd complexity features
    print("Computing PSD complexity features")
    complexityp = time.time()
    psd_complexity_feature_df = cpcf.main(locd_params, td_params, channels, open_closed_path=open_closed_path, choose_subjs='train', featurepath=featurepath, internal_folder=internal_folder, verbosity=kwargs['verbosity'], save=True)
    print(f"Finished computing PSD complexity features in {time.time()-complexityp} seconds, feature array shape: {psd_complexity_feature_df.shape}")

    # compute the network features
    print("Computing network features")
    networkt = time.time()
    network_params = extract_network_params(**kwargs)
    network_feature_df = cnf.main(network_params=network_params, data_params=locd_params, open_closed_dict=open_closed_dict, channels=channels, save=True, savepath=featurepath)
    print(f"Finished computing network features in {time.time()-networkt} seconds, feature array shape: {network_feature_df.shape}")        

    #     print("Computing Parameterized Spectra features")
    #     ps_params = extract_parameterized_spectra_params(**kwargs)
    #     ps_path = '/shared/roy/mTBI/mTBI_Classification/feature_csvs/param_spectra/'
    #     if os.path.exists('/scratch/ap60/mTBI/shared_copies/feature_csvs/param_spectra/'):
    #         ps_path = '/scratch/ap60/mTBI/shared_copies/feature_csvs/param_spectra/'
    #     parameterized_psd = psf.main(savepath=ps_path, base_folder=base_folder, **ps_params)
    #     if remove_noise:
    #         parameterized_psd = parameterized_psd[[col for col in parameterized_psd.columns if 'oise' not in col]]
    #     parameterized_psd.columns = [f'parameterized_{col}' for col in parameterized_psd.columns]
    # else:
    #     parameterized_psd = None

    ## ecg features
    # if kwargs['use_ecg']:
    # baseline_path =   '/shared/roy/mTBI/saved_processed_data/mission_connect/processed_ecg_data_features/'
    # ecg_baseline_file = os.path.join(baseline_path, 'ecg_baseline_12.6.23.csv')
    # ecg_closed_file = os.path.join(baseline_path, 'eyes_closed_12.6.23.csv')
    # ecg_open_file = os.path.join(baseline_path, 'eyes_open_12.6.23.csv')
    # ecg_baseline_df = pd.read_csv(ecg_baseline_file, index_col=0)
    # ecg_closed_df = pd.read_csv(ecg_closed_file, index_col=0)
    # ecg_open_df = pd.read_csv(ecg_open_file, index_col=0)
    # ecg_baseline_df.columns = [f'ecg_5min_{col}' for col in ecg_baseline_df.columns]
    # ecg_closed_df.columns = [f'ecg_closed_{col}' for col in ecg_closed_df.columns]
    # ecg_open_df.columns = [f'ecg_open_{col}' for col in ecg_open_df.columns]
    # ecg_df = pd.concat([ecg_baseline_df, ecg_closed_df, ecg_open_df], axis=1)
    # if kwargs['use_symptoms']:
    #     symptoms_df = epfs.process_symptoms(symptoms_only=kwargs['symptoms_only'])
    #     symptoms_df.columns = [f'symptoms_{col}' for col in symptoms_df.columns]
    #     symptoms_df.loc[[s for s in symptoms_df.index if int(s) in subjs or str(s) in subjs]]
    # else:
    #     symptoms_df = None

    # ## bispectrum features if I want
    # # print("Computing bispectrum features")
    # # make the full feature dataframe
    # out_dfs = {
    #     'band_powers': band_powers,
    #     'band_ratios': band_ratios,
    #     'regional_psd': merged_regional_psd_df,
    #     'maximal_power': maximal_power_df,
    #     'sef': sef_df,
    #     'complexity': complexity_feature_df,
    #     'psd_complexity': psd_complexity_feature_df,
    #     'network': network_feature_df,
    #     'parameterized_psd': parameterized_psd,
    #     'ecg': ecg_df,
    #     'symptoms': symptoms_df
    # }
    # for key, df in out_dfs.items():
    #     valid_subjs = epfs.load_raw_symptoms_dfs('gcs')['SubjectIDNum'].values.astype(int)
    #     if df is not None:
    #         assert all([int(s) in valid_subjs for s in df.index]), f"Subjects in {key} not in valid subjects"
    # if kwargs['load_to'] is not None:
    #     print(f"Loading to {kwargs['load_to']} subjects from the annotations")
    #     annotations_df = ld.load_annotations(base_folder=base_folder)
    #     load_subjs = annotations_df['Study ID'].iloc[:kwargs['load_to']].values
    #     for key, df in out_dfs.items():
    #         if df is not None:
    #             out_dfs[key] = df.loc[[s for s in df.index if int(s) in load_subjs]]

    # if kwargs['load_from'] is not None:
    #     print(f"Loading from {kwargs['load_from']} subjects from the annotations")
    #     annotations_df = ld.load_annotations(base_folder=base_folder)
    #     load_subjs = annotations_df['Study ID'].iloc[kwargs['load_from']:].values
    #     for key, df in out_dfs.items():
    #         if df is not None:
    #             out_dfs[key] = df.loc[[s for s in df.index if int(s) in load_subjs]]

    
    # if return_separate:
    #     print(f"Made separate dataframes in {time.time()-starttime} seconds")
    #     return out_dfs
    # else:
    #     full_feature_df = pd.concat([df for df in out_dfs.values() if df is not None], axis=1)
    #     # if use_regional:
    #     #     full_feature_df = pd.concat([band_powers, band_ratios, merged_regional_psd_df, maximal_power_df, sef_df, complexity_feature_df, psd_complexity_feature_df, network_feature_df], axis=1)
    #     # else:
    #     #     full_feature_df = pd.concat([band_powers, band_ratios, maximal_power_df, sef_df, complexity_feature_df, psd_complexity_feature_df, network_feature_df], axis=1)
    #     print(f"Made full dataframe {full_feature_df.shape} in {time.time()-starttime} seconds")
        
    #     return full_feature_df
    
def load_full_feature_df(savepath='/shared/roy/mTBI/saved_processed_data/mission_connect/all_features', **kwargs):
    full_feature_df = main(savepath=savepath, **kwargs)
    return full_feature_df

# def retrieve_band_powers(open_power, closed_power, freqs, channels, **kwargs):

#     method = (band_method, n_divisions, log_division, bin_method)
#     X_powers, feature_cols, subjs, y = cpf.load_psd_feature_from_method(method=method, when_log=when_log, ratio=False, flatten=True, channels=channels, concat=True)
    
#     X_df = pd.DataFrame(X_powers, columns=feature_cols, index=subjs)
#     assert y==fu.get_y_from_df(X_df)

#     return X_df
    

def extract_transform_params(**kwargs):
    params = {
        'interpolate_spectrum': kwargs['interpolate_spectrum'],
        'freq_interp_method': kwargs['freq_interp_method'],
        'bandwidth': kwargs['bandwidth'],
        'which_segment': 'avg'
    }
    return params

def extract_locd_params(**kwargs):
    
    params = {
        'l_freq': kwargs['l_freq'],
        'h_freq': kwargs['h_freq'],
        'fs_baseline': kwargs['fs_baseline'],
        'order': kwargs['order'],
        'notches': kwargs['notches'],
        'notch_width': kwargs['notch_width'],
        'num_subjs': kwargs['num_subjs'],
        'verbose': kwargs['verbose'],
        'reference_method': kwargs['reference_method'],
        'reference_channels': kwargs['reference_channels'],
        'keep_refs': kwargs['keep_refs'],
        'bad_channels': kwargs['bad_channels'],
        'filter_ecg': kwargs['filter_ecg'],
        'ecg_l_freq': kwargs['ecg_l_freq'],
        'ecg_h_freq': kwargs['ecg_h_freq'],
        'ecg_thresh': kwargs['ecg_thresh'],
        'ecg_method': kwargs['ecg_method'],

    }
    return params
def extract_power_params(**kwargs):

    params = {
        'band_method': kwargs['band_method'],
        'bin_methods': kwargs['bin_methods'],
        'band_method': kwargs['band_method'],
    }

    return params

def extract_regional_power_params(**kwargs):
    
        params = {
            'band_method': kwargs['regional_band_method'],
            'n_divisions': kwargs['regional_n_divisions'],
            'log_division': kwargs['regional_log_division'],
            'n_subjs': kwargs['num_subjs'],
        }
        return params

def extract_network_params(**kwargs):

    network_params = {
        'band_method': kwargs['network_band_method'],
        'network_methods': kwargs['network_methods'],
    }
    return network_params

def normalize_transform_dataset(transform_data_dict):


    try:
        channels = transform_data_dict['channels']
    except:
        channels = CHANNELS

    subjs = transform_data_dict['open_subjs']
    assert subjs == transform_data_dict['closed_subjs']

    freqs = transform_data_dict['open_freqs']
    assert np.allclose(freqs, transform_data_dict['closed_freqs'])
    assert all([np.allclose(freqs[0], f) for f in transform_data_dict['open_freqs']])
    assert all([np.allclose(freqs[0], f) for f in transform_data_dict['closed_freqs']])
    freqs = freqs[0]

    open_power = transform_data_dict['open_power']
    closed_power = transform_data_dict['closed_power']
    # open_phase = transform_data_dict['open_phase']
    # closed_phase = transform_data_dict['closed_phase']

    stacked_open_power = np.stack(open_power)
    stacked_closed_power = np.stack(closed_power)
    # stacked_open_phase = np.stack(open_phase)
    # stacked_closed_phase = np.stack(closed_phase)
    log_stacked_open_power = np.log10(stacked_open_power)
    log_stacked_closed_power = np.log10(stacked_closed_power)

    robust_psd_scaler = du.RobustChannelScaler(feature_channels=None, sample_axis=None)
    scaled_lscp = robust_psd_scaler.fit_transform(log_stacked_closed_power)
    scaled_lsop = robust_psd_scaler.transform(log_stacked_open_power)

    scaled_dict = {
        'center': robust_psd_scaler.center_,
        'scale': robust_psd_scaler.scale_,
        'scaled_open_power': scaled_lsop,
        'scaled_closed_power': scaled_lscp,
        'channels': channels,
        'subjs': subjs,
        'freqs': freqs
    }

    return scaled_dict

def check_if_same_psd_method(**kwargs):
    same_psd_method = False
    if kwargs['ratio_band_method'] == kwargs['band_method']:
        if kwargs['ratio_bin_method'] == kwargs['bin_method']:
            if kwargs['ratio_n_divisions'] == kwargs['n_divisions']:
                if kwargs['ratio_log_division'] == kwargs['log_division']:
                    if 'interbin_ratios' in kwargs.keys() and 'ratio_interbin_ratios' in kwargs.keys():
                        if kwargs['ratio_interbin_ratios'] == kwargs['interbin_ratios']:
                            same_psd_method = True
                    else:
                        same_psd_method = True
    return same_psd_method
       
def extract_parameterized_spectra_params(**kwargs):
    params={
        'bands': kwargs['ps_bands'],
        'max_n_peaks': kwargs['max_n_peaks'],
        'min_peak_height': kwargs['min_peak_height'],
        'peak_threshold': kwargs['peak_threshold'],
        'aperiodic_mode': kwargs['aperiodic_mode'],
        'n_division': kwargs['ps_n_division'],
        'log_freqs': kwargs['ps_log_freqs'],
        'prominence': kwargs['prominence'],
        'loadpath': kwargs['ps_loadpath'],
        'load_from': kwargs['load_from'],
        'load_to': kwargs['load_to']
    }
    return params
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## LOCD params
    parser.add_argument('--l_freq', type=float, default=0.3)
    parser.add_argument('--h_freq', type=float, default=None)
    parser.add_argument('--fs_baseline', type=float, default=500)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--notches', type=int, nargs='+', default=[60, 120, 180, 240])
    parser.add_argument('--notch_width', type=float, nargs='+', default=[2, 1, 0.5, 0.25])
    parser.add_argument('--num_subjs', type=int, default=151)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reference_method', type=str, default='CSD')
    parser.add_argument('--reference_channels', type=str, nargs='+', default=['A1', 'A2'])
    parser.add_argument('--keep_refs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bad_channels', type=str, nargs='+', default=['T1', 'T2'])
    parser.add_argument('--filter_ecg', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--late_filter_ecg', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ecg_l_freq', type=float, default=8)
    parser.add_argument('--ecg_h_freq', type=float, default=16)
    parser.add_argument('--ecg_thresh', type=str, default='auto')
    parser.add_argument('--ecg_method', type=str, default='correlation')
    parser.add_argument('--include_ecg', action=argparse.BooleanOptionalAction, default=True)

    ## td_params
    parser.add_argument('--interpolate_spectrum', type=int, default=1000)
    parser.add_argument('--freq_interp_method', type=str, default='linear', choices=['linear', 'log', 'log10'])
    parser.add_argument('--which_segment', type=str, default='avg', choices=['first', 'second', 'avg'], help='Which segment to use for the multitaper')
    parser.add_argument('--bandwidth', type=float, default=1)
    parser.add_argument('--n_jobs', type=int, default=1)

    ## psd params
    parser.add_argument('--band_method', type=str, default='custom', help="Possible options: 'standard', 'log-standard', 'custom', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--bin_methods', type=str, nargs='+', default=['all'], help="evaluated multiple bin_methods: ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p75', 'p90', 'p95', 'iqr']")

    ## regional psd params
    parser.add_argument('--regional_band_method', type=str, default='standard', help="Possible options: 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--regional_n_divisions', type=int, default=1, help="Number of divisions to make in the frequency band: 1,2,3,4,5 for all except the linear_50+bands")
    parser.add_argument('--regional_log_division', action=argparse.BooleanOptionalAction, default=True, help="Whether to use log division for the frequency bands")
    parser.add_argument('--regional_bin_method', type=str, nargs='+', default=['all'], help="evaluated multiple bin_methods: ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p25', 'p75', 'p95', 'iqr'] or 'pX' for Xth percentile")
    parser.add_argument('--use_regional', action=argparse.BooleanOptionalAction, default=False, help="Whether to use regional PSD features")
    
    ## maximal power params
    parser.add_argument('--power_increment', type=float, default=None, help="The increments to find the maximal power in the psd")
    parser.add_argument('--num_powers', type=int, default=20, help="The number of maximal powers to find in the psd")
    parser.add_argument('--percentile_edge_method', type=str, default='custom', choices=['custom', 'automated'], help="The method to find the spectral edge")
    
    ## spectral edge params
    parser.add_argument('--edge_increment', type=float, default=0.1, help="The increment to find the spectral edge")
    parser.add_argument('--num_edges', type=int, default=20, help="The number of spectral edges to find")
    parser.add_argument('--log_edges', action=argparse.BooleanOptionalAction, default=True, help="Whether to log the edges")
    parser.add_argument('--reverse_log', action=argparse.BooleanOptionalAction, default=False, help="Whether to reverse the log")
    parser.add_argument('--spectral_edge_method', type=str, default='custom', choices=['custom', 'automated', 'manual'], help="The method to find the spectral edge")
    
    ## complexity params
    parser.add_argument('--window_len', type=int, default=10, help="The window length for the complexity features")
    parser.add_argument('--overlap', type=float, default=1, help="The overlap for the complexity features")

    ## network params
    parser.add_argument('--network_band_method', type=str, default='custom', help="Possible options: 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--network_methods', type=str, nargs='+', default=['coherence', 'mutual_information', 'spearman', 'pearson', 'plv', 'pli']) #  'plv', 'pli'

    ## parameterized spectra
    parser.add_argument('--use_ps', action=argparse.BooleanOptionalAction, default=True, help='Whether to use parameterized spectra features')
    parser.add_argument('--ps_savepath', type=str, default='/shared/roy/mTBI/mTBI_Classification/feature_csvs/param_spectra/', help='Path to save the data to')
    parser.add_argument('--ps_bands', type=str, default='mine', help='Bands to fit the gaussians to, e.g. [(0, 4), (5, 10)]')
    parser.add_argument('--max_n_peaks', type=int, default=5, help='Maximum number of peaks to fit')
    parser.add_argument('--min_peak_height', type=float, default=0.0, help='Minimum peak height to fit')
    parser.add_argument('--peak_threshold', type=float, default=2.0, help='Peak threshold to fit')
    parser.add_argument('--aperiodic_mode', type=str, default='knee', help='Aperiodic mode to fit')
    parser.add_argument('--ps_n_division', type=int, default=1, help='Number of divisions to fit')
    parser.add_argument('--ps_log_freqs', action=argparse.BooleanOptionalAction, default=True, help='Whether to log (base e) the frequencies')
    parser.add_argument('--prominence', type=float, default=0.5, help='Prominence to fit')
    parser.add_argument('--ps_loadpath', type=str, default='/shared/roy/mTBI/data_transforms/loaded_transform_data/params/params5/', help='Path to load the data from') # also at scratch params7 # no pad many data


    ## ecg features
    parser.add_argument('--use_ecg', action=argparse.BooleanOptionalAction, default=False, help='Whether to use ecg features')
    ##
    parser.add_argument('--use_symptoms', action=argparse.BooleanOptionalAction, default=False, help='Whether to use symptoms features')
    parser.add_argument('--symptoms_only', action=argparse.BooleanOptionalAction, default=True, help='Whether to use symptoms features')
    
    ## general params
    parser.add_argument('--choose_subjs', type=str, default='train', help='Which subjects to choose from the data')
    parser.add_argument('--verbosity', type=int, default=1, help="The verbosity for the complexity features")
    
    args = parser.parse_args()
    print(args)
    uin = input("Continue? (y/n)")
    if uin == 'y':
        main(**vars(args))
    else:
        print("Exiting")


### NOTES:
## An obvious limitation of this is that the feature sets are recomputed for each unique change in the arguments
## even though changing the window_len, for example, doesn't change the ratio psd calculation
## because i have lots of options for psd computation, it may get redundant quickly
## the work around is to let those feature functions store it elsewhere... but then can't directly pass the data dict (reload each time) 
## fair trade off but i got this far so i'll just keep the redundancy...