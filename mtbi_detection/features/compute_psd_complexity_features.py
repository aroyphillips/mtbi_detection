import numpy as np
import pandas as pd
import os
import time
import argparse
import json
import mtbi_detection.data.data_utils as du
import mtbi_detection.features.compute_complexity_features as ccf
import mtbi_detection.data.transform_data as td
import mtbi_detection.features.feature_utils as fu

LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')

def get_features_from_stack(feature_func, stacked_data, feature_metrics, subjs, channels, verbosity=1):
    """
    Given an epochs_dict, some feature function, and some metrics, compute the distribution features for each epoch
    Inputs:
        epochs_dict: dictionary of structure: {state: {subj: epochs}}
        feature_func: function that takes in epochs data and returns a numpy array of shape (channels, epochs, features)
        feature_metrics: list of metrics to compute
        distribution_metrics: list of metrics to compute
        subjs: list of subjects to compute features for
        states: list of states to compute features for
        epoch_axis: axis where the epochs are
        feature_axis: axis where the features are
        chan_axis: axis where the channels are
    """
    assert stacked_data.shape[0] == len(subjs)
    assert stacked_data.shape[1] == len(channels)

    if verbosity > 2:
        print(f"Computing features for each epoch")
        overall_starttime = time.time()

    feature_array = np.zeros((len(subjs), len(channels), len(feature_metrics)))
    for sdx, subj in enumerate(subjs):
        if verbosity > 1:
            print(f"Computing features for subject {subj}: {sdx+1}/{len(subjs)}")
            starttime = time.time()
       
        feature_array[sdx, :, :] = feature_func(stacked_data[sdx,...], metrics=feature_metrics)
        if verbosity > 1:
            print(f"Finished computing features for subject {subj} in {time.time() - starttime} seconds")
    if verbosity > 2:
        print(f"Finished computing features for all subjects in {time.time() - overall_starttime} seconds")
    return feature_array

def reshape_feature_array(distribution_features, subjs, channels, feature_metrics=['perm_entropy', 'svd_entropy', 'spectral_entropy', 'app_entropy', 'sample_entropy'], verbosity=0):
    """
    Reshape the distribution features into a 2D array
    Inputs:
        distribution_features: numpy array of shape (subjs, channels, features)
        subjs: list of subjects
        channels: list of channels
        feature_metrics: list of feature metrics
    Outputs:
        reshaped_features: numpy array of shape (subjs, channels*features)
        new_columns: list of new columns
    
    """
    
    
    assert distribution_features.shape[0] == len(subjs)
    assert distribution_features.shape[1] == len(channels)
    assert distribution_features.shape[2] == len(feature_metrics)

    reshaped_features = np.zeros((len(subjs), len(channels)*len(feature_metrics)))
    new_columns = []
    startime = time.time()
    for cdx, channel in enumerate(channels):
        for fmx, feature in enumerate(feature_metrics):
            new_columns.append(f"{channel}_{feature}")
            reshaped_features[:, cdx*len(feature_metrics) + fmx] = distribution_features[:, cdx, fmx]
    if verbosity > 2:
        print(f"Finished reshaping features in {time.time() - startime} seconds")
    time2 = time.time()
    rfda_numpy = distribution_features.reshape((len(subjs), -1))
    if verbosity > 2:
        print(f"Finished reshaping features in {time.time() - time2} seconds")
    assert np.allclose(rfda_numpy, reshaped_features, equal_nan=True), f"reshaped_features and rfda_numpy are not the same"
    return reshaped_features, new_columns

def main(open_closed_params, transform_data_params, channels, open_closed_path=LOCD_DATAPATH, choose_subjs='train', featurepath=FEATUREPATH, internal_folder='data/internal/', verbosity=1, save=True):
    """
    Given the parameters to load the psd transform data, compute complexity features on the psd data
    NOTE: 
    Inputs:
        transform_data_params: dictionary of parameters to load the transform data
        channels: list of channels
        choose_subjs: str, 'train' or 'test'
        featurepath: str, path to save the features
        verbosity: int, level of verbosity
        save: bool, whether to save the features
    Outputs:
        psd_complexity_feature_df: pandas dataframe of the features
    """
    savepath = os.path.join(featurepath, 'psd_complexity_features')

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # load the psd data

    all_params = {**open_closed_params, **transform_data_params, 'choose_subjs': choose_subjs}
    du.clean_params_path(savepath)
    pcom_savepath, found_match = du.check_and_make_params_folder(savepath, all_params)
    savefilename = os.path.join(pcom_savepath, 'psd_complexity_features.csv')
    if found_match:
        complexity_feature_df = pd.read_csv(savefilename, index_col=0)
        assert set(complexity_feature_df.index) == set(fu.select_subjects_from_dataframe(complexity_feature_df, choose_subjs, internal_folder=internal_folder).index)
    else:
        loadtime = time.time()
        transform_data_dict = td.main(locd_params=open_closed_params, locd_savepath=open_closed_path, n_jobs=1, as_paths=False, **transform_data_params)
        assert list(channels) == list(transform_data_dict['channels'])
        print('Time taken to load data: ', time.time() - loadtime)
        

        unraveled_mtd = td.unravel_multitaper_dataset(transform_data_dict['subj_data'])
        stack_open_power = np.stack(unraveled_mtd['avg']['open_power'])[:, :, 1:-1]
        stack_closed_power = np.stack(unraveled_mtd['avg']['closed_power'])[:, :, 1:-1]
        
        subjs = unraveled_mtd['avg']['open_subjs']

        # freqs = transform_data_dict['common_freqs'][1:-1] # remove the first and last frequency bins

        assert subjs == unraveled_mtd['avg']['closed_subjs']
        assert stack_open_power.shape == stack_closed_power.shape
        assert stack_open_power.shape[0] == len(subjs)
        stack_open_power,_ = fu.select_subjects_from_arraylike(stack_open_power, subjs, choose_subjs, internal_folder=internal_folder)
        stack_closed_power, subjs = fu.select_subjects_from_arraylike(stack_closed_power, subjs, choose_subjs, internal_folder=internal_folder)
        
        ## compute the entropy features
        if verbosity > 0:
            print(f"Computing entropy features") 
            entropytime = time.time()
        entropy_metrics = ['perm_entropy', 'svd_entropy', 'spectral_entropy', 'app_entropy', 'sample_entropy']

        open_entropy_psd_features = get_features_from_stack(ccf.get_entropy_features, stack_open_power, entropy_metrics, subjs, verbosity=verbosity, channels=channels)
        closed_entropy_psd_features = get_features_from_stack(ccf.get_entropy_features, stack_closed_power, entropy_metrics, subjs, verbosity=verbosity, channels=channels)
        if verbosity > 0:
            print(f"Finished computing entropy features in {time.time() - entropytime} seconds")
            print("Computing complexity features")

        ##  compute the complexity features
        complexity_metrics = ['hjorth_mobility', 'hjorth_complexity', 'hurst_exponent', 'hurst_constant', 'detrended_fluctuation']
        open_complexity_psd_features = get_features_from_stack(ccf.get_complexity_features, stack_open_power, complexity_metrics, subjs, verbosity=verbosity, channels=channels)
        closed_complexity_psd_features = get_features_from_stack(ccf.get_complexity_features, stack_closed_power, complexity_metrics, subjs, verbosity=verbosity, channels=channels)
        
        ## compute the fractal dimension features

        fractal_dimension_metrics = ['katz_fd', 'higuchi_fd', 'petrosian_fd']
        open_fractal_dimension_psd_features = get_features_from_stack(ccf.get_fractal_dimension_features, stack_open_power, fractal_dimension_metrics, subjs, verbosity=verbosity, channels=channels)
        closed_fractal_dimension_psd_features = get_features_from_stack(ccf.get_fractal_dimension_features, stack_closed_power, fractal_dimension_metrics, subjs, verbosity=verbosity, channels=channels)
        
        ## compute the geometric features
        geometric_metrics = ['curve_length', 'mean', 'std', 'variance', 'skewness', 'kurtosis']
        open_geometric_psd_features = get_features_from_stack(ccf.get_geometric_feature, stack_open_power, geometric_metrics, subjs, verbosity=verbosity, channels=channels)
        closed_geometric_psd_features = get_features_from_stack(ccf.get_geometric_feature, stack_closed_power, geometric_metrics, subjs, verbosity=verbosity, channels=channels)
        

        ## now make some dataframes and save them

        if verbosity > 0:
            print(f"Making dataframes")
            dftime = time.time()
        feature_arrays = [open_entropy_psd_features, closed_entropy_psd_features, open_complexity_psd_features, closed_complexity_psd_features, open_fractal_dimension_psd_features, closed_fractal_dimension_psd_features, open_geometric_psd_features, closed_geometric_psd_features]
        feature_array_names = ['open_entropy', 'closed_entropy', 'open_complexity', 'closed_complexity', 'open_fractal_dimension', 'closed_fractal_dimension', 'open_geometric', 'closed_geometric']
        all_feature_names = [entropy_metrics, entropy_metrics, complexity_metrics, complexity_metrics, fractal_dimension_metrics, fractal_dimension_metrics, geometric_metrics, geometric_metrics]
        feature_dfs = []
        for fidx, (feature_array, feature_array_name) in enumerate(zip(feature_arrays, feature_array_names)):
            if verbosity > 2:
                print(f"Reshaping feature array {feature_array_name}")
            reshaped_feature_array, new_columns = reshape_feature_array(feature_array, subjs, channels=channels, feature_metrics=all_feature_names[fidx])
            feature_df = pd.DataFrame(reshaped_feature_array, columns=new_columns)
            feature_df.index = subjs
            feature_df.columns = [f"{feature_array_name}_{col}" for col in feature_df.columns]
            feature_dfs.append(feature_df)

        complexity_feature_df = pd.concat(feature_dfs, axis=1)
        # add "PSD_Based in front of each column"
        complexity_feature_df.columns = [f"PSD_Complexity_{col}" for col in complexity_feature_df.columns]
        assert set(complexity_feature_df.index) == set(fu.select_subjects_from_dataframe(complexity_feature_df, choose_subjs, internal_folder=internal_folder).index), f"Dataset split subjects do not match the subjects in the dataframe"
        if verbosity > 0:
            print(f"Finished making dataframes in {time.time() - dftime} seconds")
        if save:
            if verbosity > 0:
                print(f"Saving dataframes to {savefilename}")
                savetime = time.time()
            complexity_feature_df.to_csv(savefilename)
            # save all params
            with open(os.path.join(pcom_savepath, 'params.json'), 'w') as f:
                json.dump(all_params, f)
            if verbosity > 0:
                print(f"Saved dataframes in {time.time() - savetime} seconds to {savefilename}")

    return complexity_feature_df

if __name__=='__main__':
    starttime =time.time()

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

    all_params = vars(parser.parse_args())
    td_params = {key: value for key, value in all_params.items() if key in ['interpolate_spectrum', 'freq_interp_method', 'which_segment', 'bandwidth']}
    locd_params = {key: value for key, value in all_params.items() if key not in td_params}
    channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

    complexity_feature_df = main(locd_params, td_params, channels)
    print(f"Finished in {time.time()-starttime} s")