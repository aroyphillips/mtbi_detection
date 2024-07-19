import numpy as np
import pandas as pd
import os
import time
from numpy.typing import NDArray

import antropy as ant
import hurst
import mne
import scipy
import json
import mtbi_detection.data.load_open_closed_data as locd
import mtbi_detection.data.data_utils as du
import mtbi_detection.features.feature_utils as fu


LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'features')

### Feature Extraction code
def get_entropy_features_from_epochs(data, fs=500, metrics = ['perm_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy']) -> NDArray:
    """
    Computes the entropy features for a given data matrix using the antropy package, where each row is a channel
    Inputs:
        data: numpy array of shape (channels, epochs, times) # important to be 
        channels: list of channel names
        fs: sampling frequency
    Outputs:
        entropy_df: pandas dataframe of shape (1, num_channels * num_entropies)
    # reference: https://github.com/raphaelvallat/antropy
    """
    n_channels, n_epochs, n_times = data.shape
    entropies_array= np.zeros((n_channels, n_epochs, len(metrics)))
    for channel in range(n_channels):
        for epoch in range(n_epochs):
            for edx, entropy in enumerate(metrics):
                if entropy == 'perm_entropy':
                    entropies_array[channel, epoch, edx] = ant.perm_entropy(data[channel, epoch, :], normalize=True)
                elif entropy == 'svd_entropy':
                    entropies_array[channel, epoch, edx] = ant.svd_entropy(data[channel, epoch, :], normalize=True)
                elif entropy == 'spectral_entropy':
                    entropies_array[channel, epoch, edx] = ant.spectral_entropy(data[channel, epoch, :], sf=fs, method='welch', normalize=True, nperseg=512) # this is not the best way to compute this, so we compute entropy on my multitaper spectrum later
                elif entropy == 'app_entropy':
                    entropies_array[channel, epoch, edx] = ant.app_entropy(data[channel, epoch, :])
                elif entropy == 'sample_entropy':
                    entropies_array[channel, epoch, edx] = ant.sample_entropy(data[channel, epoch, :])
                else:
                    raise ValueError(f"Entropy {entropy} not recognized")
                
    return entropies_array

def get_entropy_features(data, channels=None, fs=500, metrics = ['perm_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy']) -> NDArray:
    """
    Computes the entropy features for a given data matrix using the antropy package, where each row is a channel
    Inputs:
        data: numpy array of shape (channels, epochs, times) # important to be 
        channels: list of channel names
        fs: sampling frequency
    Outputs:
        entropy_df: pandas dataframe of shape (1, num_channels * num_entropies)
    # reference: https://github.com/raphaelvallat/antropy
    """
    n_channels, n_times = data.shape
    if channels is not None:
        assert len(channels) == n_channels
    entropies_array= np.zeros((n_channels, len(metrics)))
    for channel in range(n_channels):
        for edx, entropy in enumerate(metrics):
            if entropy == 'perm_entropy':
                entropies_array[channel, edx] = ant.perm_entropy(data[channel, :], normalize=True)
            elif entropy == 'svd_entropy':
                entropies_array[channel, edx] = ant.svd_entropy(data[channel, :], normalize=True)
            elif entropy == 'spectral_entropy':
                entropies_array[channel, edx] = ant.spectral_entropy(data[channel, :], sf=fs, method='welch', normalize=True, nperseg=2048) # this is not the best way to compute this, better to do it but let's just use it for now and compute entropy on my multitaper spectrum
            elif entropy == 'app_entropy':
                entropies_array[channel, edx] = ant.app_entropy(data[channel, :])
            elif entropy == 'sample_entropy':
                entropies_array[channel, edx] = ant.sample_entropy(data[channel, :])
            else:
                raise ValueError(f"Entropy {entropy} not recognized")

            if np.isnan(entropies_array[channel, edx]):
                print(f"NaN found in entropy feature {entropy} for channel {channel}")
            
    return entropies_array


def get_hurst_exponent_from_epochs(data: NDArray) -> NDArray:
    """
    Computes the hurst exponent for a given data matrix using the hurst package, where each row is a channel
    Inputs:
        data: numpy array of shape (channels, epochs, times)
    Outputs:
        all_hurst: numpy array of shape (channels, epochs, 2) where the last dimension is the hurst exponent and the c value
    Reference:
        https://pypi.org/project/hurst/
    """
    n_channels, n_epochs, n_times = data.shape
    all_hurst = np.zeros((n_channels, n_epochs, 2))
    for channel in range(n_channels):
        h, c, _ = hurst.compute_Hc(data[:, channel], kind="random_walk", min_window=10, max_window=None, simplified=False)
        all_hurst[channel, :, 0] = h
        all_hurst[channel, :, 1] = c
    return all_hurst


def get_hurst_exponent(data: NDArray) -> NDArray:
    n_channels, n_times = data.shape
    all_hurst = np.zeros((n_channels, 2))
    for channel in range(n_channels):
        h, c, _ = hurst.compute_Hc(data[channel, :], kind="random_walk", min_window=10, max_window=None, simplified=False)
        all_hurst[channel, 0] = h
        all_hurst[channel, 1] = c
    return all_hurst

def get_complexity_features_from_epochs(data, channels=None, metrics=['hjorth_mobility', 'hjorth_complexity', 'hurst_exponent', 'hurst_constant', 'detrended_fluctuation'], fs=500):
    """
    Comoutes the complexity features (mobility, complexity, hurst exponent, detrended fluctuation) for a given data matrix using the antropy package, where each row is a channel
    """
    n_channels, n_epochs, n_times = data.shape
    if channels is not None:
        assert len(channels) == n_channels
    complexity_array = np.zeros((n_channels, n_epochs, len(metrics)))
    for channel in range(n_channels):
        for epoch in range(n_epochs):
            mobility, complexity = ant.hjorth_params(data[channel, epoch, :])
            h, c, _ = hurst.compute_Hc(data[channel, epoch, :], kind="random_walk", min_window=10, max_window=None, simplified=False)
            for mdx, metric in enumerate(metrics):
                if metric == 'hjorth_mobility':
                    complexity_array[channel, epoch, mdx] = mobility
                elif metric == 'hjorth_complexity':
                    complexity_array[channel, epoch, mdx] = complexity
                elif metric == 'hurst_exponent':
                    complexity_array[channel, epoch, mdx] = h
                elif metric == 'hurst_constant':
                    complexity_array[channel, epoch, mdx] = c
                elif metric == 'detrended_fluctuation':
                    complexity_array[channel, epoch, mdx] = ant.detrended_fluctuation(data[channel, epoch, :])
                else:
                    raise ValueError(f"Metric {metric} not recognized")

    return complexity_array



def get_complexity_features(data, channels=None, metrics=['hjorth_mobility', 'hjorth_complexity', 'hurst_exponent', 'hurst_constant', 'detrended_fluctuation'], fs=500):
    """
    Computes the complexity features (mobility, complexity, hurst exponent, detrended fluctuation) for a given data matrix using the antropy package, where each row is a channel
    """
    n_channels, n_times = data.shape
    if channels is not None:
        assert len(channels) == n_channels
    complexity_array = np.zeros((n_channels, len(metrics)))
    for channel in range(n_channels):
        mobility, complexity = ant.hjorth_params(data[channel, :])
        h, c, _ = hurst.compute_Hc(data[channel, :], kind="random_walk", min_window=10, max_window=None, simplified=False)
        for mdx, metric in enumerate(metrics):
            if metric == 'hjorth_mobility':
                complexity_array[channel, mdx] = mobility
            elif metric == 'hjorth_complexity':
                complexity_array[channel, mdx] = complexity
            elif metric == 'hurst_exponent':
                complexity_array[channel, mdx] = h
            elif metric == 'hurst_constant':
                complexity_array[channel, mdx] = c
            elif metric == 'detrended_fluctuation':
                complexity_array[channel, mdx] = ant.detrended_fluctuation(data[channel, :])
            else:
                raise ValueError(f"Metric {metric} not recognized")
            if np.isnan(complexity_array[channel, mdx]):
                print(f"NaN found in complexity feature {metric} for channel {channel}")
    return complexity_array

def get_fractal_dimension_features(data, channels=None, metrics=['katz_fd', 'higuchi_fd', 'petrosian_fd']):
    """
    Computes the fractal dimension features (katz_fd, higuchi, petrosian)
    Inputs:
        data: numpy array of shape (channels, times)
        metrics: list of metrics to compute 
    Outputs:
        fd: numpy array of shape (channels, num_metrics)
    """
    n_channels, n_times = data.shape
    fd = np.zeros((n_channels, len(metrics)))
    for channel in range(n_channels):
        for fdx, metric in enumerate(metrics):
            if metric == 'katz_fd':
                fd[channel, fdx] = ant.katz_fd(data[channel, :])
            elif metric == 'higuchi_fd':
                fd[channel, fdx] = ant.higuchi_fd(data[channel, :])
            elif metric == 'petrosian_fd':
                fd[channel, fdx] = ant.petrosian_fd(data[channel, :])
            else:
                raise ValueError(f"Metric {metric} not recognized")
            if np.isnan(fd[channel, fdx]):
                print(f"NaN found in fractal dimension feature {metric} for channel {channel}")
    return fd

def get_fractal_dimension_features_from_epochs(data, metrics=['katz_fd', 'higuchi_fd', 'petrosian_fd']):
    """
    Computes the fractal dimension features (katz_fd, higuchi, petrosian)
    Inputs:
        data: numpy array of shape (channels, epochs, times)
        metrics: list of metrics to compute
    Outputs:
        fd: numpy array of shape (channels, epochs, num_metrics)
    """
    n_channels, n_epochs, n_times = data.shape
    fd = np.zeros((n_channels, n_epochs, len(metrics)))
    for channel in range(n_channels):
        for epoch in range(n_epochs):
            for fdx, metric in enumerate(metrics):
                if metric == 'katz_fd':
                    fd[channel, epoch, fdx] = ant.katz_fd(data[channel, epoch, :])
                elif metric == 'higuchi_fd':
                    fd[channel, epoch, fdx] = ant.higuchi_fd(data[channel, epoch, :])
                elif metric == 'petrosian_fd':
                    fd[channel, epoch, fdx] = ant.petrosian_fd(data[channel, epoch, :])
                else:
                    raise ValueError(f"Metric {metric} not recognized")
    return fd

def get_geometric_feature_from_epochs(sample_eeg_epoch, metrics=['curve_length'], time_axis=2, fs=500, scale=1e11):
    """
    Compute the geometric features of a signal using the antropy package
    Inputs:
        signal: time series of shape (channels, num_epochs, times),
        feature: feature to compute  ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'curve_length']
        axis: axis where time is
        fs: sampling frequency
    Outputs:
        computed_feature_array: (channels, num_epochs, num_features)
    """
    assert len(sample_eeg_epoch.shape) > time_axis
    assert isinstance(sample_eeg_epoch, np.ndarray)
    n_channels, n_epochs, n_times = sample_eeg_epoch.shape
    computed_feature_array = np.zeros((n_channels, n_epochs, len(metrics)))
    for fdx, feature in enumerate(metrics):
        if feature=='curve_length':
            computed_feature_array[..., fdx] = curve_length(sample_eeg_epoch, time_axis=time_axis, dx=1/fs, scale=scale)
        elif feature=='mean':
            computed_feature_array[..., fdx] = np.mean(sample_eeg_epoch, axis=time_axis)
        elif feature=='std':
            computed_feature_array[..., fdx] = np.std(sample_eeg_epoch, axis=time_axis)
        elif feature=='variance':
            computed_feature_array[..., fdx] = np.var(sample_eeg_epoch, axis=time_axis)
        elif feature=='skewness':
            computed_feature_array[..., fdx] = scipy.stats.skew(sample_eeg_epoch, axis=time_axis)
        elif feature=='kurtosis':
            computed_feature_array[..., fdx] = scipy.stats.kurtosis(sample_eeg_epoch, axis=time_axis)
        else:
            raise ValueError(f"Invalid feature {feature}")
    return computed_feature_array



def get_geometric_feature(signal, metrics=['curve_length'], time_axis=1, fs=500, scale=1e11):
    """
    Compute the geometric features of a signal using the antropy package
    Inputs:
        signal: time series of shape (channels, times), such as normalized power of the band of frequencies in the EEG data (shape: (num_channels, num_time_bins))
        feature: feature to compute  ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'curve_length']
        axis: axis where time is
        fs: sampling frequency
    Outputs:
        computed_feature: geometric_feature of the signal
    """
    assert len(signal.shape) > time_axis
    assert isinstance(signal, np.ndarray)
    n_channels, n_times = signal.shape
    computed_feature_array = np.zeros((n_channels, len(metrics)))
    for fdx, feature in enumerate(metrics):
        if feature=='curve_length':
            computed_feature_array[..., fdx] = curve_length(signal, time_axis=time_axis, dx=1/fs, scale=scale)
        elif feature=='mean':
            computed_feature_array[..., fdx] = np.mean(signal, axis=time_axis)
        elif feature=='std':
            computed_feature_array[..., fdx] = np.std(signal, axis=time_axis)
        elif feature=='variance':
            computed_feature_array[..., fdx] = np.var(signal, axis=time_axis)
        elif feature=='skewness':
            computed_feature_array[..., fdx] = scipy.stats.skew(signal, axis=time_axis)
        elif feature=='kurtosis':
            computed_feature_array[..., fdx] = scipy.stats.kurtosis(signal, axis=time_axis)
        else:
            raise ValueError(f"Invalid feature {feature}")
        if np.any(np.isnan(computed_feature_array[...,fdx])):
            print(f"NaN found in geometric feature {feature}")
    
    return computed_feature_array

def curve_length(signal, time_axis=1, dx=1/500, normalize=True, scale=1):
    """ 
    sum of Euclidean distances between points 
    https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
    Input:
        signal: numpy array of shape (channels, time)
        chan_axis: axis where the channels are
        time_axis: axis where the time is
        fs: sampling frequency
    Output:
        rms_val: numpy array of shape (channels,)
    """
    if normalize:
        # rescale each channel to be between 0 and 1 using min-max scaling
        signal = (signal - np.min(signal, axis=time_axis, keepdims=True))/(np.max(signal, axis=time_axis, keepdims=True) - np.min(signal, axis=time_axis, keepdims=True))
    diff_y = -np.diff(signal, axis=time_axis)*scale
    # print(np.max(diff_y, axis=time_axis))
    # shift the diff_y vertically by adding the max of thpe original signal
    # shift_diff_y = diff_y + np.max(signal, axis=time_axis, keepdims=True)
    # scale_diff_y = shift_diff_y/np.max(shift_diff_y, axis=time_axis, keepdims=True)
    # make diff x such that it is the same shape as diff_y and it is filled with the time difference between samples (1/fs)
    diff_x = np.ones(diff_y.shape)*dx
    # print(f"shape of diff_y: {diff_y.shape}")
    # print(f"shape of diff_x: {diff_x.shape}")
    euclidean_distances = np.sqrt(diff_y**2 + diff_x**2)
    # print(f"shape of euclidean_distances: {euclidean_distances.shape}")
    curve_length_val = np.sum(euclidean_distances, axis=time_axis)
    # print(f"shape of curve_length_val: {curve_length_val.shape}")
    # return np.sum(np.sqrt(np.sum((diff_val)**2,axis=time_axis)))
    return curve_length_val

import scipy

def get_geometric_feature_from_epochs(sample_eeg_epoch, metrics=['curve_length'], time_axis=2, fs=500, scale=1e11):
    """
    Compute the geometric features of a signal using the antropy package
    Inputs:
        signal: time series of shape (channels, num_epochs, times),
        feature: feature to compute  ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'curve_length']
        axis: axis where time is
        fs: sampling frequency
    Outputs:
        computed_feature_array: (channels, num_epochs, num_features)
    """
    assert len(sample_eeg_epoch.shape) > time_axis
    assert isinstance(sample_eeg_epoch, np.ndarray)
    n_channels, n_epochs, n_times = sample_eeg_epoch.shape
    computed_feature_array = np.zeros((n_channels, n_epochs, len(metrics)))
    for fdx, feature in enumerate(metrics):
        if feature=='curve_length':
            computed_feature_array[..., fdx] = curve_length(sample_eeg_epoch, time_axis=time_axis, dx=1/fs, scale=scale)
        elif feature=='mean':
            computed_feature_array[..., fdx] = np.mean(sample_eeg_epoch, axis=time_axis)
        elif feature=='std':
            computed_feature_array[..., fdx] = np.std(sample_eeg_epoch, axis=time_axis)
        elif feature=='variance':
            computed_feature_array[..., fdx] = np.var(sample_eeg_epoch, axis=time_axis)
        elif feature=='skewness':
            computed_feature_array[..., fdx] = scipy.stats.skew(sample_eeg_epoch, axis=time_axis)
        elif feature=='kurtosis':
            computed_feature_array[..., fdx] = scipy.stats.kurtosis(sample_eeg_epoch, axis=time_axis)
        else:
            raise ValueError(f"Invalid feature {feature}")
    return computed_feature_array

def get_epoch_distribution_features(epochs_data, channels, epoch_axis=1, chan_axis=0, metrics=['mean', 'std', 'median', 'iqr', 'skewness', 'kurtosis']):
    """
    Given the epochs data, compute the distribution features for each epoch
    Inputs:
        epochs_data: numpy array of shape (channels, epochs, times)
        epoch_axis: axis where the epochs are
        feature_axis: axis where the features are
        chan_axis: axis where the channels are
        metrics: list of metrics to compute
    Outputs:
        epoch_distribution_features: numpy array of shape (channels, num_data_features, num_distribution_features)
    """
    assert epochs_data.shape[chan_axis] == len(channels)
    n_channels, n_epochs, n_feats = epochs_data.shape
    distribution_feature_array =  np.zeros((n_channels, len(metrics), n_feats))
    for mdx, metric in enumerate(metrics):
        if metric == 'mean':
            distribution_feature_array[:, mdx, :] = np.mean(epochs_data, axis=epoch_axis)
        elif metric == 'std':
            distribution_feature_array[:, mdx, :] = np.std(epochs_data, axis=epoch_axis)
        elif metric == 'median':
            distribution_feature_array[:, mdx, :] = np.median(epochs_data, axis=epoch_axis)
        elif metric == 'iqr':
            distribution_feature_array[:, mdx, :] = scipy.stats.iqr(epochs_data, axis=epoch_axis)
        elif metric == 'skewness':
            distribution_feature_array[:, mdx, :] = scipy.stats.skew(epochs_data, axis=epoch_axis)
        elif metric == 'kurtosis':
            distribution_feature_array[:, mdx, :] = scipy.stats.kurtosis(epochs_data, axis=epoch_axis)
        else:
            raise ValueError(f"Invalid metric {metric}")
        if np.any(np.isnan(distribution_feature_array[:, mdx, :])):
            print(f"NaN found in distribution feature {metric}")
    return distribution_feature_array

def curve_length(signal, chan_axis=0, time_axis=1, dx=1/500, normalize=False, scale=1):
    """ 
    sum of Euclidean distances between points 
    https://stackoverflow.com/questions/63986448/finding-arc-length-in-a-curve-created-by-numpy-array
    Input:
        signal: numpy array of shape (channels, time)
        chan_axis: axis where the channels are
        time_axis: axis where the time is
        fs: sampling frequency
    Output:
        rms_val: numpy array of shape (channels,)
    """
    if normalize:
        # rescale each channel to be between 0 and 1 using min-max scaling
        signal = (signal - np.min(signal, axis=time_axis, keepdims=True))/(np.max(signal, axis=time_axis, keepdims=True) - np.min(signal, axis=time_axis, keepdims=True))
    diff_y = -np.diff(signal, axis=time_axis)*scale
    # make diff x such that it is the same shape as diff_y and it is filled with the time difference between samples (1/fs)
    diff_x = np.ones(diff_y.shape)*dx

    euclidean_distances = np.sqrt(diff_y**2 + diff_x**2)
    curve_length_val = np.sum(euclidean_distances, axis=time_axis)
    return curve_length_val

def get_distribution_features_from_epochs(feature_func, epochs_dict, feature_metrics, distribution_metrics, subjs, channels, states=['open', 'closed'], epoch_axis=1, chan_axis=0, verbosity=1):
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
    assert all([state in epochs_dict.keys() for state in states]) and all([key in states for key in epochs_dict.keys()])
    assert len(epochs_dict[states[0]].keys()) == len(epochs_dict[states[1]].keys())
    assert all([subj in epochs_dict[states[0]].keys() for subj in subjs]) and all([subj in epochs_dict[states[1]].keys() for subj in subjs])

    if verbosity > 0:
        print(f"Computing features for each epoch")
        overall_starttime = time.time()
    feature_dict = {state: {subj: [] for subj in subjs} for state in states}
    for sdx, subj in enumerate(subjs):
        if verbosity > 0:
            print(f"Computing features for subject {subj}: {sdx+1}/{len(subjs)}")
            starttime = time.time()
        for state in states:
            feature_dict[state][subj] = feature_func(epochs_dict[state][subj], metrics=feature_metrics)
        if verbosity > 0:
            print(f"Finished computing features for subject {subj} in {time.time() - starttime} seconds")
        
    if verbosity > 0:
        print(f"Computing distribution features for each epoch")
    feature_distribution_dict = {state: {subj: [] for subj in subjs} for state in states}
    for sdx, subj in enumerate(subjs):
        if verbosity > 0:
            print(f"Computing distribution features for subject {subj}: {sdx+1}/{len(subjs)}")
            starttime = time.time()
        for state in states:
            feature_distribution_dict[state][subj] = get_epoch_distribution_features(feature_dict[state][subj], epoch_axis=1, metrics=distribution_metrics, channels=channels)
        if verbosity > 0:
            print(f"Finished computing distribution features for subject {subj} in {time.time() - starttime} seconds")


    open_feature_distribution_features = np.stack([feature_distribution_dict['open'][subj] for subj in subjs], axis=0)
    closed_feature_distribution_features = np.stack([feature_distribution_dict['closed'][subj] for subj in subjs], axis=0)

    assert open_feature_distribution_features.shape[0] == len(subjs)
    assert open_feature_distribution_features.shape[1] == len(channels)
    assert open_feature_distribution_features.shape[2] == len(distribution_metrics)
    assert open_feature_distribution_features.shape[3] == len(feature_metrics)
    assert closed_feature_distribution_features.shape == open_feature_distribution_features.shape

    if verbosity > 0:
        print(f"Finished computing features for each epoch in {time.time() - overall_starttime} seconds")

    return open_feature_distribution_features, closed_feature_distribution_features


def reshape_feature_distribution_array(distribution_features, subjs, channels, distribution_metrics=['mean', 'std', 'median', 'iqr', 'skewness', 'kurtosis'], feature_metrics=['perm_entropy', 'svd_entropy','app_entropy', 'sample_entropy'], verbosity=0):
    assert distribution_features.shape[0] == len(subjs)
    assert distribution_features.shape[1] == len(channels)
    assert distribution_features.shape[2] == len(distribution_metrics)
    assert distribution_features.shape[3] == len(feature_metrics)

    reshaped_features = np.zeros((len(subjs), len(channels)*len(distribution_metrics)*len(feature_metrics)))
    new_columns = []
    startime = time.time()
    for cdx, channel in enumerate(channels):
        for mdx, metric in enumerate(distribution_metrics):
            for fmx, feature in enumerate(feature_metrics):
                new_columns.append(f"{channel}_{metric}_{feature}")
                reshaped_features[:, cdx*len(distribution_metrics)*len(feature_metrics) + mdx*len(feature_metrics) + fmx] = distribution_features[:, cdx, mdx, fmx]
    if verbosity > 2:
        print(f"Finished reshaping features in {time.time() - startime} seconds")
    time2 = time.time()
    rfda_numpy = distribution_features.reshape((len(subjs), -1))
    if verbosity > 2:
        print(f"Finished reshaping features in {time.time() - time2} seconds")
    assert np.allclose(rfda_numpy, reshaped_features, equal_nan=True), f"reshaped_features and rfda_numpy are not the same"
    return reshaped_features, new_columns


def main(open_closed_params, channels, window_len=10, overlap=1, verbosity=1, save=False, featurepath=FEATUREPATH, choose_subjs='train', internal_folder='data/internal/'):
    """
    Given the params to load the open closed pathdict and the epoch making params
    compute the complexity features for each subject and save them to a csv
    NOTE: Currently loads entire dataset into memory, so may need adjustment for systems with smaller RAM
    Inputs:
        open_closed_params: dictionary of parameters to load the open closed pathdict
        channels: list of channels to compute features for
        window_len: length of the window in seconds
        overlap: overlap of the windows
        verbosity: level of verbosity
        save: whether to save the dataframes
        savepath: where to save the dataframes
        choose_subjs: whether to choose the train or test subjects
        internal_folder: folder that holds the dataset splits
    Outputs:
        concat_feature_df: dataframe of the features
    
    """
    complexitytime=time.time()
    distribution_metrics = ['mean', 'std', 'median', 'iqr', 'skewness', 'kurtosis']
    open_closed_dict = locd.load_open_closed_pathdict(savepath=LOCD_DATAPATH, **open_closed_params)
    
    all_params = du.make_dict_saveable({**open_closed_params, 'channels': list(channels), 'window_len': window_len, 'overlap': overlap, 'choose_subjs': choose_subjs})
    complexity_path = os.path.join(featurepath, 'complexity_features')
    du.clean_params_path(complexity_path)
    complexitysavepath, found_match = du.check_and_make_params_folder(complexity_path, all_params)
    if found_match:
        if verbosity > 0:
            print(f"Found match for params in {complexitysavepath}")
        all_complexity_feature_df = pd.read_csv(os.path.join(complexitysavepath, 'all_complexity_features.csv'), index_col=0)
        assert set(all_complexity_feature_df.index) == set(fu.select_subjects_from_dataframe(all_complexity_feature_df, choose_subjs, internal_folder).index), f"Subjects in dataframe do not match subjects in the index"
    else:
            
        open_closed_dict = fu.select_subjects_from_dict(open_closed_dict, choose_subjs, internal_folder=internal_folder)

        subjs = [key for key in open_closed_dict.keys()]
        subjs_type = max([type(s) for s in subjs], key=[type(s) for s in subjs].count)

        if subjs_type == str:
            subjs = [subj for subj in subjs if subj.isnumeric()]
        else:
            subjs = [subj for subj in subjs if type(subj) == subjs_type]

        
        states = ['open', 'closed']
        epochs_dict = {state: {subj: [] for subj in subjs} for state in states} # will become approximately 1GB (83471637 bytes)
        for sdx, subj in enumerate(subjs):
            if verbosity > 0:
                print(f"Computing epochs for subject {subj}: {sdx+1}/{len(subjs)}")
                starttime = time.time()
            for state in states:
                subj_epochs = [du.make_epochs_from_raw(mne.io.read_raw_fif(rawfile, verbose=False).pick(channels), window_len=window_len, overlap=overlap, channels=channels) for rawfile in open_closed_dict[subj][state]]
                stacked_epochs = np.concatenate(subj_epochs, axis=1)
                assert stacked_epochs.shape[0] == len(channels), f"stacked_epochs.shape[0] = {stacked_epochs.shape[0]} but len(channels) = {len(channels)}"
                epochs_dict[state][subj] = stacked_epochs

            if verbosity > 0:
                print(f"Finished computing epochs for subject {subj} in {time.time() - starttime} seconds")



        ## compute the entropy features
        if verbosity > 0:
            print(f"Computing entropy features")
            entropytime = time.time()
        entropy_metrics = ['perm_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy']
        open_entropy_distribution_features, closed_entropy_distribution_features = get_distribution_features_from_epochs(get_entropy_features_from_epochs, epochs_dict, entropy_metrics, distribution_metrics, subjs, states=states, epoch_axis=1, chan_axis=0, verbosity=verbosity, channels=channels)
        if verbosity > 0:
            print(f"Finished computing entropy features in {time.time() - entropytime} seconds")
            print("Computing complexity features")
        
        ##  compute the complexity features
        complexity_metrics = ['hjorth_mobility', 'hjorth_complexity', 'hurst_exponent', 'hurst_constant', 'detrended_fluctuation']
        open_complexity_distribution_features, closed_complexity_distribution_features = get_distribution_features_from_epochs(get_complexity_features_from_epochs, epochs_dict, complexity_metrics, distribution_metrics, subjs, states=states, epoch_axis=1, chan_axis=0, verbosity=verbosity, channels=channels)
        
        ## compute the fractal dimension features
        fractal_dimension_metrics = ['katz_fd', 'higuchi_fd', 'petrosian_fd']
        open_fractal_dimension_distribution_features, closed_fractal_dimension_distribution_features = get_distribution_features_from_epochs(get_fractal_dimension_features_from_epochs, epochs_dict, fractal_dimension_metrics, distribution_metrics, subjs, states=states, epoch_axis=1, chan_axis=0, verbosity=verbosity, channels=channels)

        ## compute the geometric features
        geometric_metrics = ['curve_length', 'mean', 'std', 'variance', 'skewness', 'kurtosis']
        open_geom_distribution_features, closed_geom_distribution_features = get_distribution_features_from_epochs(get_geometric_feature_from_epochs, epochs_dict, geometric_metrics, distribution_metrics, subjs, states=states, epoch_axis=1, chan_axis=0, verbosity=verbosity, channels=channels)

        if verbosity > 0:
            print(f"Finished computing complexity features in {time.time() - complexitytime} seconds")
            print(f"Making dataframes")
            dftime = time.time()
        feature_arrays = [open_entropy_distribution_features, closed_entropy_distribution_features, open_complexity_distribution_features, closed_complexity_distribution_features, open_fractal_dimension_distribution_features, closed_fractal_dimension_distribution_features, open_geom_distribution_features, closed_geom_distribution_features]
        feature_array_names = ['open_entropy', 'closed_entropy', 'open_complexity', 'closed_complexity', 'open_fractal_dimension', 'closed_fractal_dimension', 'open_geometric', 'closed_geometric']
        all_feature_names = [entropy_metrics, entropy_metrics, complexity_metrics, complexity_metrics, fractal_dimension_metrics, fractal_dimension_metrics, geometric_metrics, geometric_metrics]
        feature_dfs = []
        for fidx, (feature_array, feature_array_name) in enumerate(zip(feature_arrays, feature_array_names)):
            if verbosity > 1:
                print(f"Reshaping feature array {feature_array_name}")
            reshaped_feature_array, new_columns = reshape_feature_distribution_array(feature_array, subjs, channels=channels, distribution_metrics=distribution_metrics, feature_metrics=all_feature_names[fidx])
            feature_df = pd.DataFrame(reshaped_feature_array, columns=new_columns)
            feature_df.index = subjs
            feature_df.columns = [f"{feature_array_name}_{col}" for col in feature_df.columns]
            feature_dfs.append(feature_df)

        all_complexity_feature_df = pd.concat(feature_dfs, axis=1)
        all_complexity_feature_df.columns = [f"Complexity_{col}" for col in all_complexity_feature_df.columns]
        assert set(all_complexity_feature_df.index) == set(fu.select_subjects_from_dataframe(all_complexity_feature_df, choose_subjs, internal_folder).index), f"Subjects in dataframe do not match subjects in the index"
        if verbosity > 0:
            print(f"Finished making dataframes in {time.time() - dftime} seconds")
        if save:
            print(f"Saving dataframes to {complexitysavepath}")
            savetime = time.time()
            all_complexity_feature_df.to_csv(os.path.join(complexitysavepath, 'all_complexity_features.csv'))
            print(f"Saved dataframes in {time.time() - savetime} seconds to {os.path.join(complexitysavepath, 'all_complexity_features.csv')}")
            with open(os.path.join(complexitysavepath, 'params.json'), 'w') as f:
                json.dump(all_params, f)
    return all_complexity_feature_df
            
if __name__ == '__main__':
    main()