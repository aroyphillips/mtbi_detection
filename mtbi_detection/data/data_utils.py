import os
import shutil
import numpy as np
import mne
import glob
import json
import pandas as pd
import sys
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

class RobustChannelScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_channels=None, sample_axis=0):
        """
        Parameters:
            feature_channels: list of axes to compute the median and percentiles over
            sample_axis: axis to compute the median and percentiles over
        Note:
            To perform a group level scaling where the array is (n_samples, ...), set feature_channels to None or [1, 2, ...] (all the other indices) and sample_axis to 0 
            To scale over all the channels and samples, set feature_axes to 2 (or whereever non channels is) and sample_axis to 0: median will be (n_channels, )
            To scale over all the features and samples, set feature_axes to 1 (or whereever chaannels is) and sample_axis to 0: median will be (n_features, )
            To scale over the whole thing, set channel_axes to None and sample_axis to None
            The key point is that the data will get scaled over all the axes in feature_chanel and sample_axis, while treating any other axes as independent
            Seems that None, None will let each channel be scaled together with median shared across
            while feature_channels=2 and sample_axis=0 will scale each channel independently
            For EEG: since amplitude varies subject to subject and channel to channel, we want to scale over all the samples and features, but not over the channels:
                feature_channels = [2], sample_axis = None
            For PSD: since amplitude varies subject to subject and channel to channel, we want to scale over all the samples and channels, but not over the features:
                feature_channels = None, sample_axis = None
        """
        assert type(feature_channels) == list or feature_channels is None
        assert type(sample_axis) == int or sample_axis is None
        self.feature_channels = feature_channels
        self.sample_axis = sample_axis
        _test_robust_channel_scaler()
        
    def fit(self, X, y=None):
        if self.feature_channels is None:
            select_axes = self.sample_axis
        elif self.sample_axis is None:
            select_axes = self.feature_channels
        else:
            select_axes = tuple([self.sample_axis] + self.feature_channels)
        self.center_ = np.median(X, axis=select_axes)
        self.scale_ = np.percentile(X, 75, axis=select_axes) - np.percentile(X, 25, axis=select_axes)
        if select_axes is not None:
            # expand the dims
            self.center_ = np.expand_dims(self.center_, axis=select_axes)
            self.scale_ = np.expand_dims(self.scale_, axis=select_axes)
        return self

    def transform(self, X):
        check_is_fitted(self, ['center_', 'scale_'])

        X_transformed = (X- self.center_) / self.scale_
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class PSDScaler(BaseEstimator, TransformerMixin):
    def __init__(self, normalize_method='median', channel_independence=False, normalize_a=None, normalize_b=None):
        self.normalize_method = normalize_method
        self.channel_independence = channel_independence
        self.normalize_a = normalize_a
        self.normalize_b = normalize_b

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.normalize_method in ['median', 'mean']:
            if self.channel_independence:
                raise ValueError('Channel independence not supported for median or mean scaling, use robust or z_score instead')
            if self.normalize_a is not None or self.normalize_b is not None:
                print("Warning, ignoring the normalization parameters")
            scaler = mne.decoding.Scaler(scalings=self.normalize_method)
            X = scaler.fit_transform(np.log10(X))
            X = 10**X
        elif self.normalize_method == 'minmax':
            if self.normalize_a is not None and self.normalize_a is not None:
                X = (X-self.normalize_a)/(self.normalize_b - X)
            elif self.channel_independence:
                # my array is shape (n_epochs, n_channels, n_times): find the minimum value along the first axis
                min_vals = np.min(X, axis=(0, 2))
                max_vals = np.max(X, axis=(0,2))
                X = np.divide(X - min_vals[:, np.newaxis], (max_vals - min_vals)[:, np.newaxis])
            elif self.normalize_a is None and self.normalize_b is None:
                min_val = np.min(X)
                max_val = np.max(X)
                X = (X - min_val) / (max_val - min_val)
            else:
                raise ValueError('If channel_independence is False, thenself.normalize_a and self.normalize_b must be None')
        elif self.normalize_method in ['robust', 'z_score']:
            if self.normalize_a is not None and self.normalize_b is not None:
                X = (X-self.normalize_a)/(self.normalize_b)
            elif self.channel_independence:
                if self.normalize_method == 'robust':
                    middle_values = np.median(X, axis=(0, 2))
                    range_values = np.subtract(*np.percentile(X, [75, 25], axis=(0, 2)))
                elif self.normalize_method == 'z_score':
                    middle_values = np.mean(X, axis=(0, 2))
                    range_values = np.std(X, axis=(0, 2))
                X = np.divide(X - middle_values[:, np.newaxis], range_values[:, np.newaxis])
            elif self.normalize_a is None and self.normalize_b is None:
                if self.normalize_method == 'robust':
                    middle_value = np.median(X)
                    range_value = np.subtract(*np.percentile(X, [75, 25]))
                elif self.normalize_method == 'z_score':
                    middle_value = np.mean(X)
                    range_value = np.std(X)
                X = (X - middle_value) / range_value
            else:
                print(f"WARNING: If channel_independence is False, then self.normalize_a and self.normalize_b must be None")                
        else:
            print(f"WARNING: normalize_method {self.normalize_method} not recognized")
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)
    

def _test_robust_channel_scaler():
    X = np.random.rand(100, 10, 10, 10)
    median = np.median(X, axis=0)
    p75 = np.percentile(X, 75, axis=0)
    p25 = np.percentile(X, 25, axis=0)
    X_scaled = (X - median) / (p75 - p25)
    X_robuscaled = RobustChannelScaler().fit_transform(X)
    assert np.allclose(X_scaled, X_robuscaled), f"Some values are not close: {X_scaled[0, 0, 0, 0]} != {X_robuscaled[0, 0, 0, 0]}"

    # choose another
    X = np.array([[[1, 2, 3], [4, 5, 6]], 
               [[7, 8, 9], [10, 11, 12]],
               [[1,2, 3], [0, 0, 0]],
               [[1, 2, 3], [4, 5, 6]]])
    overall_median = np.median(X, axis=None)
    overall_p75 = np.percentile(X, 75, axis=None)
    overall_p25 = np.percentile(X, 25, axis=None)
    ax0_med = np.median(X, axis=0)
    ax0_p75 = np.percentile(X, 75, axis=0)
    ax0_p25 = np.percentile(X, 25, axis=0)
    ax01_med = np.median(X, axis=(0, 1))
    ax01_p75 = np.percentile(X, 75, axis=(0, 1))
    ax01_p25 = np.percentile(X, 25, axis=(0, 1))

    X_scaled = (X - ax01_med) / (ax01_p75 - ax01_p25)
    X_robuscaled = RobustChannelScaler(feature_channels=[1], sample_axis=0).fit_transform(X)
    assert np.allclose(X_scaled, X_robuscaled), f"Some values are not close: {X_scaled[0, 0, 0, 0]} != {X_robuscaled[0, 0, 0, 0]}"

    X_scaled = (X - ax0_med) / (ax0_p75 - ax0_p25)
    X_robuscaled = RobustChannelScaler(feature_channels=None, sample_axis=0).fit_transform(X)
    assert np.allclose(X_scaled, X_robuscaled), f"Some values are not close: {X_scaled[0, 0, 0, 0]} != {X_robuscaled[0, 0, 0, 0]}"

    X_scaled = (X-overall_median) / (overall_p75 - overall_p25)
    X_robuscaled = RobustChannelScaler(feature_channels=None, sample_axis=None).fit_transform(X)
    assert np.allclose(X_scaled, X_robuscaled), f"Some values are not close: {X_scaled[0, 0, 0, 0]} != {X_robuscaled[0, 0, 0, 0]}"
    # 2D
    X = np.random.rand(100, 10)
    median = np.median(X, axis=0)
    p75 = np.percentile(X, 75, axis=0)
    p25 = np.percentile(X, 25, axis=0)
    X_scaled = (X - median) / (p75 - p25)
    X_robuscaled = RobustChannelScaler().fit_transform(X)
    assert np.allclose(X_scaled, X_robuscaled)



def make_epochs_from_raw(raw, window_len=10, overlap=1, fs=500, channels=None, print_statement=None, verbosity=0):
    """
    This function takes in the raw mne object, the window length, and the amount of overlap and outputs an array of size (n_channels, n_epochs, window_length*sampling_freq)

    Inputs:
        raw: The raw mne object
        window_length: The length of the window in seconds
        overlap: The amount of overlap between windows in seconds

    Outputs:
        Epochs:  numpy array of size (n_channels, n_epochs, window_length*sampling_freq)
    """
    # check that the sampling frequency is fs
    assert overlap < window_len, "Overlap must be less than window length"
    assert raw.info['sfreq'] == fs, f"Sampling frequency is not equal to fs {fs}"

    if print_statement is not None:
        print(print_statement)
    if channels is not None:
        raw.pick(channels) # supposed to order the channels in the order of the list
    eeg_signal, times = raw.get_data(), raw.times
    n_channels = eeg_signal.shape[0]


    total_time = times[-1] - times[0]
    n_epochs = int( np.floor((total_time - window_len)/(window_len - overlap)) + 1)
    if verbosity > 0:
        print(f"Making {n_epochs} epochs")
    if verbosity > 2:
        print(f"Time diff: {(times[-1] - times[0])}, window_len: {window_len}, overlap: {overlap}, n_epochs: {n_epochs}")

    window_len_samples = int(window_len * fs)
    overlap_samples = int((window_len - overlap) * fs)

    if verbosity > 2:
        print(f"Window len samples: {window_len_samples}, overlap samples: {overlap_samples}")


    epochs = np.zeros((n_channels, n_epochs, window_len_samples))
    if verbosity > 2:
        print(f"Shape of epochs: {epochs.shape}")
        print(f"Shape of eeg_signal: {eeg_signal.shape}")
    for epc in range(n_epochs):
        if verbosity > 2:
            print(f"Segment {epc}, length: {epc*overlap_samples}:{epc*overlap_samples + window_len_samples}")
        epochs[:, epc, :] = eeg_signal[:, epc*overlap_samples:(epc*overlap_samples + window_len_samples)]

    return epochs


def remove_subdir(datapath, subdir='crops'):
    """
    Remove a subdirectory from a directory
    Inputs:
        datapath: the path to the directory
        subdir: the name of the subdirectory to remove
    """
    for root, dirs, files in os.walk(datapath):
        for dir in dirs:
            if dir == subdir:
                shutil.rmtree(os.path.join(root, dir))
    print("Removed all {} directories from {}".format(subdir, datapath))


def downsample_signal(signal, num_points, axis=1, method='log', two_sided=False, return_indices=False):
    """
    Given a signal of shape (..., axis_points, ...), downsample the signal to num_points
    Input:
        signal: the signal to downsample
        num_points: the number of points to downsample to
        axis: the axis to downsample along
    Output:
        signal: the downsampled signal (shape: (..., num_points, ...))
    """
    assert axis < len(signal.shape), "Axis {} out of bounds for signal of shape {}".format(axis, signal.shape)
    assert signal.shape[axis] >= num_points, "Cannot downsample signal of shape {} to {} points".format(signal.shape, num_points)
    if signal.shape[axis] == num_points:
        return signal
    else:
        if method == 'linear':
            signal_indices = np.linspace(0, signal.shape[axis]-1, num_points).astype(int)
        elif method == 'log':
            if two_sided:
                midpoint = signal.shape[axis]//2
                negative_index = midpoint - np.logspace(np.log(midpoint-1), 0, num_points//2, base=np.e).astype(int)
                positive_index = midpoint + np.logspace(0, np.log(signal.shape[axis]-midpoint-1), num_points//2, base=np.e).astype(int)
                signal_indices = np.concatenate((negative_index, positive_index))
            else:
                signal_indices = np.logspace(0, np.log(signal.shape[axis]-1), num_points, base=np.e).astype(int)
            # remove repeated indices
            signal_indices = np.unique(signal_indices)
        if return_indices:
            return signal.take(indices=signal_indices, axis=axis), signal_indices
        else:
            return signal.take(indices=signal_indices, axis=axis)

def robust_scaling(data, median=None, p75=None, p25=None):
    """
    https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
    formula: (data-median)/(p75-p25)
    """
    if median is None:
        median = np.median(data, axis=-1)
    if p75 is None:
        p75 = np.percentile(data, 75, method='linear', axis=-1)
    if p25 is None:
        p25 = np.percentile(data, 25, method='linear', axis=-1)
    # assert that 

    assert np.all(p75 >= median)
    assert np.all(median >= p25)
    if isinstance(median, np.ndarray):
        if median.shape[0] == 1:
            median = median[0]
        else: 
            assert median.shape[0] == data.shape[0]
            median = median.reshape(-1, 1)
    numerator = data-median
    denominator = p75-p25
    # assert that the denominator is either a scalar or a 1d array of the same shape as the numerator
    if isinstance(denominator, np.ndarray):
        if denominator.shape[0] == 1:
            denominator = denominator[0]
        else:
            assert denominator.shape[0] == numerator.shape[0]
            denominator = denominator.reshape(-1, 1)
    else:
        assert np.isscalar(denominator)
    return np.divide(numerator, denominator)

def min_max_scaling(data, min=None, max=None):
    """
    formula: (data-min)/(max-min)
    """
    if min is None:
        min = np.min(data)
    if max is None:
        max = np.max(data)
    return (data-min)/(max-min)

def scaling_from_single_array(data, scaling_method='robust', **kwargs):
    """
    data is a 2d array
    """
    if scaling_method == 'robust':
        median = np.median(data)
        p75 = np.percentile(data, 75)
        p25 = np.percentile(data, 25)
        return robust_scaling(data, median=median, p75=p75, p25=p25)
    elif scaling_method == 'min_max':
        allmin = np.min(data)
        allmax = np.max(data)
        return min_max_scaling(data, min=allmin, max=allmax)
    else:
        raise ValueError(f"scaling method {scaling_method} not recognized")


def scaling_from_list(data_list, scaling_method='robust', subj_independent=False, channel_independent=False, **kwargs):
    """
    data_list is a list of 1d arrays
    """
    # flatten the data into a Nd array
    all_shapes = [data.shape for data in data_list]
    if all([shape[0] == all_shapes[0][0] for shape in all_shapes]):
        num_data = all_shapes[0][0]
        len_flatten = sum([shape[-1] for shape in all_shapes])
    else:
        num_data = 1
        len_flatten = sum([shape[0]*shape[1] for shape in all_shapes])
    # stack the data into a vector of shape (num_data, sum(all_shapes[0][i]) for i in range(len(all_shapes[0])))
    stack_data = np.zeros((num_data, len_flatten))
    if num_data == 1:
        prev_idx = 0
        for idx, data_i in enumerate(data_list):
            flat_data = data_i.flatten()
            stack_data[0, prev_idx:(prev_idx+len(flat_data))] = flat_data
            prev_idx += len(flat_data)
    else:
        prev_idx = 0
        for idx, data_i in enumerate(data_list):
            stack_data[:, prev_idx:(prev_idx+data_i.shape[1])] = data_i
            prev_idx += data_i.shape[1]
    
    if scaling_method == 'robust':
        if not subj_independent:
            if channel_independent:
                median = np.median(stack_data, axis=1)
                p75 = np.percentile(stack_data, 75, axis=1)
                p25 = np.percentile(stack_data, 25, axis=1)
            else:
                median = np.median(stack_data)
                p75 = np.percentile(stack_data, 75)
                p25 = np.percentile(stack_data, 25)
            scaled_data = [robust_scaling(data, median=median, p75=p75, p25=p25) for data in data_list]
        else:
            if channel_independent:
                median = None
                p75 = None
                p25 = None
                scaled_data = [robust_scaling(data, median=median, p75=p75, p25=p25) for data in data_list]
            else:
                scaled_data = [scaling_from_single_array(data, scaling_method=scaling_method) for data in data_list]
            
        return scaled_data
    elif scaling_method == 'min_max':
        if not subj_independent:
            allmin = np.min(stack_data)
            allmax = np.max(stack_data)
        else:
            allmin = None
            allmax = None
        return [min_max_scaling(data, min=allmin, max=allmax) for data in data_list]
    else:
        raise ValueError(f"scaling method {scaling_method} not recognized")

def get_pos_from_channels(channels, two_d=True):

    # get the standard 10-20 positions
    pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    pos['T1'] = np.array([(-0.0860761-0.0702629)/2, -0.0249897, (-0.067986+ -0.01142)/2])
    pos['T2'] = np.array([(0.0860761+0.0702629)/2, -0.0249897, (-0.067986+ -0.01142)/2])
    assert all([channel in pos.keys() for channel in channels]), f"Not all channels in the channels dataframe are in the pos dictionary: {[channel for channel in channels if channel not in pos.keys()]}"
    channel_names = channels.copy()
    n_nodes = len(channel_names)
    # project the 3d positions to 2d
    if two_d:
        pos = {channel_names[i]: np.array([pos[channel_names[i]][0], pos[channel_names[i]][1]]) for i in range(n_nodes)}
    else:
        pos = {channel_names[i]: np.array([pos[channel_names[i]][0], pos[channel_names[i]][1], pos[channel_names[i]][2]]) for i in range(n_nodes)}
    return pos

def make_epochs(eeg_signal, times, window_len=2, overlap=0, fs=500):
    """
    Given an eegsignal (1, n_times), compute epcohs
    Inputs:
        eeg_signal: numpy array of shape (n_times, )
        times: numpy array of shape (n_times,)
        window_len: int, length of the window in seconds
        overlap: float, fraction of overlap between windows
        fs: int, sampling frequency
    Outputs:
        epochs: numpy array of shape (n_epochs, window_len*fs)
    """
    n_epochs = int(np.floor((times[-1] - times[0]) / window_len))
    window_len = int(window_len * fs)
    overlap = int(overlap * window_len)
    epochs = np.zeros((n_epochs, window_len))
    for i in range(n_epochs):
        epochs[i, :] = eeg_signal[i*overlap:(i*overlap + window_len)]

    return epochs

def make_multichannel_epochs(eeg_signal, times, window_len=2, overlap=0, fs=500):
    """
    Given an eegsignal (n_channels, n_times), compute epcohs
    Inputs:
        eeg_signal: numpy array of shape (n_times, )
        times: numpy array of shape (n_times,)
        window_len: int, length of the window in seconds
        overlap: float, fraction of overlap between windows
        fs: int, sampling frequency
    Outputs:
        epochs: numpy array of shape (n_epochs, window_len*fs)
    """
    n_channels = eeg_signal.shape[0]
    n_epochs = int(np.floor((times[-1] - times[0]) / window_len))
    try:
        assert eeg_signal.shape[1] == times.shape[0]
    except AssertionError:
        print(f"eeg_signal.shape: {eeg_signal.shape} does not match times.shape: {times.shape}")
    window_len = int(window_len * fs)
    overlap = int(overlap * window_len)
    epochs = np.zeros((n_channels, n_epochs, window_len))
    for i in range(n_epochs):
        epochs[:, i, :] = eeg_signal[:,i*overlap:(i*overlap + window_len)]

    return epochs

def _check_if_none(array):
    is_none = array == np.array(None)
    if type(is_none) == bool:
        return is_none
    else:
        return is_none.all()
    
def symlog(x, base=10, eps=1e-20):
    abs_x = np.abs(x) + eps
    sign_x = np.sign(x)
    log = np.emath.logn(base, abs_x)
    return sign_x * log

# take the log of all the features except for pnn50
def all_log_except(df, non_log_cols=[]):
    log_cols = [col for col in df.columns if col not in non_log_cols]
    df[log_cols] = np.log(df[log_cols])
    return df

def only_log_if(df, all_log_conditions=[lambda x: x < 1e-2], any_log_conditions=[lambda x: x > 10], return_log_cols=False):
    """
    Takes in a dataframe and returns the log of a dataframe on only the columns that meet the conditions
    """
    log_cols = []
    for col in df.columns:
        for log_condition in all_log_conditions:
            # check that 0 is not in the column and non
            if log_condition(df[col]).all() and (df[col] > 0).all():
                log_cols.append(col)
        for log_condition in any_log_conditions:
            if log_condition(df[col]).any() and (df[col] > 0).all():
                log_cols.append(col)
    
    log_df = df.copy()
    log_df[log_cols] = np.log(log_df[log_cols])
    if return_log_cols:
        return log_df, log_cols
    else:
        return log_df
    

def check_and_make_params_folder(savepath, params, paramfile=None, paramfilename='params.json', make_new_paramdir=True, save_early=False, skip_ui=False):
    """
    Given a savepath and a dictionary of params, check if there is a matching params file in the savepath. If there is, then set the savepath to the directory of the params file. If there is not, then create a new directory in the savepath/params directory and set the savepath to that directory.
    Inputs:
        savepath: the path to the directory to save the data
        params: a dictionary of parameters
        paramfile or paramfilename: the name of the file to save the params to
        make_new_paramdir: boolean, whether or not to create a new directory for the params
        save_early: boolean, whether or not to save the params early (if False, must save the params manually later)
        skip_ui: boolean, whether or not to skip the user interface confirmation
        
    Outputs:
        savepath: the path to the directory to save the data
        found_match: boolean, whether or not a matching params file was found
    """
    # load all the directory names that are children of os.path.join(savepath, 'params')
    found_match = False
    if paramfile is not None: # lazy fix instead of finding all calls everywhere
        paramfilename = paramfile
    # make sure that savepath is formatted as a directory
    if os.path.exists(os.path.join(savepath, 'params')):
        paramdirs = os.listdir(os.path.join(savepath, 'params'))
    else:
        # inform the user that the savepath given does not have a params directory and see if they want to create one
        if skip_ui:
            uin = 'y'
        else:
            uin = input(f"Savepath {savepath} does not have a params directory. Would you like to create one? (y/n)")
        if uin == 'y':
            if make_new_paramdir:
                os.makedirs(os.path.join(savepath, 'params'))
                paramdirs = os.listdir(os.path.join(savepath, 'params'))
            else:
                raise ValueError("Savepath does not have a params directory")
        else:
            raise ValueError("Savepath does not have a params directory")

    # now search all the json files in the paramdirs using glob
    paramfiles = []
    for paramdir in paramdirs:
        paramfiles.extend(glob.glob(os.path.join(savepath, 'params', paramdir, '*.json')))
    # if any of these paramfiles match the params, then load the data
    for paramfile in paramfiles:
        with open(paramfile, 'r') as f:
            loaded_params = json.load(f)
            if params == loaded_params:
                print("Found matching params file: {}".format(paramfile))
                newsavepath = os.path.dirname(paramfile)
                print("Setting savepath to {}".format(newsavepath))
                found_match = True
    if not found_match:
        num_paramdirs = len(paramdirs)
        new_paramdir = os.path.join(savepath, 'params', f'params{str(num_paramdirs)}')
        if not skip_ui:
            print("No matching params file found for params:")
            print(params)
            uin = input(f"Would you like to create a new params directory {new_paramdir}? (y/n)")
            if uin != 'y':
                raise ValueError("Not creating new params directory")
        print(f"No matching params file found, creating new directory {new_paramdir}")
        if make_new_paramdir:
            if not os.path.exists(new_paramdir):
                os.makedirs(new_paramdir)
            else:
                raise ValueError(f"New paramdir {new_paramdir} already exists")
            print("About to save params? ", save_early)
            if save_early:
                with open(os.path.join(new_paramdir, paramfilename), 'w') as f:
                    json.dump(params, f)
                print(f"Saved params to {os.path.join(new_paramdir, paramfilename)}")
        newsavepath = new_paramdir

    return newsavepath, found_match

# need a function to repeat the values in a column through nans until the next nonnan value appears
def replicate_value_through_nans(df, col='SubjectIDNum.21'):
    """
    df: dataframe
    col: column to replicate values through nans
    """
    new_df= df.copy(deep=True)
    old_val = np.nan
    for idx, row in df.iterrows():
        if pd.isnull(row[col]):
            new_df.loc[idx, col] = old_val
        else:
            old_val = row[col]
            new_df.loc[idx, col] = old_val
    return new_df

def make_subset_from_col(df, subj_col, val_col, col_pos_opts):
    """
    df: dataframe
    subj_col: where to find the subject IDs
    val_col: where to find the value of interest:
    col_pos_opts: dictionary containing the assignment as the key and the value in val_col as the value
        eg: {1: ['Positive'], 0: ['Negative']} or {2: ['Cocaine'], 1: ['Opiate'], 0: []}
        The value for empty list will be the catch all for everything else
    """
    nonnansubjs = []
    val_col_values = []
    subjs = df[subj_col].unique().tolist()
    for subj in subjs:
        # if not nan
        if not np.isnan(subj) and not subj == 'nan':
            print(f"Subject {subj}")
            subj_df = df[df[subj_col] == subj]
            for key, val in col_pos_opts.items():
                if any([v in val for v in subj_df[val_col].values]):
                    print(f"is {key} because: { subj_df[val_col].values[np.array([v in val for v in subj_df[val_col].values])]}")
                    val_col_values.append(key)
                    if len(val) > 0:
                        break
                elif len(val) == 0:
                    print(f"Subject {subj} does not have any of {val}")
                    val_col_values.append(key)
                    break

            nonnansubjs.append(subj)
        else:
            print('nan')

    print(f"Number of subjects: {len(nonnansubjs)}")
    print(f"Number of values: {len(val_col_values)}")

    val_list_dict = {key: [] for key in col_pos_opts.keys()}
    for subj, val in zip(nonnansubjs, val_col_values):
        val_list_dict[val].append(subj)

    return val_list_dict 

def get_prior_subjId(df, coloi, subject_colbase='SubjectIDNum'):
    """
    Given a dataframe and a column of interest,
    find the nearest preceding column that contains the subject_colbase
    """
    # get the index of the column of interest
    coloi_idx = df.columns.get_loc(coloi)

    # get the index of the nearest preceding subject column
    prior_subj_col_idx = coloi_idx

    while prior_subj_col_idx > 0:
        prior_subj_col_idx -= 1
        if subject_colbase in df.columns[prior_subj_col_idx]:
            break
    # get the subject ID column name
    return prior_subj_col_idx

# https://stackoverflow.com/questions/45393694/size-of-a-dictionary-in-bytes
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def make_dict_saveable(dwarray):
    """
    Given a dictionary with numpy arrays, convert the numpy arrays to lists so that they can be saved as json
    """
    new_dict = {}
    for k, v in dwarray.items():
        if type(v) == np.ndarray:
            new_dict[k] = v.tolist()
        elif type(v) == dict:
            new_dict[k] = make_dict_saveable(v)
        elif type(v) == pd.Index:
            new_dict[k] = v.tolist()
        else:
            try:
                new_dict[k] = v.tolist()
            except:
                try:
                    json.dumps(v)
                    new_dict[k] = v
                except (TypeError, OverflowError):
                    try:
                        json.dumps(float(v))
                        new_dict[k] = float(v)
                    except (TypeError, OverflowError):
                        try:
                            json.dumps([val.__class__.__name__ for val in v])
                            new_dict[k] = [val.__class__.__name__ for val in v]
                        except:
                            try:
                                print(f"Type {type(v)} with 0th element {type(v[0])} not recognized for key {k}")
                            except:
                                print(f"Type {type(v)} not recognized for key {k}")
    return new_dict

def sorted_cluster_labeling(data, column, n_clusters=7, random_state=0):
    # Perform k-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=random_state).fit(data[column].values.reshape(-1, 1))
    kmeans_df = pd.DataFrame({column: data[column], 'Cluster': kmeans.labels_})

    # Sort the dataframe by the specified column
    kmeans_df = kmeans_df.sort_values(column)

    # Map the cluster names to A, B, C, ...
    cluster_map = {i: chr(65 + i) for i in range(n_clusters)}  # chr(65) is 'A'
    kmeans_df['Cluster'] = kmeans_df['Cluster'].map(cluster_map)

    # Get the unique cluster labels in their new order
    cluster_orders = [idx for idx in enumerate(kmeans_df['Cluster'].unique())]

    # Create a new mapping from the old cluster labels to their new order
    new_map = {cluster: order for order, cluster in cluster_orders}

    # Apply the new mapping to the 'Cluster' column
    kmeans_df['Cluster'] = kmeans_df['Cluster'].map(new_map)

    return kmeans_df


if __name__ == '__main__':
    signal = np.random.rand(10, 105, 100)
    axis = 2
    num_points = 45
    print("Original signal shape: {}".format(signal.shape))
    print("Downsampled signal shape: {}".format(downsample_signal(signal, num_points, axis).shape))

