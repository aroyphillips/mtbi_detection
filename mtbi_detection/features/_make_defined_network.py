from typing import Dict
import numpy as np
from numpy import ndarray as NDArray
from scipy import signal
from sklearn.metrics import mutual_info_score
import scipy
import time
import mne_connectivity

def get_network(eeg_data, subjs, times, channels, bands={'Theta': (4, 8)}, fs=500, method='coherence', **kwargs):
    """
    Given the eeg_data, subjs, and times, this function will return the "method" network in the form of the adjacency matrix
    Inputs:
        eeg_data: List([list(num_chans, num_times)) len(num_subjs)
        subjs: List(str) len(num_subjs)
        times: List(num_times) len(num_subjs)
        bands: dict in the form {'band_name': (low_freq, high_freq)}
        method: str, one of ['coherence']
        verbose: bool

    Outputs:
        network: List(num_chans, num_chans) len(num_subjs) defined by
            the method and bands
    """
    valid_methods = ['coherence', 'inverse_distance', 'mutual_information', 'spearman', 'pearson', 'plv', 'pli']
    assert method in valid_methods, f"method must be one of {valid_methods}"
    assert len(eeg_data) == len(subjs) == len(times), 'eeg_data, subjs, and times must have the same length'
    assert all([len(channels) == eeg_data[idx][0].shape[0] for idx in range(1, len(eeg_data))]), f'All eeg_data must have the same number of channels: shapes: {[data[0].shape for data in eeg_data]}'
    len_times = [[len(t) for t in time] for time in times]
    if method == 'coherence':
        all_networks = make_coherence_networks(eeg_data, subjs, times, channels=channels, bands=bands, method=method, **kwargs)
    elif method == 'mutual_information':
        all_networks = np.zeros((len(subjs), eeg_data[0][0].shape[0], eeg_data[0][0].shape[0]))
        for idx, data_list in enumerate(eeg_data):
            concat_data = np.concatenate(data_list, axis=1)
            network = make_undirected_weighted_matrix_with_feature(concat_data, channels=channels, method='mutual_information')
            all_networks[idx, :, :] = network
            time.sleep(0)
    elif method == 'spearman' or method == 'pearson':
        all_networks = np.zeros((len(subjs), eeg_data[0][0].shape[0], eeg_data[0][0].shape[0]))
        for idx, data_list in enumerate(eeg_data):
            networks = []
            for data in data_list:
                network = make_undirected_weighted_matrix_with_feature(data, channels=channels, method=method)
                networks.append(network)
            sum_times = sum(len_times[idx])
            reweight_network = [g*t/sum_times for g, t in zip(networks, len_times[idx])]
            reweight_network = np.sum(reweight_network, axis=0)
            all_networks[idx, :, :] = reweight_network

    elif method == 'plv' or method == 'pli':
        all_networks = np.zeros((len(subjs), len(bands.keys()), eeg_data[0][0].shape[0], eeg_data[0][0].shape[0]))
        for idx, data_list in enumerate(eeg_data):
            networks = []
            for data in data_list:
                network = get_connectivity_measure(data, fs, bands, method=method)
                networks.append(network)
            sum_times = sum(len_times[idx])
            reweight_network = [g*t/sum_times for g, t in zip(networks, len_times[idx])]
            reweight_network = np.sum(reweight_network, axis=0)
            all_networks[idx, :, :, :] = reweight_network
    

    else:
        raise ValueError(f"Method {method} not recognized")

    return all_networks

def make_coherence_networks(eeg_data, subjs, times, bands={'Theta': (4, 8)}, **kwargs):
    """
    Given the eeg_data, subjs, and times, this function will return the coherence
    networks for each subject. The coherence networks are defined by the bands
    and the method.
    Inputs:
        eeg_data: List([list(num_chans, num_times)) len(num_subjs)
        subjs: List(str) len(num_subjs)
        times: List(num_times) len(num_subjs)
        bands: dict in the form {'band_name': (low_freq, high_freq)}
        method: str, one of ['coherence']
        verbose: bool
    Outputs:
        all_networks: NDArray(num_subjs, num_bands, num_chans, num_chans)
    
    """
    len_times = [[len(t) for t in time] for time in times]
    all_networks = np.zeros((len(subjs), len(bands), eeg_data[0][0].shape[0], eeg_data[0][0].shape[0]))
    for idx, data_list in enumerate(eeg_data):
        networks = []
        for data in data_list:
            network = get_coherence(data, 500, eeg_bands=bands) # (num_bands, num_chans, num_chans)
        # if network.shape[0] == 1:
        #     network = np.squeeze(network, axis=0)
            assert network.shape[-2] == network.shape[-1], 'network must be square'
            networks.append(network)
        sum_times = sum(len_times[idx])
        reweight_network = [g*t/sum_times for g, t in zip(networks, len_times[idx])]
        reweight_network = np.sum(reweight_network, axis=0)
        all_networks[idx, :, :, :] = reweight_network
        time.sleep(0)
    return all_networks


def get_coherence(data: NDArray, fs: int, eeg_bands = {'Delta': (0.3, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 13),
                 'Beta': (13, 36),
                 'Gamma': (36, 45),
                 'High Gamma': (45, np.inf)}) -> NDArray:
    """
    Computes the coherence between all pairs of channels in the data
    data.
    Inputs:
        data: (num_chans, num_times)
        fs: sampling frequency
        eeg_bands: dict in the form {'band_name': (low_freq, high_freq)}
    Outputs:
        coherence: (num_bands, num_chans, num_chans)
    """
    
    coherence = np.zeros((len(eeg_bands), data.shape[0], data.shape[0]))
    for row in range(0, data.shape[0]-1):
        for col in range(row+1, data.shape[0]):
            f, cxy = signal.coherence(data[row,:], data[col, :], fs=fs, nperseg=50, nfft=512)
            for idx, band in enumerate(eeg_bands):
                max_freq = eeg_bands[band][1]
                min_freq = eeg_bands[band][0]
                freq_ix = np.where((f >= eeg_bands[band][0]) &
                                   (f <= eeg_bands[band][1]))[0]
                coherence[idx, row, col] = np.mean(cxy[freq_ix])
                coherence[idx, col, row] = np.mean(cxy[freq_ix])
    return coherence

def make_undirected_weighted_matrix_with_feature(raw_data, channels, method='spearman', verbose=False):
    """
    Parameters:
        raw_data: (num_channels, num_times)
        channels: list of channels to use
        method: method to use for calculating edge weights
    Returns:
        weighted_matrix: weighted adjacency matrix
    """
    assert len(channels) == raw_data.shape[0], "All eeg data must have the same number of channels"
        
    weighted_matrix = np.zeros((len(channels), len(channels)))
     # this uses the correlation coefficient on the raw eeg_segment
    if method == 'pearson':
        # pearson's
        for i in range(0, len(channels)-1):
            for j in range(i, len(channels)):
                weighted_matrix[i,j] = np.corrcoef(np.squeeze(raw_data[i,:]), np.squeeze(raw_data[j,:]))[0,1]
    elif method == 'spearman':
        for i in range(0, len(channels)-1):
            for j in range(i, len(channels)):
                weighted_matrix[i,j] = scipy.stats.spearmanr(np.squeeze(raw_data[i,:]), np.squeeze(raw_data[j,:]), axis=0, nan_policy='propagate', alternative='two-sided')[0]
    elif method == 'mutual_information':
        # Used this equation: https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information
        for i in range(0, len(channels)-1):
            for j in range(i, len(channels)):
                mi = calc_MI(np.squeeze(raw_data[i,:]), np.squeeze(raw_data[j,:]), bins=20)
                weighted_matrix[i,j] = mi
    else:
        raise ValueError(f"Method {method} not recognized")
    # make symmetric
    if verbose:
        for row in weighted_matrix:
            print("ROW", row)
        print('Making symmetric')
    weighted_matrix = weighted_matrix + weighted_matrix.T - np.diag(weighted_matrix.diagonal())
    return weighted_matrix

def get_connectivity_measure(data: NDArray, fs: int, eeg_bands: Dict, method='pli') -> NDArray:
    """
    Inputs:
        data: (num_channels, num_times)
        fs: sampling frequency
        eeg_bands: dict in the form {'band_name': (low_freq, high_freq)}
        method: 'pli' or 'plv'
    Outputs:
        phase_lag_index: (num_bands, num_channels, num_channels)
    """
    assert method in ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased'], "method must be one of ['pli', 'plv']"
    connecivity_matrix = np.zeros((len(eeg_bands.keys()), data.shape[0], data.shape[0]))
    reshaped_data = data.reshape((1, data.shape[0], data.shape[1]))
    for idx, band in enumerate(eeg_bands):
        phase_lag_index_band = mne_connectivity.spectral_connectivity_epochs(reshaped_data, method=method, sfreq=fs, mode='multitaper',
                                                                             fmin=eeg_bands[band][0],
                                                                             fmax=eeg_bands[band][1], faverage=True, verbose=0)
        connecivity_matrix [idx, :, :] = phase_lag_index_band.xarray.data.reshape((data.shape[0], data.shape[0]))
    return connecivity_matrix 

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
