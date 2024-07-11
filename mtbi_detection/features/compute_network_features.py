import numpy as np
import pandas as pd
import time
import os
import networkx as nx
import bct
from numpy.typing import NDArray
import json
import argparse
from typing import List, Tuple
import mne

import mtbi_detection.data.data_utils as du
import mtbi_detection.data.load_dataset as ld
import mtbi_detection.features.feature_utils as fu
import mtbi_detection.features._make_defined_network as mdn
import mtbi_detection.data.load_open_closed_data as locd

LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')

# get the eigendecomposition of the laplacian:
def eigendecomposition(L):
    """
    Input: Laplacian matrix (n x n)
    Output: eigenvalues, eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(L)
    # sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1] # sort in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    assert np.allclose(np.dot(L, eigenvectors), eigenvalues*eigenvectors)
    # assert in descending order
    assert np.all(np.diff(eigenvalues) <= 0)

    return eigenvalues, eigenvectors

## Deviation Range, Ratio of Deviation Range, and Convex Sum defined in Anand et al. (2022) https://ieeexplore.ieee.org/abstract/document/10040075
def compute_deviation_range(L=None, eigenvalues=None):
    """
    l2 - lN since l1 is constant for all smooth networks
    """
    if eigenvalues is None:
        eigenvalues, _ = eigendecomposition(L)
    # assert that the eigenvalues are in descending order
    assert np.all(np.diff(eigenvalues) <= 0)
    return np.abs(eigenvalues[1] - eigenvalues[-1])

def compute_ratio_dev_range(L=None, eigenvalues=None):
    """
    np.abs((l2 - l_{n/2})) / np.abs((l_{n/2} - lN))
    Not sure what to do if n is odd, maybe just take n//2 and n//2+1
    """
    if eigenvalues is None:
        eigenvalues, _ = eigendecomposition(L)
    # assert that the eigenvalues are in descending order
    assert np.all(np.diff(eigenvalues) <= 0)
    n = len(eigenvalues)
    upper_ratio = np.abs(eigenvalues[1] - eigenvalues[n//2])
    if n % 2 == 0:
        lower_ratio = np.abs(eigenvalues[n//2] - eigenvalues[-1])
    else:
        lower_ratio = np.abs(eigenvalues[n//2+1] - eigenvalues[-1])
    ratio = upper_ratio / lower_ratio
    return ratio

def compute_convex_sum(L=None, eigenvalues=None, tau=0.5, dev_range=None, ratio_dev_range=None):
    """
    Given as tau*deviation_range + (1-tau)*ratio_dev_range
    """
    if dev_range is None:
        dev_range = compute_deviation_range(L=L, eigenvalues=eigenvalues)
    if ratio_dev_range is None:
        ratio_dev_range = compute_ratio_dev_range(L=L, eigenvalues=eigenvalues)
    return tau*dev_range + (1-tau)*ratio_dev_range


def get_network_weighted_metrics(connectivity_matrix: NDArray, channels: List) -> Tuple[List, List]:
    """
    Compute the network features given the connectivity matrix
    # reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7597206/
    # reference: https://www.biorxiv.org/content/10.1101/2021.02.15.431255v1.full.pdf
    # reference: https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6287639/8948470/9174714/ismai.t3-3018995-large.gif
    # bct reference: https://github.com/CarloNicolini/pyconnectivity/blob/master/bct.py
    """

    # Construct Graph
    assert connectivity_matrix.shape[0] == connectivity_matrix.shape[1] == len(channels)
    np.fill_diagonal(connectivity_matrix, 0)
    network = nx.from_numpy_array(connectivity_matrix, create_using=nx.Graph)
    
    
    mapping = {i: k for i, k in enumerate(channels)}
    network = nx.relabel_nodes(network, mapping)
    network_distance_dict = {(e1, e2): 1 / abs(weight) for e1, e2, weight in network.edges(data='weight')}
    nx.set_edge_attributes(network, values=network_distance_dict, name='distance')

    # Node features
    degree_centrality = [network.degree(weight='weight')[ch] for ch in channels]

    try:
        eigen_centrality = [nx.eigenvector_centrality(network, max_iter=1000, weight="weight")[ch] for ch in channels]
    except:
        print("Error with eigen_centrality")
        eigen_centrality = [0] * len(channels)
    closeness_centrality = [nx.closeness_centrality(network, distance='distance')[ch] for ch in channels]
    betweenness_centrality = [nx.betweenness_centrality(network, weight='distance', normalized=True)[ch] for ch in channels]
    try:
        pagerank = [nx.pagerank(network, weight="weight", max_iter=1000)[ch] for ch in channels]
    except:
        # print("Error with pagerank")
        pagerank = [0] * len(channels)

    # Graph features
    average_clustering = [nx.average_clustering(network, weight="weight")]
    shortest_path_length = [nx.average_shortest_path_length(network, weight="distance")]
    efficiency = [bct.efficiency_wei(connectivity_matrix, local=False)]
    number_partitions = [len(list(nx.algorithms.community.louvain_partitions(network, weight="weight"))[0])]

    deviation_range = [compute_deviation_range(nx.laplacian_matrix(network).todense())]
    ratio_dev_range = [compute_ratio_dev_range(nx.laplacian_matrix(network).todense())]

    # Combine all features and names
    all_features = degree_centrality + eigen_centrality + closeness_centrality + betweenness_centrality + pagerank + shortest_path_length + \
        average_clustering + efficiency + number_partitions + deviation_range + ratio_dev_range
    all_feature_names = []
    feature_names = ["degree centrality", "eigen centrality", "closeness centrality", "betweenness centrality", "page rank"]
    for feature in feature_names:
        for channel in channels:
            all_feature_names.append(channel + " " + feature)
    all_feature_names = all_feature_names + ["shortest path length", "average clustering", "global efficiency", "number of partitions", "deviation range", "ratio deviation range"]
    assert len(all_features) == len(all_feature_names)
    return all_features, all_feature_names

def get_all_subj_network_weighted_metrics(subj_connectivity_matrix: NDArray, channels: List, quick_print=None) -> Tuple[List, List]:
    """
    Given the connectivity matrix of a subject, compute the network features
    Inputs:
        subj_connectivity_matrix: (n_samples, n_channels, n_channels)
        channels: List of channel names
        quick_print: str, to print a quick message
    Outputs:
        all_features_array: (n_samples, n_features)
        all_feature_names: List of feature names
    """
    if quick_print is not None:
        print(quick_print)
    all_features = []
    all_feature_names = []
    n_samples = subj_connectivity_matrix.shape[0]
    assert len(subj_connectivity_matrix.shape) == 3, "subj_connectivity_matrix must be 3D"
    for idx in range(n_samples):
        subj_connectivity = subj_connectivity_matrix[idx, ...]
        subj_features, subj_feature_names = get_network_weighted_metrics(subj_connectivity, channels)
        all_features.append(subj_features)
        all_feature_names.append(subj_feature_names)

    all_features_array = np.stack(all_features)
    assert all_features_array.shape[0] == subj_connectivity_matrix.shape[0]
    assert all_features_array.shape[1] == len(all_feature_names[0])
    assert all([all_feature_names[0] == subj_feature_names for subj_feature_names in all_feature_names])
    
    return all_features_array, all_feature_names[0]

def _check_network_params(network_params):
    """
    Checks the parameters used to compute the network features
    """
    valid_params = ['band_method', 'n_divisions', 'log_division', 'network_methods']
    assert [key in valid_params for key in network_params.keys()], f"Invalid network_params: {network_params.keys()}"
    assert network_params['band_method'] in ['standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250', 'custom'], f"Invalid band_method: {network_params['band_method']}"
    valid_network_methods = ['coherence', 'mutual_information', 'inverse_distance', 'spearman', 'pearson', 'plv', 'pli']
    assert all([method in valid_network_methods for method in network_params['network_methods']]), f"Invalid methods: {network_params['methods']}"

def _check_open_closed_params(open_closed_params):
    """
    Checks the parameters used to load the open closed pathdict
    """
    valid_params = ['l_freq', 'h_freq', 'fs_baseline', 'order', 'notches', 'notch_width', 'num_subjs', 'verbose', 'reference_method', 'reference_channels', 'keep_refs', 'bad_channels', 'filter_ecg', 'late_filter_ecg', 'ecg_l_freq', 'ecg_h_freq', 'ecg_thresh', 'ecg_method', 'include_ecg']
    assert [key in valid_params for key in open_closed_params.keys()], f"Invalid open_closed_params: {open_closed_params.keys()}"
    assert open_closed_params['l_freq'] >= 0, f"Invalid l_freq: {open_closed_params['l_freq']}"
    assert open_closed_params['h_freq'] is None or open_closed_params['h_freq'] > open_closed_params['l_freq'], f"Invalid h_freq: {open_closed_params['h_freq']}"
    assert open_closed_params['fs_baseline'] > 0, f"Invalid fs_baseline: {open_closed_params['fs_baseline']}"
    if open_closed_params['h_freq'] is not None:
        assert open_closed_params['fs_baseline'] > open_closed_params['fs_baseline'], f"Invalid fs_baseline: {open_closed_params['fs_baseline']}"
    assert open_closed_params['order'] > 0, f"Invalid order: {open_closed_params['order']}"
    

def compute_networks(network_params, open_closed_pathdict, channels, verbosity=1, fs=500, n_divisions=1, log_division=True, min_freq=0.3):

    """
    Given the network parameters, and the data of the segmented open and closed datapaths, compute the networks
    Inputs:
        network_params: mapping of parameters to make the networks: {'band_method', 'n_divisions', 'log_division', 'methods', 'inverse_numerator'}
        open_closed_pathdict: mapping of the open and closed data paths for each subject
        channels: list of channels to use in the network
        verbosity: int, 0 for no print, 1 for some print, 2 for more print
        fs: int, sampling frequency of the data
        min_freq: float, minimum frequency to use in the network
    Outputs:
        all_networks: list of the networks for each subject
        all_network_names: list of the names of the networks

    """
    band_method = network_params['band_method']
    network_methods = network_params['network_methods']


    subjs = [s for s in open_closed_pathdict.keys() if str(s).isnumeric()]
    open_raws = [[mne.io.read_raw_fif(rawfile, verbose=0).pick(channels) for rawfile in open_closed_pathdict[subj]['open']] for subj in subjs]
    closed_raws = [[mne.io.read_raw_fif(rawfile, verbose=0).pick(channels) for rawfile in open_closed_pathdict[subj]['closed']] for subj in subjs]
    open_data = [[raw.get_data(picks=channels) for raw in raws] for raws in open_raws]
    closed_data = [[raw.get_data(picks=channels) for raw in raws] for raws in closed_raws]
    open_times = [[raw.times for raw in raws] for raws in open_raws]
    closed_times = [[raw.times for raw in raws] for raws in closed_raws]

    assert all([raw.ch_names == channels for raws in open_raws for raw in raws]), "Not all open raws have the same channels"
    assert all([raw.ch_names == channels for raws in closed_raws for raw in raws]), "Not all closed raws have the same channels"


    # make the bands and give them arbitrary names (band0, band1, etc)
    bands_dict = make_dummy_bands(basis=band_method, divisions=n_divisions, log_division=log_division, fs=fs, verbosity=1, min_freq=min_freq)

    open_networks = []
    closed_networks = []
    network_names = []
    if verbosity >0:
        print(f"Computing networks for {network_methods}")
        computetime = time.time()
    for method in network_methods:
        assert method in ['coherence', 'mutual_information', 'inverse_distance', 'spearman', 'pearson', 'plv', 'pli']
        closed_network = mdn.get_network(closed_data, subjs, closed_times, channels=channels, bands=bands_dict, method=method, fs=fs)
        open_network = mdn.get_network(open_data, subjs, open_times, channels=channels, bands=bands_dict, method=method, fs=fs)
        if verbosity > 1:
            print(f"Computed {method} network")
        # on methods that have band-specific networks, we need to separate them out
        if method == 'coherence' or method=='plv' or method=='pli':
            assert len(closed_network.shape) == 4 and len(open_network.shape) == 4
            assert closed_network.shape[1] == open_network.shape[1] == len(bands_dict)
            open_band_networks = [open_network[:, i, :, :] for i in range(open_network.shape[1])]
            closed_band_networks = [closed_network[:, i, :, :] for i in range(closed_network.shape[1])]
            open_networks.extend(open_band_networks)
            closed_networks.extend(closed_band_networks)
            band_network_names = [f"{method}_{bands_dict[band]}" for band in bands_dict.keys()]
            network_names.extend(band_network_names)

        else:
            open_networks.append(open_network)
            closed_networks.append(closed_network)
            if method == 'mutual_information':
                network_names.append('mi')
            elif method == 'inverse_distance':
                network_names.append('inv')
            else:
                network_names.append(method)

    if verbosity >0:
        print(f"Finished computing networks for {network_methods} in {time.time() - computetime} seconds")

    all_networks = open_networks + closed_networks
    all_network_names = [f"open_{network_name}" for network_name in network_names] + [f"closed_{network_name}" for network_name in network_names]
    return all_networks, all_network_names
            
def make_dummy_bands(basis='custom', divisions=2, log_division=False, custom_bands=None, fs=500, verbosity=1, min_freq=0.3):
    """
    Given the band parameters, make the bands
    Inputs:
        basis: str, the basis to make the bands, 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250', 'custom'
        divisions: int, the number of divisions to make the bands
        log_division: bool, whether to use log division
        custom_bands: dict, the custom bands to use
        fs: int, the sampling frequency
        verbosity: int, 0 for no print, 1 for some print, 2 for more print
        min_freq: float, the minimum frequency to use in the bands
    Outputs:
        dummy_bands: mapping of form {'band_name': (low_freq, high_freq)}
    
    """
    # dictionary of form {'band_name': (low_freq, high_freq)}
    bands = fu.make_bands(basis=basis, divisions=divisions, log_division=log_division, custom_bands=custom_bands, fs=fs, verbosity=verbosity, min_freq=min_freq)
    
    dummy_keys = [f"band{idx}" for idx in range(len(bands))]
    dummy_bands = dict(zip(dummy_keys, bands))
    return dummy_bands

def main(network_params, open_closed_params, channels, verbosity=1, save=True, featurepath=FEATUREPATH, choose_subjs='train', internal_folder='data/internal/'):
    """
    Given parameters to make the networks, compute the network features and save them to a csv file
    NOTE: loads all the segmented EO/EC data into RAM
    Inputs:
        network_params: mapping of parameters to make the networks: {'band_method', 'n_divisions', 'log_division', 'methods'}
        open_closed_params: mapping of parameters to load the open and closed data
        channels: list of channels to use in the network
        verbosity: int, 0 for no print, 1 for some print, 2 for more print
        save: bool, whether to save the network features to a csv file
        featurepath: str, path to save the network features
        choose_subjs: str, 'train' or 'test' to choose the subjects to compute the network features
        internal_folder: str, path to the internal folder to load the splits
    Outputs:
        network_features_df: DataFrame of the network features with the subjects as the index
    """
    _check_network_params(network_params)
    _check_open_closed_params(open_closed_params)
    networktime=time.time()
    all_params = du.make_dict_saveable({**network_params, **open_closed_params, 'channels': list(channels), 'choose_subjs': choose_subjs})
    network_feature_path = os.path.join(featurepath, 'network_features')
    du.clean_params_path(network_feature_path)
    network_feature_savepath, found_match = du.check_and_make_params_folder(network_feature_path, all_params)
    savefilename = os.path.join(network_feature_savepath, 'defined_network_features.csv')
    if found_match:
        if verbosity > 0:
            print(f"Found matching network features file, loading from {savefilename}")
        network_features_df = pd.read_csv(os.path.join(network_feature_savepath, 'network_features.csv'), index_col=0)
        assert set(fu.select_subjects_from_dataframe(network_features_df, choose_subjs,internal_folder).index) == set(network_features_df.index)
    else:
        if verbosity > 0:
            print(f"Didn't find matching network features file, computing and saving to {savefilename}")
    
        open_closed_pathdict = locd.load_open_closed_pathdict(savepath=LOCD_DATAPATH, **open_closed_params)
        open_closed_pathdict = fu.select_subjects_from_dict(open_closed_pathdict, choose_subjs, internal_folder=internal_folder)


        subjs = [s for s in open_closed_pathdict.keys() if str(s).isnumeric()]
        split_subjs = ld.load_splits(internal_folder=internal_folder)[choose_subjs]
        assert [int(s) in split_subjs for s in subjs]

        ## load up all the adjacency matrices
        networks, network_names = compute_networks(network_params, open_closed_pathdict=open_closed_pathdict, channels=channels, fs=open_closed_params['fs_baseline'], min_freq=open_closed_params['l_freq'], verbosity=verbosity)

        assert len(networks) == len(network_names)
        assert all([network.shape[-2] == len(channels) for network in networks]), f"Graphs shape doesn't match channels: {[(network.shape, network_name) for network, network_name in zip(networks, network_names)]}"
        assert all([network.shape[-1] == len(channels) for network in networks]), f"Graphs shape doesn't match channels: {[(network.shape, network_name) for network, network_name in zip(networks, network_names)]}"

        network_features_tups = [get_all_subj_network_weighted_metrics(network, channels, quick_print=f'Computing features for {network_names[idx]}') for idx, network in enumerate(networks)]
        network_features = [gft[0] for gft in network_features_tups]
        network_feature_names = network_features_tups[0][1]

        # now let's make a dataframe with it all
        column_names = [f"Network_{network_name}_{feature_name}" for network_name in network_names for feature_name in network_feature_names]
        network_features_df = pd.DataFrame(np.concatenate(network_features, axis=1), columns=column_names)
        subj_ints = np.array([int(subj) for subj in subjs])
        network_features_df.index = subj_ints
        
        if save:
            # save this to a csv
            network_features_df.to_csv(savefilename)
            print(f"Saved to {savefilename}")
            # save all_params to savepath/params.json
            with open(os.path.join(network_feature_savepath, 'params.json'), 'w') as f:
                json.dump(all_params, f)

    if verbosity >0:
        print(f"Finished computing network features in {time.time() - networktime} seconds")
    return network_features_df

if __name__ == '__main__':

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


    # Network params
    parser.add_argument('--band_method', type=str, default='custom', help="Possible options: 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--network_methods', type=str, nargs='+', default=['coherence', 'mutual_information', 'spearman', 'pearson', 'plv', 'pli']) #  'plv', 'pli'
    
    all_params = vars(parser.parse_args())
    network_params = {key: value for key, value in all_params.items() if key in ['band_method', 'network_methods']}
    locd_params = {key: value for key, value in all_params.items() if key not in network_params}
    channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

    main(network_params, locd_params, channels)