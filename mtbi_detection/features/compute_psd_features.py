
import argparse
import numpy as np
import pandas as pd
import os
import json
import scipy.stats as stats
import mtbi_detection.data.data_utils as du
import mtbi_detection.data.load_dataset as ld
import mtbi_detection.data.transform_data as td
import mtbi_detection.features.feature_utils as fu

CHANNELS = ['C3','C4','Cz','F3','F4','F7','F8','Fp1','Fp2','Fz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')
TDPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'psd_transform')

def compute_psd_features(transform_data_dict, choose_subjs=None, ratio=False, channels=CHANNELS, state='all', band_method='custom', bin_methods='all',verbosity=0, l_freq=0.3, fs=500, save=True, internal_folder='data/internal/', featurepath=FEATUREPATH):
    """
    Given the output of transform_data, compute the PSD features for each subject 
    and return a dataframe with the features as columns and the subjects as rows.
    Args:
        transform_data_dict (dict): Dictionary output of transform_data {subj_data: {subj: /path/to/data/}}, 'channels': channels, 'common_freqs': common_freqs}
        choose_subjs (str): the dataset cohort split to use ('train', 'ival', 'holdout', 'dev')
        ratio (bool): whether to compute ratios between bins
        channels (list): list of channels to use
        state (str): which state to use. One of ['open', 'closed', 'all']
        band_method (str): method for computing frequency bands. One of ['standard', 'custom', 'standard+custom']
        bin_methods (str): methods for binning the PSD. One of ['all', 'avg', 'sum', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'iqr']
    
    """    

    # check inputs
    all_bin_methods = ['avg', 'sum', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'iqr']
    if bin_methods == 'all' or 'all' in bin_methods:
        bin_methods = ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'iqr']
    elif type(bin_methods) == str:
        bin_methods = [bin_methods]
    assert all([method in all_bin_methods or (method[0]=='p' and method[1:].isdigit()) for method in bin_methods])
    assert choose_subjs in ['train', 'ival', 'holdout', 'dev', None]

    # make the savepath
    power_params = {'band_method': band_method, 'bin_methods': bin_methods, 'state': state, 'choose_subjs': choose_subjs}
    power_path = os.path.join(featurepath, 'psd_power_features')
    if not os.path.exists(power_path):
        os.mkdir(power_path)
    du.clean_params_path(power_path)
    powersavepath, found_match = du.check_and_make_params_folder(power_path, power_params)
    if found_match:
        power_feature_df, found_match = load_power_features(powersavepath, ratio=ratio)   
    if not found_match:
        unraveled_mtd = td.unravel_multitaper_dataset(transform_data_dict['subj_data'])
        X_open = np.stack(unraveled_mtd['avg']['open_power'])
        X_closed = np.stack(unraveled_mtd['avg']['closed_power'])

        X_open = X_open[:, :, 1:-1]  # remove the first and last frequency bins
        X_closed = X_closed[:, :, 1:-1] # remove the first and last frequency bins
        freqs = transform_data_dict['common_freqs'][1:-1] # remove the first and last frequency bins

        # show that there are no zeros in the data
        assert np.all(X_open >= 0)
        assert np.all(X_closed >= 0)

        # log transform the data
        X_open = np.log10(X_open)
        X_closed = np.log10(X_closed)

        num_axis = len(X_open.shape)

        subjs = unraveled_mtd['avg']['open_subjs']
        assert subjs == unraveled_mtd['avg']['closed_subjs']
        channels = transform_data_dict['channels']
    
        assert subjs == unraveled_mtd['avg']['closed_subjs']
        assert X_open.shape[0] == len(subjs)
        assert X_closed.shape[0] == len(subjs)
        assert X_open.shape[1] == len(channels)
        assert X_closed.shape[1] == len(channels)
        assert X_open.shape[2] == len(freqs)
        assert X_closed.shape[2] == len(freqs)

        if choose_subjs is not None:
            select_subjs = ld.load_splits(internal_folder)[choose_subjs]
            subjs_idx = np.array([idx for idx, subj in enumerate(subjs) if int(subj) in select_subjs])
            subjs = list(np.array(subjs)[subjs_idx])
            X_open = X_open[subjs_idx]
            X_closed = X_closed[subjs_idx]

        bands = fu.make_bands(basis=band_method, verbosity=verbosity, fs=fs, min_freq=l_freq)
        if state == 'open':
            X = X_open
        elif state == 'closed':
            X = X_closed
        elif state == 'all':
            X = np.concatenate((X_open, X_closed), axis=1)
        else:
            raise ValueError(f"Segment {state} not recognized, must be 'open', 'closed', or 'all'")
        
        if state != 'all':
            all_binned_psds = []
            all_bin_feature_cols = []


            all_ratios = []
            all_ratio_feature_cols = []
            
            for bin_method in bin_methods:
                binned_psd = bin_psd_by_bands(X, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
                bin_feature_cols = [f'{state}_{band}_{bin_method}' for band in bands]
                if ratio: 
                    ratios, bin_mapping = make_ratios(binned_psd, return_bin_mapping=True)
                    ratio_feature_cols = [f'({bin_feature_cols[r1]})/({bin_feature_cols[r2]})' for r1, r2 in bin_mapping]
                    all_ratios.append(ratios)
                    all_ratio_feature_cols.append(ratio_feature_cols)
                all_binned_psds.append(binned_psd)
                all_bin_feature_cols.append(bin_feature_cols)
            
            # now we flatten the (n_subjs, n_channels, n_bins) array to (n_subs, n_channels * n_bins)
            flat_binned_psds = [binned_psd.reshape(binned_psd.shape[0], -1) for binned_psd in all_binned_psds]
            concat_flat_binned_psds = np.concatenate(flat_binned_psds, axis=-1)
            flat_feature_cols_bins = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_bin_feature_cols]
            concat_flat_feature_cols_bins = [feature for feature_cols in flat_feature_cols_bins for feature in feature_cols]
            
            assert all([flat_binned_psd[:, 0] == binned_psd[:, 0, 0] for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
            assert all([flat_binned_psd[:, 3] == binned_psd[:, 0, 3] for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
            assert all([np.all(flat_binned_psd[:, len(feature_cols):2*len(feature_cols)] == binned_psd[:, 1, :]) for flat_binned_psd, binned_psd, feature_cols in zip(flat_binned_psds, all_binned_psds, all_bin_feature_cols)])

            if ratio:
                flat_ratios = [ratios.reshape(ratios.shape[0], -1) for ratios in all_ratios]
                concat_flat_ratios = np.concatenate(flat_ratios, axis=-1)
                flat_feature_cols_ratios = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_ratio_feature_cols]
                concat_flat_feature_cols_ratios = [feature for feature_cols in flat_feature_cols_ratios for feature in feature_cols]
                assert all([flat_ratio[:, 0] == ratio[:, 0, 0] for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
                assert all([flat_ratio[:, 3] == ratio[:, 0, 3] for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
                assert all([np.all(flat_ratio[:, len(feature_cols):2*len(feature_cols)] == ratio[:, 1, :]) for flat_ratio, ratio, feature_cols in zip(flat_ratios, all_ratios, all_ratio_feature_cols)])

        else:
            all_binned_psds = []
            all_bin_feature_cols = []
            all_ratios = []
            all_ratio_feature_cols = []

            for bin_method in bin_methods:
                open_binned_psd = bin_psd_by_bands(X_open, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
                closed_binned_psd = bin_psd_by_bands(X_closed, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
                binned_psd = np.concatenate((open_binned_psd, closed_binned_psd), axis=-1)
                bin_feature_cols = [f'{oc}_{band}_{bin_method}' for oc in ['open', 'closed'] for band in bands]
                all_binned_psds.append(binned_psd)
                all_bin_feature_cols.append(bin_feature_cols)

                if ratio:
                    open_ratios, bin_mapping = make_ratios(open_binned_psd, return_bin_mapping=True)
                    closed_ratios, closed_bin_mapping = make_ratios(closed_binned_psd, return_bin_mapping=True)
                    assert bin_mapping == closed_bin_mapping
                    ratios = np.concatenate((open_ratios, closed_ratios), axis=-1)
                    ratio_feature_cols = [f'({bin_feature_cols[r1]})/({bin_feature_cols[r2]})' for r1, r2 in bin_mapping]
                    ratio_feature_cols = [f'{oc}_{feature}' for oc in ['open', 'closed'] for feature in ratio_feature_cols]
                    all_ratios.append(ratios)
                    all_ratio_feature_cols.append(ratio_feature_cols)

            flat_binned_psds = [binned_psd.reshape(binned_psd.shape[0], -1) for binned_psd in all_binned_psds]
            concat_flat_binned_psds = np.concatenate(flat_binned_psds, axis=-1)
            flat_feature_cols_bins = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_bin_feature_cols]
            concat_flat_feature_cols_bins = [feature for feature_cols in flat_feature_cols_bins for feature in feature_cols]
            assert all([np.all(flat_binned_psd[:, 0] == binned_psd[:, 0, 0]) for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
            assert all([np.all(flat_binned_psd[:, 3] == binned_psd[:, 0, 3]) for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
            assert all([np.all(flat_binned_psd[:, len(feature_cols):2*len(feature_cols)] == binned_psd[:, 1, :]) for flat_binned_psd, binned_psd, feature_cols in zip(flat_binned_psds, all_binned_psds, all_bin_feature_cols)])

            if ratio:
                flat_ratios = [ratios.reshape(ratios.shape[0], -1) for ratios in all_ratios]
                concat_flat_ratios = np.concatenate(flat_ratios, axis=-1)
                flat_feature_cols_ratios = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_ratio_feature_cols]
                concat_flat_feature_cols_ratios = [feature for feature_cols in flat_feature_cols_ratios for feature in feature_cols]
                assert all([np.allclose(flat_ratio[:, 0], ratio[:, 0, 0], equal_nan=True) for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
                assert all([np.allclose(flat_ratio[:, 3], ratio[:, 0, 3], equal_nan=True) for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
                assert all([np.allclose(flat_ratio[:, len(feature_cols):2*len(feature_cols)], ratio[:, 1, :], equal_nan=True) for flat_ratio, ratio, feature_cols in zip(flat_ratios, all_ratios, all_ratio_feature_cols)])
                

        if ratio:
            X = concat_flat_ratios
            feature_cols = concat_flat_feature_cols_ratios
        else:
            X = concat_flat_binned_psds
            feature_cols = concat_flat_feature_cols_bins

        power_feature_df = pd.DataFrame(X, columns=feature_cols, index=subjs)

        if save:
            basename = f'psd_power_features.csv' if not ratio else f'psd_power_features_ratio.csv'
            savefilename = os.path.join(powersavepath, basename)
            power_feature_df.to_csv(savefilename)
            # save the parameters
            with open(os.path.join(powersavepath, 'params.json'), 'w') as f:
                json.dump(power_params, f)

    return power_feature_df

def load_power_features(savepath, ratio=False):
    """
    Load the power features from the given savepath.
    Return the dataframe and whether the file was found.
    """
    basename = f'psd_power_features.csv' if not ratio else f'psd_power_features_ratio.csv'
    found_match = os.path.exists(os.path.join(savepath, basename))
    if found_match:
        savefilename = os.path.join(savepath, basename)
        return pd.read_csv(savefilename, index_col=0), found_match
    else:
        return None, found_match

def bin_psd_by_bands(psd, freqs, bands, method='avg', verbosity=2, num_axis=2):
    """
    Bin a PSD by frequency bands.

    Parameters
    ----------
    psd : array_like (n_channels, n_freqs)
        Power spectral density.
    bands : list
        List frequency bands.
        [(low_freq, high_freq), ...]
    Out:
    ----
    binned_psd: array_like (n_channels, (len(bands.keys()))
        Binned PSD.
    """
    if num_axis==2:
        binned_psd = np.zeros((psd.shape[0], len(bands)))
    elif num_axis==3:
        binned_psd = np.zeros((psd.shape[0], psd.shape[1], len(bands)))
    for idx, band in enumerate(bands):
        if verbosity > 0:
            print("band", band)
        freqs_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if verbosity > 1:
            print("freqs", freqs_idx)
        if method == 'avg' or method == 'mean':
            if verbosity > 1:
                print("mean", np.mean(psd[:, freqs_idx], axis=-1))
            binned_psd[..., idx] = np.mean(psd[..., freqs_idx], axis=-1)
        elif method == 'sum':
            binned_psd[..., idx] = np.sum(psd[..., freqs_idx], axis=-1)
        elif method == 'median':
            binned_psd[..., idx] = np.median(psd[..., freqs_idx], axis=-1)
        elif method == 'max':
            binned_psd[..., idx] = np.max(psd[..., freqs_idx], axis=-1)
        elif method == 'min':
            binned_psd[..., idx] = np.min(psd[..., freqs_idx], axis=-1)
        elif method == 'std':
            binned_psd[..., idx] = np.std(psd[..., freqs_idx], axis=-1)
        elif method == 'var':
            binned_psd[..., idx] = np.var(psd[..., freqs_idx], axis=-1)
        elif method == 'skew':
            binned_psd[..., idx] = stats.skew(psd[..., freqs_idx], axis=-1)
        elif method[0] == 'p':
            percentile = int(method[1:])
            binned_psd[..., idx] = np.percentile(psd[..., freqs_idx], percentile, axis=-1)
        elif method == 'iqr':
            binned_psd[..., idx] = stats.iqr(psd[..., freqs_idx], axis=-1)
        else:
            raise ValueError(f'Invalid method. Must be one of: avg, sum, median, max, min, std, var, skew, pX, iqr, but was {method}')
    return binned_psd

def make_ratios(binned_psd_power, return_bin_mapping=False):
    """
    Given a binned PSD of size (n_samples, n_channels, n_bins) compute the ratios of all the bins.
    Resulting shape:
        (n_samples, n_channels, n_bins * (n_bins - 1) / 2)
    """
    n_samples, n_channels, n_bins = binned_psd_power.shape
    n_ratio_bins = int(n_bins * (n_bins - 1) / 2)
    ratio_bins = np.zeros((n_samples, n_channels, n_ratio_bins))
    idx = 0
    # let's find the higher frequency bin first and then the lower frequency bin
    bin_mapping = []
    bindx = 0
    for idx in range(n_bins):
        for jdx in range(idx + 1, n_bins):
            bin_ratio = (jdx, idx)
            ratio_bins[:, :, bindx] = binned_psd_power[:, :, bin_ratio[0]] / binned_psd_power[:, :, bin_ratio[1]]
            bindx += 1
            bin_mapping.append(bin_ratio)
    if return_bin_mapping:
        return ratio_bins, bin_mapping
    else:
        return ratio_bins

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--band_method', type=str, default='custom', help='Method for computing frequency bands. One of ["standard", "custom", "standard+custom"]')
    parser.add_argument('--ratio', type=bool, default=True, help='Whether to compute ratios between bins')
    parser.add_argument('--bin_methods', type=str, default='all', help='Methods for binning the PSD. One of ["all", "avg", "sum", "median", "max", "min", "std", "var", "skew", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "iqr"]')
    parser.add_argument('--state', type=str, default='all', help='Which state to use. One of ["open", "closed", "all"]')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for make_bands function')
    parser.add_argument('--l_freq', type=float, default=0.3, help='Low frequency cutoff for make_bands function')
    parser.add_argument('--fs', type=int, default=500, help='Sampling rate for make_bands function')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the resulting dataframe')
    parser.add_argument('--featurepath', type=str, default=LOCD_DATAPATH, help='Path to save the resulting dataframe')
    parser.add_argument('--choose_subjs', type=str, default='train', help='Which dataset cohort split to use. One of ["train", "ival", "holdout", "dev"]')
    args = parser.parse_args()

    transform_data_dict = td.main()
    out = compute_psd_features(transform_data_dict, channels=CHANNELS, **vars(args))
