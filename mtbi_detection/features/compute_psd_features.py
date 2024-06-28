
import argparse
import numpy as np
import pandas as pd

import mtbi_detection.data.data_utils as du

import src.data.transform_data as td

import mtbi_detection.features.feature_utils as fu
import src.features.compute_maximal_power_features as cmpf
import src.features.compute_spectral_edge_features as csef
import src.features.compute_complexity_features as ccf
import src.features.compute_complexity_features_from_psd as ccfp
import src.features.compute_graph_features as cgf

CHANNELS = ['C3','C4','Cz','F3','F4','F7','F8','Fp1','Fp2','Fz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']
def create_psd_feature_from_methods(transform_data_dict, choose_subjs=None, ratio=False, return_both=False, channels=CHANNELS, state='all', band_method='standard', n_division=1, log_division=True, interbin_ratios=False, bin_methods='all',verbosity=0, l_freq=0.3, fs=500, save=False, savepath='/shared/roy/mTBI/saved_processed_data/mission_connect/features_csv/'):
    """
    Given the output of transform_data, compute the PSD features for each subject 
    and return a dataframe with the features as columns and the subjects as rows.
    Args:
        transform_data_dict (dict): Dictionary output of transform_data {subj_data: {subj: /path/to/data/}}, 'channels': channels, 'common_freqs': common_freqs}
        choose_subjs (str): the dataset cohort split to use ('train', 'ival', 'holdout', 'dev')
        ratio (bool): whether to compute ratios between bins
        return_both (bool): whether to return both the binned and ratio dataframes
        channels (list): list of channels to use
        state (str): which state to use. One of ['open', 'closed', 'all']
        band_method (str): method for computing frequency bands. One of ['standard', 'custom', 'standard+custom']
        n_division (int): number of divisions to use for standard bands. Ignored if band_method is not 'standard'
        

    
    """    

    
    all_bin_methods = ['avg', 'sum', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'iqr']
    if bin_methods == 'all' or 'all' in bin_methods:
        bin_methods = ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'iqr']
    elif type(bin_methods) == str:
        bin_methods = [bin_methods]
        
    assert all([method in all_bin_methods or (method[0]=='p' and method[1:].isdigit()) for method in bin_methods])

    X_open = np.stack(transform_data_dict['open_power'])
    X_closed = np.stack(transform_data_dict['closed_power'])

    X_open = X_open[:, :, 1:-1] 
    X_closed = X_closed[:, :, 1:-1]
    freqs = transform_data_dict['open_freqs'][0][1:-1]

    # show that there are no zeros in the data
    assert np.all(X_open >= 0)
    assert np.all(X_closed >= 0)

    # log transform the data
    X_open = np.log10(X_open)
    X_closed = np.log10(X_closed)

    num_axis = len(X_open.shape)

    assert np.all([freqs == transform_data_dict['closed_freqs'][idx][1:-1] for idx in range(len(transform_data_dict['closed_freqs']))])
    subjs = transform_data_dict['open_subjs']

    try:
        channels = transform_data_dict['channels']
    except KeyError:
        print("Channels not found in transform_data_dict, using input channels: ", channels)
    assert subjs == transform_data_dict['closed_subjs']
    assert X_open.shape[0] == len(subjs)
    assert X_closed.shape[0] == len(subjs)
    assert X_open.shape[1] == len(channels)
    assert X_closed.shape[1] == len(channels)
    assert X_open.shape[2] == len(freqs)
    assert X_closed.shape[2] == len(freqs)


    if n_subjs is not None:
        X_open = X_open[:n_subjs]
        X_closed = X_closed[:n_subjs]
        subjs = subjs[:n_subjs]

    if choose_subjs is not None:
        subjs_idx = np.array([idx for idx, subj in enumerate(subjs) if subj in choose_subjs])
        subjs = list(np.array(subjs)[subjs_idx])
        X_open = X_open[subjs_idx]
        X_closed = X_closed[subjs_idx]

    bands = fu.make_bands(basis=band_method, divisions=n_division, log_division=log_division, custom_bands=None, verbosity=verbosity, fs=fs, min_freq=l_freq)
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
        all_rat_feature_cols = []
        for bin_method in bin_methods:
            binned_psd = bin_psd_by_bands(X, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
            bin_feature_cols = [f'{state}_{band}_{bin_method}' for band in bands]
            if ratio and not interbin_ratios: 
                ratios, bin_mapping = make_ratios(binned_psd, return_bin_mapping=True)
                if when_log == 'log_second' or 'double_log':
                    ratios = np.log10(ratios)
                rat_feature_cols = [f'({bin_feature_cols[r1]})/({bin_feature_cols[r2]})' for r1, r2 in bin_mapping]
                all_ratios.append(ratios)
                all_rat_feature_cols.append(rat_feature_cols)
            all_binned_psds.append(binned_psd)
            all_bin_feature_cols.append(bin_feature_cols)
        
        flat_binned_psds = [binned_psd.reshape(binned_psd.shape[0], -1) for binned_psd in all_binned_psds]
        concat_binned_psds = np.concatenate(all_binned_psds, axis=-1)
        concat_feature_cols = [feature for feature_cols in all_bin_feature_cols for feature in feature_cols]
        concat_flat_binned_psds = np.concatenate(flat_binned_psds, axis=-1)
        flat_feature_cols_bins = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_bin_feature_cols]
        concat_flat_feature_cols_bins = [feature for feature_cols in flat_feature_cols_bins for feature in feature_cols]
        assert all([flat_binned_psd[:, 0] == binned_psd[:, 0, 0] for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
        assert all([flat_binned_psd[:, 3] == binned_psd[:, 0, 3] for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
        assert all([np.all(flat_binned_psd[:, len(feature_cols):2*len(feature_cols)] == binned_psd[:, 1, :]) for flat_binned_psd, binned_psd, feature_cols in zip(flat_binned_psds, all_binned_psds, all_bin_feature_cols)])
        if ratio and interbin_ratios:
            interbin_ratios, bin_mapping = make_ratios(concat_binned_psds, return_bin_mapping=True)
            if when_log == 'log_second' or 'double_log':
                interbin_ratios = np.log10(interbin_ratios)
            interbin_rat_feature_cols = [f'({concat_feature_cols[r1]})/({concat_feature_cols[r2]})' for r1, r2 in bin_mapping]
            all_ratios.append(interbin_ratios)
            all_rat_feature_cols.append(interbin_rat_feature_cols)
        if ratio:
            flat_ratios = [ratios.reshape(ratios.shape[0], -1) for ratios in all_ratios]
            concat_flat_ratios = np.concatenate(flat_ratios, axis=-1)
            flat_feature_cols_ratios = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_rat_feature_cols]
            concat_flat_feature_cols_ratios = [feature for feature_cols in flat_feature_cols_ratios for feature in feature_cols]
            assert all([flat_ratio[:, 0] == ratio[:, 0, 0] for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
            assert all([flat_ratio[:, 3] == ratio[:, 0, 3] for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
            assert all([np.all(flat_ratio[:, len(feature_cols):2*len(feature_cols)] == ratio[:, 1, :]) for flat_ratio, ratio, feature_cols in zip(flat_ratios, all_ratios, all_rat_feature_cols)])

    else:
        all_binned_psds = []
        all_bin_feature_cols = []
        all_ratios = []
        all_rat_feature_cols = []

        for bin_method in bin_methods:
            open_binned_psd = bin_psd_by_bands(X_open, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
            closed_binned_psd = bin_psd_by_bands(X_closed, freqs, bands, method=bin_method, verbosity=verbosity, num_axis=num_axis)
            binned_psd = np.concatenate((open_binned_psd, closed_binned_psd), axis=-1)
            bin_feature_cols = [f'{oc}_{band}_{bin_method}' for oc in ['open', 'closed'] for band in bands]
            all_binned_psds.append(binned_psd)
            all_bin_feature_cols.append(bin_feature_cols)

            if ratio and not interbin_ratios:
                open_ratios, bin_mapping = make_ratios(open_binned_psd, return_bin_mapping=True)
                closed_ratios, closed_bin_mapping = make_ratios(closed_binned_psd, return_bin_mapping=True)
                assert bin_mapping == closed_bin_mapping
                if when_log == 'log_second' or 'double_log':
                    open_ratios = np.log10(open_ratios)
                    closed_ratios = np.log10(closed_ratios)
                ratios = np.concatenate((open_ratios, closed_ratios), axis=-1)
                rat_feature_cols = [f'({bin_feature_cols[r1]})/({bin_feature_cols[r2]})' for r1, r2 in bin_mapping]
                rat_feature_cols = [f'{oc}_{feature}' for oc in ['open', 'closed'] for feature in rat_feature_cols]
                all_ratios.append(ratios)
                all_rat_feature_cols.append(rat_feature_cols)

        flat_binned_psds = [binned_psd.reshape(binned_psd.shape[0], -1) for binned_psd in all_binned_psds]
        concat_flat_binned_psds = np.concatenate(flat_binned_psds, axis=-1)
        concat_binned_psds = np.concatenate(all_binned_psds, axis=-1)
        concat_feature_cols = [feature for feature_cols in all_bin_feature_cols for feature in feature_cols]
        flat_feature_cols_bins = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_bin_feature_cols]
        concat_flat_feature_cols_bins = [feature for feature_cols in flat_feature_cols_bins for feature in feature_cols]
        assert all([np.all(flat_binned_psd[:, 0] == binned_psd[:, 0, 0]) for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
        assert all([np.all(flat_binned_psd[:, 3] == binned_psd[:, 0, 3]) for flat_binned_psd, binned_psd in zip(flat_binned_psds, all_binned_psds)])
        assert all([np.all(flat_binned_psd[:, len(feature_cols):2*len(feature_cols)] == binned_psd[:, 1, :]) for flat_binned_psd, binned_psd, feature_cols in zip(flat_binned_psds, all_binned_psds, all_bin_feature_cols)])
        if ratio and interbin_ratios:
            interbin_ratios, bin_mapping = make_ratios(concat_binned_psds, return_bin_mapping=True)
            interbin_rat_feature_cols = [f'({concat_feature_cols[r1]})/({concat_feature_cols[r2]})' for r1, r2 in bin_mapping]
            all_ratios.append(interbin_ratios)
            all_rat_feature_cols.append(interbin_rat_feature_cols)
        if ratio:
            flat_ratios = [ratios.reshape(ratios.shape[0], -1) for ratios in all_ratios]
            concat_flat_ratios = np.concatenate(flat_ratios, axis=-1)
            flat_feature_cols_ratios = [[f'{ch}_{feature}' for ch in channels for feature in feature_cols] for feature_cols in all_rat_feature_cols]
            concat_flat_feature_cols_ratios = [feature for feature_cols in flat_feature_cols_ratios for feature in feature_cols]
            assert all([np.allclose(flat_ratio[:, 0], ratio[:, 0, 0], equal_nan=True) for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
            assert all([np.allclose(flat_ratio[:, 3], ratio[:, 0, 3], equal_nan=True) for flat_ratio, ratio in zip(flat_ratios, all_ratios)])
            assert all([np.allclose(flat_ratio[:, len(feature_cols):2*len(feature_cols)], ratio[:, 1, :], equal_nan=True) for flat_ratio, ratio, feature_cols in zip(flat_ratios, all_ratios, all_rat_feature_cols)])
            

    if ratio:
        if return_both:
            X_flat_bin, X_flat_rat = concat_flat_binned_psds, concat_flat_ratios
            # X_flat_bin = X_bin.reshape(X_bin.shape[0], -1)
            # X_flat_rat = X_rat.reshape(X_rat.shape[0], -1)
            flat_feature_cols_bin = concat_flat_feature_cols_bins
            flat_feature_cols_rat = concat_flat_feature_cols_ratios

            X_bin_df = pd.DataFrame(X_flat_bin, columns=flat_feature_cols_bin, index=subjs)
            X_rat_df = pd.DataFrame(X_flat_rat, columns=flat_feature_cols_rat, index=subjs)
            return X_bin_df, X_rat_df
        else:
            X = concat_flat_ratios
            feature_cols = concat_flat_feature_cols_ratios
    else:
        X = concat_flat_binned_psds
        feature_cols = concat_flat_feature_cols_bins

    X_df = pd.DataFrame(X, columns=feature_cols, index=subjs)

    return X_df


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
    parser.add_argument('--band_method', type=str, default='standard', help='Method for computing frequency bands. One of ["standard", "custom", "standard+custom"]')
    parser.add_argument('--n_division', type=int, default=1, help='Number of divisions to use for standard bands. Ignored if band_method is not "standard"')
    parser.add_argument('--log_division', type=bool, default=True, help='Whether to use log division for standard bands. Ignored if band_method is not "standard"')
    parser.add_argument('--interbin_ratios', type=bool, default=True, help='Whether to compute ratios between all bins. Ignored if ratio is False')
    parser.add_argument('--ratio', type=bool, default=True, help='Whether to compute ratios between bins')
    parser.add_argument('--bin_methods', type=str, default='all', help='Methods for binning the PSD. One of ["all", "avg", "sum", "median", "max", "min", "std", "var", "skew", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "iqr"]')
    parser.add_argument('--state', type=str, default='all', help='Which state to use. One of ["open", "closed", "all"]')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for make_bands function')
    parser.add_argument('--l_freq', type=float, default=0.3, help='Low frequency cutoff for make_bands function')
    parser.add_argument('--fs', type=int, default=500, help='Sampling rate for make_bands function')
    parser.add_argument('--save', type=bool, default=False, help='Whether to save the resulting dataframe')
    parser.add_argument('--savepath', type=str, default='/shared/roy/mTBI/saved_processed_data/mission_connect/features_npz/psd_power_features/', help='Path to save the resulting dataframe')
    parser.add_argument('--choose_subjs', type=str, default=None, help='List of subjects to use. If None, use all subjects')
    parser.add_argument('--n_subjs', type=int, default=None, help='Number of subjects to use. If None, use all subjects')
    parser.add_argument('--return_both', type=bool, default=True, help='Whether to return both the binned and ratio dataframes')
    args = parser.parse_args()

    transform_data_dict = td.main()
    out = create_psd_feature_from_methods(transform_data_dict, choose_subjs=args.choose_subjs, when_log=args.when_log, ratio=args.ratio, return_both=args.return_both, channels=CHANNELS, state=args.state, n_subjs=args.n_subjs, band_method=args.band_method, n_division=args.n_division, log_division=args.log_division, interbin_ratios=args.interbin_ratios, bin_methods=args.bin_methods, verbosity=args.verbosity, l_freq=args.l_freq, fs=args.fs, save=args.save, savepath=args.savepath)
