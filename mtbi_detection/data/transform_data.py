### functions meant to integrate with the output of load_open_closed_dict.py and transform into more usefule signals:

import time
import mne
import numpy as np
import os
import json
import argparse
import scipy
import pprint
from joblib import Parallel, delayed
import dotenv

import mtbi_detection.data.load_open_closed_data as locd
import mtbi_detection.data.data_utils as du

dotenv.load_dotenv()
DATAPATH = os.getenv('EXTRACTED_PATH')
LOCD_DATAPATH = os.getenv('OPEN_CLOSED_PATH')

# DATAPATH = open('extracted_path.txt', 'r').read().strip() 
# LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()

def main(locd_params = {
        'l_freq': 0.3,
        'h_freq': None,
        'fs_baseline': 500,
        'order': 6,
        'notches': [60, 120, 180, 240],
        'notch_width': [2, 1, 0.5, 0.25],
        'num_subjs': 151,
        'reference_method': 'CSD',
        'reference_channels': ['A1', 'A2'],
        'filter_ecg': True,
        'keep_refs': False,
        'bad_channels': ['T1', 'T2'],
        'ecg_l_freq': 8,
        'ecg_h_freq': 16,
        'ecg_thresh': 'auto',
        'ecg_method': 'correlation'
    }, interpolate_spectrum=1000, freq_interp_method='linear',bandwidth=1, which_segment='avg',
    n_jobs=1, save=True, as_paths=True, locd_savepath=LOCD_DATAPATH, skip_ui=False, verbose=True):
    # define loading parameters
    print(f"Skipping user confirmation? {skip_ui}")
    print(f"Loading parameters: {locd_params}")
    params = extract_locd_params(**locd_params)
    params['interpolate_spectrum'] = interpolate_spectrum
    params['bandwidth'] = bandwidth
    params['which_segment'] = which_segment
    
    if locd_savepath.endswith('/'):
        temp = locd_savepath[:-1]
    else:
        temp = locd_savepath
    savepath = os.path.join(os.path.dirname(temp), 'psd_transform')
    savepath = du.check_savepath(savepath, skip_ui=skip_ui)
    du.clean_params_path(savepath, skip_ui=skip_ui)

    found_match = False
    paramfile = 'params.json'
    
    savepath, found_match = du.check_and_make_params_folder(savepath, params, paramfile=paramfile, skip_ui=skip_ui)
    if found_match:
        print("Loading data from {}".format(savepath))
        output_dict = load_all_transform_data_paths(savepath=savepath, subjs=None, as_paths=as_paths)
    else:
        # load the data from load_open_closed_dict
        open_closed_dict = locd.load_open_closed_pathdict(**locd_params, savepath=locd_savepath, verbose=verbose)
        # load the psd dataset
        mjobs=1
        # mjobs = max((os.cpu_count()-2)//n_jobs//4, 1) # jobs to use for multitaper


        # makes the multitaper dataset and returns the pathdict containing the paths to the saved files: {subj: {'first': /path/to/first, 'second': /path/to/second, 'avg': /path/to/avg}}
        multitaper_dataset = make_multitaper_dataset(open_closed_dict, savepath, fmin=params['l_freq'], fmax=params['h_freq'], n_jobs=n_jobs, bandwidth=params['bandwidth'], normalization='full', verbose=False, interpolate_spectrum=interpolate_spectrum, freq_interp_method=freq_interp_method, mjobs=mjobs, save=save)
        with open(os.path.join(savepath, 'params.json'), 'w') as f:
            json.dump(params, f)
            print(f"Saved params to {os.path.join(savepath, 'params.json')}")

        subjs = [subj for subj in open_closed_dict.keys() if str(subj).isdigit()]
        output_dict = {}
        output_dict['subj_data'] = {subj: {which_segment: multitaper_dataset[subj][which_segment]} for subj in subjs}
        output_dict['common_freqs'] = multitaper_dataset['common_freqs']
        output_dict['channels'] = multitaper_dataset['channels']
        output_dict['params'] = params

        if not as_paths:
            for subj in subjs:
                output_dict[subj][which_segment] = load_single_subject_transform_data(output_dict[subj][which_segment])
    return output_dict

def process_segment(seg, segdict, channels=None, common_freqs=None, fmin=0.3, fmax=None, mjobs=1, normalization='full', bandwidth=None, verbose=False):
    """
    Returns a dictionary with the power, phase, psd, weights, freqs, and times for a given segment
    Output:
        seg_result (dict): dictionary with keys 'power', 'spectrum', 'weights', 'basefreqs', 'times'
    """
    seg_result = {'power': [], 'spectrum': [], 'weights': [], 'basefreqs':[], 'times': []}

    for raw_file in segdict[seg]:
        raw = mne.io.read_raw_fif(raw_file, verbose=False).pick(channels)
        assert du.isolate_eeg_channels(raw.ch_names) == channels, f"Channels {raw.ch_names} don't match {channels}"
        rawtime = raw.times
        eeg_arry = raw.get_data()

        # number of cpus available

        if fmax is None:
            cpsd, freqsc, weightsc = mne.time_frequency.psd_array_multitaper(eeg_arry, 500, fmin=fmin, output='complex', bandwidth=bandwidth, n_jobs=mjobs, normalization=normalization, verbose=verbose)
        else:
            cpsd, freqsc, weightsc = mne.time_frequency.psd_array_multitaper(eeg_arry, 500, fmin=fmin, fmax=fmax, output='complex', bandwidth=bandwidth, n_jobs=mjobs, normalization=normalization, verbose=verbose)
        power = _psd_from_mt(cpsd, weightsc)
        seg_result['spectrum'].append(cpsd)
        if common_freqs is not None:
            power = resample_psds(freqsc, power, common_freqs)
        seg_result['weights'].append(weightsc)
        seg_result['basefreqs'].append(freqsc)
        seg_result['times'].append(rawtime)
        seg_result['power'].append(power)

    
        time.sleep(0)
    return seg_result

def _process_single_subject(subj, statedict, states, basesavepath, interpolate_spectrum=1000, freq_interp_method='linear', fmin=0.3, idx=None, n_subjs=None, mjobs=1, fmax=None, channels=None, normalization='full', bandwidth=None, save=True, verbose=False):
       
    """
    Create multitaper psds for a single subject and save them to a file: {basesavepath}/{subj}/{segment}/open_closed_multitaper_psds_{subj}.npz
    """

    assert states == ['open', 'closed'], "States must be ['open', 'closed']"
    if verbose:
        print(f"Subject: {subj} ({idx}/{n_subjs})")
    starttime = time.time()

    common_freqs = make_interp_freqs(fmin, fmax, interpolate_spectrum, interp_method=freq_interp_method)
    subj_result = {subj: {state: process_segment(state, statedict, common_freqs=common_freqs, channels=channels, fmin=fmin, mjobs=mjobs, fmax=fmax, normalization=normalization, bandwidth=bandwidth, verbose=verbose) for state in states}}

    print("Time to multitaper for subject {}: {}".format(subj, time.time()-starttime))

    # now we save it and return the paths
    subj_savepath = os.path.join(basesavepath, f"{subj}")
    if not os.path.exists(subj_savepath):
        os.makedirs(subj_savepath)
    savefilename = f"open_closed_multitaper_psds_{subj}.npz"

    subject_paths = {subj: {}}
    savetime = time.time()
    for segment in ["first", "second", "avg"]:
        segmentsavepath = os.path.join(subj_savepath, segment)
        fullfilename = os.path.join(segmentsavepath, savefilename)
        # now save the psd_output using savez
        if save:
            if segment == "first":
                open_power = subj_result[subj]['open']['power'][0] if len(subj_result[subj]['open']['power']) > 0 else None
                closed_power = subj_result[subj]['closed']['power'][0] if len(subj_result[subj]['closed']['power']) > 0 else None
                open_spectrum = subj_result[subj]['open']['spectrum'][0] if len(subj_result[subj]['open']['spectrum']) > 0 else None
                closed_spectrum = subj_result[subj]['closed']['spectrum'][0] if len(subj_result[subj]['closed']['spectrum']) > 0 else None
                open_weights = subj_result[subj]['open']['weights'][0] if len(subj_result[subj]['open']['weights']) > 0 else None
                closed_weights = subj_result[subj]['closed']['weights'][0] if len(subj_result[subj]['closed']['weights']) > 0 else None
                open_spectrum = subj_result[subj]['open']['spectrum'][0] if len(subj_result[subj]['open']['spectrum']) > 0 else None
                closed_spectrum = subj_result[subj]['closed']['spectrum'][0] if len(subj_result[subj]['closed']['spectrum']) > 0 else None
                open_basefreqs = subj_result[subj]['open']['basefreqs'][0] if len(subj_result[subj]['open']['basefreqs']) > 0 else None
                closed_basefreqs = subj_result[subj]['closed']['basefreqs'][0] if len(subj_result[subj]['closed']['basefreqs']) > 0 else None

            elif segment == "second":
                open_power = subj_result[subj]['open']['power'][1] if len(subj_result[subj]['open']['power']) > 1 else None
                closed_power = subj_result[subj]['closed']['power'][1] if len(subj_result[subj]['closed']['power']) > 1 else None
                open_spectrum = subj_result[subj]['open']['spectrum'][1] if len(subj_result[subj]['open']['spectrum']) > 1 else None
                closed_spectrum = subj_result[subj]['closed']['spectrum'][1] if len(subj_result[subj]['closed']['spectrum']) > 1 else None
                open_weights = subj_result[subj]['open']['weights'][1] if len(subj_result[subj]['open']['weights']) > 1 else None
                closed_weights = subj_result[subj]['closed']['weights'][1] if len(subj_result[subj]['closed']['weights']) > 1 else None
                open_spectrum = subj_result[subj]['open']['spectrum'][1] if len(subj_result[subj]['open']['spectrum']) > 1 else None
                closed_spectrum = subj_result[subj]['closed']['spectrum'][1] if len(subj_result[subj]['closed']['spectrum']) > 1 else None
                open_basefreqs = subj_result[subj]['open']['basefreqs'][1] if len(subj_result[subj]['open']['basefreqs']) > 1 else None
                closed_basefreqs = subj_result[subj]['closed']['basefreqs'][1] if len(subj_result[subj]['closed']['basefreqs']) > 1 else None
            elif segment == "avg":
                open_times = subj_result[subj]['open']['times']
                closed_times = subj_result[subj]['closed']['times']
                open_power = psd_from_disjoint_arrays(subj_result[subj]['open']['power'], open_times)
                closed_power = psd_from_disjoint_arrays(subj_result[subj]['closed']['power'], closed_times)
                open_spectrum = None
                closed_spectrum = None
                open_weights = None
                closed_weights = None
                open_basefreqs = None
                closed_basefreqs = None
            if not os.path.exists(segmentsavepath):
                os.makedirs(segmentsavepath)
            np.savez_compressed(fullfilename,
                                open_power = open_power,
                                closed_power = closed_power,
                                open_weights = open_weights,
                                closed_weights = closed_weights,
                                open_spectrum = open_spectrum,
                                closed_spectrum = closed_spectrum,
                                open_basefreqs = open_basefreqs,
                                closed_basefreqs = closed_basefreqs)

            subject_paths[subj][segment] = fullfilename
        time.sleep(0)
    print(f"Time to save subject {subj}: {time.time()-savetime}")

    


    return subject_paths, time.time()-starttime

def make_multitaper_dataset(dataset, savepath, fmin=0.3, fmax=245, normalization='full', bandwidth=None, verbose=False, n_jobs=1, mjobs=1, interpolate_spectrum=1000, freq_interp_method='linear', save=True):
    """
    Input dataset in the form {subj: {'open': [], 'closed': []}}
    Returns a dataset in the form {subj: {'first': /path/to/first.npz, 'second': /path/to/second.npz, 'avg': /path/to/avg.npz}}
    where each npz file contains the power, phase, psd, weights, freqs, and times for the given segment for open and closed
    Args:
        dataset (dict): dictionary of the form {subj: {'open': [], 'closed': []}}
        fmin (float): minimum frequency to compute the multitaper
        fmax (float): maximum frequency to compute the multitaper
        normalization (str): normalization method for the multitaper
        freq_interp_method (str): method for interpolating the frequencies ('linear', 'log', 'log10')
        interpolate_spectrum (int): number of frequencies to interpolate to
        bandwidth (float): bandwidth for the multitaper
        verbose (bool): verbose output
        mjobs (int): number of jobs to use for individual multitaper
        n_jobs (int): number of jobs to use for parallelizing across subjects
    Returns:
        multitaper_dataset (dict): dictionary of the form {subj: {'first': /path/to/first.npz, 'second': /path/to/second.npz, 'avg': /path/to/avg.npz}}

    """
    states = ['open', 'closed']
    segments = ['first', 'second', 'avg']

    subjs = list(dataset.keys())
    subjs = [subj for subj in subjs if str(subj).isdigit()]
    assert all([state in states for state in dataset[list(dataset.keys())[0]].keys()]), "Inner keys must be 'open' or 'closed'"
    eeg_channels = du.isolate_eeg_channels(mne.io.read_raw_fif(dataset[list(dataset.keys())[0]][states[0]][0], verbose=False).ch_names)
    
    assert all([eeg_channels == du.isolate_eeg_channels(mne.io.read_raw_fif(dataset[subj][state][0], verbose=False).ch_names) for subj in subjs for state in states]), "Channels must be the same for all subjects"


    multitaper_dataset = {subj: {seg: {} for seg in segments} for subj in dataset.keys()}

    tottime = time.time()
    process_times = []

    if verbose:
        print(f"Beginning multitaper for {len(subjs)} subjects")
    n_subjs = len(subjs)

    parallel_results = Parallel(n_jobs=n_jobs)(delayed(_process_single_subject)(subj, dataset[subj], states, savepath, interpolate_spectrum=interpolate_spectrum, freq_interp_method=freq_interp_method, fmin=fmin, idx=idx, mjobs=mjobs, n_subjs=n_subjs, fmax=fmax, channels=eeg_channels, normalization=normalization, bandwidth=bandwidth, verbose=verbose, save=save) for idx, subj in enumerate(subjs))

    for idx, subj in enumerate(subjs):
        for seg in segments:
            multitaper_dataset[subj][seg] = parallel_results[idx][0][subj][seg]
        process_times.append(parallel_results[idx][1])

    print("Total time to multitaper: {}".format(time.time()-tottime))
    print(f"Median time: {np.median(process_times)}, max time: {np.max(process_times)}, min time: {np.min(process_times)}")

    output_dict = multitaper_dataset
    output_dict['channels'] = eeg_channels
    output_dict['common_freqs'] = make_interp_freqs(fmin, fmax, interpolate_spectrum, interp_method=freq_interp_method)
    # save the channels and common_freqs to npy
    np.save(os.path.join(savepath, 'channels.npy'), eeg_channels)
    np.save(os.path.join(savepath, 'common_freqs.npy'), output_dict['common_freqs'])
    
    return output_dict
            
def psd_from_disjoint_arrays(psds,times):
    """Compute a weighted average of the psd based on duration of original input
    Parameters
    ----------
    times : iterable 
        Iterable of arrays of times for each eeg file
    psds : iterable
        Iterable of arrays of psds for each eeg file
    """

    assert len(times) == len(psds), "Times and psds must be the same length"
    # also need psd shapes the same
    assert all([psd.shape == psds[0].shape for psd in psds]), f"All psds must have the same shape, but got {[psd.shape for psd in psds]}"
    psd_weighting = np.array([len(t) for t in times])/sum([len(t) for t in times])

    if len(psd_weighting) > 1:
        psd_output = np.average([psd for psd in psds], axis=0, weights=psd_weighting)
    else:
        psd_output = psds[0]
    return psd_output

def _psd_from_mt(x_mt, weights):
    """Compute PSD from tapered spectra.
    From: https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/multitaper.py

    Parameters
    ----------
    x_mt : array, shape=(..., n_tapers, n_freqs)
        Tapered spectra
    weights : array, shape=(n_tapers,)
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array, shape=(..., n_freqs)
        The computed PSD
    
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd

def load_all_transform_data_paths(savepath, subjs=None, which_segment='avg', as_paths=True, unravel=False):
    """
    Given a savepath, return a dictionary of the form {subj: {which_segment: /path/to/first.npz}}
    """
    if type(which_segment) == str:
        which_segment = [which_segment]
    elif type(which_segment) == list:
        assert all([seg in ['first', 'second', 'avg'] for seg in which_segment]), "Segments must be 'first', 'second', or 'avg'"
    else:
        raise ValueError(f"which_segment must be a string or a list of strings but got {which_segment}")
    
    if subjs is None:
        # search for all the subdirs in the savepath
        subjs = os.listdir(savepath)
        # only keep the ones that are directories
        subjs = [subj for subj in subjs if os.path.isdir(os.path.join(savepath, subj))]

    multitaper_dataset = {subj: {seg: {}} for subj in subjs for seg in which_segment}
    for subj in subjs:
        for seg in which_segment:
            subj_savepath = os.path.join(savepath, subj, seg)
            savefilename = f"open_closed_multitaper_psds_{subj}.npz"
            assert os.path.exists(subj_savepath), f"Path {subj_savepath} does not exist"
            mt_fullpath = os.path.join(subj_savepath, savefilename)
            multitaper_dataset[subj][seg] = mt_fullpath

            if not as_paths:
                multitaper_dataset[subj][seg] = load_single_subject_transform_data(mt_fullpath)
        
    multitaper_dataset = unravel_multitaper_dataset(multitaper_dataset, which_segment=which_segment) if unravel else multitaper_dataset
    # add the params
    output_dict = {}
    output_dict['subj_data'] = multitaper_dataset
    output_dict['channels'] = np.load(os.path.join(savepath, 'channels.npy'))
    output_dict['common_freqs'] = np.load(os.path.join(savepath, 'common_freqs.npy'))
    params_jsonfile = os.path.join(savepath, 'params.json')
    if os.path.exists(params_jsonfile):
        with open(params_jsonfile, 'r') as f:
            params = json.load(f)
            output_dict['params'] = params

    return output_dict


def load_single_subject_transform_data(mt_fullpath, inner_keys=['open_power', 'closed_power', 'open_spectrum', 'closed_spectrum', 'open_weights', 'closed_weights', 'open_basefreqs', 'closed_basefreqs']):
    """
    Given a savepath, return a dictionary of the form 
    {open_power: _, closed_power: _, open_spectrum: _, closed_spectrum: _, open_weights: _, closed_weights: _, open_basefreqs: _, closed_basefreqs: _}
    """

    assert os.path.exists(mt_fullpath), f"Path {mt_fullpath} does not exist"
    subj_data = {}
  
    data = np.load(mt_fullpath, allow_pickle=True)
    for key in inner_keys:
        subj_data[key] = data[key]
    return subj_data


def unravel_multitaper_dataset(multitaper_dataset, which_segment='avg'):
    """
    Given a dataset of the form:
    {subj: {which_segment: {}}
    return a dataset of the form:
    {'open_power': [], 'closed_power': [], 'open_subjs': [], 'closed_subjs': []}
    """
    if type(which_segment) == str:
        which_segment = [which_segment]
    elif type(which_segment) == list:
        assert all([seg in ['first', 'second', 'avg'] for seg in which_segment]), "Segments must be 'first', 'second', or 'avg'"
    else:
        raise ValueError(f"which_segment must be a string or a list of strings but got {which_segment}")
    
    assert all([key.isdigit() for key in multitaper_dataset.keys()]), "Keys must be subject numbers"
    subjs = list(multitaper_dataset.keys())
    if type(multitaper_dataset[subjs[0]][which_segment[0]]) == str:
        multitaper_dataset = {subj: {seg: load_single_subject_transform_data(multitaper_dataset[subj][seg])} for subj in subjs for seg in which_segment}
    output_dict = {seg:{'open_power': [], 'closed_power': [], 'open_subjs': [], 'closed_subjs': []} for seg in which_segment}
    for seg in which_segment:
        for subj in subjs:
            output_dict[seg]['open_power'].append(multitaper_dataset[subj][seg]['open_power'])
            output_dict[seg]['closed_power'].append(multitaper_dataset[subj][seg]['closed_power'])
            output_dict[seg]['open_subjs'].append(subj)
            output_dict[seg]['closed_subjs'].append(subj)
    return output_dict


def resample_psds(freqs, psd, common_freqs):
    """
    Interpolates the PSDs to a common set of frequencies
    Args:
        freqs (np.array): array of frequencies
        psd (np.array): array of PSDs
        common_freqs (np.array): array of frequencies to interpolate to
        min_freq (float): minimum frequency
        max_freq (float): maximum frequency
        num (int): number of frequencies to interpolate to
    Returns:
        resampled_psd (np.array): array of resampled PSDs    
    """
    resampled_func = scipy.interpolate.interp1d(freqs, psd, axis=1, kind='linear', bounds_error=False, fill_value='extrapolate')
    resampled_psd = resampled_func(common_freqs)

    return resampled_psd

def make_interp_freqs(fmin, fmax, n_freqs, fs=500, interp_method='linear'):
    """
    Makes an array of frequencies to interpolate the PSDs to
    Args:
        fmin (float): minimum frequency
        fmax (float): maximum frequency
        n_freqs (int): number of frequencies to interpolate
        fs (float): sampling frequency
        interp_method (str): method for interpolation ('linear', 'log', 'log10')
    Returns:
        interp_freqs (np.array): array of interpolated frequencies
    """
    if fmax == None or fmax > fs/2:
        fmax = fs/2
    if interp_method == 'linear':
        interp_freqs = np.linspace(fmin, fmax, n_freqs)
    elif interp_method == 'log':
        interp_freqs = np.logspace(np.log(0.2), np.log(600), n_freqs*2, base=np.exp(1))
        interp_freqs = np.unique(interp_freqs)  
        interp_freqs = interp_freqs[interp_freqs <= fmax]
        interp_freqs = interp_freqs[interp_freqs >= fmin]
        # make sure the fmax is included
        interp_freqs = np.concatenate((interp_freqs, [fmax]))
        interp_freqs = np.concatenate(([fmin], interp_freqs))
        interp_freqs = np.unique(interp_freqs)
    elif interp_method == 'log10':
        interp_freqs = np.logspace(np.log10(0.2), np.log10(600), n_freqs*2)
        interp_freqs = np.unique(interp_freqs)  
        interp_freqs = interp_freqs[interp_freqs <= fmax]
        interp_freqs = interp_freqs[interp_freqs >= fmin]
        # make sure the fmax is included
        interp_freqs = np.concatenate((interp_freqs, [fmax]))
        interp_freqs = np.concatenate(([fmin], interp_freqs))
        interp_freqs = np.unique(interp_freqs)
    else:
        raise ValueError("interp_method must be 'linear', 'log', or 'log10'")
    
    assert interp_freqs[0] >= fmin, f"Interpolated frequencies must start at {fmin}, but got {interp_freqs[0]}"
    assert interp_freqs[-1] <= fmax, f"Interpolated frequencies must end at {fmax}, but got {interp_freqs[-1]}"
    return interp_freqs


def extract_locd_params(**kwargs):
    
    params = {
        'l_freq': kwargs['l_freq'],
        'h_freq': kwargs['h_freq'],
        'fs_baseline': kwargs['fs_baseline'],
        'order': kwargs['order'],
        'notches': kwargs['notches'],
        'notch_width': kwargs['notch_width'],
        'num_subjs': kwargs['num_subjs'],
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    ### LOCD PARAMS
    parser.add_argument('--l_freq', type=float, default=0.3)
    parser.add_argument('--h_freq', type=float, default=None)
    parser.add_argument('--fs_baseline', type=float, default=500)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--notches', type=int, nargs='+', default=[60, 120, 180, 240])
    parser.add_argument('--notch_width', type=float, nargs='+', default=[2, 1, 0.5, 0.25])
    parser.add_argument('--num_subjs', type=int, default=3)
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

    ## PSD PARAMS
    parser.add_argument('--interpolate_spectrum', type=int, default=1000)
    parser.add_argument('--freq_interp_method', type=str, default='linear', choices=['linear', 'log', 'log10'])
    parser.add_argument('--which_segment', type=str, default='avg', choices=['first', 'second', 'avg'], help='Which segment to use for the multitaper')
    parser.add_argument('--bandwidth', type=float, default=1)

    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--as_paths', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    params = vars(args)
    locd_params = extract_locd_params(**params)
    # confirm that the user wants to run with these params
    print("Running with the following params:")
    pprint.pprint(params)
    response = input("Continue? (y/n)")
    if response == 'y':
        td_dict = main(locd_params=locd_params, interpolate_spectrum=params['interpolate_spectrum'], freq_interp_method=params['freq_interp_method'], bandwidth=params['bandwidth'], which_segment=params['which_segment'], n_jobs=params['n_jobs'], save=params['save'])
        print(td_dict.keys())
    else:
        print("exiting...")