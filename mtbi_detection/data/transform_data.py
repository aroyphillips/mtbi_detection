### functions meant to integrate with the output of load_open_closed_dict.py and transform into more usefule signals:

import time
import mne
import numpy as np
import mtbi_detection.data.load_open_closed_data as locd
import mtbi_detection.data.data_utils as du
import mtbi_detection.data.cleanpath as cleanpath
import os
import json
import argparse
import pprint
from joblib import Parallel, delayed


DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()

def main(locd_params = {
        'l_freq': 0.3,
        'h_freq': None,
        'fs_baseline': 500,
        'order': 6,
        'notches': [60, 120, 180, 240],
        'notch_width': [2, 1, 0.5, 0.25],
        'num_subjs': 151,
        'verbose': True,
        'method': 'CSD',
        'reference_channels': ['A1', 'A2'],
        'filter_ecg': True,
        'keep_refs': False,
        'bad_channels': ['T1', 'T2'],
        'ecg_l_freq': 8,
        'ecg_h_freq': 16,
        'ecg_thresh': 'auto',
        'ecg_method': 'correlation'
    }, pad=False, bandwidth=1, which_segment='avg',
    n_jobs=1, save=True, num_load_subjs=None, locd_savepath=LOCD_DATAPATH):
    # define loading parameters
    print("Loading parameters: {}".format(locd_params))
    params = extract_locd_params(**locd_params)
    params['pad'] = pad
    params['bandwidth'] = bandwidth
    params['which_segment'] = which_segment
    
    if locd_savepath.endswith('/'):
        temp = locd_savepath[:-1]
    else:
        temp = locd_savepath
    savepath = os.path.join(os.path.dirname(temp), 'data_transforms')
    savepath = du.check_savepath(savepath)
    du.clean_params_path(savepath)

    found_match = False
    paramfile = 'params.json'
    savepath, found_match = du.check_and_make_params_folder(savepath, params, paramfile=paramfile)
    if found_match:
        print("Loading data from {}".format(savepath))
        output_dict = load_all_transform_data_paths(savepath=savepath, subjs=None, num_load_subjs=num_load_subjs)
    else:
        # load the data from load_open_closed_dict
        open_closed_dict = locd.load_open_closed_pathdict(**locd_params, savepath=locd_savepath)
        # load the psd dataset
        mjobs = max((os.cpu_count()-2)//n_jobs//4, 1) # jobs to use for multitaper


        # makes the multitaper dataset and returns the pathdict containing the paths to the saved files: {subj: {'first': /path/to/first, 'second': /path/to/second, 'avg': /path/to/avg}}
        multitaper_dataset = make_multitaper_dataset(open_closed_dict, savepath, fmin=params['l_freq'], fmax=params['h_freq'], n_jobs=n_jobs, bandwidth=params['bandwidth'], normalization='full', verbose=False, pad=pad, mjobs=mjobs, save=save)
        with open(os.path.join(savepath, 'params.json'), 'w') as f:
            json.dump(params, f)
            print(f"Saved params to {os.path.join(savepath, 'params.json')}")

        subjs = [subj for subj in open_closed_dict.keys() if str(subj).isdigit()]
        output_dict = {subj: {which_segment: multitaper_dataset[subj][which_segment]} for subj in subjs}
        output_dict['channels'] = multitaper_dataset['channels']
    return output_dict

def process_segment(seg, segdict, channels=None, pad=False, max_time=None, fmin=0.3, fmax=None, mjobs=1, normalization='full', bandwidth=None, verbose=False):
    """
    Returns a dictionary with the power, phase, psd, weights, freqs, and times for a given segment
    Output:
        seg_result (dict): dictionary with keys 'power', 'spectrum', 'weights', 'freqs', 'times'
    """
    seg_result = {'power': [], 'spectrum': [], 'weights': [], 'freqs':[], 'times': []}

    for raw in segdict[seg]:
        assert raw.ch_names == channels, f"Channels {raw.ch_names} don't match {channels}"
        rawtime = raw.times
        eeg_arry = raw.get_data()
        if pad:
            eeg_arry = zero_pad_arrays(eeg_arry, max_time)
        # number of cpus available

        if fmax is None:
            cpsd, freqsc, weightsc = mne.time_frequency.psd_array_multitaper(eeg_arry, 500, fmin=fmin, output='complex', bandwidth=bandwidth, n_jobs=mjobs, normalization=normalization, verbose=verbose)
        else:
            cpsd, freqsc, weightsc = mne.time_frequency.psd_array_multitaper(eeg_arry, 500, fmin=fmin, fmax=fmax, output='complex', bandwidth=bandwidth, n_jobs=mjobs, normalization=normalization, verbose=verbose)
        power = _psd_from_mt(cpsd, weightsc)
        seg_result['spectrum'].append(cpsd)
        seg_result['weights'].append(weightsc)
        seg_result['freqs'].append(freqsc)
        seg_result['times'].append(rawtime)
        seg_result['power'].append(power)

    
        time.sleep(0)
    return seg_result

def _process_single_subject(subj, statedict, states, basesavepath, pad=False, max_time=None, fmin=0.3, idx=None, n_subjs=None, mjobs=1, fmax=None, channels=None, normalization='full', bandwidth=None, save=True, verbose=False):
       
    """
    Create multitaper psds for a single subject and save them to a file: {basesavepath}/{subj}/{segment}/open_closed_multitaper_psds_{subj}.npz
    """

    assert states == ['open', 'closed'], "States must be ['open', 'closed']"
    if verbose:
        print(f"Subject: {subj} ({idx}/{n_subjs})")
    starttime = time.time()


    subj_result = {subj: {state: process_segment(state, statedict, channels=channels, pad=pad, max_time=max_time, fmin=fmin, mjobs=mjobs, fmax=fmax, normalization=normalization, bandwidth=bandwidth,  verbose=verbose) for state in states}}

    print("Time to multitaper for subject {}: {}".format(subj, time.time()-starttime))

    # now we save it and return the paths
    subj_savepath = os.path.join(basesavepath, f"{subj}")
    if not os.path.exists(subj_savepath):
        os.makedirs(subj_savepath)
    savefilename = f"open_closed_multitaper_psds_{subj}.npz"

    subject_paths = {subj: {}}
    for segment in ["first", "second", "avg"]:
        segmentsavepath = os.path.join(subj_savepath, segment)
        fullfilename = os.path.join(segmentsavepath, savefilename)
        # now save the psd_output using savez
        if save:
            open_freqs = subj_result[subj]['open']['freqs'][0] if len(subj_result[subj]['open']['freqs']) > 0 else None
            closed_freqs = subj_result[subj]['closed']['freqs'][0] if len(subj_result[subj]['closed']['freqs']) > 0 else None
            if open_freqs is not None:
                assert all([np.array_equal(open_freqs, freq) for freq in subj_result[subj]['open']['freqs']]), "Not all psds have the same frequencies"
            if closed_freqs is not None:
                assert all([np.array_equal(closed_freqs, freq) for freq in subj_result[subj]['closed']['freqs']]), "Not all psds have the same frequencies"

            if segment == "first":
                open_power = subj_result[subj]['open']['power'][0] if len(subj_result[subj]['open']['power']) > 0 else None
                closed_power = subj_result[subj]['closed']['power'][0] if len(subj_result[subj]['closed']['power']) > 0 else None
                open_spectrum = subj_result[subj]['open']['cpsd'][0] if len(subj_result[subj]['open']['cpsd']) > 0 else None
                closed_spectrum = subj_result[subj]['closed']['cpsd'][0] if len(subj_result[subj]['closed']['cpsd']) > 0 else None
                open_weights = subj_result[subj]['open']['weights'][0] if len(subj_result[subj]['open']['weights']) > 0 else None
                closed_weights = subj_result[subj]['closed']['weights'][0] if len(subj_result[subj]['closed']['weights']) > 0 else None
                open_spectrum = subj_result[subj]['open']['cpsd'][0] if len(subj_result[subj]['open']['cpsd']) > 0 else None
                closed_spectrum = subj_result[subj]['closed']['cpsd'][0] if len(subj_result[subj]['closed']['cpsd']) > 0 else None

            elif segment == "second":
                open_power = subj_result[subj]['open']['power'][1] if len(subj_result[subj]['open']['power']) > 1 else None
                closed_power = subj_result[subj]['closed']['power'][1] if len(subj_result[subj]['closed']['power']) > 1 else None
                open_spectrum = subj_result[subj]['open']['cpsd'][1] if len(subj_result[subj]['open']['cpsd']) > 1 else None
                closed_spectrum = subj_result[subj]['closed']['cpsd'][1] if len(subj_result[subj]['closed']['cpsd']) > 1 else None
                open_weights = subj_result[subj]['open']['weights'][1] if len(subj_result[subj]['open']['weights']) > 1 else None
                closed_weights = subj_result[subj]['closed']['weights'][1] if len(subj_result[subj]['closed']['weights']) > 1 else None
                open_spectrum = subj_result[subj]['open']['cpsd'][1] if len(subj_result[subj]['open']['cpsd']) > 1 else None
                closed_spectrum = subj_result[subj]['closed']['cpsd'][1] if len(subj_result[subj]['closed']['cpsd']) > 1 else None

            elif segment == "avg":
                open_times = subj_result[subj]['open']['times']
                closed_times = subj_result[subj]['closed']['times']
                open_power = psd_from_disjoint_arrays(subj_result[subj]['open']['power'], open_times)
                closed_power = psd_from_disjoint_arrays(subj_result[subj]['closed']['power'], closed_times)
                open_spectrum = psd_from_disjoint_arrays(subj_result[subj]['open']['cpsd'], open_times)
                closed_spectrum = psd_from_disjoint_arrays(subj_result[subj]['closed']['cpsd'], closed_times)
                
            np.savez_compressed(fullfilename,
                                open_power = open_power,
                                closed_power = closed_power,
                                open_freqs = open_freqs,
                                closed_freqs = closed_freqs,
                                open_weights = open_weights,
                                closed_weights = closed_weights,
                                open_spectrum = open_spectrum,
                                closed_spectrum = closed_spectrum,
                                channels = channels)

            subject_paths[subj][segment] = fullfilename
        time.sleep(0)

    


    return subject_paths, time.time()-starttime

def make_multitaper_dataset(dataset, savepath, fmin=0.3, fmax=245, normalization='full', bandwidth=None, verbose=False, n_jobs=1, mjobs=1, pad=False, save=True):
    """
    Input dataset in the form {subj: {'open': [], 'closed': []}}
    Returns a dataset in the form {subj: {'first': /path/to/first.npz, 'second': /path/to/second.npz, 'avg': /path/to/avg.npz}}
    where each npz file contains the power, phase, psd, weights, freqs, and times for the given segment for open and closed
    Args:
        dataset (dict): dictionary of the form {subj: {'open': [], 'closed': []}}
        fmin (float): minimum frequency to compute the multitaper
        fmax (float): maximum frequency to compute the multitaper
        normalization (str): normalization method for the multitaper
        bandwidth (float): bandwidth for the multitaper
        verbose (bool): verbose output
        mjobs (int): number of jobs to use for individual multitaper
        n_jobs (int): number of jobs to use for parallelizing across subjects
        pad (bool): whether to zero pad the data
    Returns:
        multitaper_dataset (dict): dictionary of the form {subj: {'first': /path/to/first.npz, 'second': /path/to/second.npz, 'avg': /path/to/avg.npz}}

    """
    states = ['open', 'closed']
    segments = ['first', 'second', 'avg']
    assert all([state in states for state in dataset[list(dataset.keys())[0]].keys()]), "Inner keys must be 'open' or 'closed'"
    channels = dataset[list(dataset.keys())[0]][states[0]][0].ch_names
    assert all([channels == dataset[subj][state][0].ch_names for subj in subjs for state in states]), "Channels must be the same for all subjects"


    multitaper_dataset = {subj: {seg: {} for seg in segments} for subj in dataset.keys()}

    tottime = time.time()
    process_times = []
    max_times = []

    # get the maximum time
    for subj in dataset.keys():
        for seg in segments:
            segdict = dataset[subj][seg]
            max_times.extend([len(raw.times) for raw in segdict])

    max_time = np.max(max_times)

    subjs = list(dataset.keys())
    subjs = [subj for subj in subjs if str(subj).isdigit()]

    if verbose:
        print(f"Beginning multitaper for {len(subjs)} subjects")
    n_subjs = len(subjs)

    parallel_results = Parallel(n_jobs=n_jobs)(delayed(_process_single_subject)(subj, dataset[subj], states, savepath, pad=pad, max_time=max_time, fmin=fmin, idx=idx, mjobs=mjobs, n_subjs=n_subjs, fmax=fmax, channels=channels, normalization=normalization, bandwidth=bandwidth, verbose=verbose, save=save) for idx, subj in enumerate(subjs))

    for idx, subj in enumerate(subjs):
        for seg in segments:
            multitaper_dataset[subj][seg] = parallel_results[idx][0][subj][seg]
        process_times.append(parallel_results[idx][1])

    print("Total time to multitaper: {}".format(time.time()-tottime))
    print(f"Median time: {np.median(process_times)}, max time: {np.max(process_times)}, min time: {np.min(process_times)}")

    output_dict = multitaper_dataset
    output_dict['channels'] = channels
    return output_dict

def zero_pad_arrays(eeg_array, max_time, time_axis=1):
    zero_padded_raw_data = np.pad(eeg_array, ((0,0), (0, max_time-eeg_array.shape[time_axis])), 'constant', constant_values=0)
    return zero_padded_raw_data
            
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

def load_all_transform_data_paths(savepath, subjs=None, num_load_subjs=None, which_segment='avg'):
    """
    Given a savepath, return a dictionary of the form {subj: {which_segment: /path/to/first.npz}}
    """
    inner_keys = ['open_power', 'closed_power', 'open_freqs', 'closed_freqs', 'open_phase', 'closed_phase', 'open_bispec', 'closed_bispec', 'open_weights', 'closed_weights', 'open_spectrum', 'closed_spectrum']
    
    if subjs is None:
        # search for all the subdirs in the savepath
        subjs = os.listdir(savepath)
        # only keep the ones that are directories
        subjs = [subj for subj in subjs if os.path.isdir(os.path.join(savepath, subj))]

    if num_load_subjs is not None:
        subjs = np.random.choice(subjs, size=num_load_subjs, replace=False)

    multitaper_dataset = {subj: {} for subj in subjs}
    for subj in subjs:
        subj_savepath = os.path.join(savepath, subj, which_segment)
        savefilename = f"open_closed_multitaper_psds_{subj}.npz"
        if os.path.exists(subj_savepath):
            data = np.load(os.path.join(subj_savepath, savefilename), allow_pickle=True)
            inner_dict = {}
            for key in inner_keys:
                inner_dict[key] = data[key]
            multitaper_dataset[subj] = inner_dict
        else:
            print(f"Path {subj_savepath} does not exist")
    
    # add the params
    params_jsonfile = os.path.join(savepath, 'params.json')
    if os.path.exists(params_jsonfile):
        with open(params_jsonfile, 'r') as f:
            params = json.load(f)
            multitaper_dataset['params'] = params

    return multitaper_dataset

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
        'method': kwargs['method'],
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
    parser.add_argument('--num_subjs', type=int, default=151)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--method', type=str, default='CSD')
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
    parser.add_argument('--pad', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--which_segment', type=str, default='avg', choices=['first', 'second', 'avg'], help='Which segment to use for the multitaper')
    parser.add_argument('--bandwidth', type=float, default=1)

    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    params = vars(args)
    locd_params = extract_locd_params(**params)
    # confirm that the user wants to run with these params
    print("Running with the following params:")
    pprint.pprint(params)
    response = input("Continue? (y/n)")
    if response == 'y':
        td_dict = main(locd_params=locd_params, pad=params['pad'], bandwidth=params['bandwidth'], which_segment=params['which_segment'], n_jobs=params['n_jobs'], save=params['save'])
        print(td_dict.keys())
    else:
        print("exiting...")