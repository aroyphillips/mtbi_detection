import numpy as np
import time
import os 
import glob 
import json
import argparse
from joblib import Parallel, delayed
import dotenv

import mtbi_detection.data.load_dataset as ld
import mtbi_detection.data.rereference_data as rd
import mtbi_detection.data.filter_data as fd
import mtbi_detection.data.data_utils as du
# DATAPATH = open('extracted_path.txt', 'r').read().strip() 
dotenv.load_dotenv()
DATAPATH = os.getenv('EXTRACTED_PATH')



def load_open_closed_pathdict(datapath=DATAPATH, savepath=None, num_subjs=151, verbose=True, l_freq=0.3, h_freq=None, fs_baseline=500, order=6, notches=[60, 120, 180, 240], notch_width=[2, 1, 0.5, 0.5], reference_method='CSD', reference_channels=['A1', 'A2'], bad_channels=['T1', 'T2'], filter_ecg=True, ecg_l_freq=8, ecg_h_freq=16, ecg_thresh='auto', ecg_method='correlation', keep_refs=False, save=True, n_jobs=1, num_load_subjs=None, random_load=False, include_ecg=True, late_filter_ecg=False, skip_ui=False, tables_folder='data/tables/'):
    """
    Given parameters, load the open closed data for all subjects
    Output:
        # dictionary in form {'open': {subj: [raws]}, 'closed': {subj: [raws]}}
        dictionary in form {subj: {'open': [path2raws], 'closed': [path2raws]}
    """

    if savepath is None:
        if datapath.endswith('/'):
            temp_datapath = datapath[:-1]
        else:
            temp_datapath = datapath
        savepath = os.path.join(os.path.dirname(temp_datapath), 'open_closed_segments')

        savepath = du.check_savepath(savepath, skip_ui=skip_ui)

        du.clean_params_path(savepath, skip_ui=skip_ui)

        dotenv.set_key(dotenv.find_dotenv(), 'OPEN_CLOSED_PATH', savepath)
        # with open('open_closed_path.txt', 'w') as f:
        #     f.write(savepath)



    params = {
        'l_freq': l_freq,
        'h_freq': h_freq,
        'fs_baseline': fs_baseline,
        'order': order,
        'notches': notches,
        'notch_width': notch_width,
        'num_subjs': num_subjs,
        'reference_method': reference_method,
        'reference_channels': reference_channels,
        'keep_refs': keep_refs,
        'bad_channels': bad_channels,
        'filter_ecg': filter_ecg,
        'ecg_l_freq': ecg_l_freq,
        'ecg_h_freq': ecg_h_freq,
        'ecg_thresh': ecg_thresh,
        'ecg_method': ecg_method,
    }
    
    if include_ecg:
        params['include_ecg'] = include_ecg
    
    if late_filter_ecg:
        params['late_filter_ecg'] = late_filter_ecg

    # load all the directory names that are children of os.path.join(savepath, 'params')
    remove_ecg = False
    alt_params = params.copy()
    if params['h_freq'] == 0.5*fs_baseline:
        print(f"h_freq is 0.5*fs_baseline: {params['h_freq']}, setting h_freq to None in alt_params")
        alt_params['h_freq'] = None
    elif params['h_freq'] == None:
        print(f"h_freq is None: {params['h_freq']}, setting h_freq to 0.5*fs_baseline in alt_params")
        alt_params['h_freq'] = 0.5*fs_baseline
    else:
        print(f"h_freq={params['h_freq']}")
        alt_params = params.copy()

    du.clean_params_path(savepath, skip_ui=skip_ui)
    try_savepath, found_match = du.check_and_make_params_folder(savepath, params, skip_ui=True, make_new_paramdir=False)
    if not found_match:
        ecg_params = params.copy()
        ecg_params['include_ecg'] = True
        try_savepath, found_match = du.check_and_make_params_folder(savepath, ecg_params, skip_ui=True, make_new_paramdir=False)
        if not found_match:
            try_savepath, found_match = du.check_and_make_params_folder(savepath, alt_params, skip_ui=True, make_new_paramdir=False)
        else:
            savepath = try_savepath
        ecg_alt_params = ecg_params.copy()
        ecg_alt_params['h_freq'] = alt_params['h_freq']
        if not found_match:
            try_savepath, found_match = du.check_and_make_params_folder(savepath, ecg_alt_params, skip_ui=True, make_new_paramdir=False)
        else:
            savepath = try_savepath
        remove_ecg = True and not include_ecg
        if not found_match:
            savepath, found_match = du.check_and_make_params_folder(savepath, params, skip_ui=skip_ui)
    else:
        print(f"Found matching params file: {try_savepath}")
        savepath = try_savepath
    if found_match:
        pathdict = load_all_reref_pathdata(savepath, num_load_subjs=num_load_subjs, random_load=random_load, remove_ecg=remove_ecg)
    else:
        starttime = time.time()
        print("Loading raw data...")
        dataset = ld.load_subjects_data(datapath=datapath, subjects=None, timepoint='baseline', recording_type='raw', preload=True, verbose=False)
        print(f"Finished loading raw data in {time.time()-starttime} seconds")
        annotations = ld.load_annotations(num_rows=num_subjs, tables_folder=tables_folder) 

        subjects = annotations['Study ID'].unique().astype(int)

        # subjects with missing "EYES CLOSED START 1" annotation or "EYES CLOSED END 1" annotation
        missing = []
        for subj in subjects:
            subj_annotations = annotations[annotations['Study ID'] == subj]
            if np.isnan(subj_annotations['EYES CLOSED START 1'].values[0]) or np.isnan(subj_annotations['EYES CLOSED END 1'].values[0]):
                missing.append(subj)

        good_subjs = [subj for subj in subjects if subj not in missing]
        print("Bandpass filtering data...")
        bandpass_dataset = load_bandpass_filtered_open_closed_dict(dataset, good_subjs, l_freq=l_freq, h_freq=h_freq, fs_baseline=fs_baseline, order=order, verbose=verbose, notches=notches, notch_width=notch_width)
        
        print("Removing ECG Artifacts")
        ecg_locations = ld.get_ecg_channel_locations(subjs=good_subjs, tables_folder=tables_folder)
        ecg_channels_dict = {subj: ecg_locations[sdx] for sdx, subj in enumerate(good_subjs)}
        if filter_ecg:
            ecg_free_dataset = load_ecg_free_open_closed_dict(bandpass_dataset, good_subjs, ecg_channels=ecg_locations, method=ecg_method, verbose=verbose)
        else:
            ecg_free_dataset = bandpass_dataset
        print("Cropping data...")
        cropped_dataset = load_open_closed_crops(ecg_free_dataset, verbose=verbose, annotations=annotations)

        if late_filter_ecg:
            cropped_dataset = ecg_free_cropped_open_closed_dict(cropped_dataset, good_subjs, ecg_channels=ecg_locations, method=ecg_method, verbose=verbose)

        common_channels = ld.find_common_channels_from_dict(dataset)
        print(f"Common channels: {common_channels}")
        
        print(f"Removing bad channels: {bad_channels}")
        common_channels = [channel for channel in common_channels if channel not in bad_channels]
        print("Rereferencing data...")
        reref_dataset = load_rereferenced_dict(cropped_dataset, common_channels, reference_method=reference_method, verbose=verbose, reference_channels=reference_channels, n_jobs=n_jobs, keep_refs=keep_refs, include_ecg=include_ecg, ecg_channels_dict=ecg_channels_dict)
        print("Done loading open closed data in ", time.time()-starttime)

        pathdict = {subj: {'open': [], 'closed': []} for subj in good_subjs}
        if save:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            states = ['open', 'closed']
            for idx, subj in enumerate(good_subjs):
                subjtime = time.time()
                intermediate_savepath = f"{savepath}/{subj}/"
                for state in states:
                    state_intermediate_savepath = os.path.join(intermediate_savepath, state)
                    if not os.path.exists(state_intermediate_savepath):
                        os.makedirs(state_intermediate_savepath)
                    state_raws = reref_dataset[subj][state.split('_')[0]]
                    for idx, raw in enumerate(state_raws):
                        raw.load_data()
                        savefilename = f"{state}{idx}_{subj}_raw.fif"
                        fullfilename = os.path.join(state_intermediate_savepath, savefilename)
                        # save with mne fif
                        raw.save(fullfilename, overwrite=True)
                        print(f"Saved to {fullfilename} in {time.time()-subjtime} seconds")
                        pathdict[subj][state].append(fullfilename)
            # now save the params
            with open(os.path.join(savepath, 'params.json'), 'w') as f:
                json.dump(params, f)
                print(f"Saved params to {os.path.join(savepath, 'params.json')}")
    pathdict = {str(subj): pathdict[subj] for subj in pathdict.keys()}
    return pathdict
    
def _reference_single_subj(raws, common_channels, reference_method='ipsilateral', keep_refs=False, reference_channels=['A1', 'A2'], include_ecg=False, ecg_channel=None):
    """
    Input: list of raws
    Output: list of rereferenced raws
    """
    reref_raws = []
    for raw in raws:
        reref_raws.append(rd.rereference_raw(raw, common_channels, reference_channels=reference_channels, method=reference_method, keep_refs=keep_refs, include_ecg=include_ecg, ecg_channel=ecg_channel))
    return reref_raws

def load_rereferenced_dict(dataset, common_channels, reference_method='ipsilateral', keep_refs=False, reference_channels=['A1', 'A2'], n_jobs=1, verbose=False, include_ecg=False, ecg_channels_dict=None):
    """
    Input dataset structure: {subj: {'open': [], 'closed': []}}
    Output dataset structure: {subj: {'open': [], 'closed': []}}
    """
    inner_keys = list(dataset[list(dataset.keys())[0]].keys())
    assert all([inner_key in ['open', 'closed'] for inner_key in inner_keys]), "Inner keys must be 'open' or 'closed'"

    reref_dataset = {key: {inner_key: [] for inner_key in inner_keys} for key in dataset.keys()}
    times = []
    if n_jobs == 1:
        for sdx, subj in enumerate(dataset.keys()):
            if verbose:
                print(f"Subject: {subj} {sdx+1}/{len(dataset.keys())}")
            starttime = time.time()
            for inner_key in inner_keys:
                raws = dataset[subj][inner_key]
                for raw in raws:
                    reref_dataset[subj][inner_key].append(rd.rereference_raw(raw, common_channels, reference_channels=reference_channels, method=reference_method, keep_refs=keep_refs, include_ecg=include_ecg, ecg_channel=ecg_channels_dict[subj]))
                    if verbose:
                        print("Time to rereference: {}".format(time.time()-starttime))
                        times.append(time.time()-starttime)
    else:
        reref_dataset = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_reference_single_subj)(dataset[subj][inner_key], common_channels, reference_method=reference_method, keep_refs=keep_refs, reference_channels=reference_channels, include_ecg=include_ecg, ecg_channel=ecg_channels_dict[subj]) for subj in dataset.keys() for inner_key in inner_keys)
        reref_dataset = {subj: {inner_key: reref_dataset[idx*len(inner_keys)+idx2] for idx2, inner_key in enumerate(inner_keys)} for idx, subj in enumerate(dataset.keys())}

    if verbose:
        print(f"Total time to rereference: {np.sum(times)}")
        print(f"Average time to rereference: {np.mean(times)}, median time to rereference: {np.median(times)}")
        print(f"Min time to rereference: {np.min(times)}, Max time to rereference: {np.max(times)}")
    return reref_dataset

def load_ecg_free_open_closed_dict(dataset, subjs, ecg_channels, method='correlation', l_freq=None, h_freq=None, thresh='auto', verbose=True):
    """
    incoming dataset has the form {str(subj): raw}
    outgoing dataset has the form {str(subj): raw}
    """
    ecg_free_dataset = {}
    times = []
    for sdx, (subj, ecg_channel) in enumerate(zip(subjs, ecg_channels)):
        if verbose:
            print(f"Removing ECG artifacts for subject: {subj} {sdx+1}/{len(subjs)}")
        raw = dataset[subj]
        starttime = time.time()
        ecg_free_dataset[subj] = fd.remove_ecg_artifact(raw, ecg_channel=ecg_channel, method=method, l_freq=l_freq, h_freq=h_freq, thresh=thresh)
        if verbose:
            print(f"Time to remove ECG artifacts: {time.time()-starttime}")
            times.append(time.time()-starttime)

    if verbose:
        print(f"Total time to remove ECG artifacts: {np.sum(times)}")
        print(f"Average time to remove ECG artifacts: {np.mean(times)}, median time to remove ECG artifacts: {np.median(times)}")
        print(f"Min time to remove ECG artifacts: {np.min(times)}, Max time to remove ECG artifacts: {np.max(times)}")

    return ecg_free_dataset

def ecg_free_cropped_open_closed_dict(dataset, subjs, ecg_channels, method='correlation', l_freq=None, h_freq=None, thresh='auto', verbose=True):
    """
    incoming dataset has the form {str(subj): raw}
    outgoing dataset has the form {str(subj): raw}
    """
    ecg_free_dataset = {subj:{'open': [], 'closed': []} for subj in subjs}
    times = []
    for sdx, (subj, ecg_channel) in enumerate(zip(subjs, ecg_channels)):
        if verbose:
            print(f"Removing ECG artifacts for subject: {subj} {sdx+1}/{len(subjs)}")
        for state in ['open', 'closed']:
            for raw in dataset[subj][state]:
            
                starttime = time.time()
                ecg_free_dataset[subj][state].append(fd.remove_ecg_artifact(raw, ecg_channel=ecg_channel, method=method, l_freq=l_freq, h_freq=h_freq, thresh=thresh))
                if verbose:
                    print(f"Time to remove ECG artifacts: {time.time()-starttime}")
                    times.append(time.time()-starttime)

    if verbose:
        print(f"Total time to remove ECG artifacts: {np.sum(times)}")
        print(f"Average time to remove ECG artifacts: {np.mean(times)}, median time to remove ECG artifacts: {np.median(times)}")
        print(f"Min time to remove ECG artifacts: {np.min(times)}, Max time to remove ECG artifacts: {np.max(times)}")

    return ecg_free_dataset

def extract_subj_open_closed_seg(raw, subj_id, annotations=None, fs=500):
    """
    Function to extract eyes_open and eyes_closed segments from a raw file and the corresponding subject id
    Args:
        raw: mne.io.Raw object
        subj_id: string, subject id
        annotations: pandas dataframe, annotations for the dataset
        fs: int, sampling frequency
    Output:
        output_dict: dictionary with keys 'open' and 'closed' and values as lists of raw objects
    """
    print(f"Subject: {subj_id}")
    if annotations is None:
        start_stop_annotations = ld.load_annotations()
    else:
        start_stop_annotations = annotations
    # get the subject's annotations
    subj_annotations = start_stop_annotations[start_stop_annotations["Study ID"] == subj_id]
    output_dict = {'open': [], 'closed': []}
    for segment in range(1,3):
        # get the start and stop times for eyes open
        open_start = subj_annotations["EYES OPEN START {}".format(str(segment))].values[0]
        open_end = subj_annotations["EYES OPEN END {}".format(str(segment))].values[0]
        # make sure not nan
        if not np.isnan(open_start) and not np.isnan(open_end):
            segment_starttime = time.time()
            open_segment = raw.copy().crop(tmin=open_start, tmax=open_end)
            seg_svtim = time.time()
            print(f"Time to make segment: {seg_svtim - segment_starttime}")
            if open_segment.info['sfreq'] != fs:
                # resample
                open_segment.resample(fs)
            assert open_segment.info['sfreq'] == fs , f"Sampling frequency is not {fs} Hz, it is {open_segment.info['sfreq']} Hz"
            output_dict['open'].append(open_segment)

        # get the start and stop times for eyes closed
        closed_start = subj_annotations["EYES CLOSED START {}".format(str(segment))].values[0]
        closed_end = subj_annotations["EYES CLOSED END {}".format(str(segment))].values[0]
        # make sure not nan
        if not np.isnan(closed_start) and not np.isnan(closed_end):
            segment_starttime = time.time()
            closed_segment = raw.copy().crop(tmin=closed_start, tmax=closed_end)
            segment_savetime = time.time()
            print(f"Time to make segment: {segment_savetime - segment_starttime}")
            if closed_segment.info['sfreq'] != fs:
                # resample
                closed_segment.resample(fs)
            assert closed_segment.info['sfreq'] == fs, f"Sampling frequency is not {fs} Hz, it is {closed_segment.info['sfreq']} Hz"
            output_dict['closed'].append(closed_segment)
    
    return output_dict

def load_bandpass_filtered_open_closed_dict(dataset, subjs, l_freq=0.5, h_freq=235, fs_baseline=500, order=6, notches=[], notch_width=2, verbose=False):
    """
    incoming dataset has the form {str(subj): [raw]}
    outgoing dataset has the form {str(subj): raw}
    """
    bandpass_dataset = {}
    times = []
    for sdx, subj in enumerate(subjs):
        if verbose:
            print(f"Filtering subject: {subj} {sdx+1}/{len(subjs)}")
        raws = dataset[str(subj)]
        assert len(raws)==1
        raw = raws[0]
        starttime = time.time()
        raw.load_data()
        bandpass_dataset[subj] = fd.filter_raw_mne(raw, l_freq=l_freq, h_freq=h_freq, fs_baseline=fs_baseline, order=order, notches=notches, notch_width=notch_width)

        if verbose:
            print(f"Time to filter: {time.time()-starttime}")
            times.append(time.time()-starttime)
    if verbose:
        print(f"Total time to filter: {np.sum(times)}")
        print(f"Average time to filter: {np.mean(times)}, median time to filter: {np.median(times)}")
        print(f"Min time to filter: {np.min(times)}, Max time to filter: {np.max(times)}")
    return bandpass_dataset
    
def load_open_closed_crops(dataset, verbose=False, annotations=None, fs_baseline=500):
    open_closed_dataset = {}
    times = []
    for sdx, subj in enumerate(dataset.keys()):
        if verbose:
            print(f"Cropping subject: {subj} {sdx+1}/{len(dataset.keys())}")
        rereftime = time.time()
        raw = dataset[subj]
        open_closed_dataset[subj] = extract_subj_open_closed_seg(raw, subj, annotations=annotations, fs=fs_baseline)
        if verbose:
            print("Time to crop: {}".format(time.time()-rereftime))
            times.append(time.time()-rereftime)
    if verbose:
        print(f"Total time to crop: {np.sum(times)}")
        print(f"Average time to crop: {np.mean(times)}, median time to crop: {np.median(times)}")
        print(f"Min time to crop: {np.min(times)}, Max time to crop: {np.max(times)}")
    return open_closed_dataset

def load_all_reref_pathdata(savepath, subjs=None, num_load_subjs=None, random_load=False, remove_ecg=False):
    """
    Load from directory into dict of the structure:
    {subj: {'open': [raws], 'closed': [raws]}}
    
    """
    data_dict = {} # keys are subj and then inside is a dict with keys open_psd_power, closed_psd_power, open_freqs, closed_freqs, open_psd_imag, closed_psd_imag

    if subjs is None:
        # search for all the subdirs in the savepath
        subjs = os.listdir(savepath)
        subjs = [subj for subj in subjs if subj.isdigit()]

    if num_load_subjs is not None:
        if random_load:
            subjs = np.random.choice(subjs, size=num_load_subjs, replace=False)
        else:
            subjs = subjs[:num_load_subjs]

    inner_keys = os.listdir(os.path.join(savepath, subjs[0]))
    for subj in subjs:
        intermediate_savepath = f"{savepath}/{subj}/"
        data_dict[subj] = {inner_key.split('_')[0]: [] for inner_key in inner_keys}
        for inner_key in inner_keys:
            state_intermediate_savepath = os.path.join(intermediate_savepath, inner_key)
            # get the fif files here
            fif_files = glob.glob(os.path.join(state_intermediate_savepath, '*.fif'))
            for fif_file in fif_files:
                data_dict[subj][inner_key.split('_')[0]].append(fif_file)
    if remove_ecg:
        data_dict['remove_ecg'] = remove_ecg
    return data_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    params = vars(args)
    extract_path = open('extracted_path.txt', 'r').read().strip()
    params['datapath'] = extract_path
    # confirm that the params are what the user wanted
    print("Parameters:", params)
    ui = input("Continue? (y/n)")

    if ui == 'y':
        starttime = time.time()
        open_closed_dict = load_open_closed_pathdict(**params)
        print(f"FINISHED LOADING OPEN CLOSED DICT IN {time.time()-starttime} SECONDS")
    else:
        print("Exiting...")

