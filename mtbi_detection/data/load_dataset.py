# code to load extracted data (can only run after running extract_data.py)

import glob
import mne
import pandas as pd
import numpy as np
import platform
import os

import mtbi_detection.data.extract_data as ed

DLPATH = open('download_path.txt', 'r').read().strip()
DATAPATH = open('extracted_path.txt', 'r').read().strip()

def find_raw_files(datapath, timepoint='baseline'):
    """
    Given a timepoint (baseline or followup) and a duration and overlap, finds the crop files for that segment type
    Inputs:
        datapath: path to the saved fif files containing EEG and ECG data
        timepoint: baseline or followup (default: baseline)
        dur: duration of the crop (default: 15.0)
        overlap: overlap of the crop (default: 5.0)
    Returns:
        crop_files: list of paths to the crop files
    """
    valid_timepoints = ['baseline', 'followup']
    if timepoint not in valid_timepoints:
        raise ValueError('timepoint must be baseline or followup')
    else:
        crop_files = glob.glob(f'{datapath}*/{timepoint}/{timepoint}*_raw.fif')
    return crop_files

def find_5min_files(datapath):
    """
    Given a timepoint (baseline or followup) and a duration and overlap, finds the crop files for that segment type
    Inputs:
        datapath: path to the saved fif files containing EEG and ECG data
    Returns:
        files: list of paths to the crop files
    """    
    files = glob.glob(f'{datapath}*/*_raw.fif')
    return files

def load_fif_from_subject(subject, fif_files, timepoint='baseline', preload=False, verbose=False, as_paths=False):
    """
    Given a subject number and a list of fif files, returns the fif file for that subject
    Inputs:
        subject: subject number
        fif_files: list of fif files
    Returns:
        mne_raws: list of mne raw objects
    """
    _check_valid_timepoint(timepoint)

    subject_fif_files = []
    for file in fif_files:
        if f'_{subject}_raw' in file or f'_subj{subject}_' in file:
            subject_fif_files.append(file)
    if len(subject_fif_files) == 0:
        raise ValueError(f'No fif file found for that subject {subject} and segment type {timepoint}')
    if as_paths:
        return subject_fif_files
    else:
        mne_raws = [mne.io.read_raw_fif(file, verbose=verbose, preload=preload) for file in subject_fif_files]
        return mne_raws

def load_subjects_data(datapath, subjects=None, num_subjects=None, follow_annotations=False, timepoint='baseline', recording_type='raw', preload=False, verbose=True, annotations_folder='data/tables/', as_paths=False):
    """
    Given a list of subjects and the type of segment and epoch return a dictionary of mne raw objects where the key is the subject number and the value is the list of raw mne raw objects
    Inputs:
        datapath: path to the saved fif files containing EEG and ECG data
        subjects: list of subjects
        timepoint: baseline or followup (default: baseline)
        recording_type: raw, or 5min (default: raw)
        num_subjects: number of subjects to load
        follow_annotations: whether to follow the annotations or not
        preload: whether to preload the data or not
        verbose: whether to print the progress or not
        annotations_folder: folder containing the annotations
        as_paths: whether to return the paths or the mne

    Returns:
        data: dictionary of mne raw objects where the key is the subject number and the value is the list of raw mne raw objects
            if as_paths is True, then the value is a list of paths to the fif files
            i.e.:
             {subject: [raw1, raw2, ...], ...}
            or 
             {subject: [path1, path2, ...], ...}
    """
    _check_valid_timepoint(timepoint)
    valid_recording_types = ['raw', '5min']

    if recording_type == 'raw':
        fif_files = find_raw_files(datapath=datapath, timepoint=timepoint)
    elif recording_type == '5min':
        fif_files = find_5min_files(datapath=datapath)
    elif recording_type not in valid_recording_types:
        raise ValueError(f'recording_type must be crop or segment')
    else:
        raise ValueError(f'timepoint functionality not implemented yet')
    
    if subjects is None:
        # get the directory that is only a number within a path
        if follow_annotations:
            start_stop_annotations = load_annotations(base_folder=annotations_folder)
            subjects = np.unique(start_stop_annotations["Study ID"])
            # drop the nan values
            subjects = subjects[~np.isnan(subjects)][:num_subjects].astype(int)

        else:
            subjects = np.unique([file.split('_')[-2] for file in fif_files])
            if num_subjects is not None:
                subjects = subjects[:num_subjects]

    data = {}
    if as_paths:
        for subject in subjects:
            data[subject] = [file for file in fif_files if f'_{subject}_' in file]
    else:
        for subject in subjects:
            data[subject] = load_fif_from_subject(subject, fif_files, timepoint=timepoint, preload=preload, verbose=verbose)
    print(f'Loaded {len(data)} subjects')
    return data


def _check_valid_input(input, valid_inputs, input_name):
    if input not in valid_inputs:
        raise ValueError(f'{input_name} must be {valid_inputs}')

def _check_valid_timepoint(timepoint):
    valid_timepoints = ['baseline', 'followup']
    _check_valid_input(timepoint, valid_timepoints, 'timepoint')

# labels 
def create_label_dict(label_file_df, label_col, subj_col):
    """
    Given a dataframe containing labels and subject ids, returns a dictionary where the key is the subject id and the value is the label
    Inputs:
        label_file_df: dataframe containing labels and subject ids
        label_col: column name of the label column
        subj_col: column name of the subject id column
    Returns:
        label_dict: dictionary where the key is the subject id and the value is the label
    """
    label_dict = {}
    for i in range(label_file_df.shape[0]):
        label_val = label_file_df.iloc[i, label_file_df.columns.get_loc(label_col)]
        if type(label_val) == str:
            label_val = 0 if label_val == 'Control' else 1
        label_dict[label_file_df.iloc[i, label_file_df.columns.get_loc(subj_col)]] = label_val
    return label_dict

def load_label_dict(label_col='EEG_FITBIR.Main.CaseContrlInd', subj_col='EEG_FITBIR.Main.SubjectIDNum', sort=False):
    """
    Given a csv file containing labels and subject ids
    Inputs:
        label_col: column name of the label column
        subj_col: column name of the subject id column
    Returns:
        label_dict: dictionary where the key is the subject id and the value is the label
    """
    label_file = os.path.join(DATAPATH, 'labels.csv')
    label_file_df = pd.read_csv(label_file)
    ed.check_labels(label_file_df, DLPATH)
    label_dict = create_label_dict(label_file_df, label_col, subj_col)
    if sort:
        label_dict = {k: label_dict[k] for k in sorted(label_dict.keys())}
    return label_dict

def select_data_with_labels(data_df, labels_df, select_col, select_val, select_feature=None):
    """
    Given a dataframe of data and a dataframe of labels, select the data and labels based on the value of a column in the labels dataframe 
    Input:
        data_df: dataframe of data
        labels_df: dataframe of labels
        select_col: column in labels to select on
        select_val: value of select_col to select
        select_feature: feature to select from data_df based on the header (featuregroup_channel_featurename) (optional)
    Returns:
        data: dataframe of data
        labels: dataframe of labels
    """
    # select the rows in the labels dataframe that match the select_val
    labels = labels_df[labels_df[select_col] == select_val].copy(deep=True)
    # get the row indices of the labels dataframe
    label_indices = labels.index
    # select the rows in the data dataframe that match the subjects
    data = data_df[data_df.index.isin(label_indices)].copy(deep=True)
    # if a select_feature is provided, select the feature from the data dataframe
    if select_feature is not None:
        data = data[select_feature]
    return data, labels

def select_data_with_multiple_labels(data_df, labels_df, split_col, select_vals, select_features=None):
    """
    Given a dataframe of data and a dataframe of labels, select the data and labels based on the value of a column in the labels dataframe 
    Input:
        data_df: dataframe of data
        labels_df: dataframe of labels
        split_col: column in labels to split on
        select_vals: list of values of split_col to select
        select_feature: list of feature to select from data_df based on the header (featuregroup_channel_featurename) (optional)
    Returns:
        data: dataframe of data
        labels: dataframe of labels
    """
    # select the rows in the labels dataframe that match the select_val
    labels = labels_df[labels_df[split_col].isin(select_vals)].copy(deep=True)
    # get the row indices of the labels dataframe
    label_indices = labels.index
    # select the rows in the data dataframe that match the subjects
    data = data_df[data_df.index.isin(label_indices)].copy(deep=True)
    # if a select_feature is provided, select the feature from the data dataframe   
    if select_features is not None:
        select_feature_cols = [col for col in data.columns if any([feature in col for feature in select_features])]
        data = data[select_feature_cols]
    return data, labels

def find_common_channels_from_list(raw_list, sort=True):
    """
    Given a list of raw objects, find the common channels
    Input:
        raw_list: list of mne raw objects
    Returns:
        common_channels: list of common channels
    """
    common_channels = set(raw_list[0].ch_names)
    for raw in raw_list:
        common_channels = common_channels.intersection(set(raw.ch_names))
    if sort:
        common_channels = sorted(common_channels)
    return list(common_channels)

def find_common_channels_from_dict(data_dict, sort=True, ignore=['E']):
    """
    Given a dictionary of raw objects, find the common channels
    Input:
        data_dict: dictionary of mne raw objects of structure {subject: [raw1, raw2, ...], ...}
    Returns:
        common_channels: list of common channels
    """
    raw_list = [raw_item for value in data_dict.values() for raw_item in value]
    common_channels = find_common_channels_from_list(raw_list, sort=sort)
    common_channels = [channel for channel in common_channels if channel[0] not in ignore]
    return common_channels

def find_unique_channel_groups(data_dict):
    """
    Given a dictionary of raw objects, find the number of unique channel groups and the subjects that are in each group
    Input:
        data_dict: dictionary of mne raw objects of structure {subject: [raw1, raw2, ...], ...}
    Returns:
        channel_group_dict: dictionary of the form {group: {subjects: [subj1, subj2, ...], channels: [chan1, chan2, ...]}, ...}
    """
    channel_group_dict = {}
    unique_groups = []
    for subj, raw_list in data_dict.items():
        for raw in raw_list:
            channel_group = tuple(raw.ch_names)
            if channel_group not in unique_groups:
                channel_group_dict[len(unique_groups)] = {'subjects': [subj], 'channels': raw.ch_names}
                unique_groups.append(channel_group)
            else:
                channel_group_index = unique_groups.index(channel_group)
                channel_group_dict[channel_group_index]['subjects'].append(subj)
    return channel_group_dict

def load_single_subject(datapath, subject=None, timepoint='baseline', recording_type='raw', preload=False, verbose=True, as_paths=False):
    """
    Load a rubject from the dataset using the load_subjects_data function, and return a dictionary containing the subject number, the raw object, and the label
    Args:
        datapath: path to the saved fif files containing EEG and ECG data
        subject: subject number
        timepoint: baseline or followup (default: baseline)
        recording_type: raw, or 5min (default: raw)
        preload: whether to preload the data or not
        verbose: whether to print the progress or not
    Returns:
        data: dictionary containing the subject number, the raw object, and the label
    """
    _check_valid_timepoint(timepoint)
    valid_recording_types = ['raw', '5min']
    if recording_type == 'raw':
        fif_files = find_raw_files(datapath=datapath, timepoint=timepoint)
    elif recording_type == '5min':
        fif_files = find_5min_files(datapath=datapath)
    elif recording_type not in valid_recording_types:
        raise ValueError(f'recording_type must be crop or segment')
    else:
        raise ValueError(f'timepoint functionality not implemented yet')
    

    if subject is None:
        # get the directory that is only a number within a path
        subject = np.random.choice(np.unique([file.split('_')[-2] for file in fif_files]))
        print(f"subject: {subject}")

    label = get_subject_label(int(subject))
    if as_paths:
        fif_files = load_fif_from_subject(subject, fif_files, timepoint=timepoint, preload=preload, verbose=verbose, as_paths=as_paths)

        return {'subject': int(subject), 'raw': fif_files, 'label': label}
    else:
        raws = load_fif_from_subject(subject, fif_files, timepoint=timepoint, preload=preload, verbose=verbose)
        return {'subject': int(subject), 'raw': raws, 'label': label}

def get_subject_label(subject):
    """
    Given a subject number, return the label (0 or 1)
    """
    label_dict = load_label_dict()
    return label_dict[int(subject)]

def get_labels_from_subjects(subjects):
    """
    Given a list of subjects, return the labels
    """
    label_dict = load_label_dict()
    return [label_dict[int(subject)] for subject in subjects]

def load_annotations(filename="start_stop_annotations.csv", base_folder='data/tables/', num_rows=None):
    """
    Load the start stop annotations from the csv file
    Args:
        filename: name of the csv file
        base_folder: folder containing the csv file
        num_rows: number of rows to load
    Returns:
        start_stop_annotations: dataframe containing the start stop annotations
    """
    csv_filename = base_folder + filename
    start_stop_annotations = pd.read_csv(csv_filename)
    if num_rows is not None:
        start_stop_annotations = start_stop_annotations.iloc[:num_rows]

    # get the total time difference between the start and stop timess["start_time"]
    # strip the whitespace from beginning and end of column names
    # start_stop_annotations.columns = start_stop_annotations.columns.str.strip()
    start_stop_annotations["total_time_closed1"] = start_stop_annotations["EYES CLOSED END 1"] - start_stop_annotations["EYES CLOSED START 1"]
    start_stop_annotations["total_time_closed2"] = start_stop_annotations["EYES CLOSED END 2"] - start_stop_annotations["EYES CLOSED START 2"]
    start_stop_annotations["total_time_open1"] = start_stop_annotations["EYES OPEN END 1"] - start_stop_annotations["EYES OPEN START 1"]
    start_stop_annotations["total_time_open2"] = start_stop_annotations["EYES OPEN END 2"] - start_stop_annotations["EYES OPEN START 2"]

    # add the two time together and ignore the nan values

    start_stop_annotations["total_time_closed"] = start_stop_annotations.copy().fillna(0)["total_time_closed1"] + start_stop_annotations.fillna(0)["total_time_closed2"]
    start_stop_annotations["total_time_open"] = start_stop_annotations.fillna(0)["total_time_open1"] + start_stop_annotations.fillna(0)["total_time_open1"]


    skip_subjs = [7,9, 33, 47, 54]
    start_stop_annotations["skip"] = (start_stop_annotations["total_time_closed"] == 0) | (start_stop_annotations["total_time_open"] == 0)
    start_stop_annotations["skip"] = start_stop_annotations["skip"] | start_stop_annotations["Study ID"].isin(skip_subjs)
    start_stop_annotations["missing_both"] = (start_stop_annotations["total_time_closed"] == 0) & (start_stop_annotations["total_time_open"] == 0)  

    return start_stop_annotations


def load_labels_df(filename='/shared/roy/mTBI/saved_processed_data/mission_connect/large_disk_copy/labels.csv'):
    labels_df = pd.read_csv(filename)
    return labels_df

def make_full_labels_dict(labels_df=None):
    if labels_df is None:
        labels_df = load_labels_df()
    full_label_dict = {}
    for i, row in labels_df.iterrows():
        label_str = row['EEG_FITBIR.Main.CaseContrlInd']
        label_int = 1 if label_str == 'Case' else 0
        subject = int(row['EEG_FITBIR.Main.SubjectIDNum'])
        full_label_dict[subject] = label_int
    return full_label_dict

def make_labels_dict(df, label_buzz = 'CaseContrlInd', subj_buzz = 'SubjectIDNum'):
    labels_dict = {}
    label_col = [col for col in df.columns if label_buzz in col][0]
    subj_col = [col for col in df.columns if subj_buzz in col][0]
    for index, row in df.iterrows():
        if row[label_col] == 'Case':
            labels_dict[row[subj_col]] = 1
        elif row[label_col] == 'Control':
            labels_dict[row[subj_col]] = 0
    return labels_dict


def get_ecg_channel_locations(n_subjs=None, annotations=None, col_name='ECG Location', subjs=None, subjs_col='Study ID', base_folder='data/tables/'):
    """
    Returns a list of the ecg channel locations
    """
    if annotations is None:
        annotations = load_annotations(base_folder=base_folder)
    if subjs is not None:
        # make sure the ecg channels correspond to the subjects in subjs_col of annotations
        annotations = annotations.copy(deep=True)
        annotations = annotations.loc[[str(subj) in subjs or int(subj) in subjs or float(subj) in subjs for subj in annotations[subjs_col]]]
        assert list(annotations[subjs_col].values.astype(int)) == [int(s) for s in subjs], f"subjs_col of annotations does not match subjs: {subjs}"

    if n_subjs is not None:
        # make sure the ecg channels correspond to the first n_subjs
        annotations = annotations[:n_subjs]

    ecg_channels = annotations[col_name]
    # only apply the strip and split if the value in the column is a string
    # replace any nnan values with empty string
    ecg_channels = ecg_channels.fillna('')
    ecg_channels = ecg_channels.apply(lambda x: x.strip().split(',')[0])
    

        
    ecg_channels = list(ecg_channels)

    return ecg_channels

# now loop through each subject 
def get_merged_csv(base_folder='../data/tables/', filename='deduped_merged_csv_files.csv'):
    """
    Given a base folder and a filename, return the merged csv file
    """
    csv_filename = os.path.join(base_folder, filename)
    merged_csv = pd.read_csv(csv_filename)
    return merged_csv
def make_subset_from_col(df, subj_col, val_col, col_pos_opts, condition_col=None, verbose=False):
    """
    df: dataframe
    subj_col: where to find the subject IDs
    val_col: where to find the value of interest:
    condition_col: dictionary of columns and corresponding valid values
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
            if verbose:
                # print the subject without a line break
                print(f"Subject {subj} ", end='')
            subj_df = df[df[subj_col] == subj]
            if condition_col is not None:
                # get only the rows that satisfy the condition
                for key, val in condition_col.items():
                    subj_df = subj_df[subj_df[key].isin(val)]
            for key, val in col_pos_opts.items():
                if any([v in val for v in subj_df[val_col].values]):
                    if verbose:
                        print(f"is {key} because: { subj_df[val_col].values[np.array([v in val for v in subj_df[val_col].values])]}")
                    val_col_values.append(key)
                    if len(val) > 0:
                        break
                elif len(val) == 0:
                    if verbose:
                        print(f"Subject {subj} does not have any listed conditions")
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

    # print a summary
    for key, val in val_list_dict.items():
        print(f"Number of subjects with {key}: {len(val)}")
    return val_list_dict 

def load_splits(base_folder='data/internal/'):
    """
    Loads the train, val, dev, holdout subjs from base_folder and returns as dictionary
    Args:
        base_folder: folder containing the npy files
    Returns:
        splits: dictionary containing the train, val, dev, holdout subjs

    """
    train_subjs = np.load(os.path.join(base_folder, 'train_subjs.npy'))
    ival_subjs = np.load(os.path.join(base_folder, 'ival_subjs.npy')) # internal validation
    holdout_subjs = np.load(os.path.join(base_folder, 'holdout_subjs.npy'))
    dev_subjs = np.load(os.path.join(base_folder, 'dev_subjs.npy'))
    skip_subjs = [7,9, 33, 47, 54]
    train_subjs = [subj for subj in train_subjs if int(subj) not in skip_subjs]
    ival_subjs = [subj for subj in ival_subjs if int(subj) not in skip_subjs]
    holdout_subjs = [subj for subj in holdout_subjs if int(subj) not in skip_subjs]
    dev_subjs = [subj for subj in dev_subjs if int(subj) not in skip_subjs]
    splits = {'train': train_subjs, 'ival': ival_subjs, 'holdout': holdout_subjs, 'dev': dev_subjs, 'skip': skip_subjs}
    assert len(train_subjs) + len(ival_subjs) + len(holdout_subjs) == len(dev_subjs) + len(holdout_subjs)
    assert np.sum(get_labels_from_subjects(train_subjs)) + np.sum(get_labels_from_subjects(ival_subjs)) + np.sum(get_labels_from_subjects(holdout_subjs)) == np.sum(get_labels_from_subjects(dev_subjs)) + np.sum(get_labels_from_subjects(holdout_subjs))
    return splits

if __name__ == '__main__':
    splits = load_splits()
    print(f"Train: {len(splits['train'])}, Ival: {len(splits['ival'])}, Holdout: {len(splits['holdout'])}, Dev: {len(splits['dev'])}")
    print(f"mTBI in train: {np.sum(get_labels_from_subjects(splits['train']))}, mTBI in ival: {np.sum(get_labels_from_subjects(splits['ival']))}, mTBI in holdout: {np.sum(get_labels_from_subjects(splits['holdout']))}, mTBI in dev: {np.sum(get_labels_from_subjects(splits['dev']))}")
    print(f"Total mTBI: {np.sum(get_labels_from_subjects(splits['train'])) + np.sum(get_labels_from_subjects(splits['ival'])) + np.sum(get_labels_from_subjects(splits['holdout']))}")
    assert np.sum(get_labels_from_subjects(splits['train'])) + np.sum(get_labels_from_subjects(splits['ival'])) + np.sum(get_labels_from_subjects(splits['holdout'])) == np.sum(get_labels_from_subjects(splits['dev'])) + np.sum(get_labels_from_subjects(splits['holdout']))
    print(f"Total subjects: {len(splits['train']) + len(splits['ival']) + len(splits['holdout'])}")
    assert len(splits['train']) + len(splits['ival']) + len(splits['holdout']) == len(splits['dev']) + len(splits['holdout'])
    print(f"Control in train: {len(splits['train']) - np.sum(get_labels_from_subjects(splits['train']))}, Control in ival: {len(splits['ival']) - np.sum(get_labels_from_subjects(splits['ival']))}, Control in holdout: {len(splits['holdout']) - np.sum(get_labels_from_subjects(splits['holdout']))}, Control in dev: {len(splits['dev']) - np.sum(get_labels_from_subjects(splits['dev']))}")
    print(f"Total Control: {len(splits['train']) - np.sum(get_labels_from_subjects(splits['train'])) + len(splits['ival']) - np.sum(get_labels_from_subjects(splits['ival'])) + len(splits['holdout']) - np.sum(get_labels_from_subjects(splits['holdout']))}")
    print(f"Ratio mTBI/Total train: {np.sum(get_labels_from_subjects(splits['train']))/len(splits['train'])}, Ratio mTBI/Total ival: {np.sum(get_labels_from_subjects(splits['ival']))/len(splits['ival'])}, Ratio mTBI/Total holdout: {np.sum(get_labels_from_subjects(splits['holdout']))/len(splits['holdout'])}, Ratio mTBI/Total dev: {np.sum(get_labels_from_subjects(splits['dev']))/len(splits['dev'])}")
    print(f"Finished running {__file__}")