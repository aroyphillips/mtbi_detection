# code to convert the FITBIR Mission Connect data https://fitbir.nih.gov/study_profile/350 to python readable MNE format
import numpy as np
import pandas as pd
import os
import time
import mne
import glob
import warnings
import json
import argparse

def main(dlpath='/home/ap60/Downloads/fitbir_downloads/fitbir_mission_connect_downloads/', savepath='/scratch/ap60/mtbi_detection/data/raw_mne_files/'):
    """
    Given the path to the data as downloaded from FITBIR, converts the data to MNE readable format and saves it in the savepath
    Args:
        - dlpath: path to the downloaded FITBIR data
        - savepath: path to save the processed data
    Returns:
        - Dataframe containing the information about the data
    """
    # set up files needed for loading
    datafiles = os.listdir(dlpath)
    # finds all the csv files
    csv_files = [file for file in datafiles if file.endswith('csv')]

    # gets the list of RecordRawDataFiles
    eeg_csv_paths = [file for file in csv_files if "EEG" in file]
    if len(eeg_csv_paths) == 1:
        eeg_df = pd.read_csv(os.path.join(dlpath, eeg_csv_paths[0]))
    elif len(eeg_csv_paths) > 1:
        # ask the user to choose the correct file
        print("Multiple EEG files found. Please choose the correct one")
        for idx, file in enumerate(eeg_csv_paths):
            print(f"{idx}: {file}")
        eeg_idx = int(input("Enter the index of the correct file: "))
        eeg_df = pd.read_csv(os.path.join(dlpath, eeg_csv_paths[eeg_idx]))
    else:
        raise ValueError(f"No EEG files found in the data directory: {dlpath}")
    

    eeg_record_data = eeg_df["EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile"]

    eeg_dir_names = [z[1:-4]+"/" for z in eeg_record_data]

    start_load_time = time.time()
    all_raw_paths = find_all_nk_files(dlpath, eeg_dir_names, filetype="eeg")
    if len(all_raw_paths)<1:
        print("Error loading files")
        return
    end_load_time = time.time()
    print("Time to load all raws: ", end_load_time-start_load_time)

    rawpath_io_df = pd.DataFrame(all_raw_paths, index=list(range(len(all_raw_paths))), columns=["raw_path"]) 
                                              
    # load other data
    all_pnts = find_all_nk_files(dlpath, eeg_dir_names, filetype='pnt', verbose=True)           
    # separate this information
    meas_date_list = []
    version_list = []
    for pnt in all_pnts:
        try:
            meas_date_list.append(pnt['meas_date'])
            version_list.append(pnt['version'])
        except:
            meas_date_list.append(np.nan)
            version_list.append(np.nan)
    #     keys.update(pnt.keys()) # done to check if there are other keys

    print(len(meas_date_list))

    all_logs = find_all_nk_files(dlpath, eeg_dir_names, filetype='log', verbose=True, load=True)       

    log_keys = set(all_logs[0].keys())

    test_logs(all_logs)

    description_log_list = []
    onset_log_list = []

    # duration does not change
    for pid, log in enumerate(all_logs):
        try:
            log_keys.update(log.keys())
            description_log_list.append(log['description'])
            onset_log_list.append(log['onset'])
            
            # check if any durations are present
            if len(np.unique(log['duration']))>1:
                print("Durs change w", pid) 
                # no durations
        except:
            description_log_list.append(np.nan)
            onset_log_list.append(np.nan)

    meta_io_df = pd.DataFrame(list(zip(meas_date_list, version_list, onset_log_list, description_log_list)), index=list(range(len(all_pnts))), columns=["meas_date_pnt", "version_pnt", "onset_log","description_log"])

    data_df = pd.concat([eeg_df, rawpath_io_df, meta_io_df], axis=1)

    savetimes = save_protocol(data_df, savepath=savepath)
    save_labels(dlpath, savepath)
    # save save times using json
    with open(savepath+'preprocess_save_times.json', 'w') as fp:
        json.dump(savetimes, fp)

    return data_df

def save_protocol(data_df, savepath):
    """
    Given a dataframe of the non-data information and a list of raws, saves the protocol by going through the dataframe
    and saving the raws in the correct folder
    """
    if not os.path.exists(savepath):
        ui = input(f"Savepath {savepath} does not exist.\nCreate it? (y/n)")
        if ui == 'y':
            os.mkdirs(savepath)
        else:
            raise ValueError("Savepath does not exist")

    # sort the data_df by 'EEG_FITBIR.Main.SubjectIDNum'
    data_df = data_df.sort_values(by=['EEG_FITBIR.Main.SubjectIDNum'])

    unique_subjects = np.unique(data_df['EEG_FITBIR.Main.SubjectIDNum'])

    # iterate through the data_df
    savetimes = []

    for idx, subj in enumerate(unique_subjects):
        start = time.time()
        print(f"Saving subject {subj} ({idx}/{len(unique_subjects)})", end='\r')
        # check if there is a folder for this subject savepath+subj
        savepathdir = savepath+str(subj)
        if not os.path.exists(savepathdir):
            os.mkdir(savepathdir)
        
       
        # get all rows in the data_df with the subject
        subj_df = data_df.loc[data_df['EEG_FITBIR.Main.SubjectIDNum'] == subj]

        # get all the rows of subj_df that are have 'EEG_FITBIR.Main.GeneralNotesTxt' == 'Baseline'
        baseline_df = subj_df.loc[subj_df['EEG_FITBIR.Main.GeneralNotesTxt'] == 'Baseline']

        baseline_info_df = baseline_df.drop(columns=['raw_path'])

        if baseline_df['EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile'].nunique() == 1:
            if not os.path.exists(os.path.join(savepathdir, "baseline")):
                os.mkdir(os.path.join(savepathdir, "baseline"))
            
            raw_path = baseline_df.iloc[0]['raw_path']
            if type(raw_path) == str:
                raw = mne.io.read_raw_nihon(raw_path, preload=True)
                raw.save(os.path.join(savepathdir, "baseline", f"baseline_{subj}_raw.fif"), overwrite=True)
                baseline_info_df.to_csv(os.path.join(savepathdir, "baseline", f"baseline_{subj}_info.csv"), index=False)
        elif len(baseline_df) == 0:
            print("No baseline for ", subj)
        else:
            raise(f"Multiple filepaths {baseline_df['EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile']} for {subj}")

            
            
        # get all the rows of subj_df that do not have 'EEG_FITBIR.Main.GeneralNotesTxt' == 'Baseline'
        post_df = subj_df.loc[subj_df['EEG_FITBIR.Main.GeneralNotesTxt'] != 'Baseline']
        # datafiles = subj_df['EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile'].unique
        timepoints = post_df['EEG_FITBIR.Main.GeneralNotesTxt'].unique()
        followup_info_df = post_df.drop(columns=['raw_path'])

        for timepoint in timepoints:
            if not os.path.exists(os.path.join(savepathdir, "followup")):
                os.mkdir(os.path.join(savepathdir, "followup"))
            timepoint_df = post_df.loc[post_df['EEG_FITBIR.Main.GeneralNotesTxt'] == timepoint]
            if timepoint_df['EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile'].nunique() == 1:
                raw_path =timepoint_df.iloc[0]['raw_path']
                if type(raw_path) == str:
                    raw = mne.io.read_raw_nihon(raw_path, preload=True)
                    timepoint = timepoint.replace(" ", "-")
                    raw.save(os.path.join(savepathdir,"followup", f"followup_{timepoint}_{subj}_raw.fif"), overwrite=True)
                    followup_info_df.to_csv(os.path.join(savepathdir, "followup", f"followup_{timepoint}_{subj}_info.csv"), index=False)
            elif len(timepoint_df) == 0:
                print("No followup for ", subj)
            else:
                raise(f"Multiple filepaths {timepoint_df['EEG_FITBIR.EEG Raw Data Files.EEGRecordRawDataFile']} for {subj}")
        if len(post_df) == 0:
            print("No followup for ", subj)
        
        end = time.time()
        savetimes.append(end-start)

    print("Saved in time: ", np.sum(savetimes))
    print("Average time to save: ", np.mean(savetimes))
    return savetimes


def test_logs(all_logs):
    # test to ensure the two overlap

    logs = []
    for num, log in enumerate(all_logs):
        print("TEST on", num)
        try:
            np.isnan(log)
            print("{} is empty".format(num))
            continue
        except:
            if len(log['onset']) != len(log['description']):
                print("FAILED ON", num)
            else:
                logs.append(log)
                print("passed")
    return logs

def get_info_df(name, csv_files, load_csvs, datafiles):
    """
    given a short name (same capitilization) returns the dataframe of the info
    
    """
    targ_idx = csv_files.index([i for i in datafiles if name in i][0]) #changed from 1 to 0
    targ_df =  load_csvs[targ_idx]
    return targ_df, targ_idx

# helper functions
def endswith_test(string, str_type='EEG'):
    is_test = False
    if string[-3:].upper()==str_type:
        is_test = True
    return is_test

def endswith_eeg(string):
    """
    gets raw eeg file
    """
    is_eeg = endswith_test(string, str_type='EEG')
    return is_eeg

def endswith_pnt(string):
    """ 
    gets pnt file containing metadata such as date
    """
    is_pnt = endswith_test(string, str_type='PNT')
    return is_pnt

def endswith_log(string):
    """
    gets log file containing annotations
    """
    is_log = endswith_test(string, str_type='LOG')
    return is_log

def endswith_21e(string):
    """
    gets 21e file containing containing channel and electrode recording system
    """
    is_21e = endswith_test(string, str_type='21E')
    return is_21e


def find_all_nk_files(datapath, eeg_dir_names, filetype='eeg', verbose=False, load=False):
    """
    Given the path to the data and the names of the directories containing the data, finds all the files of the specified type
    Args:
        - datapath: path to the data
        - eeg_dir_names: list of directory names containing the data
        - filetype: type of file to load (eeg, pnt, log, 21e)
        - verbose: whether to print the files being loaded
    Returns:
        - list of filepaths
    
    """
    all_raws = []
    for count, dirname in enumerate(eeg_dir_names):
        test_files = glob.glob(datapath+dirname+'*/*/*')
        print(f"Loading files from {dirname} ({count}/{len(eeg_dir_names)})", end='\r')
        if filetype=='eeg':
            test_file = tuple(filter(endswith_eeg, test_files))[0]
        elif filetype=='pnt':
            test_file = tuple(filter(endswith_pnt, test_files))[0]
        elif filetype=='log':
            test_file = tuple(filter(endswith_log, test_files))[0]
        elif filetype=='21e':
            test_file = tuple(filter(endswith_21e, test_files))[0]
        else:
            warnings.warn("filetype not recognized")
        
        if verbose:
            print("Found file:", test_file)
        try:
            if filetype=='eeg':
                raw = mne.io.read_raw_nihon(test_file)
                all_raws.append(test_file)
                # all_raws.append(mne.io.read_raw_nihon(test_file, preload=True))
            elif filetype=='pnt':
                # all_raws.append(mne.io.nihon.nihon._read_nihon_metadata(test_file))
                raw = mne.io.nihon.nihon._read_nihon_metadata(test_file)
                all_raws.append(test_file)
            elif filetype=='log':
                # all_raws.append(mne.io.nihon.nihon._read_nihon_annotations(test_file))
                raw = mne.io.nihon.nihon._read_nihon_annotations(test_file)
                if load:
                    all_raws.append(raw)
                else:
                    all_raws.append(test_file)
            elif filetype=='21e':
                # all_raws.append(mne.io.nihon.nihon_read_nihon_annotations(test_file))
                raw = mne.io.nihon.nihon_read_nihon_annotations(test_file)
                all_raws.append(test_file)
        except Exception as e:
            print(e)
            all_raws.append(np.nan)
    return all_raws

def save_labels(datapath, savepath):


    GCS_FILE = glob.glob(datapath+"*GCS*")[0]
    GCS_df = pd.read_csv(GCS_FILE)
    # make sure all the subj/label pairs are in the GCS_df
    subjs = GCS_df['GCS.Main.SubjectIDNum']
    labels = GCS_df['GCS.Main.CaseContrlInd']
    label_df = pd.DataFrame(list(zip(subjs, labels)), columns=['SubjectIDNum', 'CaseContrlInd'])


    label_df = label_df.drop_duplicates()
    # save label_df to csv in savepath
    check_labels(label_df, datapath)
    label_df.to_csv(os.path.join(savepath,"labels.csv"), index=False)
    print("Saved labels to ", os.path.join(savepath,"labels.csv"))
    return label_df

def check_labels(label_df, datapath):
    # GCS_FILE = glob.glob(datapath+"*GCS*")[0]
    # GCS_df = pd.read_csv(GCS_FILE)
    datafiles = os.listdir(datapath)
    # finds all the csv files
    csv_files = [file for file in datafiles if file.endswith('csv')]
    # load the csv files
    load_csvs = [pd.read_csv(datapath+f) for f in csv_files]

    # gets the list of RecordRawDataFiles
    eeg_df, eeg_idx = get_info_df("EEG", csv_files, load_csvs, datafiles)
    # make a dataframe that only includes the EEG_FITBIR.Main.SubjectIDNum and the EEG_FITBIR.Main.CaseContrlInd
    label_df = eeg_df[['EEG_FITBIR.Main.SubjectIDNum', 'EEG_FITBIR.Main.CaseContrlInd']]

    # now simply the label_df by removing duplicate rows
    # make sure all the subj/label pairs are in the GCS_df
    # subjs = label_df['SubjectIDNum']
    # labels = label_df['CaseContrlInd']
    subjs = eeg_df['EEG_FITBIR.Main.SubjectIDNum']
    labels = eeg_df['EEG_FITBIR.Main.CaseContrlInd']
    conversion = {0: "Control", 1: "Case"}
    for subj, label in zip(subjs, labels):
        assert subj in label_df['GCS.Main.SubjectIDNum'], f"{subj} not in GCS_df"
        gcs_subj_row = label_df.loc[label_df['GCS.Main.SubjectIDNum'] == subj]

        assert label == gcs_subj_row['GCS.Main.CaseContrlInd'].values[0], f"Label for {subj} is not correct: expected {conversion[label]} but got {gcs_subj_row['GCS.Main.CaseContrlInd'].values[0]}"
        
    print(f"All labels are correct")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--datapath', type=str, help='path to the downloaded FITBIR data', default='/home/ap60/Downloads/fitbir_downloads/fitbir_mission_connect_downloads/')
    parser.add_argument('--savepath', type=str, help='path to save the processed data', default='/scratch/ap60/mtbi_detection/data/raw_mne_files/')
    args = parser.parse_args()
    data_df = main(dlpath=args.datapath, savepath=args.savepath)
    # save the savepath to extract_path.txt
    with open('extract_path.txt', 'w') as f:
        f.write(args.savepath)
    with open('download_path.txt', 'w') as f:
        f.write(args.datapath)