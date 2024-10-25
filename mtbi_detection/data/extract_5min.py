# five minute segments loaded to replicate Method B

import os
import pandas as pd
import mne
import glob
import time
import dotenv
import argparse

import mtbi_detection.data.rereference_data as rd
import mtbi_detection.data.load_dataset as ld

dotenv.load_dotenv()

# globals
CHANNELS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','T1','T2']
REF_CHANNELS = ['A1', 'A2']
# REF_CHANNELS = ['Cz']
DURATION = 300 # 5 minutes
SAMPLING_RATE = 500 # Hz


EXTRACTED_PATH = os.getenv('EXTRACTED_PATH')
FIVEMIN_PATH = os.path.join(os.path.dirname(EXTRACTED_PATH[:-1]), 'five_min_segments')

def main(n_subjs=152, tables_folder='data/tables/', fivemin_savepath=FIVEMIN_PATH, extracted_path=EXTRACTED_PATH):
    """
    Given extracted data, extract 5 minute segments, rereference for each subject and save them in a new directory
    Inputs:
        - n_subjs: number of subjects to load
        - tables_folder: folder where the annotations table is stored
        - fivemin_savepath: path to save the 5 minute segments
        - extracted_path: path to the extracted data
    Outputs:
        - 5 minute segments saved in fivemin_savepath    
    """

    annotations_df = ld.load_annotations(num_rows=n_subjs, tables_folder=tables_folder) 

    # only get up to n_subjs rows
    annotations_df = annotations_df.iloc[:n_subjs]

    # get the list of subject ids
    subj_ids = annotations_df['Study ID'].values
    raw_list = []

    loading_start = time.time()
    for sdx, subj_id in enumerate(subj_ids):
        print(f"Loading subject {subj_id} ({sdx+1}/{len(subj_ids)})")
        subj_start = time.time()
        # get the subject's annotations
        subj_annotations = annotations_df[annotations_df["Study ID"] == subj_id]
        # get the start and end times
        start_time = subj_annotations["5MIN SEG START"].values[0]
        end_time = subj_annotations["5MIN SEG END"].values[0]
        if (end_time-start_time) != DURATION:
            print("Start and end times are not 5 minutes apart for subject {}".format(subj_id))
            print("Skipping")
            continue

        # make segments
        subjpath = os.path.join(extracted_path, str(int(subj_id)))
        # get the raw files
        raw_file = glob.glob(os.path.join(subjpath + '/baseline/*.fif'))

        # make sure there is only one raw file
        assert len(raw_file) <= 1, "More than one raw file found for subject {}".format(subj_id)

        if len(raw_file) < 1:
            print("No raw files found for subject {}".format(subj_id))
            continue



        # read the raw file, make sure it is referenced how we want, and then keep only the channels we want
        raw = mne.io.read_raw_fif(raw_file[0], preload=True, verbose=False)

        # reref_raw = rd.rereference_raw(raw, CHANNELS + REF_CHANNELS, reference_channels=REF_CHANNELS, method=reference_method) # rereference to Cz
        # ordered_raw = reref_raw.copy().pick_channels(CHANNELS, ordered=True) # throwing away reference channels but maybe we should keep them?

        # crop the raw file
        cropped_raw = raw.copy().crop(tmin=start_time, tmax=end_time)

        # confirm sample and resample if possibly
        if cropped_raw.info['sfreq'] != SAMPLING_RATE:
            print("Resampling raw file")
            cropped_raw = cropped_raw.resample(SAMPLING_RATE, npad='auto')
        assert cropped_raw.get_data().shape[1] == DURATION * SAMPLING_RATE + 1, "Raw data is not {} seconds long".format(DURATION)

        # now we save this however we want
        save_dir = os.path.join(fivemin_savepath, str(int(subj_id)))
        # save the raw file and make the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savefilename = f'baseline_{DURATION}s_subj{int(subj_id)}_raw.fif'
        savepath = os.path.join(save_dir, savefilename)
        cropped_raw.save(savepath, overwrite=True, verbose=False)

        raw_list.append(raw)

        print(f"\tSubject {subj_id} took {time.time() - subj_start} seconds to load")

    loading_end = time.time()
    common_channels = rd.find_common_channels_from_list(raw_list)
    dotenv.set_key(dotenv.find_dotenv(), 'FIVEMIN_PATH', fivemin_savepath)
    print(f"Common channels are the same set as the channels we want? {set(common_channels) == set(CHANNELS)}")
    print(f"Common channels: {common_channels}")
    print(f"Loading took {loading_end - loading_start} seconds")
    # baseline_files
    print("DONE")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 5 minute segments")
    parser.add_argument("--n_subjs", type=int, default=152, help="Number of subjects to load")
    parser.add_argument("--tables_folder", type=str, default='data/tables/', help="Folder where the annotations table is stored")
    parser.add_argument("--fivemin_savepath", type=str, default=FIVEMIN_PATH, help="Path to save the 5 minute segments")
    parser.add_argument("--extracted_path", type=str, default=EXTRACTED_PATH, help="Path to the extracted data")
    args = parser.parse_args()
    main(n_subjs=args.n_subjs, tables_folder=args.tables_folder, fivemin_savepath=args.fivemin_savepath, extracted_path=args.extracted_path)