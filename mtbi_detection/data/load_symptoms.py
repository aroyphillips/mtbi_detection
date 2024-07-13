

import pandas as pd
import numpy as np
import os
import glob

import mtbi_detection.data.load_dataset as ld
import mtbi_detection.data.data_utils as du

 
DOWNLOADPATH = open('download_path.txt', 'r').read().strip() # run after running extract_data.py or replace with location of the downloaded FITBIR data


def process_symptoms(symptoms_only=False, with_nans=False, **kwargs):
    
    mace_final = process_mace(**kwargs)
    ace_final = process_ace(**kwargs)
    gcs_final = process_gcs(**kwargs)
    goat_fitbir_final = process_goat(**kwargs)
    imread_final = process_imread(**kwargs)
    rivermead_final = process_rivermead(**kwargs)
    inj_hx_final = process_injhx(**kwargs)
    if symptoms_only:
        final_dfs = [gcs_final, mace_final, ace_final, goat_fitbir_final, rivermead_final, inj_hx_final]
        final_df_names = ['GCS', 'MACE', 'ACE', 'GOAT', 'Rivermead', 'Injury History']
    else:
        final_dfs = [gcs_final, mace_final, ace_final, goat_fitbir_final, imread_final, rivermead_final, inj_hx_final]
        final_df_names = ['GCS', 'MACE', 'ACE', 'GOAT', 'Imaging Read', 'Rivermead', 'Injury History']
    merge_all_df = gcs_final.copy(deep=True)
    for df, df_name in zip(final_dfs[1:], final_df_names[1:]):
        merge_all_df = pd.merge(merge_all_df, df, left_index=True, right_index=True)
        print(f'Merged {df_name} with merge_all_df')
    # now merge and fill nan for any rows not present in one of the dfs
    
    if with_nans:
        merge_with_nans_df = gcs_final.copy(deep=True)
        for df, df_name in zip(final_dfs[1:], final_df_names[1:]):
            merge_with_nans_df = pd.merge(merge_with_nans_df, df, left_index=True, right_index=True, how='outer')
            print(f'Merged {df_name} with merge_with_nans_df')
        return merge_with_nans_df
    else:
        merged_nonan_df = merge_all_df.fillna(0)
        return merged_nonan_df

def process_mace(downloadpath=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing MACE")
    mace_files = glob.glob(os.path.join(downloadpath, "*MACE*.csv"))
    mace_file = [file for file in mace_files if 'Appdx' not in file][0]
    mace_df = pd.read_csv(mace_file)
    # take the average over three trials
    mace_df_prop = mace_df.copy(deep=True)
    for col in mace_df.columns:
        mace_df_prop = replicate_value_through_nans(mace_df_prop,col=col)
    numeric_cols = mace_df_prop.select_dtypes(include=np.number).columns.tolist()
    mace_df_avg_trial = mace_df_prop[numeric_cols].groupby('SubjectIDNum').mean()
    processed_mace_df = mace_df_avg_trial.drop(columns=['MACE_FITBIR.Main.AgeYrs'])

    # include the injury elapsed days
    mace_appdx_file = [file for file in mace_files if 'Appdx' in file][0]
    mace_appdx = pd.read_csv(mace_appdx_file)
    mace_appdx_proc = mace_appdx.copy(deep=True)
    mace_appdx_proc.index = mace_appdx['SubjectIDNum']
    #drop all cols except 'MACE_Appdx_0000350.Visit.InjuryElapsedDaysDur'
    mace_appdx_proc = mace_appdx_proc[['MACE_Appdx_0000350.Visit.InjuryElapsedDaysDur']]
    # make sure the index is the same as the processed_mace_df
    # merge the two dataframes
    processed_mace_df = pd.merge(processed_mace_df, mace_appdx_proc, left_index=True, right_index=True)

    # don't include the injury elapsed days:
    processed_mace_df.drop(columns=['MACE_Appdx_0000350.Visit.InjuryElapsedDaysDur'], inplace=True)
    if verbose:
    
        print(f"Number of loaded MACE subjects: {len(np.unique(mace_df['SubjectIDNum']))}, number of processed MACE subjects: {len(np.unique(processed_mace_df.index))}")
    return processed_mace_df

def process_ace(downloadpath=DOWNLOADPATH, timepoint='Screening', verbose=True, **kwargs):
    if verbose:
        print("Processing ACE")
    ace_files = glob.glob(os.path.join(downloadpath, "*ACE*.csv"))
    ace_file = [file for file in ace_files if 'Appdx' not in file and 'MACE' not in file][0]

    ace_df = pd.read_csv(ace_file)
    ace_df_prop = ace_df.copy(deep=True)
    ace_df_prop = replicate_value_through_nans(ace_df_prop, col='ACE.Main.GeneralNotesTxt')
    ace_df_prop = replicate_value_through_nans(ace_df_prop, col='ACE.Main.AgeYrs')
    ace_df_prop = replicate_value_through_nans(ace_df_prop, col='ACE.Main.CaseContrlInd')
    # drop all rows that do not have Screening in 'ACE.Main.GeneralNotesTxt'
    ace_df_prop = ace_df_prop[ace_df_prop['ACE.Main.GeneralNotesTxt'] == timepoint]
    if verbose:
        print(f"number of subjects after dropping all rows that do not have {timepoint} in 'ACE.Main.GeneralNotesTxt': {len(np.unique(ace_df_prop['SubjectIDNum']))}")
        print(f"Number of subjects in the original ACE.csv file: {len(np.unique(ace_df['SubjectIDNum']))}")
    dropped_ace_subjs = [subj for subj in np.unique(ace_df['SubjectIDNum']) if subj not in np.unique(ace_df_prop['SubjectIDNum'])]
    ace_df_midproc = ace_df_prop.copy(deep=True)
    ace_df_midproc.index = ace_df_midproc['SubjectIDNum']
    
    # drop ACE.Main.GUID, ACE.Main.AgeYrs, ACE.Main.GeneralNotesTxt, ACE.Main.CaseContrlInd
    ace_df_midproc = ace_df_midproc.drop(columns=['ACE.Main.GUID', 'ACE.Main.AgeYrs', 'ACE.Main.GeneralNotesTxt', 'ACE.Main.CaseContrlInd', 'SubjectIDNum'])
    ten_cols = ['ACE.Physical Symptom.ACEPhysicalSymptomTyp','ACE.Physical Symptom.ACESymptomPresenceInd', 'ACE.Physical Symptom.ACEPhysicalSymptomScore']
    ace_ten = ace_df_midproc[ten_cols]
    ace_four = ace_df_midproc.drop(columns=ten_cols)
    # now make each unique value in ACE.Physical Symptom.ACEPhysicalSymptomTyp its own column with value from ACE.Physical Symptom.ACESymptomPresenceInd
    ace_ten_proc = ace_ten.copy(deep=True)
    ace_ten_proc = ace_ten_proc.pivot(columns='ACE.Physical Symptom.ACEPhysicalSymptomTyp', values='ACE.Physical Symptom.ACESymptomPresenceInd')
    # include the ACE.Physical Symptom.ACEPhysicalSymptomScore column by taking the mean of all the values that correspond to the same index SubjectIDNum 
    ace_ten_proc['ACE.Physical Symptom.ACEPhysicalSymptomScore'] = ace_ten.groupby(ace_ten.index)['ACE.Physical Symptom.ACEPhysicalSymptomScore'].mean() # this is just the sum so maybe not necessary
    ace_four = ace_four.dropna()
    ace_subgroups = ['Physical', 'Cognitive', 'Emotional', 'Sleep']
    ace_sub_dfs = []
    for subgroup in ace_subgroups:
        ace_sub_df = ace_df_midproc[ace_df_midproc.columns[ace_df_midproc.columns.str.contains(subgroup)]]
        ace_sub_df = ace_sub_df.dropna()
        symptom_col = ace_sub_df.columns[ace_sub_df.columns.str.contains('Symptom')]
        presence_col = ace_sub_df.columns[ace_sub_df.columns.str.contains('Presence')]
        score_col = ace_sub_df.columns[ace_sub_df.columns.str.contains('Score')]
        ace_sub_pivot = ace_sub_df.pivot(columns=symptom_col[0], values=presence_col[0]) # removing index=None
        ace_sub_pivot[score_col] = ace_sub_df.groupby(ace_sub_df.index)[score_col].mean()
        ace_sub_dfs.append(ace_sub_pivot)
    if verbose:
        print(f"Merging the ACE sub dataframes")
    # merge all the sub dfs
    for i, df in enumerate(ace_sub_dfs):
        if i == 0:
            ace_merge_df = df
        else:
            ace_merge_df = pd.merge(ace_merge_df, df, left_index=True, right_index=True)
    ace_merge_fill_dropped = ace_merge_df.copy(deep=True)
    #  include rows for the dropped subjects
    for subj in dropped_ace_subjs:
        ace_merge_fill_dropped.loc[subj] = np.nan

    # # now fill the nans with 0 since they're all not mTBI
    ace_merge_fill_dropped = ace_merge_fill_dropped.fillna(0)
    ace_merge_fill_dropped['Total_Score'] = ace_merge_fill_dropped[[col for col in ace_merge_fill_dropped.columns if 'Score' in col]].sum(axis=1)
    # ace_merge_fill_dropped
    if verbose:
        print(f"Number of subjects in the processed ACE dataframe: {len(np.unique(ace_merge_df.index))}, number of subjects in the original ACE.csv file: {len(np.unique(ace_df['SubjectIDNum']))}")
    ## notes: obtains strong performance: 0.8965 acc with logistic regression, 0.875 balanced
    return ace_merge_fill_dropped

def process_gcs(downloadpath=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing GCS")
    gcs_files = glob.glob(os.path.join(downloadpath, "*GCS*.csv"))
    gcs_file = [file for file in gcs_files if 'Appdx' not in file][0]
    gcs_df= pd.read_csv(gcs_file)
    gcs_df_prop = gcs_df.copy(deep=True)
    gcs_df_proc = gcs_df_prop.drop(columns=['GCS.Main.GUID', 'GCS.Main.AgeYrs', 'GCS.Main.CaseContrlInd', 'SubjectIDNum'])
    gcs_df_proc.index = gcs_df['SubjectIDNum']
    # convert GCS.Glasgow Coma Scale.GCSConfounderTyp to binary indicator variables
    gcs_df_proc = pd.get_dummies(gcs_df_proc, columns=['GCS.Glasgow Coma Scale.GCSConfounderTyp'])
    if verbose:
        print(f"Number of subjects in the processed GCS dataframe: {len(np.unique(gcs_df_proc.index))}, number of subjects in the original GCS.csv file: {len(np.unique(gcs_df['SubjectIDNum']))}")
    # only 0.5 balanced accuracy logreg, 0.52 rf
    return gcs_df_proc

def process_goat(downloadpath=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing GOAT")
    goat_files = glob.glob(os.path.join(downloadpath, "*GOAT*.csv"))
    goat_file = [file for file in goat_files if 'Appdx' not in file][0]
    goat_df = pd.read_csv(goat_file)
    goat_proc_df = goat_df.copy(deep=True)
    goat_proc_df.index = goat_df['SubjectIDNum']
    # drop cols
    goat_proc_df = goat_proc_df.drop(columns=['GOAT_FITBIR.Main.GUID', 'GOAT_FITBIR.Main.AgeYrs', 'SubjectIDNum'])
    # only contains mTBI subjs 
    label_dict =ld.load_label_dict()
    non_mtbi_subjs = [key for key in label_dict.keys() if label_dict[key] == 0]
    # fill in the non-mtbi subjs with 0s
    non_mtbi_goat_df = pd.DataFrame(0, index=non_mtbi_subjs, columns=goat_proc_df.columns)
    non_mtbi_goat_df['GOAT_FITBIR.Galveston Orientation & Amnesia Test (GOAT).GOATTotalScore'] = 100
    goat_stack_df = pd.concat([goat_proc_df, non_mtbi_goat_df])
    if verbose:
        print(f"Number of subjects in the processed GOAT dataframe: {len(np.unique(goat_stack_df.index))}, number of subjects in the original GOAT.csv file: {len(np.unique(goat_df['SubjectIDNum']))}")
    
    goat_appdx_file = [file for file in goat_files if 'Appdx' in file][0]
    goat_appdx = pd.read_csv(goat_appdx_file)
    goat_elapsed = goat_appdx[['SubjectIDNum', 'GOAT_FITBIR_Appdix_0000350.Visit.InjuryElapsedDaysDur']]
    goat_elapsed.index = goat_elapsed['SubjectIDNum']
    goat_elapsed = goat_elapsed.drop(columns=['SubjectIDNum'])

    # merge the two and show where they are different
    goat_appdx_pta= goat_appdx[['SubjectIDNum', 'GOAT_FITBIR_Appdix_0000350.GOAT.PTADayDur']] # post traumatic amnesia
    goat_appdx_pta.index = goat_appdx_pta['SubjectIDNum']
    goat_appdx_pta = goat_appdx_pta.drop(columns=['SubjectIDNum'])
    nonmtbi_pta = pd.DataFrame(0, index=non_mtbi_subjs, columns=goat_appdx_pta.columns)
    stack_pta = pd.concat([goat_appdx_pta, nonmtbi_pta])
    merge_goat_stack = pd.merge(goat_stack_df, stack_pta, left_index=True, right_index=True)
    if verbose:
        print(f"Number of subjects in the processed GOAT with PTADayDur dataframe: {len(np.unique(merge_goat_stack.index))}, number of subjects in the original GOAT.csv file: {len(np.unique(goat_df['SubjectIDNum']))}")
    return merge_goat_stack

def process_imread(download_path=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing Imaging Read")
    imread_files = glob.glob(os.path.join(download_path, "*ImagingRead*.csv"))
    imread_file = [file for file in imread_files if 'Appdx' not in file][0]
    imread_df = pd.read_csv(imread_file)
    prop_imread_df = imread_df.copy(deep=True)
    prop_imread_df = replicate_value_through_nans(prop_imread_df, col='ImagingRead_FITBIR.Main.GeneralNotesTxt')
    prop_imread_df = replicate_value_through_nans(prop_imread_df, col='ImagingRead_FITBIR.Main.AgeYrs')
    prop_imread_df = replicate_value_through_nans(prop_imread_df, col='ImagingRead_FITBIR.Main.CaseContrlInd')

    # get the rows where "Baseline" is in (but not equal) ImagingRead_FITBIR.Main.GeneralNotesTxt
    unique_vals = prop_imread_df['ImagingRead_FITBIR.Main.GeneralNotesTxt'].unique()
    baseline_values = [val for val in unique_vals if 'Baseline' in val and 'Baseline' != val]
    imread_baseline_df = imread_df[imread_df['ImagingRead_FITBIR.Main.GeneralNotesTxt'].isin(baseline_values)]
    
    imread_baseline_df.index = imread_baseline_df['SubjectIDNum']
    imread_baseline_df = imread_baseline_df.drop(columns=['ImagingRead_FITBIR.Main.GUID', 'ImagingRead_FITBIR.Main.AgeYrs', 'ImagingRead_FITBIR.Technical Information.ImgFileHashCode', 'ImagingRead_FITBIR.Main.GeneralNotesTxt', 'ImagingRead_FITBIR.Main.CaseContrlInd', 'SubjectIDNum'])
    
    # get the columns of interest
    imread_cols = ['ImagingRead_FITBIR.Findings.ImgBrainAssessmtReslt', 'ImagingRead_FITBIR.Findings.ImgNormalityNonTraumaInd']
    imread_select = imread_baseline_df[imread_cols]

    # drop all rows that have nans in both columns - does not change the number of subjects just doesnt count multiple findings
    imread_select = imread_select.dropna(subset=imread_cols, how='all')
    imread_select = imread_select.fillna(0)
    # make the columns binary
    imread_select[imread_cols]

    # convert Abnormal to 1 and Normal to 0
    imread_select[imread_cols[0]] = imread_select[imread_cols[0]].apply(lambda x: 1 if x == 'Abnormal' else 0)
    # convert Yes to 1 and No to 0
    imread_select[imread_cols[1]] = imread_select[imread_cols[1]].apply(lambda x: 1 if x == 'No' else 0)
    if verbose:
        print(f"Number of subjects in the processed Imaging Read dataframe: {len(np.unique(imread_select.index))}, number of subjects in the original ImagingRead_FITBIR.csv file: {len(np.unique(imread_df['SubjectIDNum']))}")
    ## notes: obtains poor performance: 0.5 balanced acc logreg, 0.52 rf
    return imread_select

def process_rivermead(downloadpath=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing Rivermead")
    rivermead_files = glob.glob(os.path.join(downloadpath, "*Rivermead*.csv"))
    rivermead_file = [file for file in rivermead_files if 'Appdx' not in file][0]
    rivermead_df = pd.read_csv(rivermead_file)
    rivermead_baseline = rivermead_df[rivermead_df['Rivermead.Main.GeneralNotesTxt'] == 'Baseline']

    rivermead_baseline.index = rivermead_baseline['SubjectIDNum']
    rivermead_baseline = rivermead_baseline.drop(columns=['Rivermead.Main.GUID', 'Rivermead.Main.AgeYrs', 'Rivermead.Main.GeneralNotesTxt', 'Rivermead.Main.CaseContrlInd', 'SubjectIDNum'])

    rivermead_baseline = rivermead_baseline.fillna(0)
    if verbose:
        print(f"Number of subjects in the processed Rivermead dataframe: {len(np.unique(rivermead_baseline.index))}, number of subjects in the original Rivermead.csv file: {len(np.unique(rivermead_df['SubjectIDNum']))}")
    
    return rivermead_baseline

def process_injhx(downloadpath=DOWNLOADPATH, verbose=True, **kwargs):
    if verbose:
        print("Processing Injury History")
    inj_hx_files = glob.glob(os.path.join(downloadpath, "*InjHx*.csv"))
    inj_hx_file = [file for file in inj_hx_files if 'Appdx' not in file][0]
    inj_hx_df = pd.read_csv(inj_hx_file)
    inj_hx_proc = inj_hx_df.copy(deep=True)
    inj_hx_proc.index = inj_hx_df['SubjectIDNum']
    # drop cols
    inj_hx_proc = inj_hx_proc.drop(columns=['InjHx_FITBIR.Main.GUID', 'InjHx_FITBIR.Main.AgeYrs', 'InjHx_FITBIR.Main.CaseContrlInd', 'SubjectIDNum'])

    drop_column_names = ['InjHx_FITBIR.Injury General Info.InjPlcOccncTyp', 'InjHx_FITBIR.Injury General Info.InjPlcOccncOTH', 'InjHx_FITBIR.AIS.AISBodyRegionChapterCat',
                    'InjHx_FITBIR.AIS.AISInjurySeverityScore', 'InjHx_FITBIR.Hospital Discharge.InjElapsedTime', 'InjHx_FITBIR.Neurological Assessment Symptoms and Signs.InjElapsedTime']

    # drop columns with "Type Place Cause" in the name
    inj_hx_proc = inj_hx_proc.drop(columns=drop_column_names)
    type_place_cause_cols = [col for col in inj_hx_proc.columns if 'Type Place Cause' in col and not 'Airbag' in col]
    inj_hx_proc = inj_hx_proc.drop(columns=type_place_cause_cols)
    airbag_col = [col for col in inj_hx_proc.columns if 'Airbag' in col]
    # drop any rows with nan in InjHx_FITBIR.Injury General Info.InjElapsedTime
    inj_hx_proc = inj_hx_proc.dropna(subset=['InjHx_FITBIR.Injury General Info.InjElapsedTime'])
    inj_hx_no_airbag = inj_hx_proc.drop(columns=airbag_col)
    # now drop InjHx_FITBIR.Injury General Info.InjElapsedTime
    inj_hx_no_airbag = inj_hx_no_airbag.drop(columns=['InjHx_FITBIR.Injury General Info.InjElapsedTime'])

    inj_hx_no_airbag_filled = inj_hx_no_airbag.fillna(0)
    # now convert no to 0 and yes to 1
    inj_hx_no_airbag_filled = inj_hx_no_airbag_filled.replace('No', 0)
    inj_hx_no_airbag_filled = inj_hx_no_airbag_filled.replace('Yes', 1)
    inj_hx_filled_airbag = inj_hx_no_airbag_filled.copy(deep=True)
    # now add the airbag column back in
    inj_hx_filled_airbag[airbag_col] = inj_hx_proc[airbag_col]
    inj_hx_filled_airbag = inj_hx_filled_airbag.fillna(0)
    inj_hx_filled_airbag = inj_hx_filled_airbag.replace('No', 0)
    inj_hx_filled_airbag = inj_hx_filled_airbag.replace('Yes', 1)
    if verbose:
        print(f"Number of subjects in the processed Injury History dataframe: {len(np.unique(inj_hx_no_airbag_filled.index))}, number of subjects in the original InjHx_FITBIR.csv file: {len(np.unique(inj_hx_df['SubjectIDNum']))}")
    return inj_hx_no_airbag_filled

def load_raw_symptoms_dfs(which_symptoms='all', csv_filepath='/shared/roy/mTBI/raw_data/csv_files/'):
    if which_symptoms == 'gcs':
        df = pd.read_csv(os.path.join(csv_filepath, 'GCS.csv'))
    elif which_symptoms == 'rivermead':
        df = pd.read_csv(os.path.join(csv_filepath, 'Rivermead.csv'))
    elif which_symptoms == 'mace':
        df = pd.read_csv(os.path.join(csv_filepath, 'MACE_FITBIR.csv'))
    elif which_symptoms == 'ace':
        df = pd.read_csv(os.path.join(csv_filepath, 'ACE.csv'))
    elif which_symptoms == 'goat':
        df = pd.read_csv(os.path.join(csv_filepath, 'GOAT_FITBIR.csv'))
    elif which_symptoms == 'imread':
        df = pd.read_csv(os.path.join(csv_filepath, 'ImagingRead_FITBIR.csv'))
    elif which_symptoms == 'injhx':
        df = pd.read_csv(os.path.join(csv_filepath, 'InjHx_FITBIR.csv'))
    elif which_symptoms == 'all':
        gcs_df = pd.read_csv(os.path.join(csv_filepath, 'GCS.csv'))
        rivermead_df = pd.read_csv(os.path.join(csv_filepath, 'Rivermead.csv'))
        mace_df = pd.read_csv(os.path.join(csv_filepath, 'MACE_FITBIR.csv'))
        ace_df = pd.read_csv(os.path.join(csv_filepath, 'ACE.csv'))
        goat_df = pd.read_csv(os.path.join(csv_filepath, 'GOAT_FITBIR.csv'))
        imread_df = pd.read_csv(os.path.join(csv_filepath, 'ImagingRead_FITBIR.csv'))
        injhx_df = pd.read_csv(os.path.join(csv_filepath, 'InjHx_FITBIR.csv'))
        # merge on SubjectIDNum outer
        df = pd.merge(gcs_df, rivermead_df, on='SubjectIDNum', how='outer')
        df = pd.merge(df, mace_df, on='SubjectIDNum', how='outer')
        df = pd.merge(df, ace_df, on='SubjectIDNum', how='outer')
        df = pd.merge(df, goat_df, on='SubjectIDNum', how='outer')
        df = pd.merge(df, imread_df, on='SubjectIDNum', how='outer')
        df = pd.merge(df, injhx_df, on='SubjectIDNum', how='outer')
    else:
        raise ValueError(f"which_symptoms must be one of ['gcs', 'rivermead', 'mace', 'ace', 'goat', 'imread', 'injhx', 'all']")
    df.index = df['SubjectIDNum']
    return df


def replicate_value_through_nans(df, col='SubjectIDNum.21'):
    """
    df: dataframe
    col: column to replicate values through nans
    """
    new_df= df.copy(deep=True)
    old_val = np.nan
    for idx, row in df.iterrows():
        if pd.isnull(row[col]):
            new_df.loc[idx, col] = old_val
        else:
            old_val = row[col]
            new_df.loc[idx, col] = old_val
    return new_df

def get_simple_symptoms(verbose=True):
    """
    Loads a simple set of 8 symptoms to implement the methods in McNerney et al. 2019: https://pubmed.ncbi.nlm.nih.gov/31001724/
    Args:
    verbose: whether to print the number of subjects in the final dataframe
    Returns:
    simple_symptom_df: a dataframe with the 8 symptoms: LOC, Headache, Nausea, LightSens, NoiseSens, LongToThink, Memory, Average_Score
    in reference to:
    we used seven yes-no symptom questions. These questions were in reference to 
    loss of consciousness, headache, nausea or vomiting, sensitivity to light, 
    sensitivity to sound, confusion, and memory disfunction. 
    These questions are similar to part of the ImPACT and SCAT5 tests. 
    The subjects were also asked to rate the symptom severity 
    on a scale of 0 to 6, 0 signifying no symptoms. 
    """
    symptoms = process_symptoms(verbose=verbose)
    symp_cols = ['InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal', 'Rivermead.Questionnaire.RPQHeadachesScale', 'Rivermead.Questionnaire.RPQNauseaScale',  'Rivermead.Questionnaire.RPQLightSensScale', 'Rivermead.Questionnaire.RPQNoiseSensScale', 'Rivermead.Questionnaire.RPQLongToThinkScale', 'MACE_FITBIR.Scores.MACEImmdtMemScore']

    selected_symp = symptoms[symp_cols]
    # make the memory column = 15-orignal
    selected_symp.loc[:, 'MACE_FITBIR.Scores.MACEImmdtMemScore'] = 15 - selected_symp['MACE_FITBIR.Scores.MACEImmdtMemScore']

    binary_selected_symp = selected_symp.copy()
    # now turn all non 0 values to 1
    binary_selected_symp[binary_selected_symp != 0] = 1

    kmeans_mem_df = du.sorted_cluster_labeling(selected_symp, 'MACE_FITBIR.Scores.MACEImmdtMemScore', n_clusters=7) # some slight data leakage here
    kmeans_loc_df = du.sorted_cluster_labeling(selected_symp, 'InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal', n_clusters=7) # some slight data leakage here although it is unsupervised

    selected_symp_scores = selected_symp.copy()
    selected_symp_scores['MACE_FITBIR.Scores.MACEImmdtMemScore'] = kmeans_mem_df['Cluster']
    selected_symp_scores['InjHx_FITBIR.LOC AOC and PTA.LOCDurationVal'] = kmeans_loc_df['Cluster']

    new_col_names = ['LOC', 'Headache', 'Nausea', 'LightSens', 'NoiseSens', 'LongToThink', 'Memory']
    selected_symp_scores.columns = new_col_names
    binary_selected_symp.columns = new_col_names

    avg_selected_symp = selected_symp_scores.mean(axis=1)
    mcnerney_symptom_df = binary_selected_symp.copy()
    mcnerney_symptom_df['Average_Score'] = avg_selected_symp
    return mcnerney_symptom_df

