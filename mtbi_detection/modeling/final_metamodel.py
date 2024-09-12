import numpy as np
import pandas as pd
import os
import time
import sklearn
import xgboost
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sklearn

import mtbi_detection.features.feature_utils as fu
import mtbi_detection.modeling.model_utils as mu
import mtbi_detection.data.data_utils as du
import mtbi_detection.data.load_dataset as ld


CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'features')
RESULTS_SAVEPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'results')
BASELEARNERS_SAVEPATH = os.path.join(RESULTS_SAVEPATH, 'baselearners')

### LOADING AND SAVING
def load_model_results(which_featuresets=['eeg', 'ecg'], savepath=BASELEARNERS_SAVEPATH, late_fuse=True, base_model_names=['XGBClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'GaussianNB', 'AdaBoost', 'LogisticRegression']):
    """
    Given a list of featuresets, load the model results for each featureset
    Inputs:
        which_featuresets: list of featuresets to load
        savepath: path to the results folder
        late_fuse: whether to load featureset basemodels separately (late fusion) or together (early fusion)

    Outputs:
        basemodel_results: a dictionary containing keys for each featureset and a dictionary of basemodels with their paths
            {'eeg': {'basemodel_name1': basemodelpath1, ...}, 'ecg': {'basemodel_name1': basemodelpath1, ...}}
    """
    basemodel_results = {}
    if late_fuse:
        # load the basemodels separately
        for fset in which_featuresets:
            basemodel_results[fset] = {}
            for base_model in base_model_names:
                param_mapping = {'which_features': [fset], 'model_name': base_model}
                basemodel_results[fset][base_model] = du.load_path_from_params(savepath, param_mapping)
    else:
        # load the basemodels together
        basemodel_results['-'.join(which_featuresets)] = {}
        for base_model in base_model_names:
            param_mapping = {'which_features': which_featuresets, 'model_name': base_model}
            basemodel_results['-'.join(which_featuresets)][base_model] = du.load_path_from_params(savepath, param_mapping)

    return basemodel_results

def check_if_split_results_exist(fset_savepath: str, n_splits: int):
    """
    Given a featureset savepath, check if the split results exist (saved from save_split_results)
    Inputs:
        fset_savepath: path to the featureset savepath
        n_splits: number of splits
    Outputs:
        (bool) whether the split results exist
    """
    assert type(n_splits) == int, "n_splits must be an integer"
    assert type(fset_savepath) == str, "fset_savepath must be a string"
    initial_results_exists = os.path.exists(os.path.join(fset_savepath, 'initial_split', 'saved_results_paths.json'))
    split_results_exists = os.path.exists(os.path.join(fset_savepath, f'shuffle_split_{n_splits}', 'saved_results_paths.json'))
    return initial_results_exists and split_results_exists

def save_split_results(results_dict, savepath, results_table=None, fitted_metamodels=None, optional_savables=None):
    """
    Given a results dictionary, save the results to a savepath
    Inputs:
        results_dict: dictionary with the results
        results_table: dataframe with the results
        savepath: path to save the results with json, joblib, and csv files
    Outputs:
        saved_results: dictionary with the paths to the saved results
            {'results_dict': /path/to/split_results.json, 'results_table': /path/to/results_table.csv, 'fitted_metamodels': {metamodel_name: metamodelpath}}
    """
    os.makedirs(savepath, exist_ok=True)
    saved_results = {}

    # save the results dictionary
    saveable_results_dict = du.make_dict_saveable(results_dict)
    split_results_savepath = os.path.join(savepath, 'split_results.json')
    json.dump(saveable_results_dict, open(split_results_savepath, 'w'))
    saved_results['results_dict'] = split_results_savepath

    # save the fitted metamodels
    if fitted_metamodels is not None:
        for key, val in fitted_metamodels.items():
            metamodelpath = os.path.join(savepath, f"{key}.joblib")
            joblib.dump(val, open(metamodelpath, 'wb'))
            saved_results[key] = metamodelpath

    # save the dataframe
    if results_table is not None:
        results_table_path = os.path.join(savepath, 'results_table.csv')
        results_table.to_csv(results_table_path)
        saved_results['results_table'] = results_table_path

    # save the optional savables
    if optional_savables is not None:
        for key, val in optional_savables.items():
            if type(val) == pd.DataFrame:
                val.to_csv(os.path.join(savepath, f"{key}.csv"))
                saved_results[key] = os.path.join(savepath, f"{key}.csv")
            elif type(val) == dict:
                json.dump(val, open(os.path.join(savepath, f"{key}.json"), 'w'))
                saved_results[key] = os.path.join(savepath, f"{key}.json")
            else:
                raise ValueError(f"Type {type(val)} not implemented")

    # save the saved results to a json file
    saved_results_path = os.path.join(savepath, 'saved_results_paths.json')
    json.dump(saved_results, open(saved_results_path, 'w'))

    return saved_results

def load_split_results(savepath):
    """
    Given a savepath, load the split results (saved from save_split_results)
    Inputs:
        savepath: path to the saved results
    Outputs:
        loaded_results: dictionary with the loaded results
    """
    saved_results_paths = json.load(open(os.path.join(savepath, 'saved_results_paths.json')))
    loaded_results = {}
    for key, val in saved_results_paths.items():
        if key.endswith('.json'):
            loaded_results[key] = json.load(open(val))
        elif key.endswith('.csv'):
            loaded_results[key] = pd.read_csv(val, index_col=0)
        elif key.endswith('.joblib'):
            loaded_results[key] = joblib.load(open(val, 'rb'))
        else:
            loaded_results[key] = val
    return loaded_results

### Model selection
def select_base_models(dev_pred_df=None, ival_pred_df=None, holdout_pred_df=None, internal_folder='data/internal/', verbose=False):
    """
    Select a base set of binary classifier models using McNemar midp val. Choose all models that are above 0.05 likelihood simlar to the best model on the internal validation set
    Inputs:
        dev_pred_df: dataframe with the predictions on the development set (rows are subjects, columns are model predictions e.g. model1_0, model1_1, model2_0, model2_1)
        ival_pred_df: dataframe with the predictions on the internal validation set (THIS IS THE SET USED TO SELECT THE MODELS)
        holdout_pred_df: dataframe with the predictions on the holdout set (optional)
    Returns:
        dev_select_pred_df: dataframe with the selected predictions on the development set
        ival_select_pred_df: dataframe with the selected predictions on the internal validation set
        holdout_select_pred_df: dataframe with the selected predictions on the holdout set
        model_midps: dictionary with the likelihood that each model is the same as the best model {model0_0: 0.05, model0_1: 0.1, ...}
    """
    assert ival_pred_df is not None, "Must provide internal validation set predictions"

    # create dummy predictions if dev and holdout are not provided
    if dev_pred_df is None:
        dev_subjs = [g for g in ld.load_splits(internal_folder)['train'] if int(g) not in ival_pred_df.index and g not in holdout_pred_df.index]
        if len(dev_subjs) == 0:
            dev_subjs = [max(ival_pred_df.index) + max(holdout_pred_df.index) + 1]
        dev_pred_df = pd.DataFrame(np.random.random((len(dev_subjs), len(ival_pred_df.columns))), columns=ival_pred_df.columns, index=dev_subjs)
    if holdout_pred_df is None:
        holdout_subjs = [g for g in ld.load_splits(internal_folder)['holdout'] if int(g) not in ival_pred_df.index and g not in dev_pred_df.index]
        if len(holdout_subjs) == 0:
            holdout_subjs = [max(ival_pred_df.index) + max(dev_pred_df.index) + 1]
        holdout_pred_df = pd.DataFrame(np.random.random((len(holdout_subjs), len(ival_pred_df.columns))), columns=ival_pred_df.columns, index=holdout_subjs)
        
    # binarize the predictions
    dev_pos_pred_df = dev_pred_df[dev_pred_df.columns[dev_pred_df.columns.str.endswith('_1')]] >= 0.5
    ival_pos_pred_df = ival_pred_df[ival_pred_df.columns[ival_pred_df.columns.str.endswith('_1')]] >= 0.5
    dev_pos_pred_df = dev_pos_pred_df.astype(int)
    ival_pos_pred_df = ival_pos_pred_df.astype(int)


    # choose the best model based on the internal validation set
    ival_y_test = fu.get_y_from_df(ival_pred_df)
    all_model_mccs_ival = [sklearn.metrics.matthews_corrcoef(ival_y_test, ival_pos_pred_df[col]) for col in ival_pos_pred_df.columns]

    assert ival_pos_pred_df.shape[1] == dev_pos_pred_df.shape[1]
    best_model_idx = np.argmax(all_model_mccs_ival)

    # calculate the mcnemar midp likelihood that each other model could be the same as the best model
    model_midps = validation_mcnemar_midp(ival_pos_pred_df, best_model_idx, verbose=verbose)

    # select all models that are above 0.05 likelihood
    selected_cols = [col[:-2] for col, midp in model_midps.items() if midp > 0.05]
    dev_select_pred_df = dev_pred_df[[col for col in dev_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    ival_select_pred_df = ival_pred_df[[col for col in ival_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    holdout_select_pred_df = holdout_pred_df[[col for col in holdout_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    
    # assert no overlap
    assert len(set(dev_select_pred_df.index).intersection(set(ival_select_pred_df.index))) == 0, f"{set(dev_select_pred_df.index).intersection(set(ival_select_pred_df.index))} overlap between dev and ival"
    assert len(set(dev_select_pred_df.index).intersection(set(holdout_select_pred_df.index))) == 0, f"{set(dev_select_pred_df.index).intersection(set(holdout_select_pred_df.index))} overlap between dev and holdout"
    assert len(set(ival_select_pred_df.index).intersection(set(holdout_select_pred_df.index))) == 0, f"{set(ival_select_pred_df.index).intersection(set(holdout_select_pred_df.index))} overlap between ival and holdout"
    assert all([all(ival_select_pred_df.columns == dev_select_pred_df.columns), all(ival_select_pred_df.columns == holdout_select_pred_df.columns)]), "Columns do not match"
    
    return dev_select_pred_df, ival_select_pred_df, holdout_select_pred_df, model_midps

def validation_mcnemar_midp(pred_df, best_model_idx, verbose=False):
    """
    Given a dataframe of predictions, calculate the likelihood that each model is the same as the best model
    Inputs:
        pred_df: dataframe with the predictions (rows are subjects, columns are model predictions e.g. model1_0, model1_1, model2_0, model2_1)
        best_model_idx: index of the best model in the dataframe 
    Outputs:
        midp_vals: dictionary with the likelihood that each model is the same as the best model {model0_0: 0.05, model0_1: 0.1, ...}
    """
    y_test = fu.get_y_from_df(pred_df)
    best_preds = pred_df.iloc[:, best_model_idx]
    tn1, fp1, fn1, tp1 = sklearn.metrics.confusion_matrix(y_test, best_preds).ravel()
    midp_vals = {}
    for col in pred_df.columns:
        if col == pred_df.columns[best_model_idx]:
            midp_vals[col] = 1
            continue

        preds = pred_df[col]
        tn2, fp2, fn2, tp2 = sklearn.metrics.confusion_matrix(y_test, preds).ravel()
        if verbose:
            print(f"Model 1 sensitivity: {tp1/(tp1 + fn1)}, specificity: {tn1/(tn1 + fp1)}")
            print(f"Model 2 sensitivity: {tp2/(tp2 + fn2)}, specificity: {tn2/(tn2 + fp2)}")
        n12 = np.sum(np.logical_and(best_preds == y_test, preds != y_test))
        n21 = np.sum(np.logical_and(preds == y_test, best_preds != y_test))
        # calculate the mcnemar test
        if verbose:
            print(f"Model 1 and Model 2 discordant pairs: n12: {n12}, n21: {n21}, overlap: {np.sum(np.logical_and(best_preds == y_test, preds == y_test))}")
        midp = mu.mcnemar_midp(n12, n21)
        if verbose:
            print(f"Likelihood that the expected predictions of the two models are the same: {midp}")
        midp_vals[col] = midp
    return midp_vals

### Data splitting and model development
def return_basemodel_preds(basemodel_results, n_basetrain_cv=None, verbose=False):
    """
    Given a dictionary of basemodel results, load the predictions for each basemodel
    Input:
        - basemodel_results: a dictionary containing keys for each featureset and a 
            {'eeg': {'basemodel_name1': basemodelpath1, ...}, 'ecg': {'basemodel_name1': basemodelpath1, ...}}
        - n_basetrain_cv: number of splits to do for the train cv
    Output:
        - basemodel_preds: a dictionary containing keys for each featureset, and a dictionary of basemodel predictions
            - dev_preds: a dataframe with the predictions on the development set
                - columns: [basemodel_name0_0, basemodel_name0_1, basemodel_name1_0, ...]
            - unseen_preds: a dataframe with the predictions on the unseen set (internal validation + holdout)
                - columns: [basemodel_name0_0, basemodel_name0_1, basemodel_name1_0, ...]
            - loaded_model_data: dictionary of loaded model data {basemodel_name: trained_model}
            - feature_data: dictionary of training data {featureset: {'Xtr': Xtr, 'Xts': Xts, 'X_hld': X_hld}}
    """

    all_dev_preds = {}
    all_unseen_preds = {}
    loaded_model_data = {}
    feature_data = {}
    which_featuresets = list(basemodel_results.keys())
    model_names = list(set().union(*basemodel_results.values())) # https://stackoverflow.com/questions/45652155/extract-all-keys-in-the-second-level-of-nested-dictionary/45652179#45652179
    
    model_counter = 1
    for fdx, fset in enumerate(which_featuresets):
        all_dev_preds[fset] = {}
        all_unseen_preds[fset] = {}
        loaded_model_data[fset] = {}
        feature_data[fset] = {}
        for mdx, model_name in enumerate(model_names):
            print(f"Loading model {model_counter}/{len(which_featuresets)*len(model_names)}")
            st = time.time()
            modelpath = basemodel_results[fset][model_name]
            model, Xtr, Xts, X_hld = mu.load_model_data(modelpath, load_model=True, load_holdout=True)
            Xunseen = pd.concat([Xts, X_hld], axis=0)
            assert len(set(Xtr.index).intersection(set(Xunseen.index))) == 0, f"Overlap between train and unseen: {set(Xtr.index).intersection(set(Xunseen.index))}"
            _, dev_cv_preds = return_cv_train_test_preds(Xtr, model, n_cv=n_basetrain_cv, verbose=verbose)
            dev_cv_preds.columns = [f"{model_name}_{idx}" for idx in range(2)]
            yunseen = model.predict_proba(Xunseen)
            yunseen = pd.DataFrame(yunseen, columns=[f"{model_name}_{idx}" for idx in range(2)], index=Xunseen.index)

            all_dev_preds[fset][model_name] = dev_cv_preds
            all_unseen_preds[fset][model_name] = yunseen
            loaded_model_data[fset][model_name]=model
            if mdx == 0:
                feature_data[fset] = {'Xtr': Xtr, 'Xts': Xts, 'X_hld': X_hld}
            else:
                # make sure the training data is the same
                assert np.array_equal(Xtr.values, feature_data[fset]['Xtr'].values, equal_nan=True), "Training data does not match"
                assert np.array_equal(Xts.values, feature_data[fset]['Xts'].values, equal_nan=True), "Testing data does not match"
                assert np.array_equal(X_hld.values, feature_data[fset]['X_hld'].values, equal_nan=True), "Holdout data does not match"
                assert all([all(Xtr.index == feature_data[fset]['Xtr'].index), all(Xts.index == feature_data[fset]['Xts'].index), all(X_hld.index == feature_data[fset]['X_hld'].index)]), "Indices do not match"
            model_counter += 1
    # assert all([all(all_dev_preds[which_featuresets[0]][model_names[0]].index == all_dev_preds[fs][mn].index) for fs in which_featuresets for mn in model_names])
    # assert all([all(all_unseen_preds[which_featuresets[0]][model_names[0]].index == all_unseen_preds[fs][mn].index) for fs in which_featuresets for mn in model_names])
    
    # dev_pred_groups = all_dev_preds[which_featuresets[0]][model_names[0]].index
    dev_pred_groups = set(all_dev_preds[which_featuresets[0]][model_names[0]].index)
    dev_pred_groups = list(dev_pred_groups.intersection(*[set(all_dev_preds[fs][mn].index) for fs in which_featuresets for mn in model_names]))
    unseen_pred_groups = set(all_unseen_preds[which_featuresets[0]][model_names[0]].index)
    unseen_pred_groups = list(unseen_pred_groups.intersection(*[set(all_unseen_preds[fs][mn].index) for fs in which_featuresets for mn in model_names]))

    # unseen_pred_groups = all_unseen_preds[which_featuresets[0]][model_names[0]].index

    dev_preds = pd.concat([all_dev_preds[fs][mn] for fs in which_featuresets for mn in model_names], axis=1)
    unseen_preds = pd.concat([all_unseen_preds[fs][mn] for fs in which_featuresets for mn in model_names], axis=1)

    # only keep the groups that are in all predictions
    dev_preds = dev_preds.loc[dev_pred_groups]
    unseen_preds = unseen_preds.loc[unseen_pred_groups]

    # dev_pred_df = pd.DataFrame(dev_preds, columns=[f"{basemodel_results[fs][mn].split('/')[-1]}_{idx}" for fs in which_featuresets for mn in model_names for idx in range(2)], index=dev_pred_groups)
    # unseen_pred_df = pd.DataFrame(unseen_preds, columns=[f"{basemodel_results[fs][mn].split('/')[-1]}_{idx}" for fs in which_featuresets for mn in model_names for idx in range(2)], index=unseen_pred_groups)

    assert len(dev_preds.index) == len(set(dev_preds.index)), "Duplicate indices in dev test"
    return dev_preds, unseen_preds, loaded_model_data, feature_data

def split_unseen_preds(unseen_preds, internal_folder='data/internal/', default=True, random_seed=0):
    """
    Given a dataframe unseen_preds with subjects as the index, split the predictions into internal validation and holdout
    using the default splits in internal_folder or a random split
    Inputs:
        - unseen_preds: dataframe with the predictions
        - internal_folder: folder with the internal splits
        - default: whether to use the default splits or a random split
    Outputs:
        - ival_preds: dataframe with the internal validation predictions
        - holdout_preds: dataframe with the holdout predictions
    """
    splits = ld.load_splits(internal_folder=internal_folder)
    ival_subjs = splits['ival']
    holdout_subjs = splits['holdout']
    assert all([int(subj) in ival_subjs or subj in holdout_subjs for subj in unseen_preds.index]), "Some subjects not in splits"

    if default:
        ival_preds = unseen_preds.loc[[s for s in unseen_preds.index if int(s) in ival_subjs]]
        holdout_preds = unseen_preds.loc[[s for s in unseen_preds.index if int(s) in holdout_subjs]]
    else:
        rng = np.random.RandomState(random_seed)
        rand_ival_subjs = rng.choice(unseen_preds.index, int(len(unseen_preds.index)*0.8), replace=False)
        rand_holdout_subjs = np.array([s for s in unseen_preds.index if s not in rand_ival_subjs])
        ival_preds = unseen_preds.loc[[s for s in unseen_preds.index if s in rand_ival_subjs]]
        holdout_preds = unseen_preds.loc[[s for s in unseen_preds.index if s in rand_holdout_subjs]]
    return ival_preds, holdout_preds

def return_split_results(default_savepaths, n_splits=10, n_metatrain_cv=5, metalearners=['rf', 'lr', 'xgb'], n_jobs=1):
    """
    Given a dictionary that contains the paths to:
        - dev_basepreds: path to the development set predictions 
        - unseen_basepreds: path to the unseen set predictions
    
    Return n_splits perturbations of the development and unseen set predictions

    Inputs:
        - default_savepaths: dictionary with the paths to the base predictions
        - n_splits: number of splits to do
        - n_train_cv: number of splits to do for the train cv
        - n_jobs: number of jobs to run in parallel for each split's metamodel grid search
    
    Returns:
        - split_results: dictionary with the split results
            {'split_k':
                {'select_cols': selected_columns,
                'ival_groups': ival_groups,
                'holdout_groups': holdout_groups,
                'holdout_labels': holdout_labels,
                'holdout_results': output of test_models_on_unseen_data,
                }
            'perturbation_scores': dictionary with the perturbation scores for each model
                {'rf': {'matthews_corrcoef': 0.5, 'roc_auc': 0.6, ...}, 'lr': {...}, 'xgb': {...}}
    - split_results_df: dataframe with the split results
    - perturbation_score_df: dataframe with the perturbation scores
    """
    assert 'dev_basepreds' in default_savepaths.keys(), "Must provide path to the development set predictions"
    assert 'unseen_basepreds' in default_savepaths.keys(), "Must provide path to the unseen set predictions"
    dev_basepreds = pd.read_csv(default_savepaths['dev_basepreds'], index_col=0)
    unseen_basepreds = pd.read_csv(default_savepaths['unseen_basepreds'], index_col=0)
    unseen_labels = fu.get_y_from_df(unseen_basepreds)
    unseen_groups = unseen_basepreds.index.values
    all_split_results = {}
    holdout_cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)
    for split_idx, (train_idx, test_idx) in enumerate(holdout_cv.split(unseen_basepreds, unseen_labels)):
        print(f"Running split {split_idx+1}/{n_splits}")
        ival_groups = unseen_groups[train_idx]
        holdout_groups = unseen_groups[test_idx]

        ival_preds = unseen_basepreds.loc[ival_groups]
        holdout_preds = unseen_basepreds.loc[holdout_groups]
        
        select_split_dev_preds, select_split_ival_preds, select_split_holdout_preds, _ = select_base_models(dev_pred_df=dev_basepreds, ival_pred_df=ival_preds, holdout_pred_df=holdout_preds)
        select_split_dev_ival_preds = pd.concat([select_split_dev_preds, select_split_ival_preds], axis=0)

        ival_labels, holdout_labels = fu.get_y_from_df(ival_preds), fu.get_y_from_df(holdout_preds)

        fitted_split_metamodels, fitted_split_fitscores = train_metamodel_on_preds(select_split_dev_ival_preds, n_cv=n_metatrain_cv, n_jobs=n_jobs)

        split_results, _ = test_models_on_unseen_data(fitted_split_metamodels, select_split_holdout_preds, metalearners=metalearners)


        all_split_results[f'split_{split_idx}'] = {
            'select_cols': select_split_dev_preds.columns,
            'ival_groups': ival_groups,
            'holdout_groups': holdout_groups,
            'holdout_labels': holdout_labels,
            'ival_labels': ival_labels,
            'holdout_results': split_results
        }

    # tabularize the split results
    score_types = all_split_results['split_0']['holdout_results']['rf']['scores'].keys()
    split_results_for_pandas = {score_type: [] for score_type in score_types}
    split_results_for_pandas['split'] = []
    split_results_for_pandas['metalearner'] = []
    for metalearner in metalearners:
        for splitdx in range(n_splits):
            split_results_for_pandas['split'].append(f"split_{splitdx}")
            split_results_for_pandas['metalearner'].append(metalearner)
            for score in score_types:
                split_results_for_pandas[score].append(all_split_results[f'split_{splitdx}']['holdout_results'][metalearner]['scores'][score])

    split_results_df = pd.DataFrame(split_results_for_pandas)
    
    # now aggregate the results
    overall_labels = [label for split_idx in range(n_splits) for label in all_split_results[f'split_{split_idx}']['holdout_labels']]
    overall_preds = {}
    overall_pred_probas = {}
    for split_idx in range(n_splits):
        subsplit_results = all_split_results[f'split_{split_idx}']['holdout_results']
        for model in subsplit_results.keys():
            if model not in overall_preds.keys():
                overall_preds[model] = []
                overall_pred_probas[model] = []
            overall_preds[model].extend(subsplit_results[model]['preds'])
            overall_pred_probas[model].extend(subsplit_results[model]['pred_probas'])

    perturbation_scores = {}
    for model in overall_preds.keys():
        overall_preds[model] = np.array(overall_preds[model])
        overall_pred_probas[model] = np.array(overall_pred_probas[model])
        perturbation_scores[model] = mu.compute_select_binary_scores(overall_labels, overall_preds[model], overall_pred_probas[model])

    perturbation_score_df = pd.DataFrame({metalearner: [perturbation_scores[metalearner][score] for score in score_types] for metalearner in metalearners}, index=score_types)
    all_split_results = {
        'split_results': all_split_results,
        'perturbation_scores': perturbation_scores
    }
    return all_split_results, split_results_df, perturbation_score_df

### Model development
def return_cv_train_test_preds(X, model, n_cv=None, base_model=True, verbose=False):
    """
    For a base model, refits the model on n_cv splits and returns the training and testing cv predictions
    (will be biased optimistically because hyperparams and features selected on entire development set)
    Inputs:
        X: development data to look through
        model: a bayes cv fitted model
        n_cv: how many splits to do
        base_model: whether the model is a base model, in which case the classifier is extracted from the pipeline and the data is transformed
        verbose: whether to print out the progress
    Outputs:
        training_preds: dataframe with the training predictions
        testing_preds: dataframe with the testing predictions
    """


    st = time.time()
    if base_model:
        X_proctr = mu.get_transformed_data(X, model, verbose=verbose)
        clf_name = str(model.best_estimator_.named_steps['classifier'].__class__.__name__)
        clf = model.best_estimator_.named_steps['classifier']
    else:
        X_proctr = X.copy(deep=True)
        clf_name = model.best_estimator_.__class__.__name__
        clf = model
    
    print(f"Computing CV predictions for {clf_name}")

    if n_cv is None:
        cv_splitter = sklearn.model_selection.LeaveOneGroupOut()
    elif type(n_cv) == int:
        cv_splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    else:
        raise ValueError(f"n_cv must be None or an integer: got {n_cv}, {type(n_cv)}")
    training_preds = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0], 2))*-1
    testing_preds = np.ones((X_proctr.shape[0], 2))*-1
    training_groups = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0]))*-1
    testing_groups = np.ones((X_proctr.shape[0]))*-1
    split_col = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0]))*-1
    y_true = fu.get_y_from_df(X)
    groups = X.index.values.astype(int)
    training_block_sizes = []

    for idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_proctr, y_true, groups=groups)):
        print(f"Running base model split {idx+1}/{cv_splitter.get_n_splits(X_proctr, y_true, groups=groups)}")
        X_train, X_test = X_proctr.iloc[train_idx], X_proctr.iloc[test_idx]
        y_train = y_true[train_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        assert len(set(groups_train).intersection(set(groups_test))) == 0, f"Overlap between train and test: {set(groups_train).intersection(set(groups_test))}"
        
        # # fits the model
        # try:
        #     # set the clf n_jobs to 1 to avoid memory issues
        #     clf.set_params(n_jobs=1)
        # except:
        #     print(f"Could not set n_jobs for {clf_name}")
        clf.fit(X_train, y_train)
        testing_preds[test_idx, :] = clf.predict_proba(X_test)
        # assert len(train_idx) == X_proctr.shape[0]-1
        # store the predictions
        train_block_size = len(train_idx)
        training_preds[idx*train_block_size:(idx+1)*train_block_size, :] = clf.predict_proba(X_train)
        training_groups[idx*train_block_size:(idx+1)*train_block_size] = groups_train
        testing_groups[test_idx] = groups_test
        training_block_sizes.append(train_block_size)
        split_col[idx*train_block_size:(idx+1)*train_block_size] = idx
    testing_preds = pd.DataFrame(testing_preds, columns=[f"class{idx}" for idx in range(2)], index=testing_groups)
    training_preds = pd.DataFrame(training_preds, columns=[f"class{idx}" for idx in range(2)], index=training_groups)
    training_preds['split'] = split_col
    if verbose:    
        print(f"Made my training and testing predictions for {clf_name} in {time.time()-st} seconds, shape: {training_preds.shape}, {testing_preds.shape}, split_col: {len(split_col)}")
        print(f"Computing CV predictions took {time.time()-st} seconds: ({cv_splitter.get_n_splits(X_proctr, y_true, groups=groups)} splits)")
    return training_preds, testing_preds
  
def get_ensemble_cv_res_preds(dev_test_pred_df, n_cv=None, scores=['matthews_corrcoef', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity'], metalearners=['rf', 'lr', 'xgb'], n_jobs=5, to_select_base_models=False, internal_folder='data/internal/'):
    """
    Evaluates my metalearners in somewhat nested CV fashion (the test preds are found first and then the metalearner is evaluated in nested cv manner)
    Each test prediction had not been seen by the base model but there is surely some optimistic bias due to FS on whole dev set
    """
    split_dict = {'cv_results': {}}
    logo= False
    if n_cv is None or n_cv >= dev_test_pred_df.shape[0]:
        cv = sklearn.model_selection.LeaveOneGroupOut()
        logo = True
        n_cv = cv.get_n_splits(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df), groups=dev_test_pred_df.index)
    else:
        cv = sklearn.model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        n_cv = cv.get_n_splits(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df))

    meta_preds = {metalearner: np.ones((dev_test_pred_df.shape[0], 2))*-1 for metalearner in metalearners}
    meta_y = np.ones(dev_test_pred_df.shape[0])
    meta_groups = np.ones(dev_test_pred_df.shape[0])

    # mdx_start = {key: idx*2 for idx, key in enumerate(metalearners)}
    for splitdx, (train_idx, test_idx) in enumerate(cv.split(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df), groups=dev_test_pred_df.index)):
        print(f"Running split {splitdx+1}/{n_cv}")
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
        lr = sklearn.linear_model.LogisticRegression(random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5, n_jobs=1, max_iter=1000)
        xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
        lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
        rf_grid = {'max_depth': [2, 3, 5, 7, 11, None], 'min_samples_leaf':[1,2,4,8], 'max_leaf_nodes': [2, 3, 5, 11, 13, 17, 24, 32, 64, None]}
        xgb_grid = {
            'n_estimators': [500],
            'learning_rate': [0.03, 0.3], 
            'colsample_bytree': [0.7, 0.8, 0.9],
            'max_depth': [2, 5, 7, 15],
            'reg_alpha': [0, 1, 1.5],
            'reg_lambda': [0, 1, 1.5],
            'subsample': [0.7, 0.8, 0.9]
        }
        metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        metalearner_xgb = sklearn.model_selection.GridSearchCV(xgb, xgb_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42))


        train_pred_df = dev_test_pred_df.iloc[train_idx]
        test_pred_df = dev_test_pred_df.iloc[test_idx]
        if to_select_base_models:
            test_pred_df, train_pred_df, _, _, _ = select_base_models(test_pred_df, train_pred_df, internal_folder=internal_folder)
        y_train = fu.get_y_from_df(train_pred_df)
        y_test = fu.get_y_from_df(test_pred_df)
        groups_train = train_pred_df.index
        groups_test = test_pred_df.index
        meta_groups[test_idx] = groups_test
        meta_y[test_idx] = y_test
        assert len(set(train_pred_df.index).intersection(set(test_pred_df.index))) == 0, f"Overlap between train and test: {set(train_pred_df.index).intersection(set(test_pred_df.index))}"

        split_dict['cv_results'][f'split{splitdx}'] = {}
        split_dict['cv_results'][f'split{splitdx}']['test_groups'] = test_pred_df.index
        if 'rf' in metalearners:
            print(f"Training RandomForest metalearner on split {splitdx+1}/{n_cv}", end='\r')
            rfst= time.time()
            metalearner_rf.fit(train_pred_df, y_train, groups=groups_train)
            best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(5)]
            metalearner_rf_pred = metalearner_rf.predict_proba(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_rf'] = best_scores_rf
            split_dict['cv_results'][f'split{splitdx}']['metalearner_rf_pred'] = metalearner_rf_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_rf'] = metalearner_rf
            meta_preds['rf'][test_idx,:] = metalearner_rf_pred
            print(f"Training RandomForest metalearner on split {splitdx+1}/{n_cv} took {time.time()-rfst} seconds")

        if 'lr' in metalearners:
            print(f"Training LogisticRegression metalearner on split {splitdx+1}/{n_cv}", end='\r')
            lrst = time.time()
            metalearner_lr.fit(train_pred_df, y_train, groups=groups_train)
            best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(5)]
            metalearner_lr_pred = metalearner_lr.predict_proba(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_lr'] = best_scores_lr
            split_dict['cv_results'][f'split{splitdx}']['metalearner_lr_pred'] = metalearner_lr_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_lr'] = metalearner_lr
            meta_preds['lr'][test_idx,:] = metalearner_lr_pred
            print(f"Training LogisticRegression metalearner on split {splitdx+1}/{n_cv} took {time.time()-lrst} seconds")
        if 'xgb' in metalearners:
            print(f"Training XGBoost metalearner on split {splitdx+1}/{n_cv}", end='\r')
            xgbst = time.time()
            metalearner_xgb.fit(train_pred_df, y_train, groups=groups_train)
            best_scores_xgb = [metalearner_xgb.cv_results_[f'split{k}_test_score'][metalearner_xgb.best_index_] for k in range(5)]
            metalearner_xgb_pred = metalearner_xgb.predict_proba(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_xgb'] = best_scores_xgb
            split_dict['cv_results'][f'split{splitdx}']['metalearner_xgb_pred'] = metalearner_xgb_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_xgb'] = metalearner_xgb
            meta_preds['xgb'][test_idx,:] = metalearner_xgb_pred
            print(f"Training XGBoost metalearner on split {splitdx+1}/{n_cv} took {time.time()-xgbst} seconds")
        if not logo:
            for metalearner in metalearners:
                if metalearner == 'rf':
                    mdl_pred = np.argmax(metalearner_rf_pred, axis=1)
                    mdl_pred_proba = metalearner_rf_pred
                elif metalearner == 'lr':
                    mdl_pred = np.argmax(metalearner_lr_pred, axis=1)
                    mdl_pred_proba = metalearner_lr_pred
                elif metalearner == 'xgb':
                    mdl_pred = np.argmax(metalearner_xgb_pred, axis=1)
                    mdl_pred_proba = metalearner_xgb_pred
                else:
                    raise(ValueError(f"Metalearner {metalearner} not implemented"))
                for score in scores:
                    if score == 'specificity':
                        metric_score = sklearn.metrics.recall_score(y_test, mdl_pred, pos_label=0)
                    elif score == 'sensitivity':
                        metric_score = sklearn.metrics.recall_score(y_test, mdl_pred, pos_label=1)
                    elif score == 'balanced_accuracy':
                        metric_score = sklearn.metrics.balanced_accuracy_score(y_test, mdl_pred)
                    elif score == 'roc_auc':
                        metric_score = sklearn.metrics.roc_auc_score(y_test, mdl_pred_proba[:,1])
                    elif score == 'matthews_corrcoef':
                        metric_score = sklearn.metrics.matthews_corrcoef(y_test, mdl_pred)
                    else:
                        raise ValueError(f"Score {score} not implemented")
                        # scorer = sklearn.metrics.get_scorer(score)
                        # metric_score = scorer(y_test, mdl_pred) # won't work since scorer takes clf, X, y
                    split_dict['cv_results'][f'split{splitdx}'][f'test_scores_{metalearner}_{score}'] = metric_score
        time.sleep(0)

    for score in scores:
        for medx, metalearner in enumerate(metalearners):
            meta_pred_proba = meta_preds[metalearner]
            meta_bin_pred = np.argmax(meta_pred_proba, axis=1)
            if score == 'specificity':
                metric_score = sklearn.metrics.recall_score(meta_y, meta_bin_pred, pos_label=0)
            elif score == 'sensitivity':
                metric_score = sklearn.metrics.recall_score(meta_y, meta_bin_pred, pos_label=1)
            elif score == 'balanced_accuracy':
                metric_score = sklearn.metrics.balanced_accuracy_score(meta_y, meta_bin_pred)
            elif score == 'roc_auc':
                metric_score = sklearn.metrics.roc_auc_score(meta_y, meta_pred_proba[:,1])
            elif score == 'matthews_corrcoef':
                metric_score = sklearn.metrics.matthews_corrcoef(meta_y, meta_bin_pred)
            else:
                raise ValueError(f"Score {score} not implemented")
            split_dict['cv_results'][f'ovallall_meta_test_scores_{metalearner}_{score}'] = metric_score
            if not logo:
                split_dict['cv_results'][f'mean_meta_test_scores_{metalearner}_{score}'] = np.mean([split_dict['cv_results'][f'split{k}'][f'test_scores_{metalearner}_{score}'] for k in range(n_cv)])
                split_dict['cv_results'][f'std_meta_test_scores_{metalearner}_{score}'] = np.std([split_dict['cv_results'][f'split{k}'][f'test_scores_{metalearner}_{score}'] for k in range(n_cv)])

    split_dict['cv_results']['meta_groups'] = meta_groups
    split_dict['cv_results']['meta_y'] = meta_y
    split_dict['cv_results']['meta_pred_probas'] = meta_preds
    return split_dict

def get_avg_model_best_estimators(loaded_model_data, clf_names=None):
    if clf_names is None:
        clf_names = [str(clf[0].best_estimator_.named_steps['classifier'].__class__.__name__) for clf in loaded_model_data]
    best_scores = [eelmd[0].best_score_ for eelmd in loaded_model_data]
    best_stds = [eelmd[0].cv_results_['std_test_score'][eelmd[0].best_index_] for eelmd in loaded_model_data]
    assert best_scores == [eelmd[0].cv_results_['mean_test_score'][eelmd[0].best_index_] for eelmd in loaded_model_data]
    avg_best_score = np.mean(best_scores)
    pooled_std = np.sqrt(np.sum([eelmd[0].cv_results_['std_test_score'][eelmd[0].best_index_]**2 for eelmd in loaded_model_data]))

    print(f"Average best score: {avg_best_score}, pooled std: {pooled_std}")


    n_splits = len([key for key in loaded_model_data[0][0].cv_results_ if "_test_score" in key])-3
    model_splits = {}
    for (mdl, _, _), clf_name in zip(loaded_model_data, clf_names):
        assert n_splits == len([key for key in mdl.cv_results_ if "_test_score" in key])-3
        split_scores = [mdl.cv_results_[f'split{k}_test_score'][mdl.best_index_] for k in range(n_splits)]
        model_splits[clf_name] = split_scores
    out_dict = {"clf_names": clf_names, "best_scores": best_scores, "best_stds": best_stds, 
            "avg_best_score": avg_best_score, "pooled_std": pooled_std, "model_splits": model_splits}

    return out_dict

def plot_results(results, cv_ensemble_split_dict=None, include_average=True, fontsize=20, figsize=(10, 4), title="Training Set CV Scores for Each Classifier"):
    # Create a DataFrame for easier plotting
    df_splits = pd.DataFrame(results['model_splits'])

    # Calculate the mean scores and sort the DataFrame by them in descending order
    mean_scores = df_splits.mean().sort_values(ascending=False)
    df_splits = df_splits[mean_scores.index]

    # Convert the list of best scores into a DataFrame
    if cv_ensemble_split_dict is not None:
        best_scores = results['best_scores'].copy()
        # n_add = int(len(best_scores)/0.75)o
        # for nad in range(n_add):
        #     if nad%2 == 0:
        #         best_scores.append(max(best_scores)*1.1)
        #     else:
        #         best_scores.append(min(best_scores)-.1*min(best_scores))
        rf_best_scores = [cv_ensemble_split_dict['cv_results'][f'split{k}']['test_scores_rf_matthews_corrcoef'] for k in range(5)]
        lr_best_scores = [cv_ensemble_split_dict['cv_results'][f'split{k}']['test_scores_lr_matthews_corrcoef'] for k in range(5)]
        if include_average:
            df_best_scores =[rf_best_scores, lr_best_scores, best_scores]
        else:
            df_best_scores =[rf_best_scores, lr_best_scores]
    else:
        # we add extra on the edges to force the boxplot to go to the edges
        
        best_scores = results['best_scores'].copy()
        # n_add = np.floor(len(best_scores)/0.75).astype(int)
        # # make sure n_add is even
        # n_add = n_add + 1 if n_add%2 == 1 else n_add
        # for nad in range(n_add):
        #     if nad%2 == 0:
        #         best_scores.append(np.mean(best_scores)+np.std(best_scores)*1.5)
        #     else:
        #         best_scores.append(np.mean(best_scores)-np.std(best_scores)*1.5)
        print(best_scores)
        df_best_scores = pd.DataFrame(best_scores, columns=['Average of selected models'])

    # Create a figure with two subplots: one for the overall score and one for the classifier scores
    if cv_ensemble_split_dict is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 5]})
    else:
        if include_average:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 5]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 5]})
    # Create a box plot for the overall average best score
    sns.boxplot(data=df_best_scores, orient='h', ax=ax1, whis=0, showfliers=False)

    # Add error bars for the pooled standard deviation
    if cv_ensemble_split_dict is not None:
        x = [np.mean(ress) for ress in df_best_scores]
        rf_err = cv_ensemble_split_dict['std_test_scores_rf_matthews_corrcoef']
        lr_err = cv_ensemble_split_dict['std_test_scores_lr_matthews_corrcoef']
        if include_average:
            x[-1] = np.mean(results['best_scores'])

            xerrs = [rf_err, lr_err, results['pooled_std']]
        else:
            xerrs = [rf_err, lr_err]


        print(x, xerrs)
        for idx in range(len(x)):
            ax1.errorbar(x[idx], [idx], xerr=xerrs[idx], fmt='', color='k', capsize=12)
        ax1.set_yticks([idx for idx in range(len(x))])
        if include_average:
            ax1.set_yticklabels([f"Random Forest Metalearner", f"Logistic Regression Metalearner", f"Average of selected models"], fontsize=fontsize*.8)
        else:
            ax1.set_yticklabels([f"Random Forest Metalearner", f"Logistic Regression Metalearner"], fontsize=fontsize*.8)
    else:
        x =  np.mean(results['best_scores'])
        xerrs = results['pooled_std']
        ax1.errorbar(x, 0, xerr=xerrs, fmt='', color='k', capsize=12)
    # ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=fontsize*.8)  # Change 14 to your desired size

    # Create a horizontal box plot for each classifier
    sns.boxplot(data=df_splits, orient='h', ax=ax2)
    ax1.set_title(title, fontsize=fontsize)

    # Add a dotted line to separate the subplots
    # ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, linestyle='dotted', color='black')
    ax2.set_xlabel('Matthews correlation coefficient', fontsize=fontsize*.8)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=fontsize*.8)  # Change 14 to your desired size
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=fontsize*.6)  # Change 14 to your desired size
    plt.tight_layout()
    plt.show()

    if cv_ensemble_split_dict is not None:
        if include_average:
            means = [np.mean(ress) for ress in df_best_scores[:-1]] + [results['avg_best_score']] + df_splits.mean(axis=0).tolist()
            stds = [np.std(ress) for ress in df_best_scores[:-1]] + [results['pooled_std']] + df_splits.std(axis=0).tolist()
            mean_std_df = pd.DataFrame({'mean': means, 'std': stds},
                                        index=[f"Random Forest Metalearner", f"Logistic Regression Metalearner", f"Average of selected models"] + df_splits.columns.tolist())
        else:
            means = [np.mean(ress) for ress in df_best_scores] + df_splits.mean(axis=0).tolist()
            stds = [np.std(ress) for ress in df_best_scores] + df_splits.std(axis=0).tolist()
            mean_std_df = pd.DataFrame({'mean': means, 'std': stds},
                                        index=[f"Random Forest Metalearner", f"Logistic Regression Metalearner"] + df_splits.columns.tolist())
    else:
        if include_average:
            means = [df_best_scores.mean()] + df_splits.mean(axis=0).tolist()
            stds = [results['pooled_std']] + df_splits.std(axis=0).tolist()
            mean_std_df = pd.DataFrame({'mean': means, 'std': stds},
                                        index=[f"Average of selected models"] + df_splits.columns.tolist())
        else:
            means = df_splits.mean(axis=0).tolist()
            stds =  df_splits.std(axis=0).tolist()
            mean_std_df = pd.DataFrame({'mean': means, 'std': stds},
                                        index=df_splits.columns.tolist())
    # else:
    #     if 
    #     means = [np.mean(ress) for ress in df_best_scores] + df_splits.mean(axis=0).tolist()
    #     stds = [np.std(ress) for ress in df_best_scores] + df_splits.std(axis=0).tolist()
    #     mean_std_df = pd.DataFrame({'mean': means, 'std': stds},
    #                                 index=[f"Random Forest Metalearner", f"Logistic Regression Metalearner"] + df_splits.columns.tolist())
        
    return mean_std_df

def train_metamodel_on_preds(basepred_df: pd.DataFrame, n_cv:int=5, metalearners=['rf', 'lr', 'xgb'], n_jobs=5):
    """
    Trains the hypertuned metalearners on the predictions of the base models
    Inputs:
        - basepred_df: DataFrame containing the predictions of the base models
        - n_cv: number of splits for the cross-validation
        - metalearners: list of metalearners to use (default: ['rf', 'lr', 'xgb'])
        - n_jobs: number of jobs to use for the grid search (default: 5)
    Outputs:
        - fitted_models: dictionary containing the fitted metalearners, 
            e.g. {'metalearner_rf': fitted_rf, 'metalearner_lr': fitted_lr, 'metalearner_xgb': fitted_xgb}
        - metamodel_fitscores: dictionary containing the best scores for each metalearner,
             e.g. {'best_scores_rf': mcc_score_list, 'best_scores_lr': mcc_score_list, 'best_scores_xgb': mcc_score_list}
    """
    assert type(n_cv) == int, "cv must be an integer"
    assert n_cv > 1, "cv must be greater than 1"
    assert type(basepred_df) == pd.DataFrame, "pred_df must be a DataFrame"


    if n_cv is None:
        n_cv = sklearn.model_selection.LeaveOneOut()
        n_cv = n_cv.get_n_splits(basepred_df, fu.get_y_from_df(basepred_df))
    else:
        n_cv = n_cv

    fitted_models = {}
    metamodel_fitscores = {}
    for metalearner in metalearners:
        if metalearner == 'rf':
            rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
            rf_grid = {'max_depth': [2, 3, 5, 7, 11, None], 'min_samples_leaf':[1,2,4,8], 'max_leaf_nodes': [2, 3, 5, 11, 13, 17, 24, 32, 64, None]}
            metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=n_cv)
            metalearner_rf.fit(basepred_df, fu.get_y_from_df(basepred_df))
            best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(n_cv)]
            metamodel_fitscores[f'best_scores_{metalearner}'] = best_scores_rf
            fitted_models[f'metalearner_{metalearner}'] = metalearner_rf
        elif metalearner == 'lr':
            lr = sklearn.linear_model.LogisticRegression(random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=1000, n_jobs=1)
            lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
            metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=n_cv)
            metalearner_lr.fit(basepred_df, fu.get_y_from_df(basepred_df))
            best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(n_cv)]
            metamodel_fitscores[f'best_scores_{metalearner}'] = best_scores_lr
            fitted_models[f'metalearner_{metalearner}'] = metalearner_lr
        elif metalearner == 'xgb':
            xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
            xgb_grid = {
                'n_estimators': [100, 500, 1000],
                'colsample_bytree': [0.7, 0.8],
                'max_depth': [2, 5, 7, 15,20,25],
                'reg_alpha': [1.1, 1.2, 1.3],
                'reg_lambda': [1.1, 1.2, 1.3],
                'subsample': [0.7, 0.8, 0.9]
            }
            metalearner_xgb = sklearn.model_selection.GridSearchCV(xgb, xgb_grid, scoring='matthews_corrcoef', n_jobs=n_jobs, cv=n_cv)
            metalearner_xgb.fit(basepred_df, fu.get_y_from_df(basepred_df))
            best_scores_xgb = [metalearner_xgb.cv_results_[f'split{k}_test_score'][metalearner_xgb.best_index_] for k in range(n_cv)]
            metamodel_fitscores[f'best_scores_{metalearner}'] = best_scores_xgb
            fitted_models[f'metalearner_{metalearner}'] = metalearner_xgb
        else:
            raise ValueError(f"Metalearner {metalearner} not implemented")
    return fitted_models, metamodel_fitscores

def test_models_on_unseen_data(metalearner_dict, test_pred_df, metalearners=['rf', 'lr', 'xgb']): 
    """
    Given a dictionary that contains the fitted metalearners, test the metalearners on unseen data
    Inputs:
        - metalearner_dict: dictionary containing the fitted metalearners,
            e.g. {'metalearner_rf': fitted_rf, 'metalearner_lr': fitted_lr, 'metalearner_xgb': fitted_xgb}
        - test_pred_df: DataFrame containing the predictions of the base models
        - metalearners: list of metalearners to use (default: ['rf', 'lr', 'xgb'])
    Outputs:
        - unseen_score_pred_dict: dictionary containing the scores and predictions of the metalearners on the unseen data
            e.g. {'rf': {'scores': {'MCC': mcc, 'ROC AUC': roc_auc, 'Balanced Accuracy': balanced_accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity}, 
                        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresh': thresh},
                        'preds': metalearner_pred,
                        'pred_probs': metalearner_pred_proba},
                        
        - score_df: DataFrame containing the scores of the metalearners on the unseen data
            - rows: ['MCC', 'ROC AUC', 'Balanced Accuracy', 'Sensitivity', 'Specificity']
            - columns: metalearners
    """
    assert type(metalearner_dict) == dict, "metalearner_dict must be a dictionary"
    assert type(test_pred_df) == pd.DataFrame, "test_pred_df must be a DataFrame"
    assert all([f'metalearner_{metalearner}' in metalearner_dict.keys() for metalearner in metalearners]), "metalearner_dict must contain all metalearners"
    [sklearn.utils.validation.check_is_fitted(metalearner_dict[f'metalearner_{metalearner}']) for metalearner in metalearners]
    
    
    unseen_score_pred_dict = {}
    for metalearner in metalearners:
        fitted_mdl = metalearner_dict[f'metalearner_{metalearner}']
        metalearner_pred = fitted_mdl.predict_proba(test_pred_df)
        y_test = fu.get_y_from_df(test_pred_df)
        try:
            metalearner_pred_proba = fitted_mdl.predict_proba(test_pred_df)
            metalearner_pred = np.argmax(metalearner_pred_proba, axis=1) 
            
        except:
            print(f"Error in {metalearner}")
            metalearner_pred = fitted_mdl.predict(test_pred_df)

        # # scores
        # mcc = sklearn.metrics.matthews_corrcoef(y_test, metalearner_pred)
        # roc_auc = sklearn.metrics.roc_auc_score(y_test, metalearner_pred)
        # balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, metalearner_pred)
        # sensitivity = sklearn.metrics.recall_score(y_test, metalearner_pred, pos_label=1)
        # specificity = sklearn.metrics.recall_score(y_test, metalearner_pred, pos_label=0)

        # #roc_curve
        # fpr, tpr, thresh = sklearn.metrics.roc_curve(y_test, metalearner_pred_proba[:,1])
        # unseen_score_pred_dict[metalearner] = {'scores': {'MCC': mcc, 'ROC AUC': roc_auc, 'Balanced Accuracy': balanced_accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity},
        #                                     'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresh': thresh},
        #                                     'preds': metalearner_pred,
        #                                     'pred_probs': metalearner_pred_proba}   
        unseen_score_pred_dict[metalearner] = mu.compute_select_binary_scores(y_test, metalearner_pred, metalearner_pred_proba)
        unseen_score_pred_dict[metalearner]['preds'] = metalearner_pred
        unseen_score_pred_dict[metalearner]['pred_probs'] = metalearner_pred_proba
        
    score_names = ['MCC', 'ROC AUC', 'Balanced Accuracy', 'Sensitivity', 'Specificity']
    score_df = pd.DataFrame({metalearner: [unseen_score_pred_dict[metalearner]['scores'][score] for score in score_names] for metalearner in metalearners}, index=score_names)

    return unseen_score_pred_dict, score_df

def score_split_dfs(all_test_score_dfs):

    scores = all_test_score_dfs[0].index # ['MCC', 'ROC AUC', 'Balanced Accuracy', 'Sensitivity', 'Specificity']
    metalearners =  all_test_score_dfs[0].columns # ['Random Forest', 'Logistic Regression']

    n_splits = len(all_test_score_dfs)
    split_scores = np.ones((n_splits, len(scores)*len(metalearners)))*-100
    all_melt_columns = None
    for idx, test_score_df in enumerate(all_test_score_dfs):
        melt_columns = []
        score_values = []
        for learner in metalearners:
            for score in scores:
                melt_columns.append(f"{learner} {score}")
                score_values.append(test_score_df.loc[score, learner])
        split_scores[idx,:] = score_values
        if idx == 0:
            all_melt_columns = melt_columns
        else:
            assert all_melt_columns == melt_columns
    split_scores_df = pd.DataFrame(split_scores, columns=melt_columns, index=[f"split{idx}" for idx in range(n_splits)])
    return split_scores_df
    
def main(which_featuresets=["eeg", "ecg"], n_splits=10, late_fuse=True, savepath=BASELEARNERS_SAVEPATH,  \
         internal_folder='data/internal/', metalearners=['rf', 'lr', 'xgb'], reload_results=True, n_jobs=1, n_basetrain_cv=None, n_metatrain_cv=5, verbose=False):
    """
    Runs the final meta model using baselearners trained with train_all_baselearners.py
    Returns the results of the final model on the default datasplit and n_split perturbations of the unseen internal validation and holdout data
    
    Inputs:
        - which_featuresets: list of strings, which featuresets to use for the final model ['eeg', 'ecg', 'symptoms', 'selectsym']
        - n_splits: int, number of splits to run the final model on
        - late_fuse: bool, whether to fuse the features late or early
        - savepath: str, path where the baselearners were saved
        - internal_folder: str, path to the internal folder
        - metalearners: list of strings, which metalearners to use ['rf', 'lr', 'xgb']
        - reload_results: bool, whether to reload the results if they already exist
        - n_jobs: int, number of jobs to run the metamodel gridsearch
        - n_basetrain_cv: int, number of cross validation splits to use for training the baselearners (if None, uses the default logo cv)
        - n_metatrain_cv: int, number of cross validation splits to use for training the metamodels
        - verbose: bool, whether to print out the progress of the function
    Outputs:
        - all_results: dict, dictionary containing the results of the default and n_splits of the final model

    """
    ### check inputs 
    valid_featuresets= ['eeg', 'ecg', 'symptoms', 'selectsym']
    assert all([fs in valid_featuresets for fs in which_featuresets])

    ### make the save path
    final_savepath = os.path.join(os.path.dirname(savepath[:-1]), 'metamodels')
    fset_savepath = os.path.join(final_savepath, f'{"late" if late_fuse else "early"}_{"-".join(which_featuresets)}')
    default_savepath = os.path.join(fset_savepath, 'initial_split')
    split_savepath = os.path.join(fset_savepath, f"shuffle_split_{n_splits}")

    if not os.path.exists(split_savepath):
        os.makedirs(split_savepath)
    if not os.path.exists(default_savepath):    
        os.makedirs(default_savepath)


    ### check if the results are already there
    already_run = check_if_split_results_exist(fset_savepath, n_splits)
    run_ensemble= True if reload_results else not already_run

    if run_ensemble:
        ### Get the dev/ival/holdout predictions of the baselearners
        print(f"Loading the dev/unseen predictions")
        basemodel_results = load_model_results(which_featuresets, savepath, late_fuse=late_fuse)

        ### get the train and test predictions
        print(f"Getting train and test predictions")
        dev_pred_df, unseen_pred_df, loaded_model_data, feature_data= return_basemodel_preds(basemodel_results, n_basetrain_cv=n_basetrain_cv, verbose=verbose)
        ival_pred_df, holdout_pred_df = split_unseen_preds(unseen_pred_df, internal_folder=internal_folder, default=True)

        ### Select the best models
        print(f"Selecting the best models")
        select_dev_pred_df, select_ival_pred_df, select_holdout_pred_df, _ = select_base_models(dev_pred_df, ival_pred_df, holdout_pred_df, internal_folder=internal_folder)
        select_dev_ival_pred_df = pd.concat([select_dev_pred_df, select_ival_pred_df], axis=0)
        

        ### Train the metamodels using the dev and ival predictions
        print(f"Fitting metamodels on dev predictions")
        fitted_metamodels, metamodel_fitscores = train_metamodel_on_preds(select_dev_ival_pred_df, n_cv=n_metatrain_cv, n_jobs=n_jobs)

        ### Test the metamodels on the holdout data
        print(f"Testing metamodels on unseen data")
        default_split_results, default_results_table = test_models_on_unseen_data(fitted_metamodels, select_holdout_pred_df, metalearners=metalearners)

        ### Save the results
        print(f"Saving the default split results")
        default_results_saved = save_split_results(default_split_results,default_savepath, results_table=default_results_table, fitted_metamodels=fitted_metamodels, optional_savables={'metamodel_fitscores': metamodel_fitscores, 'dev_basepreds': dev_pred_df, 'unseen_basepreds': unseen_pred_df})

        ### Repeat the process for n_splits
        print(f"Running the final metamodel on n_splits")
        n_split_results, split_results_df, overall_perturbation_results_df = return_split_results(default_results_saved, n_splits, n_metatrain_cv)

        ### Save the results
        print(f"Saving the perturbation split results")
        split_results_saved = save_split_results(n_split_results, split_savepath, optional_savables={'perturbation_split_results_table': split_results_df, 'overall_perturbation_results': overall_perturbation_results_df})

        all_results = {
            'default': default_split_results,
            'n_splits': n_split_results,
        }

    else: 
        print(f"Results already exist, reloading them")
        default_split_results = load_split_results(default_savepath)['split_results']
        n_split_results = load_split_results(split_savepath)['split_results']
        all_results = {
            'default': default_split_results,
            'n_splits': n_split_results,
        }

    print(f"Finished running the final metamodel")

    return all_results

def plot_cv_results(*dfs, fontsize=20, figsize=(10, 4), title="Training Set CV Scores for Each Classifier"):
    n_dfs = len(dfs)
    fig, axs = plt.subplots(n_dfs, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [df.shape[1] for df in dfs]})
    for idx, df in enumerate(dfs):
        sns.boxplot(data=df, orient='h', ax=axs[idx])
        axs[idx].set_yticklabels(axs[idx].get_yticklabels(), fontsize=fontsize*0.8)

    axs[0].set_title(title, fontsize=fontsize)
    axs[-1].set_xticklabels(axs[-1].get_xticklabels(), fontsize=fontsize*0.7)
    axs[-1].set_xlabel('Matthews Correlation Coefficient', fontsize=fontsize*0.8)
    # set the xlim to be -1 to 1 for all axes
    for ax in axs:
        ax.set_xlim(-1, 1)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the final ensemble model')
    parser.add_argument('--delay', type=int, default=0, help='Delay in seconds before running the script')
    parser.add_argument('--which_featuresets', type=str, nargs='+', default=['eeg', 'ecg'], help='Which featuresets to use')
    parser.add_argument('--savepath', type=str, default=BASELEARNERS_SAVEPATH, help='Path to the baselearners')
    parser.add_argument('--late_fuse', action=argparse.BooleanOptionalAction, help='Whether to late fuse the features', default=False)
    parser.add_argument('--internal_folder', type=str, default='data/internal/', help='Path to the internal folder')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for the ensemble')
    parser.add_argument('--metalearners', type=str, nargs='+', default=['rf', 'lr', 'xgb'], help='Metalearners to use')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run the metamodel gridsearch')
    parser.add_argument('--n_basetrain_cv', type=int, default=None, help='Number of splits to use for training the baselearners')
    parser.add_argument('--n_metatrain_cv', type=int, default=5, help='Number of splits for the ensemble')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print out the progress of the function', default=True)
    args = parser.parse_args()

    # ask the user if the arguments are correct
    print(f"Running the ensemble model with the following arguments: {args}")
    input("Press Enter to continue...")
    time.sleep(args.delay)
    # remove the delay
    del args.delay#
    all_outs = main(**vars(args))