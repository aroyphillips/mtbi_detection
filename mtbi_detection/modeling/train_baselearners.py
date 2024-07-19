import math
import argparse
import time
import numpy as np
import pandas as pd
import os
import json
import sklearn
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pprint
import joblib
import glob

from typing import List
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
# import a bunch of models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


import mtbi_detection.features.compute_all_features as caf
import mtbi_detection.features.decollinearizer as deco
# import mtbi_detection.features.load_ecg_features as lef

from mtbi_detection.features.feature_utils import pearson_corr, spearman_corr, kendall_corr, anova_pinv, UnivariateThresholdSelector
import mtbi_detection.features.feature_utils as fu
import mtbi_detection.modeling.model_utils as mu
# import mtbi_detection.models.other_method_replication as omr
import mtbi_detection.data.data_utils as du


CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1]), 'features'))
RESULTS_SAVEPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1]), 'results'))


# set the random seed
np.random.seed(88)
def main(model_name='GaussianNB', which_features=['eeg'], wrapper_method='recursive', n_jobs=10, n_hyper_cv=5, n_fs_cv=3, step=5, 
         search_method='bayes', n_points=1, n_iterations=100, results_savepath=RESULTS_SAVEPATH, choose_subjs='train', featurepath=FEATUREPATH,
         verbosity=10,  n_fs_repeats=2, n_hyper_repeats=2, internal_folder='data/internal', **kwargs):
    """
    Trains the baselearner specified by model_name on the features specified by which_features
 
    Inputs
        - model_name (str): The name of the model to train. Default is 'GaussianNB'.
        - which_features (list): List of features to use for training. Default is ['ecg'].
        - wrapper_method (str): The feature selection method to use. Default is 'recursive'.
        - n_jobs (int): Number of parallel jobs to run. Default is 10.
        - n_hyper_cv (int): Number of cross-validation folds for hyperparameter tuning. Default is 5.
        - n_fs_cv (int): Number of cross-validation folds for feature selection. Default is 3.
        - step (int): Step size for the feature selection process. Default is 5.
        - search_method (str): Method to use for hyperparameter search. Default is 'bayes'.
        - n_points (int): Number of points to sample in each iteration of the search method. Default is 1.
        - n_iterations (int): Number of iterations for the search method. Default is 10.
        - results_savepath (str): Path to save the results. Default is RESULTS_SAVEPATH.
        - featurepath (str): Path to the feature data. Default is FEATUREPATH.
        - verbosity (int): Level of verbosity for logging. Default is 10.
        - n_fs_repeats (int): Number of repeats for feature selection. Default is 2.
        - n_hyper_repeats (int): Number of repeats for hyperparameter tuning. Default is 2.
        - internal_folder (str): Path to the internal folder for data storage. Default is 'data/internal'.
        - **kwargs: keyword arguments for compute_all_features.py

    Returns:
        None. Trains the model and saves the results to the specified path.
    """
    assert wrapper_method in ['none', 'recursive', 'nofsatall'], f"Wrapper method {wrapper_method} not recognized, must be 'none', 'recursive', or 'nofsatall'"
    assert search_method in ['grid', 'random', 'bayes'], f"Search method {search_method} not recognized, must be 'grid', 'random', or 'bayes'"
    assert model_name in ['GaussianNB', 'LogisticRegression', 'AdaBoost', 'KNeighborsClassifier', 'RandomForestClassifier', 'XGBClassifier', 'all'], f"Model name {model_name} not recognized"
    
    # save the best modelß
    results_savepath = os.path.join(results_savepath, 'base_learners')
    du.clean_params_path(results_savepath)
    caf_params = caf.extract_all_params(choose_subjs=choose_subjs, **kwargs)
    all_params = {**caf_params, 'which_feaures': which_features}
    savepath, found_match = du.check_and_make_params_folder(results_savepath, all_params)

    totaltime = time.time()
    
    # mattews correlation coefficient
    scoring = sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)

    fs_cv = sklearn.model_selection.RepeatedKFold(n_splits=n_fs_cv, n_repeats=n_fs_repeats, random_state=88)
    hyper_cv = sklearn.model_selection.RepeatedKFold(n_splits=n_hyper_cv, n_repeats=n_hyper_repeats, random_state=88) 

    # let's calculate the number of jobs
    max_outer_jobs = n_points * n_hyper_cv * n_hyper_repeats
    max_inner_jobs = n_fs_cv * n_fs_repeats

    outer_jobs = min(max_outer_jobs, n_jobs)
    inner_jobs = min(max_inner_jobs, max(n_jobs//outer_jobs, 1))
    model_jobs = max(n_jobs//(outer_jobs*inner_jobs), 1)
    if inner_jobs == 1 and model_jobs == 1:
        outer_jobs = n_jobs

    print(f"N_jobs: {n_jobs}, outer_jobs: {outer_jobs}, inner_jobs: {inner_jobs}, model_jobs: {model_jobs}")

    valid_featuresets= ['eeg', 'ecg', 'symptoms', 'selectsym']

    assert set(which_features).issubset(valid_featuresets), f"which_features must be a subset of {valid_featuresets}, but got {which_features}"
    
    all_feature_df, col_mapping = load_all_features(which_features, featurepath=featurepath, n_jobs=n_jobs, verbosity=verbosity, choose_subjs='train', internal_folder=internal_folder, **kwargs)


    currtime = time.strftime("%Y%m%d%H%M%S")
    filebasename = f"{model_name}_{currtime}_{'-'.join(which_features)}"

    ## Double check that there is no data leakage between dataset splits
    X_train = all_feature_df.copy(deep=True)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    y_train = fu.get_y_from_df(X_train)
    assert(len(fu.print_infs(X_train)[0])==0)
    groups_train = X_train.index.values.astype(int)


    X_ival, _ = load_all_features(which_features, featurepath=featurepath, n_jobs=n_jobs, verbosity=verbosity, choose_subjs='ival', internal_folder=internal_folder, **kwargs)
    X_holdout,_ = load_all_features(which_features, featurepath=featurepath, n_jobs=n_jobs, verbosity=verbosity, choose_subjs='holdout', internal_folder=internal_folder, **kwargs)
    y_ival = fu.get_y_from_df(X_ival)
    y_holdout = fu.get_y_from_df(X_holdout)
    groups_ival = X_ival.index.values.astype(int)
    groups_holdout = X_holdout.index.values.astype(int)

    assert np.intersect1d(groups_train, groups_ival).size == 0, "Train and ival sets are not disjoint"
    assert np.intersect1d(groups_train, groups_holdout).size == 0, "Train and holdout sets are not disjoint"
    assert np.intersect1d(groups_ival, groups_holdout).size == 0, "Ival and holdout sets are not disjoint"
    # assert np.intersect1d(X_train.index, X_ival.index).size == 0, "Train and ival sets are not disjoint"
    # assert np.intersect1d(X_train.index, X_holdout.index).size == 0, "Train and holdout sets are not disjoint"
    # assert np.intersect1d(X_ival.index, X_holdout.index).size == 0, "Ival and holdout sets are not disjoint"

    # save the data
    trainfeaturesavepath = os.path.join(savepath, f"{'-'.join(which_features)}_X_train.csv")
    ivalfeaturesavepath = os.path.join(savepath, f"{'-'.join(which_features)}_X_ival.csv")
    holdoutfeaturesavepath = os.path.join(savepath, f"{'-'.join(which_features)}_X_holdout.csv")
    if found_match:
        assert pd.read_csv(trainfeaturesavepath, index_col=0).equals(X_train)
        assert pd.read_csv(ivalfeaturesavepath, index_col=0).equals(X_ival)
        assert pd.read_csv(holdoutfeaturesavepath, index_col=0).equals(X_holdout)
    else:
        X_train.to_csv(trainfeaturesavepath)
        X_ival.to_csv(ivalfeaturesavepath)
        X_holdout.to_csv(holdoutfeaturesavepath)

    print("Making pipeline...")
    param_grid = {}
    deduper = fu.DropDuplicatedAndConstantColumns(min_unique=n_hyper_cv+n_fs_cv+1)
    var_thresh = sklearn.feature_selection.VarianceThreshold(threshold=0.0)
    scalers = [RobustScaler(), None]
    param_grid['scaler'] = scalers

    param_grid['vart__threshold'] = [0.0, 0.01, 0.05, 0.1]

    feature_filter = UnivariateThresholdSelector(sklearn.feature_selection.mutual_info_classif, threshold=0.05, discrete_features=False, min_features=100)
    param_grid['filter__score_func'] = [sklearn.feature_selection.mutual_info_classif, pearson_corr, spearman_corr, kendall_corr, anova_pinv]
    param_grid['filter__threshold'] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0]

    # narrow the threshold for EEG features due to computational complexity
    if 'eeg' in which_features:
        param_grid['filter__threshold'] = [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
        param_grid['filter__score_func'] = [pearson_corr, spearman_corr, kendall_corr, anova_pinv]

    if wrapper_method == 'recursive':
        # rfecv_estimator = XGBClassifier(max_depth=2)
        rfecv_estimator = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=4, max_features=32)
        wrapper = RFECV(estimator=rfecv_estimator, step=step, cv=fs_cv, scoring=scoring, n_jobs=inner_jobs, min_features_to_select=32, verbose=1)
        param_grid['wrapper__estimator'] = [RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=4, max_features=32)]
        param_grid['wrapper__estimator__max_depth'] = [2, 4, 8]
        param_grid['wrapper__estimator__min_samples_leaf'] = [1, 2]
        param_grid['wrapper__estimator__max_features'] = ['sqrt', 'log2']
        param_grid['wrapper__step'] = [0.001, 0.01, 0.05, 0.1, 0.2]
        param_grid['wrapper__scoring'] = ['roc_auc', 'neg_brier_score', 'balanced_accuracy', 'neg_log_loss', sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)] #, sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)]
    elif wrapper_method is None or wrapper_method.lower() == 'none' or wrapper_method.lower() == 'nofsatall':
        wrapper = 'passthrough'
    else:
        raise ValueError(f"Wrapper method {wrapper_method} not recognized")
    
    classifier = GaussianNB()

    
    if model_name.lower() == 'all':
        param_grid['classifier'] = [GaussianNB(), KNeighborsClassifier(), RandomForestClassifier(n_estimators=10000, min_samples_leaf=1, min_samples_split=4, n_jobs=model_jobs), LogisticRegression(penalty='l1', solver='saga', C=1, max_iter=1000)]
    elif model_name == 'GaussianNB':
        param_grid['classifier'] = [GaussianNB()]
    elif model_name == 'LogisticRegression':
        param_grid['classifier'] = [LogisticRegression(penalty='elasticnet', solver='saga', C=1, max_iter=1000, warm_start=True)]
        param_grid['classifier__penalty'] = ['elasticnet']
        param_grid['classifier__solver'] = ['saga']
        param_grid['classifier__C'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        param_grid['classifier__l1_ratio'] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    elif model_name == 'AdaBoost':
        param_grid['classifier'] = [AdaBoostClassifier()]
        param_grid['classifier__n_estimators'] = [100, 1000, 10000]
        param_grid['classifier__learning_rate'] = [0.001, 0.01, 0.1, 1.0]
    elif model_name == 'KNeighborsClassifier':
        param_grid['classifier'] = [KNeighborsClassifier()]
        param_grid['classifier__n_neighbors'] = [3, 5, 7]
    elif model_name == 'RandomForestClassifier':
        param_grid['classifier'] = [RandomForestClassifier(n_estimators=100, max_features=0.01, max_depth=None, max_samples=1.0, min_samples_leaf=1, min_samples_split=2, oob_score=True, n_jobs=model_jobs)]
        param_grid['classifier__n_estimators'] = [100, 1000, 10000]
        param_grid['classifier__max_features'] = ['log2', 'sqrt']
        param_grid['classifier__max_depth'] = [1, 2, 4, 8, 16, None]
        param_grid['classifier__min_samples_leaf'] = [1, 2, 4]
    elif model_name == 'XGBClassifier':
        param_grid['classifier'] = [XGBClassifier(n_jobs=model_jobs, verbosity=1)]
        param_grid['classifier__n_estimators'] = [100, 1000, 10000]
        param_grid['classifier__max_depth'] = [None, 2, 3, 4, 5, 7]
        param_grid['classifier__min_child_weight'] = [1, 2]
        param_grid['classifier__learning_rate'] = [0.001, 0.01, 0.05, 0.1, 0.2]
        param_grid['classifier__subsample'] = [0.6, 0.8, 1.0]
        param_grid['classifier__colsample_bytree'] = [0.6, 0.8, 1.0]
        param_grid['classifier__gamma'] = [0, 0.1, 0.2]
        param_grid['classifier__reg_alpha'] = [0.1, 1, 10, 100, 1000, 10000]
        param_grid['classifier__reg_lambda'] = [0.1, 1, 10, 100, 1000, 10000]
    else:
        raise ValueError(f"Model name {model_name} not implemented'")

    if wrapper_method == 'nofsatall':
        preprocpipe = Pipeline([('deduper', deduper), ('scaler', scalers[0]), ('vart', var_thresh), ('median imputer', SimpleImputer(strategy='median')), ('post_nan_imputer', fu.DataFrameImputer(fill_value=0))])
        # remove all params in param grid that start with 'filter'
        param_grid = {key: value for key, value in param_grid.items() if not key.startswith('filter')}
    elif wrapper_method == 'none':
        preprocpipe = Pipeline([('deduper', deduper), ('scaler', scalers[0]), ('vart', var_thresh), ('median imputer', SimpleImputer(strategy='median')), ('post_nan_imputer', fu.DataFrameImputer(fill_value=0)), ('filter', feature_filter)])
    else:
        decollinearizer = deco.Decollinarizer(targ_threshold=0.1, feat_threshold=0.5, prune_method='pearson', num_features=5000, targ_method='anova_pinv', n_jobs=1, min_features=64, verbosity=1)
        param_grid['deco__feat_threshold']= [0.3, 0.5, 0.7, 1.0, 1.001]
        param_grid['deco__targ_threshold']= [-.001, 0, 0.1, 0.2, 0.3, 0.9, 0.95, 0.99, 0.999]
        param_grid['deco__num_features']= [5000] 
        param_grid['deco__targ_method']= ['anova_pinv', 'mutual_information', 'kendall', 'spearman', 'pearson']
        param_grid['deco__prune_method']= ['pearson', 'spearman']
        preprocpipe = Pipeline([('deduper', deduper), ('scaler', scalers[0]), ('median imputer', SimpleImputer(strategy='median')), ('post_nan_imputer', fu.DataFrameImputer(fill_value=0)), ('vart', var_thresh),('filter', feature_filter), ('deco', decollinearizer)])

    # ensures the feature selection procedure is performed to each modality separately (ensures the final feature matrix contains features from each modality)
    preproctransform = ColumnTransformer([(f'{ftname}', preprocpipe, col_mapping[ftname]) for ftname in col_mapping.keys()])

    pipe = Pipeline([('preproc', preproctransform), ('wrapper', wrapper), ('out_nan_imputer', fu.DataFrameImputer(fill_value=0)), ('classifier', classifier)])
    preprocsteps = list(preprocpipe.named_steps.keys())
    new_param_grid = {}
    for param_name, param_values in param_grid.items():
        found_ft = False
        for step in preprocsteps:
            if param_name.startswith(step):
                for ftname in col_mapping.keys():
                    new_param_name = param_name.replace(step, f'preproc__{ftname}__{step}')
                    new_param_grid[new_param_name] = param_values
                found_ft = True
                
        if not found_ft:
            new_param_grid[param_name] = param_values
    # now we need to handle the EEG features separately due to computational complexity
    if 'eeg' in which_features:
        eeg_prefix = 'preproc__eeg__'
        new_param_grid[f'{eeg_prefix}deco__feat_threshold']= [0.5, 0.7, 1.0, 1.001]
        new_param_grid[f'{eeg_prefix}deco__targ_threshold']= [-.001, 0, 0.1, 0.2, 0.3, 0.9, 0.95, 0.99, 0.999]
        new_param_grid[f'{eeg_prefix}deco__num_features']= [100, 1000, 5000, 10000]
        new_param_grid[f'{eeg_prefix}deco__targ_method']= ['anova_pinv', 'kendall', 'spearman', 'pearson']
        new_param_grid[f'{eeg_prefix}deco__prune_method']= ['pearson', 'spearman']
                
        new_param_grid[f'{eeg_prefix}filter__threshold'] = [0.7, 0.9, 0.95, 0.99, 0.999, 1.0]
        new_param_grid[f'{eeg_prefix}filter__score_func'] = [pearson_corr, spearman_corr, anova_pinv, sklearn.feature_selection.mutual_info_classif]
        new_param_grid[f'{eeg_prefix}filter__min_features'] = [100, 500, 1000]
        new_param_grid[f'{eeg_prefix}filter__scale_scores'] = [True]
        

    param_grid = new_param_grid

    print("Using model:", model_name)
    print("Using multivariate selector method:", wrapper_method)
    print("Using search method:", search_method, "with", n_iterations, "random iterations")
    print("Using param grid:", param_grid)
    print("Pipeline:", pipe)

    ### create search
    starttime = time.time()
    if search_method == 'bayes':
        bayes_param_grid = convert_param_grid_to_search_space(param_grid)
        print("Search space:", bayes_param_grid)
        search = BayesSearchCV(estimator=pipe, search_spaces=bayes_param_grid, cv=hyper_cv, n_iter=n_iterations, n_points=n_points, n_jobs=outer_jobs, verbose=verbosity, scoring=scoring, error_score=-100, refit=True)
    elif search_method == 'grid':
        search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=hyper_cv, n_jobs=outer_jobs, verbose=verbosity, scoring=scoring, refit=True)
    elif search_method == 'random':
        search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, n_iter=n_iterations, cv=hyper_cv, n_jobs=outer_jobs, verbose=verbosity, scoring=scoring, error_score='raise', refit=True)
    else:
        raise ValueError(f"Search method {search_method} not recognized")

    # fit search

    print("Train dataframe shape", X_train.shape, "Original dataframe shape", all_feature_df.shape)
    print("Fitting search...")
    search.fit(X_train, fu.get_y_from_df(X_train))
    print(f"{search_method}Search CV took: {time.time() - starttime}")

    # print the best hyperparameters and corresponding score
    model = search.best_estimator_
    print(f"Estimator: {model}, selector: {wrapper_method}")
    print("Possible hyperparameters:", param_grid)
    print("Best hyperparameters:", search.best_params_)
    print("CV best score:", search.best_score_)


    train_results = mu.score_binary_model(search, X_train, y_train)
    ival_results = mu.score_binary_model(search, X_ival, y_ival)

    print("___TRAIN RESULTS___")
    mu.print_binary_scores(train_results)
    print("___________________")
    print("___INTERNAL VALIDATION (TEST) RESULTS___")
    mu.print_binary_scores(ival_results)
    print("__________________")
    print("Saving results...")


    currtime = time.strftime("%Y%m%d-%H%M%S")
    # pickle the grid search
    filebasename = f"{model_name}_{currtime}_{'-'.join(which_features)}"
    joblib.dump(search, os.path.join(savepath, f"{filebasename}.joblib"))
    print(f"Saved model: total time to here: {time.time() - totaltime}")
    
    saveable_test_results = du.make_dict_saveable(ival_results)
    saveable_train_results = du.make_dict_saveable(train_results)
    if wrapper_method == 'recursive':
        try:
            k_feature_names = [X_train.columns[idx] for idx in model.named_steps['wrapper'].get_support(indices=True)]
            k_feature_names_attr = None
            preproc_X_train = X_train.copy(deep=True)
            for named_step in model.named_steps:
                if named_step=='wrapper':
                    break
                if model.named_steps[named_step] is not None and model.named_steps[named_step] != 'passthrough':
                    preproc_X_train = model.named_steps[named_step].transform(preproc_X_train)
                
            # preproc_X_train = model.named_steps['deduper'].transform(X_train)
            # preproc_X_train = model.named_steps['scaler'].transform(preproc_X_train)
            # preproc_X_train = model.named_steps['filter'].transform(preproc_X_train)
            k_score = model.named_steps['wrapper'].score(preproc_X_train, y_train)
            sffs_subset = None
        except:
            print(f"Something went wrong with recursive feature selection k feature names")
            k_feature_names = X_train.columns.tolist()
            k_feature_names_attr = None
            k_score = None
            sffs_subset = None


    elif wrapper_method is None or wrapper_method.lower() == 'none' or wrapper_method.lower()=='nofsatall':
        try:
            if model_name == 'RandomForestClassifier':
                k_feature_names = X_train.columns[[idx for idx in model.named_steps['classifier'].feature_importances_.argsort()[::-1]]]
                k_feature_names_attr = [idx for idx in model.named_steps['classifier'].feature_importances_]
                # print the oob score
                k_score = model.named_steps['classifier'].oob_score_
                print("OOB score:", k_score)
                print("Top 10 features:", k_feature_names[:10])
                    
            else:
                # get the pipeline up to the classifier
                preproc_X_train = X_train.copy(deep=True)
                for named_step in model.named_steps:
                    if model.named_steps[named_step] is not None and model.named_steps[named_step] != 'passthrough' and named_step != 'classifier' and named_step != 'scaler':
                        preproc_X_train = model.named_steps[named_step].transform(preproc_X_train)

                # identify the columns that are in preproc_X_train using the original X_train: https://stackoverflow.com/questions/45313301/find-column-name-in-pandas-that-matches-an-array
                k_feature_names = X_train.columns[(X_train.values ==  np.asarray(preproc_X_train))[:, np.newaxis].all(axis=0)]
                        
                k_feature_names_attr = None
                k_score = None
        except:
            k_feature_names = X_train.columns.tolist()
            k_feature_names_attr = None
            k_score = None
        sffs_subset = None
    else:
        print(f"Wrapper method {wrapper_method} not recognized")

    # print the number of features selected
    print(f"Number of features selected: {len(k_feature_names)}")
    out_dict = {
        'search_method': search_method,
        'wrapper_method': wrapper_method,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'model_name': model_name,
        'search_params': param_grid,
        'test_metrics': saveable_test_results,
        'train_metrics': saveable_train_results,
        'k_features': [k for k in k_feature_names],
        'k_feature_names_attr': k_feature_names_attr,
        'data_params': kwargs,
        'wrapper_method': wrapper_method,
        'input_columns': X_train.columns.tolist(),
        'k_score': k_score,
        'sffs_subset': sffs_subset,
        'groups_train': groups_train.tolist(),
        'groups_test': groups_ival.tolist(),
        'groups_holdout': groups_holdout.tolist(),
        'which_features': which_features,
        'scoring': scoring,
        }
    out_dict = du.make_dict_saveable(out_dict)

    with open(os.path.join(savepath, f"{filebasename}.json"), 'w') as f:
        json.dump(out_dict, f, indent=4)

    # save the kwargs used to generate the data
    with open(os.path.join(savepath, f"{filebasename}_caf_kwargs.json"), 'w') as f:
        json.dump(kwargs, f, indent=4)

    print(f"Saved results to {savepath}: total time to here: {time.time() - totaltime}")

    return search

def load_all_features(which_features, featurepath=FEATUREPATH, n_jobs=1, verbosity=0, choose_subjs='train', internal_folder='data/internal/', **kwargs):
    """
    Load the EEG, ECG, or symptoms data
    Inputs:
        - which_features: list of featuresets to load ('eeg','ecg', 'symptoms', 'selectsym')
        - featurepath: base filepath to features
        - n_jobs: number of jobs to use during loading
        - verbosity: specifies amount of print statements
        - kwargs: used to load all features (see mtbi_detection.features.compute_all_features for more)
    Returns:
        - all_feature_df: the feature matrix as a dataframe with subjects as indices
        - col_mapping: dictionary specifying which columns belong to each feature set
    """ 
    print("Returning only the following features:", which_features)
    feature_subset_dfs = caf.main(verbosity=verbosity, n_jobs=n_jobs, return_separate=True, choose_subjs=choose_subjs, **kwargs)
    print(f"Dataset: {feature_subset_dfs.keys()}")
    print(f"Features wanted: {which_features}")
    all_feature_dfs = []
    col_mapping = {} # maps each featureset to the corresponding columns in the feature matrix

    if 'eeg' in which_features:
        all_eeg_feature_df = pd.concat([df for df in feature_subset_dfs.values()], axis=1)
        col_mapping['eeg'] = all_eeg_feature_df.columns
        all_feature_dfs.append(all_eeg_feature_df)
        
    if 'ecg' in which_features:
        all_ecg_feature_df =  lecg.main(featurepath=featurepath, choose_subjs=choose_subjs)
        col_mapping['ecg'] = all_ecg_feature_df.columns
        all_feature_dfs.append(all_ecg_feature_df)
        
    if 'symptoms' in which_features:
        all_symptom_feature_df =  ls.main(featurepath=featurepath, choose_subjs=choose_subjs)
        col_mapping['symptoms'] = all_symptom_feature_df.columns
        all_feature_dfs.append(all_symptom_feature_df)

    if 'selectsym' in which_features:
        selectsym_feature_df = omr.load_select_symptoms(featurepath=featurepath, choose_subjs=choose_subjs)
        col_mapping['selectsym'] = selectsym_feature_df.columns
        all_feature_dfs.append(selectsym_feature_df)

    assert len(col_mapping.keys()) == len(all_feature_dfs)
    assert set(col_mapping) == set(which_features)
    if len(all_feature_dfs) == 1:
        all_feature_df = all_feature_dfs[0]
    elif len(all_feature_dfs) > 1:

        overlapping_subjs = np.intersect1d([df.index for df in all_feature_dfs])
        all_feature_df = pd.concat(all_feature_dfs, axis=1)
        all_feature_df.loc[overlapping_subjs]
    else:
        raise RuntimeError(f"Error no feature sets detected for features {which_features}")
    subjs = all_feature_df.index
    assert set(subjs) == set(fu.select_subjects_from_dataframe(all_feature_df, choose_subjs=choose_subjs, internal_folder=internal_folder).index)
    return all_feature_df, col_mapping


def determine_prior(values: List):
    """
    Given a list of values, determine if they should be sampled on a log scale or not
    Inputs:
        value: list of values used in hyperparameter search
    Outputs:
        - prior: string specificying the prior to use in bayesian search
        - base: integer specificying the base value (if 'log-uniform')
    """
    # if zero is in the list, return uniform
    if 0 in values:
        return 'uniform', 10
    elif sum([math.log2(v).is_integer() for v in values])>=3:
        return 'log-uniform', 2
    elif sum([math.log10(v).is_integer() for v in values])>=3:
        return 'log-uniform', 10
    else:
        return 'uniform', 10 # default useless


# function to convert param_grid to search space for skopt
def convert_param_grid_to_search_space(param_grid):
    """
    Given a param_grid, convert it to a search space for skopt
    """
    search_space = {}
    for key, value in param_grid.items():
        # check if the value is iterable
        if hasattr(value, '__iter__'):
            if all([type(v) == int for v in value]) and len(value) > 1:
                prior, base = determine_prior(value)
                search_space[key] = Integer(value[0], value[-1], prior=prior, base=base)
            elif all([type(v) == float for v in value]) and len(value) > 1:
                prior, base = determine_prior(value)
                search_space[key] = Real(value[0], value[-1], prior=prior, base=base)
            else:
                search_space[key] = Categorical(value, transform='identity')
        else:
            raise ValueError(f"Type {type(value)} not iterable so not recognized")
    return search_space

def process_feature_df(which_features, feature_subset_dfs):
    if 'eeg' not in which_features:
        all_eeg_feature_df = pd.concat([df for key, df in feature_subset_dfs.items() if key in which_features], axis=1)
        col_mapping = {ftname: feature_subset_dfs[ftname].columns for ftname in feature_subset_dfs.keys() if ftname in which_features}
    elif 'eeg' in which_features:
        unfused_sets = [ f for f in ['ecg', 'symptoms'] if f not in which_features]
        all_eeg_feature_df = pd.concat([df for key, df in feature_subset_dfs.items() if key not in unfused_sets], axis=1)
        col_mapping = {}
        if 'ecg' in which_features:
            col_mapping['ecg'] = feature_subset_dfs['ecg'].columns
            if len(col_mapping['ecg']) == 0:
                raise ValueError(f"ECG features not found in {feature_subset_dfs.keys()}")
        if 'symptoms' in which_features:
            col_mapping['symptoms'] = feature_subset_dfs['symptoms'].columns
            if len(col_mapping['symptoms']) == 0:
                raise ValueError(f"Symptom features not found in {feature_subset_dfs.keys()}")
        col_mapping['eeg'] = [col for col in all_eeg_feature_df.columns if col not in feature_subset_dfs['ecg'].columns and col not in feature_subset_dfs['symptoms'].columns]
    return all_eeg_feature_df, col_mapping

def extract_diff_dict(big_dict, sub_dict):
    """
    Given a large dictionary and a smaller dictionary, get all the keys and values that are in the larger dictionary but not in the smaller dictionary
    """
    diff_dict = {}
    for key in big_dict.keys():
        if key not in sub_dict.keys():
            diff_dict[key] = big_dict[key]
        else:
            if big_dict[key] != sub_dict[key]:
                diff_dict[key] = big_dict[key]
    return diff_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## LOCD params
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

    ## td_params
    parser.add_argument('--interpolate_spectrum', type=int, default=1000)
    parser.add_argument('--freq_interp_method', type=str, default='linear', choices=['linear', 'log', 'log10'])
    parser.add_argument('--which_segment', type=str, default='avg', choices=['first', 'second', 'avg'], help='Which segment to use for the multitaper')
    parser.add_argument('--bandwidth', type=float, default=1)

    ## psd params
    parser.add_argument('--band_method', type=str, default='custom', help="Possible options: 'standard', 'log-standard', 'custom', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--bin_methods', type=str, nargs='+', default=['all'], help="evaluated multiple bin_methods: ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p10', 'p25', 'p75', 'p90', 'p95', 'iqr']")

    ## regional psd params
    parser.add_argument('--regional_band_method', type=str, default='standard', help="Possible options: 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--regional_n_divisions', type=int, default=1, help="Number of divisions to make in the frequency band: 1,2,3,4,5 for all except the linear_50+bands")
    parser.add_argument('--regional_log_division', action=argparse.BooleanOptionalAction, default=True, help="Whether to use log division for the frequency bands")
    parser.add_argument('--regional_bin_method', type=str, nargs='+', default=['all'], help="evaluated multiple bin_methods: ['avg', 'median', 'max', 'min', 'std', 'var', 'skew', 'p5', 'p25', 'p75', 'p95', 'iqr'] or 'pX' for Xth percentile")
    parser.add_argument('--use_regional', action=argparse.BooleanOptionalAction, default=False, help="Whether to use regional PSD features")
    
    ## maximal power params
    parser.add_argument('--power_increment', type=float, default=None, help="The increments to find the maximal power in the psd")
    parser.add_argument('--num_powers', type=int, default=20, help="The number of maximal powers to find in the psd")
    parser.add_argument('--percentile_edge_method', type=str, default='custom', choices=['custom', 'automated'], help="The method to find the spectral edge")
    
    ## spectral edge params
    parser.add_argument('--edge_increment', type=float, default=0.1, help="The increment to find the spectral edge")
    parser.add_argument('--num_edges', type=int, default=20, help="The number of spectral edges to find")
    parser.add_argument('--log_edges', action=argparse.BooleanOptionalAction, default=True, help="Whether to log the edges")
    parser.add_argument('--reverse_log', action=argparse.BooleanOptionalAction, default=False, help="Whether to reverse the log")
    parser.add_argument('--spectral_edge_method', type=str, default='custom', choices=['custom', 'automated', 'manual'], help="The method to find the spectral edge")
    
    ## complexity params
    parser.add_argument('--window_len', type=int, default=10, help="The window length for the complexity features")
    parser.add_argument('--overlap', type=float, default=1, help="The overlap for the complexity features")

    ## network params
    parser.add_argument('--network_band_method', type=str, default='custom', help="Possible options: 'standard', 'anton', 'buzsaki', 'linear_50', 'linear_100', 'linear_250'")
    parser.add_argument('--network_methods', type=str, nargs='+', default=['coherence', 'mutual_information', 'spearman', 'pearson', 'plv', 'pli']) #  'plv', 'pli'

    ## parameterized spectra
    parser.add_argument('--ps_band_basis', type=str, default='custom', help='Bands to fit the gaussians to, e.g. [(0, 4), (5, 10)]')
    parser.add_argument('--aperiodic_mode', type=str, default='knee', help='Aperiodic mode to fit')

    ## ecg features
    parser.add_argument('--use_ecg', action=argparse.BooleanOptionalAction, default=False, help='Whether to use ecg features')
    ##
    parser.add_argument('--use_symptoms', action=argparse.BooleanOptionalAction, default=False, help='Whether to use symptoms features')
    parser.add_argument('--symptoms_only', action=argparse.BooleanOptionalAction, default=True, help='Whether to use symptoms features')
    
    ## general params
    parser.add_argument('--verbosity', type=int, default=1, help="The verbosity for the complexity features")
    parser.add_argument('--n_jobs', type=int, default=1)
    
    # main params
    parser.add_argument('--n_hyper_cv', type=int, default=2, help="The number of folds to use for the grid search")
    parser.add_argument('--n_fs_cv', type=int, default=2, help="The number of folds to use for the inner cross validation")
    parser.add_argument('--n_fs_repeats', type=int, default=3, help="The number of times to repeat the feature cv")
    parser.add_argument('--n_hyper_repeats', type=int, default=3, help="The number of times to repeat the hyperparameter cv")
    parser.add_argument('--kfolds', type=int, default=10, help="The number of folds to use for the kfold cross validation")
    parser.add_argument('--step', type=float, default=0.05, help="The step to use for the recursive feature elimination")
    parser.add_argument('--search_method', type=str, default='bayes', help="The search method to use for the grid search")
    parser.add_argument('--n_iterations', type=int, default=100, help="The number of random iterations to use for the random search")
    parser.add_argument('--n_points', type=int, default=1, help="The number of points to use for the bayesian search")
    parser.add_argument('--wrapper_method', type=str, default='recursive', help="The selector method to use for the grid search")
    parser.add_argument('--sequential_tol', type=float, default=0.0, help="The tolerance to use for the sequential feature selector")
    parser.add_argument('--results_savepath', type=str, default=RESULTS_SAVEPATH, help="The path to save the results of the grid search")
    parser.add_argument('--model_name', type=str, default='XGBClassifier', help="The ndame of the model to use")
    parser.add_argument('--scoring', type=str, default='mcc', help="The scoring method to use for the grid search")



    ## data subset
    parser.add_argument('--which_features', nargs='+', type=str, default=['eeg'], help='Which features to use') # ['eeg', 'ecg', 'symptoms', 'selectsym']
    
    args = parser.parse_args()
    
    pprint.pprint(args)
    # ask the user to continue
    data_args = caf.extract_all_params(choose_subjs='train', **vars(args))
    main_args = extract_diff_dict(vars(args), data_args)

    main(**main_args, **data_args)
    print("Finished running with these inputs", main_args, data_args)
