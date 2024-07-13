import numpy as np
import pandas as pd
import os
import time
import sklearn
import skopt
import mlxtend
import xgboost
import json
import joblib
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
font_path = '/home/ap60/inter.ttf'
font_prop = fm.FontProperties(fname=font_path, size=20)
import argparse
import sklearn
from sklearn.utils.validation import check_is_fitted

import src.features.feature_utils as fu
from src.features.feature_utils import DataFrameImputer, DropDuplicatedAndConstantColumns, DropInfColumns, DropNanColumns, DropNanRows, UnivariateThresholdSelector, anova_f, anova_pinv, pearson_corr, spearman_corr, kendall_corr
import src.models.model_selection as ms
import src.models.model_analysis as ma
import src.models.eval_model as em
import src.models.model_utils as mu
import src.data.data_utils as du
import src.features.compute_all_features as caf
import src.data.load_dataset as ld
import src.models.final_ensemble_perturbations_resfit as fepr
import src.models.compile_ival_results_reg as cirr
import src.models.transfer_binary2regression as tbr

# code to evaluate my final ensemble model of regressors.
from sklearn.metrics import make_scorer
SCORING_METHODS = {'avg_pearson': make_scorer(fu.avg_pearson_pred), 
                    'avg_spearman': make_scorer(fu.avg_spearman_pred), 
                    'avg_rmse': make_scorer(fu.avg_rmse), 
                    'avg_mae': make_scorer(fu.avg_mae_pred, greater_is_better=False),
                    'avg_medae': make_scorer(fu.avg_medae_pred, greater_is_better=False),
                    'stacked_rmse': make_scorer(fu.stacked_rmse_pred, greater_is_better=False),
                    'stacked_mae': make_scorer(fu.stacked_mae_pred, greater_is_better=False),
                    'stacked_medae': make_scorer(fu.stacked_medae_pred, greater_is_better=False),
                    'stacked_pearson': make_scorer(fu.stacked_pearson_pred),   
                    'stacked_spearman': make_scorer(fu.stacked_spearman_pred),
                    'max_pearson': make_scorer(fu.max_pearson_pred),
                    'max_spearman': make_scorer(fu.max_spearman_pred),
                    'max_rmse': make_scorer(fu.max_rmse_pred, greater_is_better=False),
                    'max_mae': make_scorer(fu.max_mae_pred, greater_is_better=False),
                    'max_medae': make_scorer(fu.max_medae_pred, greater_is_better=False),
                    }  
SCORE_FUNCS = {
    'avg_pearson': fu.avg_pearson_pred, 
    'avg_spearman': fu.avg_spearman_pred, 
    'avg_rmse': fu.avg_rmse, 
    'avg_mae': fu.avg_mae_pred,
    'avg_medae': fu.avg_medae_pred,
    'stacked_rmse': fu.stacked_rmse_pred,
    'stacked_mae': fu.stacked_mae_pred,
    'stacked_medae': fu.stacked_medae_pred,
    'stacked_pearson': fu.stacked_pearson_pred,   
    'stacked_spearman': fu.stacked_spearman_pred,
    'max_pearson': fu.max_pearson_pred,
    'max_spearman': fu.max_spearman_pred,
    'max_rmse': fu.max_rmse_pred,
    'max_mae': fu.max_mae_pred,
    'max_medae': fu.max_medae_pred,
    'min_pearson': fu.min_pearson_pred,
    'min_spearman': fu.min_spearman_pred,
    'min_rmse': fu.min_rmse_pred,
    'min_mae': fu.min_mae_pred,
    'min_medae': fu.min_medae_pred,

}

def check_for_results(new_savepath, which_dataset):
    already_present = os.path.exists(os.path.join(new_savepath,  f"all_score_df_{which_dataset}.csv"))
    if already_present:
        print(f"Results already present in {new_savepath}/all_score_df_{which_dataset}.csv")
    return already_present

def make_savepaths(savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/final_reg_results/', which_dataset='eeg_ecg', n_splits=10, to_select_base_models=False):
    dset_savepath = os.path.join(savepath, which_dataset)

    if to_select_base_models:
        dset_savepath = os.path.join(dset_savepath, 'select_base_models')

    new_savepath = os.path.join(dset_savepath, f"shuffle_split_{n_splits}")


    if not os.path.exists(new_savepath):
        os.makedirs(new_savepath)

    return dset_savepath, new_savepath

def load_dataset_results(tables_path='data/tables/', dset='eeg'):
    """
    Returns all results for a dataset with a specific feature selection method
    """
    if dset in ['eeg', 'ecg', 'eeg_ecg']:
        dataset_results = dataset_results[dataset_results['dataset'] == dset]
    else:
        raise ValueError(f"Dataset {dset} not recognized")
    
    return dataset_results

def get_ensemble_cv_res_preds(dev_test_pred_df, n_cv=None, scores=['avg_rmse', 'max_rmse', 'min_rmse', 'stacked_spearman', 'stacked_pearson', 'max_spearman', 'min_spearman'], metalearners=['rf', 'lr', 'xgb'], n_jobs=5, to_select_base_models=False, internal_folder='data/internal/'):
    """
    Evaluates my metalearners in somewhat nested CV fashion (the test preds are found first and then the metalearner is evaluated in nested cv manner)
    Each test prediction had not been seen by the base model but there is surely some optimistic bias due to FS on whole dev set
    """
    split_dict = {'cv_results': {}}
    logo = False
    if n_cv is None or n_cv >= dev_test_pred_df.shape[0]:
        outer_cv = sklearn.model_selection.LeaveOneGroupOut()
        cv = sklearn.model_selection.LeaveOneGroupOut()
        logo = True
        n_cv = cv.get_n_splits(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df), groups=dev_test_pred_df.index)
    else:
        outer_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        cv = sklearn.model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=42)
        n_cv = cv.get_n_splits(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df))

    meta_preds = {metalearner: np.ones((dev_test_pred_df.shape[0], 2))*-100 for metalearner in metalearners}
    meta_y = np.ones((dev_test_pred_df.shape[0], 2))*-100
    meta_groups = np.ones(dev_test_pred_df.shape[0])

    # mdx_start = {key: idx*2 for idx, key in enumerate(metalearners)}
    scoring = sklearn.metrics.make_scorer(fu.avg_rank_rmse, greater_is_better=False)

    for splitdx, (train_idx, test_idx) in enumerate(outer_cv.split(dev_test_pred_df, fu.get_y_from_df(dev_test_pred_df))):
        print(f"Running split {splitdx+1}/{n_cv}")
        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
        lr = sklearn.linear_model.ElasticNet(random_state=42, max_iter=1000)
        xgb = xgboost.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
        lr_grid = {'alpha': [0, 0.01, 0.1, 1, 10, 100], 
                    'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
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
        metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)
        metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)
        metalearner_xgb = sklearn.model_selection.GridSearchCV(xgb, xgb_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)


        train_pred_df = dev_test_pred_df.iloc[train_idx]
        test_pred_df = dev_test_pred_df.iloc[test_idx]
        if to_select_base_models:
            test_pred_df, train_pred_df, _, _, _ = select_base_models(test_pred_df, train_pred_df)
        y_train = tbr.get_reg_from_df(train_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
        y_test = tbr.get_reg_from_df(test_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
        meta_groups[test_idx] = test_pred_df.index
        meta_y[test_idx] = y_test
        assert len(set(train_pred_df.index).intersection(set(test_pred_df.index))) == 0, f"Overlap between train and test: {set(train_pred_df.index).intersection(set(test_pred_df.index))}"

        y_train_means = y_train.mean(axis=0)
        y_train_stds = y_train.std(axis=0)
        split_dict['cv_results'][f'split{splitdx}'] = {}
        split_dict['cv_results'][f'split{splitdx}']['test_groups'] = test_pred_df.index
        split_dict['cv_results'][f'split{splitdx}']['norm_means'] = y_train_means
        split_dict['cv_results'][f'split{splitdx}']['norm_stds'] = y_train_stds
        if 'rf' in metalearners:
            print(f"Training RandomForest metalearner on split {splitdx+1}/{n_cv}", end='\r')
            rfst= time.time()
            metalearner_rf.fit(train_pred_df, y_train)
            best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(n_cv)]
            metalearner_rf_pred = metalearner_rf.predict(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_rf'] = best_scores_rf
            split_dict['cv_results'][f'split{splitdx}']['metalearner_rf_pred'] = metalearner_rf_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_rf'] = metalearner_rf
            split_dict['cv_results'][f'split{splitdx}']['all_scores_rf'] = mu.score_multireg_model(metalearner_rf, test_pred_df, y_test, norm_means=y_train_means, norm_stds=y_train_stds)

            meta_preds['rf'][test_idx,:] =  metalearner_rf_pred
            print(f"Training RandomForest metalearner on split {splitdx+1}/{n_cv} took {time.time()-rfst} seconds")

        if 'lr' in metalearners:
            print(f"Training LogisticRegression metalearner on split {splitdx+1}/{n_cv}", end='\r')
            lrst = time.time()
            metalearner_lr.fit(train_pred_df, y_train)
            best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(n_cv)]
            metalearner_lr_pred = metalearner_lr.predict(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_lr'] = best_scores_lr
            split_dict['cv_results'][f'split{splitdx}']['metalearner_lr_pred'] = metalearner_lr_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_lr'] = metalearner_lr
            split_dict['cv_results'][f'split{splitdx}']['all_scores_lr'] = mu.score_multireg_model(metalearner_lr, test_pred_df, y_test, norm_means=y_train_means, norm_stds=y_train_stds)
            meta_preds['lr'][test_idx,:] = metalearner_lr_pred
            print(f"Training LogisticRegression metalearner on split {splitdx+1}/{n_cv} took {time.time()-lrst} seconds")
        if 'xgb' in metalearners:
            print(f"Training XGBoost metalearner on split {splitdx+1}/{n_cv}", end='\r')
            xgbst = time.time()
            metalearner_xgb.fit(train_pred_df, y_train)
            best_scores_xgb = [metalearner_xgb.cv_results_[f'split{k}_test_score'][metalearner_xgb.best_index_] for k in range(n_cv)]
            metalearner_xgb_pred = metalearner_xgb.predict(test_pred_df)
            split_dict['cv_results'][f'split{splitdx}']['best_scores_xgb'] = best_scores_xgb
            split_dict['cv_results'][f'split{splitdx}']['metalearner_xgb_pred'] = metalearner_xgb_pred
            split_dict['cv_results'][f'split{splitdx}']['metalearner_xgb'] = metalearner_xgb
            split_dict['cv_results'][f'split{splitdx}']['all_scores_xgb'] = mu.score_multireg_model(metalearner_xgb, test_pred_df, y_test, norm_means=y_train_means, norm_stds=y_train_stds)
            meta_preds['xgb'][test_idx,:] = metalearner_xgb_pred
            print(f"Training XGBoost metalearner on split {splitdx+1}/{n_cv} took {time.time()-xgbst} seconds")
        if not logo:
            for metalearner in metalearners:
                if metalearner == 'rf':
                    mdl_pred = metalearner_rf_pred
                    mdl = metalearner_rf
                elif metalearner == 'lr':
                    mdl_pred = metalearner_lr_pred
                    mdl = metalearner_lr
                elif metalearner == 'xgb':
                    mdl_pred = metalearner_xgb_pred
                    mdl = metalearner_xgb
                else:
                    raise(ValueError(f"Metalearner {metalearner} not implemented"))
                for score in scores:
                    try:
                        # scorer = SCORING_METHODS[score]
                        metric_score = SCORE_FUNCS[score](y_test.values, mdl_pred)
                        # metric_score = scorer(mdl,test_pred_df, y_test)#, mdl_pred)
                    except Exception as e:
                        print(f"Error in {metalearner} {score} {splitdx}")
                        raise(e)
                        # scorer = sklearn.metrics.get_scorer(score)
                        # metric_score = scorer(y_test, mdl_pred) # won't work since scorer takes clf, X, y
                    split_dict['cv_results'][f'split{splitdx}'][f'test_scores_{metalearner}_{score}'] = metric_score
        time.sleep(0)

    for score in scores:
        for medx, metalearner in enumerate(metalearners):
            meta_pred = meta_preds[metalearner]
            try:
                score_func = SCORE_FUNCS[score]
                metric_score = score_func(meta_y, meta_pred)
            except Exception as e:
                print(f"Error in {metalearner} {score}")
                raise(e)
            split_dict['cv_results'][f'ovallall_meta_test_scores_{metalearner}_{score}'] = metric_score
            if not logo:
                split_dict['cv_results'][f'mean_meta_test_scores_{metalearner}_{score}'] = np.mean([split_dict['cv_results'][f'split{k}'][f'test_scores_{metalearner}_{score}'] for k in range(n_cv)])
                split_dict['cv_results'][f'std_meta_test_scores_{metalearner}_{score}'] = np.std([split_dict['cv_results'][f'split{k}'][f'test_scores_{metalearner}_{score}'] for k in range(n_cv)])

    split_dict['cv_results']['meta_groups'] = meta_groups
    split_dict['cv_results']['meta_y'] = meta_y
    split_dict['cv_results']['meta_pred_probas'] = meta_preds
    return split_dict

def validation_wilcoxon_signed(pred_df, best_model_idx):
    """
    Given a pred df with columns as models and rows as subjects, return the p-values for the wilcoxon signed rank test for each model against the best model
    args:
    pred_df: dataframe with the predictions
    best_model_idx: index of the best model
    returns:
    p_vals: list of p-values for each model
    """
    best_model_preds = pred_df.iloc[:, best_model_idx]
    p_vals = []
    for col in pred_df.columns:
        if col == pred_df.columns[best_model_idx]:
            p_vals.append(1)
        else:
            p_vals.append(scipy.stats.wilcoxon(best_model_preds, pred_df[col])[1])
    return p_vals

def select_base_models(dev_pred_df=None, ival_pred_df=None, holdout_pred_df=None, internal_folder='data/internal/'):
    """
    Select a base set of multiout regressor models using Friedman test or Wilcoxon signed rank https://www.nature.com/articles/s41598-024-56706-x. Choose all models that are above 0.05 likelihood simlar to the best model on the internal validation set
    dev_pred_df: dataframe with the predictions on the development set
    ival_pred_df: dataframe with the predictions on the internal validation set (THIS IS THE SET USED TO SELECT THE MODELS)
    holdout_pred_df: dataframe with the predictions on the holdout set
    Outputs:
    dev_select_pred_df: dataframe with the selected models on the development set
    ival_select_pred_df: dataframe with the selected models on the internal validation set
    holdout_select_pred_df: dataframe with the selected models on the holdout set
    selected_cols: list of selected columns
    col_dict: dictionary of column: pval for each column
    """
    assert ival_pred_df is not None, "Must provide internal validation set predictions"

    if dev_pred_df is None:
        dev_subjs = [g for g in ld.load_splits()['train'] if int(g) not in ival_pred_df.index and g not in holdout_pred_df.index]
        if len(dev_subjs) == 0:
            dev_subjs = [max(ival_pred_df.index) + max(holdout_pred_df.index) + 1]
        dev_pred_df = pd.DataFrame(np.random.random((len(dev_subjs), len(ival_pred_df.columns))), columns=ival_pred_df.columns, index=dev_subjs)
    if holdout_pred_df is None:
        holdout_subjs = [g for g in ld.load_splits()['holdout'] if int(g) not in ival_pred_df.index and g not in dev_pred_df.index]
        if len(holdout_subjs) == 0:
            holdout_subjs = [max(ival_pred_df.index) + max(dev_pred_df.index) + 1]
        holdout_pred_df = pd.DataFrame(np.random.random((len(holdout_subjs), len(ival_pred_df.columns))), columns=ival_pred_df.columns, index=holdout_subjs)
        
    dev_y_test = tbr.get_reg_from_df(dev_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
    ival_y_test = tbr.get_reg_from_df(ival_pred_df, pcs_dir=internal_folder, questionnaires_only=True)

    dev_y_test_ace = dev_y_test[[col for col in dev_y_test.columns if 'ACE' in col]]
    dev_y_test_rivermead = dev_y_test[[col for col in dev_y_test.columns if 'Rivermead' in col]]
    dev_pred_df_ace = dev_pred_df[[col for col in dev_pred_df.columns if 'ACE' in col]]# Score_ACE_Baseline, Score_Rivermead_Baseline
    dev_pred_df_rivermead = dev_pred_df[[col for col in dev_pred_df.columns if 'Rivermead' in col]]
    ival_pred_df_ace = ival_pred_df[[col for col in ival_pred_df.columns if 'ACE' in col]]
    ival_pred_df_rivermead = ival_pred_df[[col for col in ival_pred_df.columns if 'Rivermead' in col]]
    # choose the best model
    best_ace_model_idx = np.argmin([fu.avg_rank_rmse(dev_y_test_ace.values, dev_pred_df_ace[col]) for col in dev_pred_df_ace.columns])
    best_rivermead_model_idx = np.argmin([fu.avg_rank_rmse(dev_y_test_rivermead.values, dev_pred_df_rivermead[col]) for col in dev_pred_df_rivermead.columns])

    assert dev_pred_df_ace.shape[1] == dev_pred_df_rivermead.shape[1], "num columns do not match"
    assert ival_pred_df_ace.shape[1] == ival_pred_df_rivermead.shape[1], "num columns do not match"


    # calculate the mcnemar midp likelihood that each other model could be the same as the best model
    ace_p_vals = validation_wilcoxon_signed(ival_pred_df_ace, best_ace_model_idx)
    rivermead_p_vals = validation_wilcoxon_signed(ival_pred_df_rivermead, best_rivermead_model_idx)

    # select all models that are above 0.05 likelihood
    selected_cols = [col for col, midp in zip(ival_pred_df_ace.columns, ace_p_vals) if midp > 0.05] + [col for col, midp in zip(ival_pred_df_rivermead.columns, rivermead_p_vals) if midp > 0.05]
    col_dict = {col: pval for col, pval in zip(ival_pred_df_ace.columns, ace_p_vals)}
    for col, pval in zip(ival_pred_df_rivermead.columns, rivermead_p_vals):
        col_dict[col] = pval

    dev_select_pred_df = dev_pred_df[[col for col in dev_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    ival_select_pred_df = ival_pred_df[[col for col in ival_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    holdout_select_pred_df = holdout_pred_df[[col for col in holdout_pred_df.columns if any([col.startswith(s) for s in selected_cols])]]
    # assert no overlap
    assert len(set(dev_select_pred_df.index).intersection(set(ival_select_pred_df.index))) == 0, f"{set(dev_select_pred_df.index).intersection(set(ival_select_pred_df.index))} overlap between dev and ival"
    assert len(set(dev_select_pred_df.index).intersection(set(holdout_select_pred_df.index))) == 0, f"{set(dev_select_pred_df.index).intersection(set(holdout_select_pred_df.index))} overlap between dev and holdout"
    assert len(set(ival_select_pred_df.index).intersection(set(holdout_select_pred_df.index))) == 0, f"{set(ival_select_pred_df.index).intersection(set(holdout_select_pred_df.index))} overlap between ival and holdout"
    assert all([all(ival_select_pred_df.columns == dev_select_pred_df.columns), all(ival_select_pred_df.columns == holdout_select_pred_df.columns)]), "Columns do not match"
    return dev_select_pred_df, ival_select_pred_df, holdout_select_pred_df, selected_cols, col_dict

def load_reg_model_data(tables_path='data/tables/', dset='eeg'):
    """
    Returns a dict of filename: tuples (model, Xtrain, Xtest)
    """

    dataset_results = load_dataset_results(tables_path=tables_path, dset=dset)

    loaded_model_data = {filename: ms.load_model_data(filename) for filename in dataset_results['filename'].values}
    train_index = loaded_model_data[dataset_results['filename'].values[0]][1].index
    test_index = loaded_model_data[dataset_results['filename'].values[0]][2].index
    assert all([all(train_index == loaded_model_data[filename][1].index) for filename in dataset_results['filename'].values]), f" Train indices don't match for {dataset_results['filename'].values[[not all(train_index == loaded_model_data[filename][1].index) for filename in dataset_results['filename'].values]]}"
    assert all([all(test_index == loaded_model_data[filename][2].index) for filename in dataset_results['filename'].values]), f" Test indices don't match for {dataset_results['filename'].values[[not all(test_index == loaded_model_data[filename][2].index) for filename in dataset_results['filename'].values]]}"
    [check_is_fitted(loaded_model_data[filename][0].best_estimator_) for filename in dataset_results['filename'].values]

    # loaded_model_data = [ms.load_model_data(filename) for filename in dataset_results['filename'].values]
    # train_index = loaded_model_data[0][1].index
    # test_index = loaded_model_data[0][2].index
    # assert all([all(train_index == loaded_model_data[f][1].index) for f in range(len(loaded_model_data))]), f" Train indices don't match for {dataset_results['filename'].values[[not all(train_index == loaded_model_data[f][1].index) for f in range(len(loaded_model_data))]]}"
    # assert all([all(test_index == loaded_model_data[f][2].index) for f in range(len(loaded_model_data))]), f" Test indices don't match for {dataset_results['filename'].values[[not all(test_index == loaded_model_data[f][2].index) for f in range(len(loaded_model_data))]]}"
    # assert all([check_is_fitted(loaded_model_data[f][0]) for f in range(len(loaded_model_data))]), "Not all models are fitted"
    return loaded_model_data

def return_cv_train_test_preds(X, model, n_cv=None, internal_folder='data/internal/'):
    """
    For a base model, returns the logo predictions
    (will be slightly optimistic bc hyperparams and features selected on whole dev set)
    args:
        X full training data
        cv model
    returns:
        training_preds: dataframe with the training predictions on cv splits
        testing_preds: dataframe with the testing predictions on cv splits
    """

    clf_name = str(model.best_estimator_.named_steps['classifier'].__class__.__name__)
    print(f"Computing CV predictions for {clf_name}")
    st = time.time()
    X_proctr = ma.get_transformed_data(X, model, verbose=False)
    clf = model.best_estimator_.named_steps['classifier']
    if n_cv is None:
        logo_cv = sklearn.model_selection.LeaveOneGroupOut()
        outer_cv = sklearn.model_selection.LeaveOneGroupOut()
    elif type(n_cv) == int:
        outer_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        logo_cv = sklearn.model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=42)
    else:
        raise ValueError(f"n_cv must be None or an integer: got {n_cv}, {type(n_cv)}")
    training_preds = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0], 2))*-1
    testing_preds = np.ones((X_proctr.shape[0], 2))*-1
    training_groups = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0]))*-1
    testing_groups = np.ones((X_proctr.shape[0]))*-1
    split_col = np.ones(((X_proctr.shape[0]-1)*X_proctr.shape[0]))*-1
    y_true = tbr.get_reg_from_df(X, pcs_dir=internal_folder, questionnaires_only=True)
    groups = X.index.values.astype(int)
    training_block_sizes = []

    y_true_bin = fu.get_y_from_df(X)
    for idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_proctr, y_true_bin, groups=groups)):
        print(f"Running base model split {idx+1}/{outer_cv.get_n_splits(X_proctr, y_true_bin, groups=groups)}")
        X_train, X_test = X_proctr.iloc[train_idx], X_proctr.iloc[test_idx]
        y_train = y_true.iloc[train_idx, :]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        assert len(set(groups_train).intersection(set(groups_test))) == 0, f"Overlap between train and test: {set(groups_train).intersection(set(groups_test))}"
        
        # fits the model
        try:
            # set the clf n_jobs to 1 to avoid memory issues
            clf.set_params(n_jobs=1)
        except:
            print(f"Could not set n_jobs for {clf_name}")
        clf.fit(X_train, y_train)
        testing_preds[test_idx, :] = clf.predict(X_test)
        # assert len(train_idx) == X_proctr.shape[0]-1
        # store the predictions
        train_block_size = len(train_idx)
        training_preds[idx*train_block_size:(idx+1)*train_block_size, :] = clf.predict(X_train)
        training_groups[idx*train_block_size:(idx+1)*train_block_size] = groups_train
        testing_groups[test_idx] = groups_test
        training_block_sizes.append(train_block_size)
        split_col[idx*train_block_size:(idx+1)*train_block_size] = idx
    total_training_block_size = sum(training_block_sizes)

    assert np.allclose(training_preds[total_training_block_size:, :], -1), f"Training preds not filled: {training_preds[total_training_block_size:, :]}"
    training_preds = training_preds[:total_training_block_size, :]
    training_groups = training_groups[:total_training_block_size]
    
    symptom_cols = y_true.columns
    testing_preds = pd.DataFrame(testing_preds, columns=symptom_cols, index=testing_groups)
    training_preds = pd.DataFrame(training_preds, columns=symptom_cols, index=training_groups)
    # training_preds['split'] = [f"split{idx}" for idx, block_size in enumerate(training_block_sizes) for _ in range(block_size)]
    print(f"Made my training and testing predictions for {clf_name} in {time.time()-st} seconds, shape: {training_preds.shape}, {testing_preds.shape}, split_col: {len(split_col)}")
    training_preds['split'] = split_col
    print(f"Computing CV predictions took {time.time()-st} seconds: ({outer_cv.get_n_splits(X_proctr, y_true_bin, groups=groups)} splits)")
    
    assert training_preds.shape[1]-1  == testing_preds.shape[1] == y_true.shape[1], f"Shapes do not match: {training_preds.shape[1]}, {testing_preds.shape[1]}, {y_true.shape[1]}"
    return training_preds, testing_preds


def return_dev_unseen_res_preds(loaded_model_data, dset='eeg', internal_folder='data/internal/', tables_folder='data/tables/'):
    """
    Go through a dataframe and load the development and unseen predictions into a dataframe   
    Input: loaded_model_data: dict of filename: (model, Xtrain, Xtest)
    Output:
    dev_test_pred_df: dataframe with the predictions on the development set
    dev_train_pred_df: dataframe with the predictions on the development set
    unseen_pred_df: dataframe with the predictions on the unseen set
    loaded_model_data: list of tuples with the model, Xtr, Xunseen
    """
    all_dev_cv_test_preds = []
    all_dev_cv_train_preds = []
    unseen_preds = []
    filenames = list(loaded_model_data.keys())
    prediction_cols = tbr.get_reg_from_df(loaded_model_data[filenames[0]][1], pcs_dir=internal_folder, questionnaires_only=True).columns
    for idx, filename in enumerate(filenames):
        print(f"Getting development and unseen predictions for model {idx+1}/{len(loaded_model_data)}")
        mdl, Xtr, Xts = loaded_model_data[filename]
        assert len(set(Xtr.index).intersection(set(Xts.index))) == 0, f"Overlap between train and test: {set(Xtr.index).intersection(set(Xts.index))}"
        print(f"Getting predictions for model {idx+1}/{len(loaded_model_data)}")
        Xunseen = fepr.load_unseen_data(filename, dset, Xtr.index, base_folder=tables_folder, internal_folder=internal_folder))
        trts_subjects = list(Xtr.index.values.astype(int)) + list(Xts.index.values.astype(int))
        # Xholdout = Xunseen.loc[[s for s in Xunseen.index if int(s) not in trts_subjects ]]
        # ytr = tbr.get_reg_from_df(Xtr, questionnaires_only=True)
        # yts = tbr.get_reg_from_df(Xtr, questionnaires_only=True)
        # yunseen = tbr.get_reg_from_df(Xunseen, questionnaires_only=True)
        dev_cv_train_preds, dev_cv_test_preds = return_cv_train_test_preds(Xtr, mdl)

        unseen_pred = mdl.predict(Xunseen)
        unseen_pred = pd.DataFrame(unseen_pred, columns=prediction_cols, index=Xunseen.index)

        unseen_preds.append(unseen_pred)
        all_dev_cv_test_preds.append(dev_cv_test_preds)
        all_dev_cv_train_preds.append(dev_cv_train_preds)

        assert len(set(Xtr.index).intersection(set(Xts.index))) == 0
        assert len(set(Xtr.index).intersection(set(Xunseen.index))) == 0

    assert len(all_dev_cv_test_preds) == len(all_dev_cv_train_preds) == len(unseen_preds), f"Lengths do not match: {len(all_dev_cv_test_preds)}, {len(all_dev_cv_train_preds)}, {len(unseen_preds)}"
    assert all([all(all_dev_cv_test_preds[0].index == dctp.index) for dctp in all_dev_cv_test_preds])
    assert all([all(all_dev_cv_train_preds[0].index == dctp.index) for dctp in all_dev_cv_train_preds])
    assert all([all(unseen_preds[0].index == up.index) for up in unseen_preds])

    dev_cv_test_groups = all_dev_cv_test_preds[0].index
    dev_cv_train_groups = all_dev_cv_train_preds[0].index
    unseen_groups = unseen_preds[0].index

    dev_test_preds = np.concatenate(all_dev_cv_test_preds, axis=1)
    dev_train_preds = np.concatenate(all_dev_cv_train_preds, axis=1)
    unseen_preds = np.concatenate(unseen_preds, axis=1)
    train_sym_cols = [f"{filename.split('/')[-1]}_{col}" for filename in filenames for col in all_dev_cv_train_preds[0].columns]

    column_names = [f"{filename.split('/')[-1]}_{col}" for filename in filenames for col in prediction_cols]
    assert dev_test_preds.shape[1] == unseen_preds.shape[1] == len(column_names), f"Shapes do not match: {dev_test_preds.shape[1]}, {unseen_preds.shape[1]}, {len(column_names)}"
    dev_test_pred_df = pd.DataFrame(dev_test_preds, columns=column_names, index=dev_cv_test_groups)
    dev_train_pred_df = pd.DataFrame(dev_train_preds, columns=train_sym_cols, index=dev_cv_train_groups)
    unseen_pred_df = pd.DataFrame(unseen_preds, columns=column_names, index=unseen_groups)

    return dev_test_pred_df, dev_train_pred_df, unseen_pred_df

def train_reg_ensemble_on_preds(train_pred_df, cv:int=5, metalearners=['rf', 'lr', 'xgb'], n_jobs=5, internal_folder='data/internal/'):
    """
    Given a dataframe with the predictions, train a metalearner on the predictions
    Inputs:
    train_pred_df: dataframe with the predictions
    cv: number of cross validation splits
    metalearners: list of metalearners to train
    Returns:
    fitted_mdl_dict: dictionary of fitted models
    out_dict: dictionary of scores
    """
    if cv is None:
        cv = sklearn.model_selection.LeaveOneOut()
        n_cv = cv.get_n_splits(train_pred_df, fu.get_y_from_df(train_pred_df))
    else:
        n_cv = cv
    fitted_mdl_dict = {}
    out_dict = {}

    scoring = sklearn.metrics.make_scorer(fu.avg_rmse, greater_is_better=False)

    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    lr = sklearn.linear_model.ElasticNet(random_state=42, max_iter=1000)
    xgb = xgboost.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    lr_grid = {'alpha': [0, 0.01, 0.1, 1, 10, 100], 
                'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
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
    metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)
    metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)
    metalearner_xgb = sklearn.model_selection.GridSearchCV(xgb, xgb_grid, scoring=scoring, n_jobs=n_jobs, cv=cv)

    lr = sklearn.linear_model.ElasticNet(random_state=42)
    y_train = tbr.get_reg_from_df(train_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
    for metalearner in metalearners:
        if metalearner == 'rf':
            metalearner_rf.fit(train_pred_df, y_train)
            best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(n_cv)]
            out_dict[f'best_scores_{metalearner}'] = best_scores_rf
            fitted_mdl_dict[f'metalearner_{metalearner}'] = metalearner_rf
        elif metalearner == 'lr':
            metalearner_lr.fit(train_pred_df, y_train)
            best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(n_cv)]
            out_dict[f'best_scores_{metalearner}'] = best_scores_lr
            fitted_mdl_dict[f'metalearner_{metalearner}'] = metalearner_lr
        elif metalearner == 'xgb':
            metalearner_xgb.fit(train_pred_df, y_train)
            best_scores_xgb = [metalearner_xgb.cv_results_[f'split{k}_test_score'][metalearner_xgb.best_index_] for k in range(n_cv)]
            out_dict[f'best_scores_{metalearner}'] = best_scores_xgb
            fitted_mdl_dict[f'metalearner_{metalearner}'] = metalearner_xgb
    return fitted_mdl_dict, out_dict

def test_reg_model_on_unseen_data(metalearner_dict, test_pred_df, metalearners=['rf', 'lr'], internal_folder='data/internal/'):
    """
    Given a dictionary that contains the fitted metalearners, test the metalearners on unseen data
    """
    unseen_score_pred_dict = {}
    score_df_dict = {} # metalearner: {ACE_rmse, ACE_spearman, ACE_pearson, Rivermead_rmse, Rivermead_spearman, Rivermead_pearson, avg_rmse, stacked_spearman, stacked_pearson}
    for metalearner in metalearners:
        fitted_mdl = metalearner_dict[f'metalearner_{metalearner}']
        metalearner_pred = fitted_mdl.predict(test_pred_df)
        y_test = tbr.get_reg_from_df(test_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
        multiout_scores = mu.score_multireg_model(fitted_mdl, test_pred_df, y_test)

        unseen_score_pred_dict[metalearner] = {'preds': metalearner_pred, 'scores': multiout_scores, 'test_groups': test_pred_df.index, 'test_y_true': y_test}
        
        ace_test = y_test[[col for col in y_test.columns if 'ACE' in col]]
        rivermead_test = y_test[[col for col in y_test.columns if 'Rivermead' in col]]
        ace_pred = metalearner_pred[:, np.array([idx for idx, col in enumerate(y_test.columns) if 'ACE' in col])]
        rivermead_pred = metalearner_pred[:, np.array([idx for idx, col in enumerate(y_test.columns) if 'Rivermead' in col])]
        # stacked_test = y_test.values.flatten()
        # stacked_pred = metalearner_pred.flatten()

        ace_rmse = sklearn.metrics.mean_squared_error(ace_test, ace_pred, squared=False)
        ace_rrmse = fu.avg_rank_rmse(ace_test, ace_pred) 
        ace_spearman = scipy.stats.spearmanr(ace_test, ace_pred, axis=0)[0]
        ace_pearson = scipy.stats.pearsonr(ace_test.values.flatten(), ace_pred.flatten())[0]

        rivermead_rmse = sklearn.metrics.mean_squared_error(rivermead_test, rivermead_pred, squared=False)
        rivermead_rrmse = fu.avg_rank_rmse(rivermead_test, rivermead_pred)
        rivermead_spearman = scipy.stats.spearmanr(rivermead_test, rivermead_pred, axis=0)[0]
        rivermead_pearson = scipy.stats.pearsonr(rivermead_test.values.flatten(), rivermead_pred.flatten())[0]

        avg_rmse = fu.avg_rmse(y_test.values, metalearner_pred)
        avg_rrmse = fu.avg_rank_rmse(y_test.values, metalearner_pred)
        stacked_spearman = fu.stacked_spearman_pred(y_test.values, metalearner_pred)
        stacked_pearson = fu.stacked_pearson_pred(y_test.values, metalearner_pred)

        score_df_dict[metalearner] = {'ACE_rmse': ace_rmse, 'ACE_rrmse': ace_rrmse, 'ACE_spearman': ace_spearman,
                                      'ACE_pearson': ace_pearson, 'Rivermead_rmse': rivermead_rmse, 'Rivermead_rrmse': rivermead_rrmse,
                                      'Rivermead_spearman': rivermead_spearman, 'Rivermead_pearson': rivermead_pearson, 
                                      'avg_rmse': avg_rmse, 'avg_rrmse': avg_rrmse, 'stacked_spearman': stacked_spearman, 'stacked_pearson': stacked_pearson}
        
        unseen_score_pred_dict[metalearner]['selected_scores'] = score_df_dict[metalearner]


    
    score_df = pd.DataFrame(score_df_dict).T # index: scores, columns: metalearners

    return unseen_score_pred_dict, score_df

def main(n_splits=10, which_dataset="eeg_ecg", savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/final_reg_results/',  \
         results_table_path='data/tables/', internal_folder='data/internal/', metalearners=['rf', 'lr', 'xgb'],
            reload_results=True, dev_eval_cv=5, ival_eval_cv=5, n_train_cv=5, n_jobs=5, to_select_base_models=False):
    # load up the dataframe for the dataset
    print(f"Running on {which_dataset} with feature selection")
    dset_savepath, new_savepath = make_savepaths(savepath=savepath, which_dataset=which_dataset, n_splits=n_splits, to_select_base_models=to_select_base_models)

    # check if the results are already there

    print(f"Checking if the results exist in {new_savepath}/all_score_df_{which_dataset}.csv")
    already_present = check_for_results(new_savepath=new_savepath, which_dataset=which_dataset)

    run_ensemble = reload_results or not already_present
    print(f"Rerunning the dev/unseen predictions? {run_ensemble}")
    loaded_model_data = False

    if run_ensemble:

        if which_dataset.split('_')[0] == 'late':
            full_results, dev_cv_ensemble_split_dict, dev_test_pred_df, dev_train_pred_df, unseen_pred_df, all_score_df, fitted_metamodels = run_late_ensemble_resmodel(savepath=savepath, results_table_path=results_table_path, internal_folder=internal_folder, which_dataset=which_dataset, reload_results=reload_results, n_splits=n_splits, dev_eval_cv=dev_eval_cv, ival_eval_cv=ival_eval_cv, metalearners=metalearners, n_jobs=n_jobs, n_train_cv=n_train_cv, to_select_base_models=to_select_base_models)
            print(f"Finished running late ensemble model")
        else:
            holdout_cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)

            loaded_model_data = load_reg_model_data(tables_path=results_table_path)
            if os.path.exists(os.path.join(dset_savepath, f"unseen_pred_df_{which_dataset}.csv")) and not reload_results:
                print(f"Found the dev/unseen predictions in {dset_savepath}")
                dev_test_pred_df = pd.read_csv(os.path.join(dset_savepath, f"dev_test_pred_df_{which_dataset}.csv"), index_col=0)
                dev_train_pred_df = pd.read_csv(os.path.join(dset_savepath, f"dev_train_pred_df_{which_dataset}.csv"), index_col=0)
                unseen_pred_df = pd.read_csv(os.path.join(dset_savepath, f"unseen_pred_df_{which_dataset}.csv"), index_col=0)
            else:
                print(f"Loading the dev/unseen predictions")
                # get the train and test predictions
                print(f"Getting train and test predictions")
                dev_test_pred_df, dev_train_pred_df, unseen_pred_df = return_dev_unseen_res_preds(loaded_model_data)
                dev_test_pred_df.to_csv(os.path.join(dset_savepath, f"dev_test_pred_df_{which_dataset}.csv"))
                dev_train_pred_df.to_csv(os.path.join(dset_savepath, f"dev_train_pred_df_{which_dataset}.csv"))
                unseen_pred_df.to_csv(os.path.join(dset_savepath, f"unseen_pred_df_{which_dataset}.csv"))

            # let's remove skips
            dev_test_pred_df = dev_test_pred_df.loc[[int(s) for s in ld.load_splits()['train'] if int(s) in dev_test_pred_df.index]]
            unseen_pred_df = unseen_pred_df.loc[[int(s) for s in ld.load_splits()['ival'] if int(s) in unseen_pred_df.index] + [int(s) for s in ld.load_splits()['holdout'] if int(s) in unseen_pred_df.index]]


            # get the ensemble cv predictions

            # now store the cv results
            print(f"Storing cv results")
            if os.path.exists(os.path.join(dset_savepath, f"dev_cv_ensemble_split_dict_{which_dataset}.json")) and not reload_results:
                with open(os.path.join(dset_savepath, f"dev_cv_ensemble_split_dict_{which_dataset}.json"), 'r') as f:
                    dev_cv_ensemble_split_dict = json.load(f)

            else:
                print(f"Getting ensemble cv predictions on dev set")
                dev_cv_ensemble_split_dict = get_ensemble_cv_res_preds(dev_test_pred_df, metalearners=metalearners, n_cv=dev_eval_cv)
                if not loaded_model_data:
                    loaded_model_data = load_reg_model_data(tables_path=results_table_path)

                # save the results to dset_savepath
                saveable_dev_cv_out_dict = du.make_dict_saveable(dev_cv_ensemble_split_dict)
                saveable_dev_cv_out_dict['basefilenames'] = list(loaded_model_data.keys())
                with open(os.path.join(dset_savepath, f"dev_cv_ensemble_split_dict_{which_dataset}.json"), 'w') as f:
                    json.dump(saveable_dev_cv_out_dict, f)

            full_results = {}
            trained_metalearners = {}
            split_test_score_dfs = []
            split_holdout_score_dfs = []


            y_unseen = tbr.get_reg_from_df(unseen_pred_df, pcs_dir=internal_folder, questionnaires_only=True)
            y_bin_unseen = fu.get_y_from_df(unseen_pred_df)
            for splitdx, (ival_idx, holdout_idx) in enumerate(holdout_cv.split(unseen_pred_df, y_bin_unseen)):
                unseen_ival_pred_df = unseen_pred_df.iloc[ival_idx]
                unseen_holdout_pred_df = unseen_pred_df.iloc[holdout_idx]
                groups_ival = unseen_ival_pred_df.index
                groups_holdout = unseen_holdout_pred_df.index
                assert len(set(groups_ival).intersection(set(groups_holdout))) == 0
                # dev_select_pred_df, ival_select_pred_df, holdout_select_pred_df, selected_cols, col_dict = select_base_models(dev_pred_df, ival_pred_df, holdout_pred_df)

                dev_ival_pred_df = pd.concat([dev_test_pred_df, unseen_ival_pred_df], axis=0)

                print(f"Fitting metamodels on dev logo +ival predictions")
                if to_select_base_models:
                    select_dev_pred_df, select_ival_pred_df, unseen_holdout_pred_df, selected_cols, col_dict = select_base_models(dev_test_pred_df, unseen_ival_pred_df, unseen_holdout_pred_df)
                    dev_ival_pred_df = pd.concat([select_dev_pred_df, select_ival_pred_df], axis=0)

                fitted_metamodels, meta_out_dict = train_reg_ensemble_on_preds(dev_ival_pred_df, cv=n_train_cv)

                # evaluate the model on the ival set
                split_results = {
                    'ival_idx': ival_idx,
                    'holdout_idx': holdout_idx,
                }

                split_savepath = os.path.join(new_savepath, f"split{splitdx}")
                if not os.path.exists(split_savepath):
                    os.makedirs(split_savepath)

                
                # get the holdout predictions
                print(f"Getting holdout predictions")
                split_holdout_score_pred_dict, split_holdout_score_df = test_reg_model_on_unseen_data(fitted_metamodels, unseen_holdout_pred_df, metalearners=metalearners)
                split_results['holdout_results'] = split_holdout_score_pred_dict    
                split_holdout_score_df.to_csv(os.path.join(split_savepath, f"split{splitdx}_holdout_score_df_{which_dataset}.csv"))
            

                # save the metalearner models
                for metalearner in metalearners:
                    joblib.dump(fitted_metamodels[f'metalearner_{metalearner}'], os.path.join(split_savepath, f"metalearner_{metalearner}_{splitdx}_{which_dataset}.joblib"))

                full_results[f"split{splitdx}"] = split_results
                trained_metalearners[f"split{splitdx}"] = fitted_metamodels

                split_holdout_score_dfs.append(split_holdout_score_df)
                time.sleep(0)

            holdout_score_df = fepr.score_split_dfs(split_holdout_score_dfs)

            all_score_df = holdout_score_df
            all_score_df.columns = [f"Holdout {col}" for col in holdout_score_df.columns]
        # json to save the results
        print(f"Saving results to {new_savepath}")
        full_results = du.make_dict_saveable(full_results)
        with open(os.path.join(new_savepath, f"full_results_{which_dataset}.json"), 'w') as f:
            json.dump(full_results, f)

        # test_score_df.to_csv(os.path.join(new_savepath, f"test_score_df_{which_dataset}.csv"))
        # holdout_score_df.to_csv(os.path.join(new_savepath, f"holdout_score_df_{which_dataset}.csv"))
        # unseen_score_df.to_csv(os.path.join(new_savepath, f"unseen_score_df_{which_dataset}.csv"))
        all_score_df.to_csv(os.path.join(new_savepath, f"all_score_df_{which_dataset}.csv"))

        all_outs = {
        'full_results': full_results,
        'dev_test_pred_df': dev_test_pred_df,
        'dev_train_pred_df': dev_train_pred_df,
        'unseen_pred_df': unseen_pred_df,
        'all_score_df': all_score_df,
        'metamodels': fitted_metamodels,

        }
        return all_outs

    else:
        # all_outs  = load_ensemble_results(savepath=savepath, which_dataset=which_dataset, n_splits=n_splits, metalearners=metalearners)
        # full_results = all_outs['full_results']
        # dev_test_pred_df = all_outs['dev_test_pred_df']
        # dev_train_pred_df = all_outs['dev_train_pred_df']
        # unseen_pred_df = all_outs['unseen_pred_df']
        # all_score_df = all_outs['all_score_df']
        # fitted_metamodels = all_outs['metamodels']
        print(f"Loading results from {new_savepath}... in the future")
        return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--which_dataset", type=str, default="eeg")
    parser.add_argument("--savepath", type=str, default='/shared/roy/mTBI/mTBI_Classification/cv_results/final_reg_results/')
    parser.add_argument("--results_table_path", type=str, default='data/tables/')
    parser.add_argument("--internal_folder", type=str, default='data/internal/')
    parser.add_argument("--metalearners", type=str, default='rf,lr,xgb')
    parser.add_argument("--reload_results", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dev_eval_cv", type=int, default=5)
    parser.add_argument("--ival_eval_cv", type=int, default=5)
    parser.add_argument("--n_train_cv", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=5)
    parser.add_argument("--to_select_base_models", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(n_splits=args.n_splits, which_dataset=args.which_dataset, savepath=args.savepath, results_table_path=args.results_table_path, internal_folder=args.internal_folder, metalearners=args.metalearners.split(','), reload_results=args.reload_results, dev_eval_cv=args.dev_eval_cv, ival_eval_cv=args.ival_eval_cv, n_train_cv=args.n_train_cv, n_jobs=args.n_jobs, to_select_base_models=args.to_select_base_models)