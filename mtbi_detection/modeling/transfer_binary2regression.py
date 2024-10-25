"""
Given the models selected in model_selection.py, use the selected features and equivalent multiregressor model to predict the continuous outcomes

# models:
LinearRegression (etc)
KNeighbors
RandomForestRegression
XGBoost

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import joblib
import json
import sklearn
import argparse
from scipy import stats
import scipy
from xgboost import XGBClassifier, XGBRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import src.models.model_selection as ms
import src.models.model_analysis as ma

DATASETS_NAME_DICT = {
    "eeg": "eeg",
    "ecgsym": "ecg_symptoms",
    "eegsym": "eeg_symptoms",
    "eegecg": "eeg_ecg",
    "eegecgsym": "eeg_ecg_symptoms",
    "ecg": "ecg",
    "sym": "symptoms",
}

MODEL_NAME_DICT = {
    "xgb": "XGBClassifier",
    "rf": "RandomForestClassifier",
    "knn": "KNeighborsClassifier",
    # "svc": "SVC",
    "gnb": "GaussianNB",
    "ada": "AdaBoost",
    "lre": "LogisticRegression",
}

class NaiveAverageRegressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        assert type(y) == pd.DataFrame
        self.y = y.mean(axis=0)
        return self

    def predict(self, X):
        return pd.DataFrame([self.y for _ in range(X.shape[0])], columns=self.y.index).values
    
def main(results_folder='data/tables/', savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results/',  scores = ['r2', 'mae', 'mse', 'pearsonr', 'spearmanr'], n_jobs=10, which_results='new_split'):
    # load the selected model information
    st = time.time()
    selected_feature_json = ma.load_selected_features(which_results=which_results)
    results_comparison = ms.load_results_comparison(results_folder=results_folder, which_results=which_results)
    candidate_results = process_results(results_comparison, selected_feature_json)
    # for each model, load the training and testing data and train and test using the found hyperparameters ... can use load_model from model_selection.py
    regression_results = convert_binary2regression(candidate_results, selected_feature_json, n_jobs=n_jobs, savepath=savepath, scores=scores)
    
    regression_df = convert_regression_results_to_df(regression_results, candidate_results)

    # save the results
    regression_df.to_csv(os.path.join(results_folder, f'{which_results}_regression_results.csv'))
    print(f"Finished processing in {time.time() - st:.2f} seconds")
    return regression_df

def convert_binary2regression(candidate_results, selected_feature_json, n_jobs=10, scores = ['r2', 'mae', 'mse', 'pearsonr', 'spearmanr'], savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results/'):
    # for each model, load the training and testing data and train and test using the found hyperparameters ... can use load_model from model_selection.py
    regression_results = Parallel(n_jobs=n_jobs)(delayed(_bin2reg)(row, selected_feature_json, scores, savepath) for i, row in candidate_results.iterrows())
    return regression_results

def _bin2reg(row, selected_feature_json, scores = ['r2', 'mae', 'mse', 'pearsonr', 'spearmanr'], savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results/'):
    print(f"Processing {row['model_name']} on {row['filename']}")
    st = time.time()
    # load the model
    model_name = row['model_name']
    model, X_train, X_test = ms.load_model_data(row['filename'], load_model=True)
    # load the selected features
    # selected_features = selected_feature_json[row['dataset']][row['filename']]['selected_features']
    X_train = ma.get_transformed_data(X_train, model)
    X_test = ma.get_transformed_data(X_test, model)
    # X_train = X_train[selected_features]
    # X_test = X_test[selected_features]
    y_train = get_reg_from_df(X_train)
    y_test = get_reg_from_df(X_test)
    reg_model = binary_model2regression_model(model, model_name)
    # train and test the model
    reg_model.fit(X_train, y_train)
    naive_avg_reg = NaiveAverageRegressor()
    naive_avg_reg.fit(X_train, y_train)

    reg_test_result = score_multiregression_model(reg_model, X_test, y_test, y_cols=y_train.columns, scores=scores, verbose=True)
    reg_train_result = score_multiregression_model(reg_model, X_train, y_train, y_cols=y_train.columns, scores=scores, verbose=False)
    naive_avg_test_result = score_multiregression_model(naive_avg_reg, X_test, y_test, y_cols=y_train.columns, scores=scores, verbose=False)
    naive_avg_train_result = score_multiregression_model(naive_avg_reg, X_train, y_train, y_cols=y_train.columns, scores=scores, verbose=False)
    savefilename = save_reg_model(reg_model, row['filename'], reg_test_result, savepath=savepath)
    
    print(f"Finished processing {row['model_name']} on {row['filename']} in {time.time() - st:.2f} seconds")
    return {'train_result': reg_train_result, 'test_result': reg_test_result, 'naive_test_result': naive_avg_test_result, 'naive_train_result': naive_avg_train_result, 'savefilename': savefilename}

def convert_regression_results_to_df(regression_results, candidate_results, scores = ['r2', 'mae', 'mse', 'pearsonr', 'spearmanr']):
    # convert the results to a dataframe
    cols = regression_results[0]['train_result'][scores[0]].keys()
    all_cols = [f"{score}_{col}" for score in scores for col in cols] + [f"avg_{score}" for score in scores]
    
    train_results_dict = {f"train_{col}": [] for col in all_cols}
    avg_train_results_dict = {f"train_avg_{col}": [] for col in scores}
    train_results_dict.update(avg_train_results_dict)
    
    test_results_dict = {f"test_{col}": [] for col in all_cols}
    avg_test_results_dict = {f"test_avg_{col}": [] for col in scores}
    test_results_dict.update(avg_test_results_dict)

    naive_train_results_dict = {f"naive_train_{col}": [] for col in all_cols}
    naive_avg_train_results_dict = {f"naive_train_avg_{col}": [] for col in scores}
    naive_train_results_dict.update(naive_avg_train_results_dict)

    naive_test_results_dict = {f"naive_test_{col}": [] for col in all_cols}
    naive_avg_test_results_dict = {f"naive_test_avg_{col}": [] for col in scores}
    naive_test_results_dict.update(naive_avg_test_results_dict)

    filenames = []
    model_names = []
    dataset = []
    for rdx, result in enumerate(regression_results):
        for score in scores:
            for col in cols:
                train_results_dict[f"train_{score}_{col}"].append(result['train_result'][score][col])
                test_results_dict[f"test_{score}_{col}"].append(result['test_result'][score][col])

                naive_train_results_dict[f"naive_train_{score}_{col}"].append(result['naive_train_result'][score][col])
                naive_test_results_dict[f"naive_test_{score}_{col}"].append(result['naive_test_result'][score][col])
       
            train_results_dict[f"train_avg_{score}"].append(np.mean([result['train_result'][score][col] for col in cols]))
            test_results_dict[f"test_avg_{score}"].append(np.mean([result['test_result'][score][col] for col in cols]))
            naive_avg_test_results_dict[f"naive_test_avg_{score}"].append(np.mean([result['naive_test_result'][score][col] for col in cols]))
            naive_avg_train_results_dict[f"naive_train_avg_{score}"].append(np.mean([result['naive_train_result'][score][col] for col in cols]))


        filenames.append(result['savefilename'])
        model_name = candidate_results[candidate_results['filename'].apply(lambda x: x.split('/')[-1]) == result['savefilename'].split(f'_regmodel')[0].replace('zq', '/').split('/')[-1] + '.json']['model_name'].values[0]
        assert model_name is not None and candidate_results.iloc[rdx]['model_name'] == model_name, f"model_name: {model_name}, candidate_results.iloc[rdx]['model_name']: {candidate_results.iloc[rdx]['model_name']}"
        model_names.append(model_name)
        dataset.append(candidate_results.iloc[rdx]['dataset'])

    train_results_df = pd.DataFrame(train_results_dict)
    test_results_df = pd.DataFrame(test_results_dict)
    naive_train_results_df = pd.DataFrame(naive_train_results_dict)
    naive_test_results_df = pd.DataFrame(naive_test_results_dict)

    preamble_candidate_df = candidate_results.iloc[:, :5]
    #drop the index
    preamble_candidate_df = preamble_candidate_df.reset_index(drop=True)

    assert all(preamble_candidate_df['model_name'].values == model_names), f"preamble_candidate_df['model_name'].values: {preamble_candidate_df['model_name'].values}, model_names: {model_names}"
    regression_df = pd.concat([preamble_candidate_df, pd.DataFrame({'savefilename': filenames}), train_results_df,  naive_train_results_df, naive_test_results_df, test_results_df], axis=1)
    return regression_df

def save_reg_model(model, filename, test_result, savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results/'):
    metrics = test_result.keys()
    avg_metrics = {metric: np.mean([test_result[metric][col] for col in test_result[metric].keys()]) for metric in metrics}
    metric_insert = '_'.join([f"{key}{value*100:.0f}" for key, value in avg_metrics.items()])
    savefilename = os.path.join(savepath, f"{filename.replace('.json', '').replace('/', 'zq')}_regmodel_{metric_insert}regression_model.joblib")
    joblib.dump(model, savefilename)
    return savefilename

def load_reg_model(filename, savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results/', binarymodel_datapath='/shared/roy/mTBI/mTBI_Classification/cv_results/'):
    """
    Load the regression model and X_train, X_test, y_train, y_test
    """
    model = joblib.load(filename)
    return model

def load_X_from_regfile(filename):
    """
    Load the regression model and X_train, X_test, y_train, y_test
    """
    binfilename = '/'+filename.split('/zq')[-1].replace('zq', '/')
    binfilename = binfilename.split('_regmodel')[0] + '.json'
    model, X_train, X_test = ms.load_model_data(binfilename, load_model=True)
    X_train = ma.get_transformed_data(X_train, model)
    X_test = ma.get_transformed_data(X_test, model)
    return model, X_train, X_test

def score_multiregression_model(model, X_test, y_test, y_cols=None, scores = ['r2', 'mae', 'mse', 'pearsonr', 'spearmanr'], verbose=True):
    """ Given a fitted multiregression model, score the model on other data"""

    if type(y_test) == pd.DataFrame:
        pass
    elif type(y_test) == np.ndarray:
        y_test = pd.DataFrame(y_test, columns=y_cols)
    else:
        raise ValueError(f"y_test must be a DataFrame or a numpy array")
    results = {}
    results = {score: [] for score in scores}
    y_pred = model.predict(X_test)
    results['r2'] = [sklearn.metrics.r2_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y_test.columns)]
    results['mae'] = [sklearn.metrics.mean_absolute_error(y_test[col], y_pred[:, i]) for i, col in enumerate(y_test.columns)]
    results['mse'] = [sklearn.metrics.mean_squared_error(y_test[col], y_pred[:, i]) for i, col in enumerate(y_test.columns)]
    # pearsonr
    results['pearsonr'] = [scipy.stats.pearsonr(y_test[col], y_pred[:, i])[0] for i, col in enumerate(y_test.columns)]
    # spearmanr
    results['spearmanr'] = [scipy.stats.spearmanr(y_test[col], y_pred[:, i])[0] for i, col in enumerate(y_test.columns)]
    assert all([key in results.keys() for key in scores]) and all([key in scores for key in results.keys()])
    if verbose:
        print(f"Result for each target:")
        for cdx, col in enumerate(y_test.columns):
            print(f'\t{col}')
            for metric in scores:
                print(f'\t\t{metric}: {results[metric][cdx]}')
        
        print(f"Score across all targets:")
        for metric in scores:
            print(f'\t{metric}: {np.mean(results[metric])} (std: {np.std(results[metric])}), median: {np.median(results[metric])}, max: {np.max(results[metric])}, min: {np.min(results[metric])}, iqr: {np.percentile(results[metric], 75) - np.percentile(results[metric], 25)}')
   
    for score in scores:
        results[score] = {col: results[score][cdx] for cdx, col in enumerate(y_test.columns)}
        
    return results

def binary_model2regression_model(model, model_name):
    """ Given a model of the form HyperCV with the final classifier at model.best_estimator_.named_steps['classifier'], return the equivalent regression model
    Input:
    model: HyperCV model
    Output:
    reg_model: HyperCV model
    
    """
    candidate_models = {'XGBClassifier': XGBRegressor(), 'RandomForestClassifier': RandomForestRegressor(), 'KNeighborsClassifier': KNeighborsRegressor(), 'LogisticRegression': LinearRegression()}    
    reg_model = candidate_models[model_name]
    model_hyperparams = model.best_params_
    cls_hyperparams = {key.split('__')[1]: value for key, value in model_hyperparams.items() if 'classifier__' in key}
    try:
        reg_model.set_params(**cls_hyperparams)
    except:
        # one by one find the hyperparameters that can be applied
        valid_hyperparams = {}
        for key, value in cls_hyperparams.items():
            try:
                
                reg_model.set_params(**{key: value})
                valid_hyperparams[key] = value
                print(f"Confirmed {key}: {value} is a valid hyperparameter for {model_name}")
            except:
                print(f"Failed to set {key}: {value} as a valid hyperparameter for {model_name}")
        reg_model.set_params(**valid_hyperparams)
    print(f"Successfully set the hyperparameters for {model_name} to {reg_model.get_params()}")
    return reg_model

def get_reg_from_df(df, pcs_dir='data/internal/', questionnaires_only=False, exclude_goat=True):
    reg = pd.read_csv(os.path.join(pcs_dir, 'pcs_targets.csv'), index_col=0)
    reg = reg.fillna(0)
    baseline_score = reg[[col for col in reg.columns if 'Baseline' in col and 'Score' in col]]
    out_reg = baseline_score.loc[[int(ind) for ind in df.index]]
    if questionnaires_only:
        out_reg = out_reg[[col for col in out_reg.columns if 'Duration' not in col]]
    if exclude_goat:
        out_reg = out_reg[[col for col in out_reg.columns if 'GOAT' not in col]]
    assert np.all(out_reg.index == df.index)
    return out_reg

def get_reg_from_subj(subj, pcs_dir='data/internal/', questionnaires_only=False, exclude_goat=True):
    reg = pd.read_csv(os.path.join(pcs_dir, 'pcs_targets.csv'), index_col=0)
    reg = reg.fillna(0)
    baseline_score = reg[[col for col in reg.columns if 'Baseline' in col and 'Score' in col]]
    out_reg = baseline_score.loc[[int(subj)]]
    if questionnaires_only:
        out_reg = out_reg[[col for col in out_reg.columns if 'Duration' not in col]]
    if exclude_goat:
        out_reg = out_reg[[col for col in out_reg.columns if 'GOAT' not in col]]
    out_reg = out_reg.loc[int(subj)]
    return out_reg

def process_results(results_comparison, selected_feature_json):
    midp_selected = results_comparison[results_comparison['midp'] > 0.05]
    # plot the average of 'search_score' and 'ival_mcc' against the geometric mean of 'search_score' and 'ival_mcc'
    avg_score = results_comparison[['search_score', 'ival_mcc']].mean(axis=1)
    geom_mean = np.sqrt((results_comparison['search_score'] +1) * (results_comparison['ival_mcc']+1))-1
    num_selected_features = [len(selected_feature_json[row['dataset']][row['filename']]['selected_features']) for i, row in midp_selected.iterrows()]
    midp_avg_score = midp_selected[['search_score', 'ival_mcc']].mean(axis=1)
    midp_geom_mean = np.sqrt((midp_selected['search_score'] +1) * (midp_selected['ival_mcc']+1))-1

    midp_selected_results = pd.concat([midp_selected, pd.DataFrame({'midp_avg_score': midp_avg_score, 'midp_geom_mean': midp_geom_mean, 'num_selected_features': num_selected_features})], axis=1)
    nosym_results = midp_selected_results.loc[~midp_selected_results['dataset'].str.contains('sym')]
    nosym_results.sort_values(by='search_score', ascending=False)

    # only get the models that are the counterpart of multiregressors: LinearRegression, KNeighbors, RandomForestRegression, XGBoost
    candidate_models = {'XGBClassifier': 'XGBRegressor', 'RandomForestClassifier': 'RandomForestRegressor', 'KNeighborsClassifier': 'KNeighborsRegressor', 'LogisticRegression': 'LinearRegression'}
    candidate_results = nosym_results[nosym_results['model_name'].isin(candidate_models.keys())]
    return candidate_results

def load_tbr_results(results_folder='data/tables/', which_results='new_split'):
     tbr_results = pd.read_csv(os.path.join(results_folder, f'{which_results}_regression_results.csv'), index_col=0)
     return tbr_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer binary classification models to regression models')
    parser.add_argument('--results_folder', type=str, default='data/tables/', help='The folder where the results are stored')
    parser.add_argument('--savepath', type=str, default='/shared/roy/mTBI/mTBI_Classification/cv_results/reg_results_new_splits/', help='The folder where the regression models will be saved')
    parser.add_argument('--scores', type=str, nargs='+', default=['r2', 'mae', 'mse', 'pearsonr', 'spearmanr'], help='The scores to be used to evaluate the regression models')
    parser.add_argument('--n_jobs', type=int, default=1, help='The number of jobs to be used in parallel')
    parser.add_argument('--which_results', type=str, default='new_split', help='The results to be used')
    args = parser.parse_args()
    main(results_folder=args.results_folder, savepath=args.savepath, scores=args.scores, n_jobs=args.n_jobs, which_results=args.which_results)