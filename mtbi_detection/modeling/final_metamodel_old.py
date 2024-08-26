import numpy as np
import pandas as pd
import os   
import time
import sklearn
import skopt
import mlxtend
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
font_path = '/home/ap60/inter.ttf'
font_prop = fm.FontProperties(fname=font_path, size=20)
import argparse
import sklearn

import src.features.feature_utils as fu
from src.features.feature_utils import DataFrameImputer, DropDuplicatedAndConstantColumns, DropInfColumns, DropNanColumns, DropNanRows, UnivariateThresholdSelector, anova_f, anova_pinv, pearson_corr, spearman_corr, kendall_corr
import src.models.model_selection as ms
import src.models.model_analysis as ma
import src.models.eval_model as em
import src.data.data_utils as du
import src.data.load_dataset as ld



def return_train_test_preds(df, loaded_model_data=None):
    """
    Go through a dataframe and load the train and test prediction probabilities into a dataframe   
    The dataframe contains the filename needed to load the models 
    """
    train_preds = []
    test_preds = []
    return_model_data = False
    if loaded_model_data is None:
        loaded_model_data = []
        return_model_data = True
    for idx, (i, row) in enumerate(df.iterrows()):
        print(f"Loading model {idx+1}/{df.shape[0]}")
        json_filename = row['filename']
        if not return_model_data:
            model, Xtr, Xts = loaded_model_data[idx]
        else:
            model, Xtr, Xts = ms.load_model_data(json_filename)
        assert len(set(Xtr.index).intersection(set(Xts.index))) == 0, f"Index overlap between training and test data: {set(Xtr.index).intersection(set(Xts.index))}"
        ytr = model.predict_proba(Xtr)
        yts = model.predict_proba(Xts)
        train_preds.append(ytr)
        test_preds.append(yts)
        if return_model_data:
            loaded_model_data.append((model, Xtr, Xts))
        
    assert all([all(loaded_model_data[0][1].index == loaded_model_data[i][1].index) for i in range(1, len(loaded_model_data))])
    assert all([all(loaded_model_data[0][2].index == loaded_model_data[i][2].index) for i in range(1, len(loaded_model_data))])
    
    train_preds = np.concatenate(train_preds, axis=1)
    test_preds = np.concatenate(test_preds, axis=1)
    train_pred_df = pd.DataFrame(train_preds, columns=[f"{filename.split('/')[-1]}_{idx}" for filename in df['filename'] for idx in range(2)], index=loaded_model_data[0][1].index)
    test_pred_df = pd.DataFrame(test_preds, columns=[f"{filename.split('/')[-1]}_{idx}" for filename in df['filename'] for idx in range(2)], index=loaded_model_data[0][2].index)
    if return_model_data:
        return train_pred_df, test_pred_df, loaded_model_data
    else:
        return train_pred_df, test_pred_df

def return_cv_preds(df, loaded_model_data=None):
    """
    Go through a dataframe and load the train and test prediction probabilities into a dataframe   
    The dataframe contains the filename needed to load the models 
    """
    train_preds = {}
    test_preds = {}
    return_model_data = False
    clf_names = []
    if loaded_model_data is None:
        loaded_model_data = []
        return_model_data = True
    for idx, (i, row) in enumerate(df.iterrows()):
        print(f"Loading model {idx+1}/{df.shape[0]}")
        json_filename = row['filename']
        if not return_model_data:
            model, Xtr, _ = ms.load_model_data(json_filename)
        else:
            model, Xtr, _ = loaded_model_data[idx]
        clf_name = str(model.best_estimator_.named_steps['classifier'].__class__.__name__)
        clf_names.append(clf_name)
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_proctr = ma.get_transformed_data(Xtr, model, verbose=False)
        clf = model.best_estimator_.named_steps['classifier']
        testing_preds = np.empty((X_proctr.shape[0], 2))# = cross_val_predict(clf, X_proctr, fu.get_y_from_df(Xtr), cv=cv, method='predict_proba')
        training_preds = np.empty((X_proctr.shape[0], 2*5))
        # fill with -1
        training_preds.fill(-1)
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for split, (train_idx, test_idx) in enumerate(cv.split(X_proctr, fu.get_y_from_df(Xtr))):
            X_train, X_test = X_proctr.iloc[train_idx], X_proctr.iloc[test_idx]
            y_train = fu.get_y_from_df(Xtr)[train_idx]
            clf.fit(X_train, y_train)
            testing_preds[test_idx, :] = clf.predict_proba(X_test)
            training_preds[train_idx, 2*split:2*(split+1)] = clf.predict_proba(X_train)
        testing_preds = pd.DataFrame(testing_preds, columns=[f"class{idx}" for idx in range(2)], index=X_proctr.index)
        training_preds = pd.DataFrame(training_preds, columns=[f"split{k}_class{idx}" for k in range(5) for idx in range(2)], index=X_proctr.index)
        train_preds[clf_name] = training_preds
        test_preds[clf_name] = testing_preds
    assert all([all(loaded_model_data[0][1].index == loaded_model_data[i][1].index) for i in range(1, len(loaded_model_data))])
    assert all([all(loaded_model_data[0][2].index == loaded_model_data[i][2].index) for i in range(1, len(loaded_model_data))])
    
    if return_model_data:
        return train_preds, test_preds, loaded_model_data, clf_names
    else:
        return train_preds, test_preds, clf_names
    
def get_ensemble_cv_preds(cv_train_preds, cv_test_preds, clf_names, scores=['matthews_corrcoef', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity']):

    n_cv = len(cv_train_preds[clf_names[0]].columns)//2

    split_dict = {'cv_results': {}}
    for cv in range(n_cv): 
        print(f"Running split {cv+1}/{n_cv}")
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        lr = sklearn.linear_model.LogisticRegression(random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=1000)
        lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
        rf_grid = {'max_depth': [2, 3, 5, 7, 11, None], 'min_samples_leaf':[1,2,4,8], 'max_leaf_nodes': [2, 3, 5, 11, 13, 17, 24, 32, 64, None]}
        metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring='matthews_corrcoef', n_jobs=5, cv=n_cv)
        metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring='matthews_corrcoef', n_jobs=5, cv=n_cv)

        train_preds_concat = []
        test_preds_concat = []
        for clf_name in clf_names:
            train_df = cv_train_preds[clf_name][[f"split{cv}_class{idx}" for idx in range(2)]]
            # now drop all rows with -1 and store the index of the dropped rows
            test_idx = train_df[train_df[f'split{cv}_class0'] == -1].index
            train_df = train_df.drop(test_idx)
            train_preds_concat.append(train_df)
            test_df = cv_test_preds[clf_name].loc[test_idx]
            test_preds_concat.append(test_df)
        
        train_common_idx = list(set(train_preds_concat[0].index).intersection(*[set(df.index) for df in train_preds_concat[1:]]))
        test_common_idx = list(set(test_preds_concat[0].index).intersection(*[set(df.index) for df in test_preds_concat[1:]]))
        train_preds_concat = [df.loc[train_common_idx].values for df in train_preds_concat]
        test_preds_concat = [df.loc[test_common_idx].values for df in test_preds_concat]
                            
        train_preds_concat = np.concatenate(train_preds_concat, axis=1)
        test_preds_concat = np.concatenate(test_preds_concat, axis=1)
        train_pred_df = pd.DataFrame(train_preds_concat, columns=[f"{clf_name}_{idx}" for clf_name in clf_names for idx in range(2)], index=train_common_idx)
        test_pred_df = pd.DataFrame(test_preds_concat, columns=[f"{clf_name}_{idx}" for clf_name in clf_names for idx in range(2)], index=test_common_idx)
        print(f"Training metalearner on split {cv+1}/{n_cv}", end='\r')
        metalearner_rf.fit(train_pred_df, fu.get_y_from_df(train_pred_df))
        print(f"Training metalearner on split {cv+1}/{n_cv}", end='\r')
        metalearner_lr.fit(train_pred_df, fu.get_y_from_df(train_pred_df))

        best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(cv)]
        best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(cv)]

        metalearner_rf_pred = metalearner_rf.predict_proba(test_pred_df)
        metalearner_lr_pred = metalearner_lr.predict_proba(test_pred_df)
        true_y = fu.get_y_from_df(test_pred_df)

        split_dict['cv_results'][f'split{cv}'] = {
            'metalearner_rf': metalearner_rf,
            'metalearner_lr': metalearner_lr,
            'best_scores_rf': best_scores_rf,
            'best_scores_lr': best_scores_lr,
            'metalearner_rf_pred': metalearner_rf_pred,
            'metalearner_lr_pred': metalearner_lr_pred,
        }
        for score in scores:
            if score == 'specificity':
                rf_pred = np.argmax(metalearner_rf_pred, axis=1)
                lr_pred = np.argmax(metalearner_lr_pred, axis=1)
                metric_score_rf = sklearn.metrics.recall_score(true_y, rf_pred, pos_label=0)
                metric_score_lr = sklearn.metrics.recall_score(true_y, lr_pred, pos_label=0)
            elif score == 'sensitivity':
                rf_pred = np.argmax(metalearner_rf_pred, axis=1)
                lr_pred = np.argmax(metalearner_lr_pred, axis=1)
                metric_score_rf = sklearn.metrics.recall_score(true_y, rf_pred, pos_label=1)
                metric_score_lr = sklearn.metrics.recall_score(true_y, lr_pred, pos_label=1)
            elif score == 'balanced_accuracy':
                rf_pred = np.argmax(metalearner_rf_pred, axis=1)
                lr_pred = np.argmax(metalearner_lr_pred, axis=1)
                metric_score_rf= sklearn.metrics.balanced_accuracy_score(true_y, rf_pred)
                metric_score_lr= sklearn.metrics.balanced_accuracy_score(true_y, lr_pred)
            elif score == 'roc_auc':
                rf_pred = metalearner_rf_pred
                lr_pred = metalearner_lr_pred
                metric_score_rf = sklearn.metrics.roc_auc_score(true_y, rf_pred[:,1])
                metric_score_lr = sklearn.metrics.roc_auc_score(true_y, lr_pred[:,1])
            elif score == 'matthews_corrcoef':
                rf_pred = np.argmax(metalearner_rf_pred, axis=1)
                lr_pred = np.argmax(metalearner_lr_pred, axis=1)
                metric_score_rf = sklearn.metrics.matthews_corrcoef(true_y, rf_pred)
                metric_score_lr = sklearn.metrics.matthews_corrcoef(true_y, lr_pred)
            else:
                raise ValueError(f"Score {score} not implemented")
                # scorer = sklearn.metrics.get_scorer(score)
                # rf_pred = metalearner_rf_pred
                # lr_pred = metalearner_lr_pred
                # metric_score = scorer(true_y, rf_pred) # won't work since scorer takes clf, X, y
            split_dict['cv_results'][f'split{cv}'][f'test_scores_rf_{score}'] = metric_score_rf
            split_dict['cv_results'][f'split{cv}'][f'test_scores_lr_{score}'] = metric_score_lr

    for score in scores:
        split_dict[f'mean_test_scores_rf_{score}'] = np.mean([split_dict['cv_results'][f'split{cv}'][f'test_scores_rf_{score}'] for cv in range(n_cv)])
        split_dict[f'mean_test_scores_lr_{score}'] = np.mean([split_dict['cv_results'][f'split{cv}'][f'test_scores_lr_{score}'] for cv in range(n_cv)])
        split_dict[f'std_test_scores_rf_{score}'] = np.std([split_dict['cv_results'][f'split{cv}'][f'test_scores_rf_{score}'] for cv in range(n_cv)])
        split_dict[f'std_test_scores_lr_{score}'] = np.std([split_dict['cv_results'][f'split{cv}'][f'test_scores_lr_{score}'] for cv in range(n_cv)])
    return split_dict

def get_holdout_preds(df, all_model_data, which_dataset='eeg_ecg', internal_folder='data/internal/', tables_folder='data/tables/'):
        
    holdouts = [em.load_holdout_data(json_filename=f, dataset=which_dataset, base_folder=tables_folder, internal_folder=internal_folder)[0] for f in df['filename']]
    filled_holdouts = []
    holdout_preds = []
    for X_holdout, (model, Xtr, Xts) in zip(holdouts, all_model_data):
        X = X_holdout.replace([np.inf, -np.inf], np.nan)
        assert len(set(X.index).intersection(set(Xtr.index))) == 0, f"Index overlap between holdout and training data: {set(X.index).intersection(set(Xtr.index))}"
        assert len(set(X.index).intersection(set(Xts.index))) == 0, f"Index overlap between holdout and test data: {set(X.index).intersection(set(Xts.index))}"
        assert len(set(Xtr.index).intersection(set(Xts.index))) == 0, f"Index overlap between training and test data: {set(Xtr.index).intersection(set(Xts.index))}"
        filled_holdouts.append(X)
        y_pred = model.predict_proba(X)
        holdout_preds.append(y_pred)
    assert all([all(filled_holdouts[0].index == filled_holdouts[i].index) for i in range(1, len(filled_holdouts))])
    holdout_preds = np.concatenate(holdout_preds, axis=1)
    holdout_pred_df = pd.DataFrame(holdout_preds, columns=[f"{filename.split('/')[-1]}_{idx}" for filename in df['filename'] for idx in range(2)], index=holdouts[0].index)
    return holdout_pred_df, filled_holdouts

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

def store_cv_results(base_model_results, cv_ensemble_split_dict):

    base_split_df = pd.DataFrame(base_model_results['model_splits'])
    rf_best_scores = [cv_ensemble_split_dict['cv_results'][f'split{k}']['test_scores_rf_matthews_corrcoef'] for k in range(5)]
    lr_best_scores = [cv_ensemble_split_dict['cv_results'][f'split{k}']['test_scores_lr_matthews_corrcoef'] for k in range(5)]
        
    out_dict = {
        'model_splits': base_model_results['model_splits'],
        'model_best_scores': base_model_results['best_scores'],
        'model_best_stds': base_model_results['best_stds'],
        'model_names': base_model_results['clf_names'],
        'model_mean_best_score': base_model_results['avg_best_score'],
        'model_pooled_std': base_model_results['pooled_std'],
        'metarf_best_scores': rf_best_scores,
        'metalr_best_scores': lr_best_scores,
    }
    mean_std_df = pd.DataFrame({
        'mean': [np.mean(ress) for ress in [rf_best_scores, lr_best_scores]] + [base_model_results['avg_best_score']] + base_split_df.mean(axis=0).tolist(),
        'std': [np.std(ress) for ress in [rf_best_scores, lr_best_scores]] + [base_model_results['pooled_std']] + base_split_df.std(axis=0).tolist()
    }, index=[f"Random Forest Metalearner", f"Logistic Regression Metalearner", f"Average of selected models"] + base_split_df.columns.tolist())
    
    return out_dict, mean_std_df
def plot_results(results, cv_ensemble_split_dict=None, include_average=True, fontsize=20, figsize=(10, 4), title="Training Set CV Scores for Each Classifier"):
    # Create a DataFrame for easier plotting
    df_splits = pd.DataFrame(results['model_splits'])

    # Calculate the mean scores and sort the DataFrame by them in descending order
    mean_scores = df_splits.mean().sort_values(ascending=False)
    df_splits = df_splits[mean_scores.index]

    # Convert the list of best scores into a DataFrame
    if cv_ensemble_split_dict is not None:
        best_scores = results['best_scores'].copy()
        # n_add = int(len(best_scores)/0.75)
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

def train_ensemble_on_preds(train_pred_df, cv:int=5):
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    lr = sklearn.linear_model.LogisticRegression(random_state=42, solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=1000)
    lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}
    rf_grid = {'max_depth': [2, 3, 5, 7, 11, None], 'min_samples_leaf':[1,2,4,8], 'max_leaf_nodes': [2, 3, 5, 11, 13, 17, 24, 32, 64, None]}
    metalearner_rf  = sklearn.model_selection.GridSearchCV(rf, rf_grid, scoring='matthews_corrcoef', n_jobs=5, cv=cv)
    metalearner_lr  = sklearn.model_selection.GridSearchCV(lr, lr_grid, scoring='matthews_corrcoef', n_jobs=5, cv=cv)


    metalearner_rf.fit(train_pred_df, fu.get_y_from_df(train_pred_df))
    metalearner_lr.fit(train_pred_df, fu.get_y_from_df(train_pred_df))

    best_scores_rf = [metalearner_rf.cv_results_[f'split{k}_test_score'][metalearner_rf.best_index_] for k in range(cv)]
    best_scores_lr = [metalearner_lr.cv_results_[f'split{k}_test_score'][metalearner_lr.best_index_] for k in range(cv)]

    out_dict = {
        'best_scores_rf': best_scores_rf,
        'best_scores_lr': best_scores_lr,
    }
    return metalearner_rf, metalearner_lr, out_dict

def test_model_on_unseen_data(metalearner_rf, metalearner_lr, test_pred_df):
    metalearner_rf_pred = metalearner_rf.predict_proba(test_pred_df)
    metalearner_lr_pred = metalearner_lr.predict_proba(test_pred_df)
    true_y = fu.get_y_from_df(test_pred_df)
    rf_pred = np.argmax(metalearner_rf_pred, axis=1)
    lr_pred = np.argmax(metalearner_lr_pred, axis=1)

    # mcc
    metric_score_rf = sklearn.metrics.matthews_corrcoef(true_y, rf_pred)
    metric_score_lr = sklearn.metrics.matthews_corrcoef(true_y, lr_pred)

    # roc_auc
    metric_score_rf_roc = sklearn.metrics.roc_auc_score(true_y, metalearner_rf_pred[:,1])
    metric_score_lr_roc = sklearn.metrics.roc_auc_score(true_y, metalearner_lr_pred[:,1])

    # balanced accuracy
    metric_score_rf_balanced = sklearn.metrics.balanced_accuracy_score(true_y, rf_pred)
    metric_score_lr_balanced = sklearn.metrics.balanced_accuracy_score(true_y, lr_pred)

    # sensitivity
    metric_score_rf_sensitivity = sklearn.metrics.recall_score(true_y, rf_pred, pos_label=1)
    metric_score_lr_sensitivity = sklearn.metrics.recall_score(true_y, lr_pred, pos_label=1)

    # specificity
    metric_score_rf_specificity = sklearn.metrics.recall_score(true_y, rf_pred, pos_label=0)
    metric_score_lr_specificity = sklearn.metrics.recall_score(true_y, lr_pred, pos_label=0)

    #roc_curve
    fpr_rf, tpr_rf, thresh_rf = sklearn.metrics.roc_curve(true_y, metalearner_rf_pred[:,1])
    fpr_lr, tpr_lr, thresh_lr = sklearn.metrics.roc_curve(true_y, metalearner_lr_pred[:,1])

    unseen_score_pred_dict = {'rf': {'scores': {
                            'mcc': metric_score_rf,
                            'roc_auc': metric_score_rf_roc,
                            'balanced_accuracy': metric_score_rf_balanced,
                            'sensitivity': metric_score_rf_sensitivity,
                            'specificity': metric_score_rf_specificity,
                            },
                        'roc_curve': {'fpr': fpr_rf, 'tpr': tpr_rf, 'thresh': thresh_rf},
                        'preds': rf_pred,
                        'pred_probs': metalearner_rf_pred
                    },
                'lr': {'scores': {
                            'mcc': metric_score_lr,
                            'roc_auc': metric_score_lr_roc,
                            'balanced_accuracy': metric_score_lr_balanced,
                            'sensitivity': metric_score_lr_sensitivity,
                            'specificity': metric_score_lr_specificity,
                            },
                        'roc_curve': {'fpr': fpr_lr, 'tpr': tpr_lr, 'thresh': thresh_lr},
                        'preds': lr_pred,
                        'pred_probs': metalearner_lr_pred
                    },
    }

    # as a dataframe
    score_df = pd.DataFrame({
        'Random Forest': [metric_score_rf, metric_score_rf_roc, metric_score_rf_balanced, metric_score_rf_sensitivity, metric_score_rf_specificity],
        'Logistic Regression': [metric_score_lr, metric_score_lr_roc, metric_score_lr_balanced, metric_score_lr_sensitivity, metric_score_lr_specificity]
    }, index=['MCC', 'ROC AUC', 'Balanced Accuracy', 'Sensitivity', 'Specificity'])
    return unseen_score_pred_dict, score_df

def load_model_results(results_table_path='data/tables/', dataset='eeg_ecg'):
    

    all_results = ms.load_results_comparison(results_folder=results_table_path)
    selected_feature_json = ma.load_selected_features()
    num_selected_features = [len(selected_feature_json[row['dataset']][row['filename']]['selected_features']) for i, row in all_results.iterrows()]
    all_results['num_selected_features'] = num_selected_features
    all_results = all_results[all_results['midp'] > 0.05]
    dataset_results = all_results[all_results['dataset'] == dataset]
    print(f"Number of models: {dataset_results.shape[0]}")
    return dataset_results


def run_late_ensemble_model(results_table_path='data/tables/', internal_folder='data/internal/', which_dataset='late_eeg-ecg'):
    # loads the eeg and ecg results separately to late fuse them on subjects with both data
    dset1 = which_dataset.split('_')[1]
    dset2 = which_dataset.split('_')[2]
    if len(which_dataset.split('_')) > 3:
        dset3 = which_dataset.split('_')[3]
    if dset1 in ['eeg', 'ecg']:
        dset1_str = dset1.upper()
    else:
        dset1_str = dset1[0].upper() + dset1[1:]
    if dset2 in ['eeg', 'ecg']:
        dset2_str = dset2.upper()
    else:
        dset2_str = dset2[0].upper() + dset2[1:]
    
    use_symptoms = False
    if len(which_dataset.split('_')) > 3 and any(['sym' in dset for dset in which_dataset.split('_')]):
        use_symptoms = True
        if dset3 in ['eeg', 'ecg']:
            dset3_str = dset3.upper()
        else:
            dset3_str = dset3[0].upper() + dset3[1:]
            

    eeg_dataset_results = load_model_results(results_table_path=results_table_path, dataset=dset1)
    ecg_dataset_results = load_model_results(results_table_path=results_table_path, dataset=dset2)
    if use_symptoms:
        symptom_dataset_results = load_model_results(results_table_path=results_table_path, dataset=dset3)

    # get the train and test predictions
    print(f"Getting train and test predictions")
    eeg_train_pred_df, eeg_test_pred_df, eeg_loaded_model_data = return_train_test_preds(eeg_dataset_results)
    ecg_train_pred_df, ecg_test_pred_df, ecg_loaded_model_data = return_train_test_preds(ecg_dataset_results)
    if use_symptoms:
        symptom_train_pred_df, symptom_test_pred_df, symptom_loaded_model_data = return_train_test_preds(symptom_dataset_results)


    # get the cv results 
    print(f"Getting cv predictions")
    eeg_cv_train_preds, eeg_cv_test_preds, eeg_clf_names = return_cv_preds(eeg_dataset_results, eeg_loaded_model_data)
    ecg_cv_train_preds, ecg_cv_test_preds, ecg_clf_names = return_cv_preds(ecg_dataset_results, ecg_loaded_model_data)
    if use_symptoms:
        symptom_cv_train_preds, symptom_cv_test_preds, symptom_clf_names = return_cv_preds(symptom_dataset_results, symptom_loaded_model_data)

    # get the ensemble cv predictions
    print(f"Getting ensemble cv predictions")
    cv_train_preds = {f"{dset1_str} {key}": eeg_cv_train_preds[key] for key in eeg_cv_train_preds.keys()}
    for key in ecg_cv_train_preds.keys():
        cv_train_preds[f"{dset2_str} {key}"] = ecg_cv_train_preds[key]

    cv_test_preds = {f"{dset1_str} {key}": eeg_cv_test_preds[key] for key in eeg_cv_test_preds.keys()}
    for key in ecg_cv_test_preds.keys():
        cv_test_preds[f"{dset2_str} {key}"] = ecg_cv_test_preds[key]

    if use_symptoms:
        for key in symptom_cv_train_preds.keys():
            cv_train_preds[f"{dset3_str} {key}"] = symptom_cv_train_preds[key]
        for key in symptom_cv_test_preds.keys():
            cv_test_preds[f"{dset3_str} {key}"] = symptom_cv_test_preds[key]

    clf_names = list(cv_train_preds.keys())
    cv_ensemble_split_dict = get_ensemble_cv_preds(cv_train_preds, cv_test_preds, clf_names, scores=['matthews_corrcoef', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity'])

    # now store the cv results
    print(f"Storing cv results")
    loaded_model_data = eeg_loaded_model_data + ecg_loaded_model_data
    if use_symptoms:
        loaded_model_data += symptom_loaded_model_data
        train_pred_df = pd.concat([eeg_train_pred_df, ecg_train_pred_df, symptom_train_pred_df], axis=1).dropna()
        test_pred_df = pd.concat([eeg_test_pred_df, ecg_test_pred_df, symptom_test_pred_df], axis=1).dropna()
    else:
        train_pred_df = pd.concat([eeg_train_pred_df, ecg_train_pred_df], axis=1).dropna()
        test_pred_df = pd.concat([eeg_test_pred_df, ecg_test_pred_df], axis=1).dropna()
        
    assert len(set(train_pred_df.index).intersection(set(test_pred_df.index))) == 0, f"Index overlap between train and test data: {set(train_pred_df.index).intersection(set(test_pred_df.index))}"
    cv_out_dict, cv_mean_std_df = store_cv_results(get_avg_model_best_estimators(loaded_model_data, clf_names=clf_names), cv_ensemble_split_dict)
    # get the test predictions
    print(f"Getting test predictions")
    metarf, metalr, meta_out_dict = train_ensemble_on_preds(train_pred_df)
    test_score_pred_dict, test_score_df = test_model_on_unseen_data(metarf, metalr, test_pred_df)

    # get the holdout predictions
    print(f"Getting holdout predictions")
    eeg_holdout_pred_df, _ = get_holdout_preds(eeg_dataset_results, eeg_loaded_model_data, which_dataset=dset1, tables_folder=results_table_path, internal_folder=internal_folder)
    ecg_holdout_pred_df, _ = get_holdout_preds(ecg_dataset_results, ecg_loaded_model_data, which_dataset=dset2, tables_folder=results_table_path, internal_folder=internal_folder)
    if use_symptoms:
        symptom_holdout_pred_df, _ = get_holdout_preds(symptom_dataset_results, symptom_loaded_model_data, which_dataset=dset3, tables_folder=results_table_path, internal_folder=internal_folder)
        holdout_pred_df = pd.concat([eeg_holdout_pred_df, ecg_holdout_pred_df, symptom_holdout_pred_df], axis=1).dropna()
    else: 
        holdout_pred_df = pd.concat([eeg_holdout_pred_df, ecg_holdout_pred_df], axis=1).dropna()
    
    assert len(set(train_pred_df.index).intersection(set(holdout_pred_df.index))) == 0, f"Index overlap between test and holdout data: {set(test_pred_df.index).intersection(set(holdout_pred_df.index))}"

    holdout_score_pred_dict, holdout_score_df = test_model_on_unseen_data(metarf, metalr, holdout_pred_df)

    # get unseen results
    print(f"Getting unseen results")
    unseen_score_pred_dict, unseen_score_df = test_model_on_unseen_data(metarf, metalr, pd.concat([test_pred_df, holdout_pred_df], axis=0))

    full_results = {
        'cv_results': cv_out_dict,
        'test_results': test_score_pred_dict,
        'holdout_results': holdout_score_pred_dict,
        'unseen_results': unseen_score_pred_dict,
        'train_metascores': meta_out_dict,
    }

    return full_results, cv_mean_std_df, test_score_df, holdout_score_df, unseen_score_df, metarf, metalr

def main(which_dataset="eeg_ecg", savepath='/shared/roy/mTBI/mTBI_Classification/cv_results/final_results_csd/', results_table_path='data/tables/', internal_folder='data/internal/', reload_results=False):
    # load up the dataframe for the dataset
    new_savepath = os.path.join(savepath, which_dataset)
    if not os.path.exists(new_savepath):
        os.makedirs(new_savepath)
    # check if the results are already there
    run_ensemble=True
    if os.path.exists(os.path.join(new_savepath,  f"metalr_{which_dataset}.joblib")):
        print(f"Results already exist for and {which_dataset}")
        run_ensemble=False
    run_ensemble = reload_results or run_ensemble
    if run_ensemble:

        if which_dataset.split('_')[0] == 'late':
            full_results, cv_mean_std_df, test_score_df, holdout_score_df, unseen_score_df, metarf, metalr = run_late_ensemble_model(results_table_path=results_table_path, internal_folder=internal_folder, which_dataset=which_dataset)
            print(f"Finished running late ensemble model")
        else:

            dataset_results = load_model_results(results_table_path=results_table_path, dataset=which_dataset)

            # get the train and test predictions
            print(f"Getting train and test predictions")
            train_pred_df, test_pred_df, loaded_model_data = return_train_test_preds(dataset_results)

            # get the cv results 
            print(f"Getting cv predictions")
            cv_train_preds, cv_test_preds, clf_names = return_cv_preds(dataset_results, loaded_model_data)

            # get the ensemble cv predictions
            print(f"Getting ensemble cv predictions")
            cv_ensemble_split_dict = get_ensemble_cv_preds(cv_train_preds, cv_test_preds, clf_names, scores=['matthews_corrcoef', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity'])

            # now store the cv results
            print(f"Storing cv results")
            cv_out_dict, cv_mean_std_df = store_cv_results(get_avg_model_best_estimators(loaded_model_data), cv_ensemble_split_dict)
            # get the test predictions
            print(f"Getting test predictions")
            metarf, metalr, meta_out_dict = train_ensemble_on_preds(train_pred_df)
            test_score_pred_dict, test_score_df = test_model_on_unseen_data(metarf, metalr, test_pred_df)

            # get the holdout predictions
            print(f"Getting holdout predictions")
            holdout_pred_df, filled_holdouts = get_holdout_preds(dataset_results, loaded_model_data,which_dataset=which_dataset, tables_folder=results_table_path, internal_folder=internal_folder)
            holdout_score_pred_dict, holdout_score_df = test_model_on_unseen_data(metarf, metalr, holdout_pred_df)


            # get unseen results
            print(f"Getting unseen results")
            unseen_score_pred_dict, unseen_score_df = test_model_on_unseen_data(metarf, metalr, pd.concat([test_pred_df, holdout_pred_df], axis=0))

            # save the results
            full_results = {
                'cv_results': cv_out_dict,
                'test_results': test_score_pred_dict,
                'holdout_results': holdout_score_pred_dict,
                'unseen_results': unseen_score_pred_dict,
                'train_metascores': meta_out_dict,
            }

        # json to save the results
        print(f"Saving results to {new_savepath}")
        full_results = du.make_dict_saveable(full_results)
        with open(os.path.join(new_savepath, f"full_results_{which_dataset}.json"), 'w') as f:
            json.dump(full_results, f)

        cv_mean_std_df.to_csv(os.path.join(new_savepath, f"cv_mean_std_{which_dataset}.csv"))
        test_score_df.to_csv(os.path.join(new_savepath, f"test_score_df_{which_dataset}.csv"))
        holdout_score_df.to_csv(os.path.join(new_savepath, f"holdout_score_df_{which_dataset}.csv"))
        unseen_score_df.to_csv(os.path.join(new_savepath, f"unseen_score_df_{which_dataset}.csv"))

        # save the metalearner models
        joblib.dump(metarf, os.path.join(new_savepath, f"metarf_{which_dataset}.joblib"))
        joblib.dump(metalr, os.path.join(new_savepath, f"metalr_{which_dataset}.joblib"))

    else:
        with open(os.path.join(new_savepath, f"full_results_{which_dataset}.json"), 'r') as f:
            full_results = json.load(f)
        cv_mean_std_df = pd.read_csv(os.path.join(new_savepath, f"cv_mean_std_{which_dataset}.csv"), index_col=0)
        test_score_df = pd.read_csv(os.path.join(new_savepath, f"test_score_df_{which_dataset}.csv"), index_col=0)
        holdout_score_df = pd.read_csv(os.path.join(new_savepath, f"holdout_score_df_{which_dataset}.csv"), index_col=0)
        unseen_score_df = pd.read_csv(os.path.join(new_savepath, f"unseen_score_df_{which_dataset}.csv"), index_col=0)
        metarf = joblib.load(os.path.join(new_savepath, f"metarf_{which_dataset}.joblib"))
        metalr = joblib.load(os.path.join(new_savepath, f"metalr_{which_dataset}.joblib"))

    all_outs = {
        'full_results': full_results,
        'cv_mean_std_df': cv_mean_std_df,
        'test_score_df': test_score_df,
        'holdout_score_df': holdout_score_df,
        'unseen_score_df': unseen_score_df,
        'metarf': metarf,
        'metalr': metalr,

    }
    return all_outs

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
    parser.add_argument('--which_dataset', type=str, default='late_eeg_ecg', help='Dataset to run the ensemble on')
    parser.add_argument('--savepath', type=str, default='/shared/roy/mTBI/mTBI_Classification/cv_results/final_results_csd/', help='Path to save the results')
    parser.add_argument('--results_table_path', type=str, default='data/tables/', help='Path to the results table')
    parser.add_argument('--internal_folder', type=str, default='data/internal/', help='Path to the internal folder')
    parser.add_argument('--reload_results', action=argparse.BooleanOptionalAction, default=True, help='Whether to reload the results')
    args = parser.parse_args()
    all_outs = main(**vars(args))