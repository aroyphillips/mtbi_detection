
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as stats
import json
import joblib
import os 
import pandas as pd

from sklearn.utils.validation import check_is_fitted

import mtbi_detection.features.feature_utils as fu


### SCORING 
def score_binary_preds(y_pred_proba=None, y_test=None, y_pred=None, verbose=0):
    """
    Calculate various evaluation metrics for binary classification predictions.

    Parameters:
    - y_pred_proba (numpy.ndarray, optional): Predicted probabilities for the positive class.
    - y_test (numpy.ndarray): True labels for the test set.
    - y_pred (numpy.ndarray, optional): Predicted labels for the test set.
    - verbose (int, optional): Verbosity level. Set to 0 by default.

    Returns:
    - score_dict (dict): Dictionary containing the evaluation metrics:
        - tp (int): True positives.
        - tn (int): True negatives.
        - fp (int): False positives.
        - fn (int): False negatives.
        - accuracy (float): Accuracy.
        - precision (float): Precision.
        - recall (float): Recall.
        - f1 (float): F1 score.
        - f2 (float): F2 score.
        - sensitivity (float): Sensitivity.
        - specificity (float): Specificity.
        - ba (float): Balanced accuracy.
        - correct (int): Number of correct predictions.
        - total (int): Total number of predictions.
        - y_pred (numpy.ndarray): Predicted labels.
        - y_test (numpy.ndarray): True labels.
        - mcc (float): Matthews correlation coefficient.
        - y_pred_proba (numpy.ndarray, optional): Predicted probabilities for the positive class.
        - roc_auc (float): Area under the ROC curve.
        - fpr (numpy.ndarray): False positive rates.
        - tpr (numpy.ndarray): True positive rates.
        - neg_log_loss (float): Negative log loss.
        - neg_brier_score (float): Negative Brier score.
        - precision_curve (numpy.ndarray): Precision values for different thresholds.
        - recall_curve (numpy.ndarray): Recall values for different thresholds.
        - auc_pr (float): Area under the precision-recall curve.
    """  
    
    if y_pred_proba is not None:
        # positive inf to 1 and negative inf to 0
        y_pred_proba[y_pred_proba == np.inf] = 1
        y_pred_proba[y_pred_proba == -np.inf] = 0
        y_pred_proba[y_pred_proba > 1] = 1
        y_pred_proba[y_pred_proba < 0] = 0
    
    if y_pred is None:
        if y_pred_proba is None:
            raise ValueError("y_pred_proba or y_pred must be provided")
        if len(y_pred_proba.shape) == 1:
            y_pred = np.round(y_pred_proba)
        else:
            if y_pred_proba.shape[1] == 1:
                y_pred = np.squeeze(np.round(y_pred_proba))
            elif y_pred_proba.shape[1] == 2:
                y_pred = np.argmax(y_pred_proba, axis=1)
        
    y_pred[y_pred == np.inf] = 1
    y_pred[y_pred == -np.inf] = 0
    y_pred[y_pred > 1] = 1
    y_pred[y_pred < 0] = 0


    if verbose > 0:
        print(f"Shapes: y_pred: {y_pred.shape}, y_test: {y_test.shape}")
        
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    correct = tp + tn
    total = tp + tn + fp + fn
    accuracy = correct / total
    if tp+fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan
    if tp+fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan
    if tp+fn != 0 and tp+fp != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        f2 = 5 * (precision * recall) / (4 * precision + recall)
    else:
        f1 = np.nan
        f2 = np.nan
    if tp+fn != 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = np.nan
    
    if tn+fp != 0:
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan
    if not np.isnan(sensitivity) and not np.isnan(specificity):
        ba = (sensitivity + specificity) / 2
    else:
        ba = np.nan

    score_dict = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1, 
        'f2': f2,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ba': ba,
        'correct': correct,
        'total': total,
        'y_pred': y_pred,
        'y_test': y_test,
        'mcc': sklearn.metrics.matthews_corrcoef(y_test, y_pred),
    }
    if y_pred_proba is not None:
        # print("y_pred_proba is not None")
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = np.expand_dims(y_pred_proba, axis=1)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, -1])
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, -1])

        log_loss = -1*np.abs(sklearn.metrics.log_loss(y_test, y_pred_proba[:, -1]))
        brier_score = -1*np.abs(sklearn.metrics.brier_score_loss(y_test, y_pred_proba[:, -1]))
        precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(y_test, y_pred_proba[:, -1])
        auc_pr = sklearn.metrics.auc(recall_curve, precision_curve)
        score_dict['y_pred_proba'] = y_pred_proba
        score_dict['roc_auc'] = roc_auc
        score_dict['fpr'] = fpr
        score_dict['tpr'] = tpr
        score_dict['neg_log_loss'] = log_loss
        score_dict['neg_brier_score'] = brier_score
        score_dict['precision_curve'] = precision_curve
        score_dict['recall_curve'] = recall_curve
        score_dict['auc_pr'] = auc_pr
        # print("log_loss: ", log_loss)
    return score_dict

def score_binary_model(model, X_test, y_test):
    """
    Given a fitted sklearn model, X_test, and y_test, return a dictionary of scores

    
    """

    y_pred = model.predict(X_test)
    try:
        y_pred_proba = model.predict_proba(X_test)[:,1]
    except:
        y_pred_proba = model.predict(X_test)
    
    score_dict = score_binary_preds(y_pred_proba, y_test, y_pred=y_pred)
    return score_dict

def print_binary_scores(score_dict):
    """
    Prints the scores from a score_dict (output of score_binary_preds)
    """
    print(f"Num TP: {score_dict['tp']}, Num TN: {score_dict['tn']}, Num FP: {score_dict['fp']}, Num FN: {score_dict['fn']}")
    print(f"Accuracy: {score_dict['accuracy']}")
    print(f"Sensitivity: {score_dict['sensitivity']}")
    print(f"Specificity: {score_dict['specificity']}")
    print(f"Balanced Accuracy: {score_dict['ba']}")
    print(f"Precision: {score_dict['precision']}")
    print(f"Recall: {score_dict['recall']}")
    print(f"F1: {score_dict['f1']}")
    print(f"F2: {score_dict['f2']}")
    

    possible_scores = {'roc_auc': 'ROC AUC', 'neg_log_loss': 'Neg Log Loss', 'neg_brier_score': 'Neg Brier Score', 'auc_pr': 'AUC PR', 'mcc': 'MCC'}
    for score in possible_scores.keys():
        try:
            print(f"{possible_scores[score]}: {score_dict[score]}")
        except:
            pass

def compute_select_binary_scores(y_true, y_pred, y_pred_proba):
    """
    Given the true labels, predicted labels, and predicted probabilities, calculate binary classification scores (MCC, ROC AUC, Balanced Accuracy, Sensitivity, Specificity)
    Inputs:
        - y_true: true labels
        - y_pred: predicted labels (binary)
        - y_pred_proba: predicted probabilities
    Outputs:
        - score_dict: dictionary containing the scores and predictions of the metalearners on the unseen data
            e.g. {'scores': {'MCC': mcc, 'ROC AUC': roc_auc, 'Balanced Accuracy': balanced_accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresh': thresh}
            'preds': y_pred
            'pred_probs': y_pred_proba}
    
    """
    # scores
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    assert len(y_true) == len(y_pred_proba), "y_true and y_pred_proba must have the same length"
    assert len(np.unique(y_true)) == 2, "y_true must be binary"
    assert len(np.unique(y_pred)) == 2, "y_pred must be binary"
    assert y_pred_proba.shape[1] == 2, "y_pred_proba must have 2 columns"
    assert np.allclose(y_pred_proba.sum(axis=1), 1), "y_pred_proba must sum to 1"

    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    sensitivity = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1)
    specificity = sklearn.metrics.recall_score(y_true, y_pred, pos_label=0)

    #roc_curve
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true, y_pred_proba[:,1])
    score_dict = {'scores': {'MCC': mcc, 'ROC AUC': roc_auc, 'Balanced Accuracy': balanced_accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity},
                                        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresh': thresh},
                                        'preds': y_pred,
                                        'pred_probs': y_pred_proba}  
    return score_dict

### Statistical Tests

def mcnemar_exact_conditional_test(n12, n21, p=0.5):
    """
    Calculate the exact p-value for McNemar's exact conditional test.
    This is the probability of at least n12 sucesses out of n12 + n21 trials times 2.
    H0: p = 0.5
    H1: p > 0.5 and p < 0.5
    https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-91
    
    Parameters:
    n12: int
        The number of discordant pairs where the first model was correct.
    n21: int
        The number of discordant pairs where the second model was correct.
    p: float
        The probability of success (default 0.5).
    Returns:
    p: float
        The p-value of the test.
    """
    n = n12 + n21
    x = min(n12, n21)
    p = 2* stats.binom.cdf(x, n+1, p) # probability of at most x successes (two-tailed test)

    return p

def mcnemar_midp(n12, n21, p=0.5):
    """
    Calculate the mid-p value for McNemar's exact conditional test.
    This is the probability of at least n12 sucesses out of n12 + n21 trials times 2 with the mid-p correction.
    H0: p = 0.5
    H1: p > 0.5 and p < 0.5
    https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-91
    

    midp = exact_p - binompdf(n12, n, p)

    Parameters:
    n12: int
        The number of discordant pairs where the first model was correct.
    n21: int
        The number of discordant pairs where the second model was correct.
    p: float
        The probability of success (default 0.5).
    Returns:
    p: float
        The p-value of the test.
    """
    n = n12 + n21
    a = min(n12, n21)

    exact_p = mcnemar_exact_conditional_test(n12, n21, p)
    mid_p = exact_p - stats.binom.pmf(a, n+1, p) # probability of exactly n12 successes
    return mid_p

### Model Saving and Loading
def load_model_data(modelpath, load_model=True, load_holdout=False):
    """
    Given the path to a joblib model, return the model, X_train, and X_test data
    Inputs:
        - modelpath (str): path to the joblib model
        - load_model (bool): whether to load the model (default True)
        - load_holdout (bool): whether to load the holdout data (default False)
    Returns:
        - model (sklearn model): the fitted model
        - X_train (pd.DataFrame): training data
        - X_test (pd.DataFrame): testing data
        - X_holdout (pd.DataFrame): holdout data (if load_holdout is True)
    """
    modelfiles = os.listdir(modelpath)
    joblib_filenames = [f for f in modelfiles if f.endswith('.joblib')]
    assert len(joblib_filenames) == 1, f"Expected 1 joblib file in {modelpath}, got {len(joblib_filenames)}"
    joblib_filename = os.path.join(modelpath, joblib_filenames[0])
    if load_model:
        model = joblib.load(joblib_filename)
    else:
        model = None
    X_train = pd.read_csv(os.path.join(modelpath, [f for f in modelfiles if 'X_train' in f][0]), index_col=0)
    X_test = pd.read_csv(os.path.join(modelpath, [f for f in modelfiles if 'X_ival' in f][0]), index_col=0)
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    if load_holdout:
        X_holdout = pd.read_csv(os.path.join(modelpath, [f for f in modelfiles if 'X_holdout' in f][0]), index_col=0)
        assert len(set(X_train.index).intersection(set(X_holdout.index))) == 0
        assert len(set(X_test.index).intersection(set(X_holdout.index))) == 0
        return model, X_train, X_test, X_holdout
    else:
        return model, X_train, X_test
    

def get_transformed_data(X, model, verbose=True):
    """
    Given a data frame and a pipeline model, return the transformed data before the ML model
    Inputs:
        - X (pd.DataFrame): the data to transform
        - model (sklearn pipeline): the fitted model
        - verbose (bool): whether to print the transformations
    Returns:
        - out_X (pd.DataFrame): the transformed data
    """
    out_X = X.copy(deep=True)
    for step_name, step in model.best_estimator_.named_steps.items():
        if step_name == 'classifier' or step_name == 'regressor':
            continue
        if step == 'passthrough':  
            continue
        out_X = step.transform(out_X)
    _, column_transform_dict = track_preproc_fs(X, model.best_estimator_.named_steps['preproc'], verbose=verbose)
    preproc_features = column_transform_dict['final']
    selected_features = get_wrapper_features(model.best_estimator_.named_steps['wrapper'], preproc_features)
    out_X = pd.DataFrame(out_X, columns=selected_features, index=X.index)
    return out_X


def track_preproc_fs(X_train, preproc, verbose=True):
    """
    Given an input dataframe and a preprocessor trained on the dataframe, return the columns and data at each step of the preprocessor
    Input:
        X_train: dataframe, (n_samples, n_features) with column names
        preproc: sklearn.compose.ColumnTransformer, trained on X_train
    Output:
        transformed_df_dict: dict, with keys as the index of the step in the preprocessor and values as a tuple of the step name and the transformed data at that step
        column_transform_dict: dict, with keys as the index of the step in the preprocessor and values as a tuple of the step name and the column names at that step
        e.g.:
            {transformer: {step: (step_name, transformed_data)}} and {transformer: {step: (step_name, column_names)}}

    """
    # preproc = model.best_estimator_.named_steps['preproc']
    check_is_fitted(preproc)
    transformed_df = X_train.copy(deep=True)
    column_transformers = preproc.transformers_
    transformed_df_dict = {ct[0]: {0: ('baseline', transformed_df[ct[2]])} for ct in column_transformers}
    column_transform_dict = {ct[0]: {0: ('baseline', transformed_df[ct[2]].columns)} for ct in column_transformers}
    # loop through each column transformer
    for ct_idx, (ct_name, ct, ct_cols) in enumerate(column_transformers):
        if verbose:
            print(f"Transforming {ct_name}... {ct_idx+1}/{len(column_transformers)}")
            print(f"Initial shape: {transformed_df_dict[ct_name][0][1].shape}")
        # loop through each step in the column transformer
        for step_idx, (step_name, step) in enumerate(ct.named_steps.items()):
            if verbose:
                print(f"Transforming through {step_name}... {step_idx+1}/{len(ct.named_steps)}")
            if step_name == 'scaler':
                # if the step is a scaler, we don't need to transform the data, just keep the same data
                transformed_df_dict[ct_name][step_idx+1] = ('no-scaler',  transformed_df_dict[ct_name][step_idx][1])
                column_transform_dict[ct_name][step_idx+1] = ('no-scaler', column_transform_dict[ct_name][step_idx][1])
            else:
                transformed_df_dict[ct_name][step_idx+1] = (step_name, step.transform(transformed_df_dict[ct_name][step_idx][1]))
                if hasattr(step, 'get_support'):
                    # ASSUMPTION: if the step changes the number of columns, it has a get_support method
                    support = step.get_support()
                    column_transform_dict[ct_name][step_idx+1] = (step_name, column_transform_dict[ct_name][step_idx][1][support])
                    transformed_df_dict[ct_name][step_idx+1] = (step_name, transformed_df_dict[ct_name][step_idx][1][column_transform_dict[ct_name][step_idx+1][1]])
                elif hasattr(step, 'support_'):
                    # ASSUMPTION: if the step changes the number of columns, it has a support_ attribute
                    support = step.support_
                    column_transform_dict[ct_name][step_idx+1] = (step_name, column_transform_dict[ct_name][step_idx][1][support])
                    transformed_df_dict[ct_name][step_idx+1] = (step_name, transformed_df_dict[ct_name][step_idx][1][column_transform_dict[ct_name][step_idx+1][1]])
                else:
                    # NO column selection
                    column_transform_dict[ct_name][step_idx+1] = (step_name, column_transform_dict[ct_name][step_idx][1])
                    transformed_df_dict[ct_name][step_idx+1] = (step_name, transformed_df_dict[ct_name][step_idx][1])
            if verbose:
                print(transformed_df_dict[ct_name][step_idx+1][0], transformed_df_dict[ct_name][step_idx+1][1].shape, "columns", len(column_transform_dict[ct_name][step_idx+1][1]))
            assert  transformed_df_dict[ct_name][step_idx+1][1].shape[1] == len(column_transform_dict[ct_name][step_idx+1][1]), f"Step {step_name} changed the number of columns but did not have a get_support method"
        
    transformed_df_dict['final'] = np.concatenate([transformed_df_dict[ct_name][len(ct.named_steps)][1] for ct_name in transformed_df_dict], axis=1)
    column_transform_dict['final'] = [column_transform_dict[ct_name][len(ct.named_steps)][1] for ct_name in column_transform_dict]
    column_transform_dict['final'] = np.concatenate(column_transform_dict['final'])

    return transformed_df_dict, column_transform_dict


def get_wrapper_features(wrapper, columns, verbose=True):
    """
    Given a wrapper object and the column names, return the selected features
    Inputs:
        - wrapper: sklearn wrapper object
        - columns: list of column names
        - verbose: bool, whether to print the selected features
    Returns:
        - selected_features: list of selected
    """
    dummy_df = pd.DataFrame(np.arange(columns.shape[0]).reshape(-1, 1).T, columns=columns)
    if type(wrapper) == str:
        if wrapper == 'passthrough':
            return columns
        else:
            raise ValueError("wrapper must be a sklearn wrapper object or 'passthrough'")
    else:
        support =  wrapper.transform(dummy_df)
    selected_features = columns[support]
    if len(selected_features.shape) > 1:
        selected_features = selected_features[0]
    else:
        pass
    return selected_features

### multi output regression
def score_multireg_model(model, X_test, y_test, norm_means=None, norm_stds=None):
    """
    Given a fitted sklearn model, X_test, and y_test, return a dictionary of scores for multi-output regression
    Inputs:
        - model (sklearn model): the fitted model
        - X_test (pd.DataFrame): the testing data
        - y_test (pd.DataFrame): the testing labels
        - norm_means (optional): the normalization means
        - norm_stds (optional): the normalization standard deviations
    Returns:
        - out_dict (dict): dictionary of scores {metric: value}    
    """
    check_is_fitted(model)
    y_test = convert_obj_to_array(y_test)
    y_pred = model.predict(X_test)

    if norm_means is not None and norm_stds is not None:

        norm_means = convert_obj_to_array(norm_means)
        norm_stds = convert_obj_to_array(norm_stds)
        if type(norm_means) == int or type(norm_means) == float:
            y_pred = (y_pred - norm_means) / norm_stds
            y_test = (y_test - norm_means) / norm_stds
        elif norm_means.shape[0] == y_pred.shape[0] and norm_means.shape[1] == y_pred.shape[1]:
            
            y_pred = np.divide(y_pred - norm_means, norm_stds)
            y_test = np.divide(y_test - norm_means,  norm_stds)
        elif norm_means.shape[0] == y_pred.shape[1]:
            y_pred = np.divide(y_pred - norm_means[np.newaxis, :], norm_stds[np.newaxis, :])
            y_test = np.divide(y_test - norm_means[np.newaxis, :],  norm_stds[np.newaxis, :])
        else:
            raise ValueError(f"Unknown type for norm_means: {type(norm_means)} and norm_stds: {type(norm_stds)}, or incompatible shapes: means {norm_means.shape} and preds {y_pred.shape}")

    out_dict = score_multireg_preds(y_pred, y_test)
    return out_dict

def score_multireg_preds(y_pred, y_test):
    """
    Given predicted and true labels for multi-output regression, return a dictionary of scores
    Inputs:
        - y_pred (np.ndarray): the predicted labels
        - y_test (np.ndarray): the true labels
    Returns:
        - out_dict (dict): dictionary of scores {metric: value}
    """ 
    perf_zeros = ['rmse', 'mae', 'medae', 'r2']
    perf_ones = ['pearson', 'spearman', 'r2']
    eval_scoring = {'avg_pearson': fu.avg_pearson_pred, 
                'avg_spearman': fu.avg_spearman_pred, 
                'avg_rmse': fu.avg_rmse,
                'avg_mae': fu.avg_mae_pred,
                'avg_medae': fu.avg_medae_pred,
                'avg_r2': fu.avg_r2_pred,
                'max_pearson': fu.max_pearson_pred,
                'max_spearman': fu.max_spearman_pred,
                'max_rmse': fu.max_rmse_pred,
                'max_mae': fu.max_mae_pred,
                'max_medae': fu.max_medae_pred,
                'max_r2': fu.max_r2_pred,
                'median_pearson': fu.med_pearson_pred,
                'median_spearman': fu.med_spearman_pred,
                'median_rmse': fu.med_rmse_pred,
                'median_mae': fu.med_mae_pred,
                'median_medae': fu.med_medae_pred,
                'median_r2': fu.med_r2_pred,
                'min_pearson': fu.min_pearson_pred,
                'min_spearman': fu.min_spearman_pred,
                'min_rmse': fu.min_rmse_pred,
                'min_mae': fu.min_mae_pred,
                'min_medae': fu.min_medae_pred,
                'min_r2': fu.min_r2_pred,
                'stacked_rmse': fu.stacked_rmse_pred,
                'stacked_mae': fu.stacked_mae_pred,
                'stacked_medae': fu.stacked_medae_pred,
                'stacked_pearson': fu.stacked_pearson_pred,   
                'stacked_spearman': fu.stacked_spearman_pred,
                'stacked_r2': fu.stacked_r2_pred,
            }
    out_dict = {}
    for key, scorer in eval_scoring.items():
        if scorer(y_test, y_pred) is None or np.isnan(scorer(y_test, y_pred)):
            if any([perf in key for perf in perf_zeros]):
                out_dict[key] = 100
            elif any([perf in key for perf in perf_ones]):
                out_dict[key] = -100
        else:
            out_dict[key] = scorer(y_test, y_pred)
    return out_dict  

def convert_obj_to_array(obj):
    """
    Given an object, return the object as a numpy array
       
    """
    if type(obj) == pd.DataFrame:
        return obj.values
    elif type(obj) == pd.Series:
        return obj.values
    elif type(obj) in [list, tuple]:  
        return np.array(obj)
    elif type(obj) == set:
        return np.array(list(obj))
    elif type(obj) == int or type(obj) == float:
        return np.array([obj])
    elif type(obj) == np.ndarray:
        return obj
    else:
        raise ValueError(f"Unknown type: {type(obj)}")
    
def print_multireg_scores(out_dict):
    """
    Print the scores from a score_multireg_model output dictionary
    out_dict: output dictionary from score_multireg_model
    """

    print(f"Pearson: avg: {out_dict['avg_pearson']}, max: {out_dict['max_pearson']}, median: {out_dict['median_pearson']}, min: {out_dict['min_pearson']}")
    print(f"Spearman: avg: {out_dict['avg_spearman']}, max: {out_dict['max_spearman']}, median: {out_dict['median_spearman']}, min: {out_dict['min_spearman']}")
    print(f"RMSE: avg: {out_dict['avg_rmse']}, max: {out_dict['max_rmse']}, median: {out_dict['median_rmse']}, min: {out_dict['min_rmse']}")
    print(f"MAE: avg: {out_dict['avg_mae']}, max: {out_dict['max_mae']}, median: {out_dict['median_mae']}, min: {out_dict['min_mae']}")
    print(f"MedAE: avg: {out_dict['avg_medae']}, max: {out_dict['max_medae']}, median: {out_dict['median_medae']}, min: {out_dict['min_medae']}")
    print(f"R2: avg: {out_dict['avg_r2']}, max: {out_dict['max_r2']}, median: {out_dict['median_r2']}, min: {out_dict['min_r2']}")

    print(f"Stacked RMSE: {out_dict['stacked_rmse']}")
    print(f"Stacked MAE: {out_dict['stacked_mae']}")
    print(f"Stacked MedAE: {out_dict['stacked_medae']}")
    print(f"Stacked Pearson: {out_dict['stacked_pearson']}")
    print(f"Stacked Spearman: {out_dict['stacked_spearman']}")
    
def compute_select_multireg_scores(y_true, y_pred, col_names):
    """
    Compute the scores for the multi-output regression predictions

    Note: since the ACE and RIVERMEAD scores are both 0 to 100 we shouldn't need to normalize them for the stacked scores
    
    Inputs:
        - y_true: true labels
        - y_pred: predicted labels
        - col_names: column names
    Outputs:
        - out_scores: dictionary of scores 
            {'col_name': {'rmse': rmse, 'rrmse': rrmse, 'spearman': spearman, 'pearson': pearson}, 'aggregate': {'avg_rmse': avg_rmse, 'avg_rrmse': avg_rrmse, 'stacked_spearman': stacked_spearman, 'stacked_pearson': stacked_pearson}}
    """
    
    out_scores = {}
    for idx, col in enumerate(col_names):

        rmse = sklearn.metrics.mean_squared_error(y_true[col], y_pred[:, idx], squared=False)
        rrmse = fu.avg_rank_rmse(y_true[col], y_pred[:, idx])
        spearman = stats.spearmanr(y_true[col], y_pred[:, idx], axis=0)[0]
        pearson = stats.pearsonr(y_true[col], y_pred[:, idx])[0]

        out_scores[col] = {'rmse': rmse, 'rrmse': rrmse, 'spearman': spearman, 'pearson': pearson}
        
    avg_rmse = fu.avg_rmse(y_true.values, y_pred)
    avg_rrmse = fu.avg_rank_rmse(y_true.values, y_pred)
    stacked_spearman = fu.stacked_spearman_pred(y_true.values, y_pred)
    stacked_pearson = fu.stacked_pearson_pred(y_true.values, y_pred)

    out_scores['aggregate'] = {'avg_rmse': avg_rmse, 'avg_rrmse': avg_rrmse, 'stacked_spearman': stacked_spearman, 'stacked_pearson': stacked_pearson}

    return out_scores