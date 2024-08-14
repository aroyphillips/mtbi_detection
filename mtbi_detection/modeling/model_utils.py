
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score

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
