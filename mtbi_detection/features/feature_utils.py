import pandas as pd
import numpy as np
import mne
import time
import sklearn

import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from numpy.typing import NDArray
from typing import Dict, List, Tuple
from pandas.core.frame import DataFrame

import mtbi_detection.data.load_dataset as ld
import mtbi_detection.features.gradiompy_integrate as gp_integrate
CHANNELS = ['C3','C4','Cz','F3','F4','F7','F8','Fp1','Fp2','Fz','O1','O2','P3','P4','Pz','T1','T2','T3','T4','T5','T6']
LABEL_DICT = ld.load_label_dict()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=0, pos_inf_value=1e6, neg_inf_value=-1e6):
        self.fill_value = fill_value
        self.pos_inf_value = pos_inf_value
        self.neg_inf_value = neg_inf_value

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.support_ = np.array([col not in X.columns[X.isna().any()].tolist() for col in X.columns])
            filled = X.fillna(self.fill_value)
            inf_replace = filled.replace([np.inf, -np.inf], [self.pos_inf_value, self.neg_inf_value])
            return inf_replace
        else:
            self.support_ = np.array([not np.any(np.isnan(X[:, i])) for i in range(X.shape[1])])
            filled = np.nan_to_num(X, nan=self.fill_value, posinf=self.pos_inf_value, neginf=self.neg_inf_value)
            return filled

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            filled = X.fillna(self.fill_value)
            inf_replace = filled.replace([np.inf, -np.inf], [self.pos_inf_value, self.neg_inf_value])
            return inf_replace
        else:
            filled = np.nan_to_num(X, nan=self.fill_value, posinf=self.pos_inf_value, neginf=self.neg_inf_value)
            return filled
        
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y)
    
    def get_params(self, deep=True):
        return {'fill_value': self.fill_value, 'pos_inf_value': self.pos_inf_value, 'neg_inf_value': self.neg_inf_value}
    
    def get_support(self):
        return self.support_
    
    
class DropDuplicatedAndConstantColumns(BaseEstimator, TransformerMixin):
    def __init__(self, min_unique=2):
        self.min_unique = min_unique

    def fit(self, X, y=None, cols=None):
        # if X is not a dataframe, make it one
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.copy(), columns=cols)
        X_copy = X.copy(deep=True)
        # find duplicated columns
        dup_mask = X_copy.transpose().duplicated(keep='first').transpose()
        dup_cols = X_copy.columns[dup_mask]

        # drop duplicated columns
        X_dedup = X_copy.drop(columns=dup_cols)

        # find constant columns
        const_mask = X_dedup.apply(lambda x: x.nunique() < self.min_unique)
        const_cols = X_dedup.columns[const_mask]

        self.drop_cols_ = dup_cols.append(const_cols) # append bc series
        
        # get all the columns that are not dropped
        self.support_ = np.array([col not in self.drop_cols_ for col in X.columns])
        # now make it binary for support_
        self.save_cols_ = X.columns[self.support_]

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X, y=None):
        check_is_fitted(self, 'drop_cols_')
        # drop constant columns
        if isinstance(X, pd.DataFrame):
            X = X.copy(deep=True).drop(columns=self.drop_cols_)
        else:
            X = X.copy()[:, self.support_]

        return X

    def get_feature_drop_names(self):
        return self.drop_cols_
    
    def get_feature_save_names(self):
        return self.save_cols_
    
    def get_support(self):
        return self.support_
    
    def get_params(self, deep=True):
        return {'min_unique': self.min_unique}
    

# make a transformer that drops columns with nans
class DropNanColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):

        self.nan_columns = X.columns[X.isna().any()].tolist()
        self.support_ = [not col in self.nan_columns for col in X.columns]
        self.drop_cols_ = [col for col in X.columns if col in self.nan_columns]
        self.save_cols_ = [col for col in X.columns if not col in self.nan_columns]
        return self
    def transform(self, X, y=None):
        return X.copy(deep=True)[self.save_cols_]
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    def get_support(self):
        return self.support_
    def get_drop_cols(self):
        return self.drop_cols_
    def get_save_cols(self):
        return self.save_cols_
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
class DropInfColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.inf_columns = X.columns[X.isin([np.inf, -np.inf]).any()].tolist()
        self.support_ = [not col in self.inf_columns for col in X.columns]
        self.drop_cols_ = [col for col in X.columns if col in self.inf_columns]
        self.save_cols_ = [col for col in X.columns if not col in self.inf_columns]
        return self
    
    def transform(self, X, y=None):
        return X.copy(deep=True)[self.save_cols_]
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_support(self):
        return self.support_
    
    def get_drop_cols(self):
        return self.drop_cols_
    
    def get_save_cols(self):
        return self.save_cols_
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
# make a transformer that drops rows with nans
class DropNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.nan_rows = X[X.isna().any(axis=1)].index.tolist()
        self.support_ = [not row in self.nan_rows for row in X.index]
        self.drop_rows_ = [row for row in X.index if row in self.nan_rows]
        self.save_rows_ = [row for row in X.index if not row in self.nan_rows]
        return self
    def transform(self, X, y=None):
        return X.copy(deep=True).loc[self.save_rows_]
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    def get_support(self):
        return self.support_
    def get_drop_rows(self):
        return self.drop_rows_
    def get_save_rows(self):
        return self.save_rows_
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

### make a feature filter selector with arbitrary scoring function
class UnivariateThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, score_func=None, threshold=None, verbose=0, min_features=1, scale_scores=False, scale_min=0, scale_max=1.0, **kwargs):
        if verbose>2:
            print(f"Initializing UnivariateThresholdSelector with score_func: {score_func}, threshold: {threshold}, verbose: {verbose}, min_features: {min_features}, scale_scores: {scale_scores}, scale_min: {scale_min}, scale_max: {scale_max}")
        if type(score_func) == str:
            print(f"score_func: {score_func}")
            self.score_func = self._str2func(score_func)
        elif callable(score_func):
            self.score_func = score_func
        else:
            raise ValueError(f"score_func must be a string or a callable, got {type(score_func)}")
        self.threshold = threshold
        self.kwargs = kwargs
        self.verbose = verbose
        self.min_features = min_features
        self.scale_scores = scale_scores
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _str2func(self, func_name):
        if func_name == 'pearson_corr':
            return pearson_corr
        elif func_name == 'spearman_corr':
            return spearman_corr
        elif func_name == 'kendall_corr':
            return kendall_corr
        elif func_name == 'anova_f':
            return anova_f
        elif func_name == 'anova_pinv':
            return anova_pinv
        elif func_name == 'mi_classif':
            return mutual_info_classif
        elif func_name == 'mi_regression':
            return mutual_info_regression   
        else:
            raise ValueError(f"Invalid score function name {func_name}")
    def fit(self, X, y=None):
        if type(self.score_func) == str:
            if self.verbose>0:
                print(f"score_func: {self.score_func} was str")
            self.score_func = self._str2func(self.score_func)
        scores = self.score_func(X, y)
        if self.scale_scores:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) * (self.scale_max - self.scale_min) + self.scale_min
        self.support_ = np.where(scores > self.threshold)[0]
        if self.verbose>0:
            print(f"Number of features selected: {len(self.support_)}")
        if len(self.support_) < self.min_features:
            if self.verbose > 0:
                print(f'No features selected for {self.threshold} and {self.score_func}, picking top {self.min_features} features by default.')
            self.support_ = np.argsort(scores)[::-1][:self.min_features]
        if self.verbose>3:
            print(f"Selected features: {self.support_}")
        return self

    def transform(self, X):
        # if X is a dataframe, return a dataframe
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_]
        # otherwise return a numpy array
        else:
            return X[:, self.support_]
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self):
        return self.support_
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
        


def pearson_corr(X, y):
    return np.abs(np.corrcoef(X, y, rowvar=False)[:-1, -1])

def spearman_corr(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return np.abs(X.corrwith(pd.Series(y), method='spearman').values)

def kendall_corr(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return np.abs(X.corrwith(pd.Series(y), method='kendall').values)

def anova_f(X, y):
    return sklearn.feature_selection.f_classif(X, y)[0]

def anova_pinv(X, y):
    return 1-sklearn.feature_selection.f_classif(X, y)[1]

## multivar funcs

def avg_pearson_corr(X, y):
    assert y.shape[1] > 1, "y must have more than one column"
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pearson_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='pearson', axis=0).values) for i in range(y.shape[1])]
    avg_pearson = np.nanmean(pearson_vals, axis=0)
    assert len(avg_pearson) == X.shape[1]
    return avg_pearson

def avg_spearman_corr(X, y):
    assert y.shape[1] > 1, "y must have more than one column"
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    spearman_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='spearman', axis=0).values) for i in range(y.shape[1])]
    avg_spearman = np.nanmean(spearman_vals, axis=0)
    assert len(avg_spearman) == X.shape[1]
    return avg_spearman

def avg_kendall_corr(X, y):
    assert y.shape[1] > 1, "y must have more than one column"
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    kendall_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='kendall', axis=0).values) for i in range(y.shape[1])]
    avg_kendall = np.nanmean(kendall_vals, axis=0)
    assert len(avg_kendall) == X.shape[1]
    return avg_kendall

def avg_mi_reg(X, y):
    """
    Average mutual information for multitask regression to determine feature importance
    """
    mi_scores = [sklearn.feature_selection.mutual_info_regression(X, y[:, idx]) for idx in range(y.shape[1])]
    avg_mi_scores = np.nanmean(mi_scores, axis=0)
    assert len(avg_mi_scores) == X.shape[1]
    return avg_mi_scores

def max_mi_reg(X, y):
    """
    Max mutual information for multitask regression to determine feature importance
    """
    mi_scores = [sklearn.feature_selection.mutual_info_regression(X, y[:, idx]) for idx in range(y.shape[1])]
    max_mi_scores = np.nanmax(mi_scores, axis=0)
    assert len(max_mi_scores) == X.shape[1]
    return max_mi_scores

def max_spearman_corr(X, y):
    """
    Max Spearman correlation
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    spearman_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='spearman', axis=0).values) for i in range(y.shape[1])]
    max_spearman = np.nanmax(spearman_vals, axis=0)
    assert len(max_spearman) == X.shape[1]
    return max_spearman

def max_pearson_corr(X, y):
    """
    Max Pearson correlation
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pearson_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='pearson', axis=0).values) for i in range(y.shape[1])]
    max_pearson = np.nanmax(pearson_vals, axis=0)
    assert len(max_pearson) == X.shape[1]
    return max_pearson

def max_kendall_corr(X, y):
    """
    Max Kendall correlation
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    kendall_vals = [np.abs(X.corrwith(pd.Series(y[:, i]), method='kendall', axis=0).values) for i in range(y.shape[1])]
    max_kendall = np.nanmax(kendall_vals, axis=0)
    assert len(max_kendall) == X.shape[1]
    return max_kendall

# metrics for prediction evaluation
def avg_spearman_pred(y_true, y_pred):
    """
    Average Pearson correlation for multitask regression
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    spearman_vals = [np.abs(spearmanr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    avg_spearman = np.nanmean(spearman_vals)
    return avg_spearman

def avg_pearson_pred(y_true, y_pred):
    """
    Average Pearson correlation for multitask regression
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    pearson_vals = [np.abs(pearsonr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    avg_pearson = np.nanmean(pearson_vals)
    return avg_pearson

# def avg_rmse_pred(y_true, y_pred):
#     """
#     Average RMSE for multitask regression
#     """
#     rmse_vals = [np.sqrt(sklearn.metrics.mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
#     avg_rmse = np.nanmean(rmse_vals)
#     return avg_rmse


def avg_rmse(y_true, y_pred):
    """
    Average RMSE for multitask regression
    usage in sklearn make scorer: sklearn.metrics.make_scorer(avg_rmse, greater_is_better=False)
    """
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred, multioutput='uniform_average'))
    return rmse

def avg_rank_rmse(y_true, y_pred):
    """
    Towards Better Evaluation of Multi-Target Regression Models
    """
    rank_y_true = scipy.stats.rankdata(y_true, axis=0)
    rank_y_pred = scipy.stats.rankdata(y_pred, axis=0)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(rank_y_true, rank_y_pred, multioutput='uniform_average'))
    return rmse




def avg_mae_pred(y_true, y_pred):
    """
    Average MAE for multitask regression
    """
    mae_vals = [sklearn.metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    avg_mae = np.nanmean(mae_vals)
    return avg_mae

def avg_mape_pred(y_true, y_pred):
    """
    Average MAPE for multitask regression
    """
    mape_vals = [np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100 for i in range(y_true.shape[1])]
    avg_mape = np.nanmean(mape_vals)
    return avg_mape

def avg_medae_pred(y_true, y_pred):
    """
    Average median absolute error for multitask regression
    """
    medae_vals = [sklearn.metrics.median_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    avg_medae = np.nanmean(medae_vals)
    return avg_medae

def avg_r2_pred(y_true, y_pred):
    """
    Average R^2 for multitask regression
    """
    r2_vals = [sklearn.metrics.r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    avg_r2 = np.nanmean(r2_vals)
    return avg_r2
    
def stacked_pearson_pred(y_true, y_pred):
    """
    Pearson pred of unravelled y_true and y_pred
    
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    pearson_val = np.abs(pearsonr(y_true.ravel(), y_pred.ravel())[0])
    return pearson_val

def stacked_spearman_pred(y_true, y_pred):
    """
    Spearman pred of unravelled y_true and y_pred
    
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    spearman_val = np.abs(spearmanr(y_true.ravel(), y_pred.ravel())[0])
    return spearman_val

def stacked_rmse_pred(y_true, y_pred):
    """
    RMSE of unravelled y_true and y_pred
    
    """
    rmse_val = np.sqrt(sklearn.metrics.mean_squared_error(y_true.ravel(), y_pred.ravel()))
    return rmse_val

def stacked_mae_pred(y_true, y_pred):
    """
    MAE of unravelled y_true and y_pred
    
    """
    mae_val = sklearn.metrics.mean_absolute_error(y_true.ravel(), y_pred.ravel())
    return mae_val

def stacked_medae_pred(y_true, y_pred):
    """
    Median absolute error of unravelled y_true and y_pred
    
    """
    medae_val = sklearn.metrics.median_absolute_error(y_true.ravel(), y_pred.ravel())
    return medae_val

def stacked_r2_pred(y_true, y_pred):
    """
    R^2 of unravelled y_true and y_pred
    
    """
    r2_val = sklearn.metrics.r2_score(y_true.ravel(), y_pred.ravel())
    return r2_val

def max_spearman_pred(y_true, y_pred):
    """
    Max Spearman correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    spearman_vals = [np.abs(spearmanr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    max_spearman = np.nanmax(spearman_vals)
    return max_spearman

def max_pearson_pred(y_true, y_pred):
    """
    Max Pearson correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    pearson_vals = [np.abs(pearsonr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    max_pearson = np.nanmax(pearson_vals)
    return max_pearson

def max_rmse_pred(y_true, y_pred):
    """
    Max RMSE
    """
    rmse_vals = [np.sqrt(sklearn.metrics.mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    max_rmse = np.nanmax(rmse_vals)
    return max_rmse

def max_mae_pred(y_true, y_pred):
    """
    Max MAE
    """
    mae_vals = [sklearn.metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    max_mae = np.nanmax(mae_vals)
    return max_mae

def max_medae_pred(y_true, y_pred):
    """
    Max median absolute error
    """
    medae_vals = [sklearn.metrics.median_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    max_medae = np.nanmax(medae_vals)
    return max_medae

def max_r2_pred(y_true, y_pred):
    """
    Max R^2
    """
    r2_vals = [sklearn.metrics.r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    max_r2 = np.nanmax(r2_vals)
    return max_r2

# write out the mean for each of the metrics
def med_spearman_pred(y_true, y_pred):
    """
    Median Spearman correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    spearman_vals = [np.abs(spearmanr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    med_spearman = np.nanmedian(spearman_vals)
    return med_spearman

def med_pearson_pred(y_true, y_pred):
    """
    Median Pearson correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    pearson_vals = [np.abs(pearsonr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    med_pearson = np.nanmedian(pearson_vals)
    return med_pearson

def med_rmse_pred(y_true, y_pred):
    """
    Median RMSE
    """
    rmse_vals = [np.sqrt(sklearn.metrics.mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    med_rmse = np.nanmedian(rmse_vals)
    return med_rmse

def med_mae_pred(y_true, y_pred):
    """
    Median MAE
    """
    mae_vals = [sklearn.metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    med_mae = np.nanmedian(mae_vals)
    return med_mae

def med_medae_pred(y_true, y_pred):
    """
    Median median absolute error
    """
    medae_vals = [sklearn.metrics.median_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    med_medae = np.nanmedian(medae_vals)
    return med_medae

def med_r2_pred(y_true, y_pred):
    """
    Median R^2
    """
    r2_vals = [sklearn.metrics.r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    med_r2 = np.nanmedian(r2_vals)
    return med_r2

def min_spearman_pred(y_true, y_pred):
    """
    Min Spearman correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    spearman_vals = [np.abs(spearmanr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    min_spearman = np.nanmin(spearman_vals)
    return min_spearman

def min_pearson_pred(y_true, y_pred):
    """
    Min Pearson correlation
    """
    assert y_true.shape[1] == y_pred.shape[1], "y_true and y_pred must have the same number of columns, got {y_true.shape[1]} and {y_pred.shape[1]}"
    pearson_vals = [np.abs(pearsonr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    min_pearson = np.nanmin(pearson_vals)
    return min_pearson

def min_rmse_pred(y_true, y_pred):
    """
    Min RMSE
    """
    rmse_vals = [np.sqrt(sklearn.metrics.mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    min_rmse = np.nanmin(rmse_vals)
    return min_rmse

def min_mae_pred(y_true, y_pred):
    """
    Min MAE
    """
    mae_vals = [sklearn.metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    min_mae = np.nanmin(mae_vals)
    return min_mae

def min_medae_pred(y_true, y_pred):
    """
    Min median absolute error
    """
    medae_vals = [sklearn.metrics.median_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    min_medae = np.nanmin(medae_vals)
    return min_medae

def min_r2_pred(y_true, y_pred):
    """
    Min R^2
    """
    r2_vals = [sklearn.metrics.r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    min_r2 = np.nanmin(r2_vals)
    return min_r2


def flatten_df(df, join_char='_'):
    """
    Given a dataframe with only features in it and the columns and indices contain feature names,
    Returns a dataframe with a single row and the columns are the feature names joined by join_char
    """
    values_flat = df.values.flatten('C')
    df_flat = pd.DataFrame(values_flat).T
    new_columns = []
    for index in df.index:
        for col in df.columns:
            new_columns.append( index + '_' + col)
    df_flat.columns = new_columns
    return df_flat

def bin_psd_by_bands(psd, freqs, bands, method='avg', verbosity=2):
    """
    Bin a PSD by frequency bands.

    Parameters
    ----------
    psd : array_like (n_channels, n_freqs)
        Power spectral density.
    bands : list
        List frequency bands.
        [(low_freq, high_freq), ...]
    Out:
    ----
    binned_psd: array_like (n_channels, (len(bands.keys()))
        Binned PSD.
    """

    binned_psd = np.zeros((psd.shape[0], len(bands)))
    for idx, band in enumerate(bands):
        if verbosity > 0:
            print("band", band)
        freqs_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if verbosity > 1:
            print("freqs", freqs_idx)
        if method == 'avg':
            if verbosity > 1:
                print("mean", np.mean(psd[:, freqs_idx], axis=1))
            binned_psd[:, idx] = np.mean(psd[:, freqs_idx], axis=1)
        elif method == 'sum':
            binned_psd[:, idx] = np.sum(psd[:, freqs_idx], axis=1)
        elif method == 'median':
            binned_psd[:, idx] = np.median(psd[:, freqs_idx], axis=1)
        elif method == 'max':
            binned_psd[:, idx] = np.max(psd[:, freqs_idx], axis=1)
        elif method == 'min':
            binned_psd[:, idx] = np.min(psd[:, freqs_idx], axis=1)
        elif method == 'std':
            binned_psd[:, idx] = np.std(psd[:, freqs_idx], axis=1)
        elif method == 'var':
            binned_psd[:, idx] = np.var(psd[:, freqs_idx], axis=1)
        elif method == 'skew':
            binned_psd[:, idx] = np.skew(psd[:, freqs_idx], axis=1)
        else:
            raise ValueError('Invalid method. Must be one of: avg, sum, median, max, min, std, var, skew')
    
    return binned_psd

def make_bands(basis='custom', divisions=1, log_division=False, custom_bands=None, fs=500, verbosity=1, min_freq=0.3):
    """
    Given a string to specify the bands to use, return a list of tuples of the bands

    
    """
    
    basis = str(basis)
    if basis == 'standard':
        bands =  [(0.3, 1.5), (1.5, 4), (4, 8), (8, 12), (12, 30), (30, 70), (70, 150), (150, fs//2)]
    elif basis == 'log-standard':
        bands = [(1/5, 1/2), (1/2, 1/0.7), (1.5, 4), (4, 10), (10, 30), (30, 80), (80, 200), (200, 600)]# from https://www.science.org/doi/pdf/10.1126/science.1099745
    elif basis=='custom':
        bands = [(0.3, 1.5), (1.5, 4), (4, 8), (8, 12.5), (12.5, 25), (25, 36), (36, 45), (45, 70), (70, 150), (150, 250)]
    elif 'linear' in basis:
        if len(basis.split('_'))  > 1:
            max_f = int(basis.split('_')[-1])
            bands = [(f, f+1) for f in range(max_f)]
        else:
            bands = [(f, f+1) for f in range(50)]
    elif 'log' in basis:
        if len(basis.split('_'))  > 1:
            max_f = int(basis.split('_')[-1])
            bounds = np.logspace(0, np.log(max_f),7, base=np.e)
            bands = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
        else:
            bounds = np.logspace(0, np.log(50),5, base=np.e)
            bands = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    elif basis == 'custom':
        if custom_bands is None:
            raise ValueError('Must provide custom_bands argument when using custom basis')
        bands = custom_bands
    else:
        raise ValueError(f'basis must be one of anton, standard, buzaki, linear, or log, but was {basis}')
    
    if divisions > 1:
        num_bands = len(bands) * divisions
        if log_division:
            # divide each band into {divisions} evenly spaced (on a log scale) subbands
            new_bands = []
            for band in bands:
                bounds = np.logspace(np.log(band[0]), np.log(band[1]),divisions+1, base=np.e)
                if verbosity > 1:
                    print("band, bounds", band, bounds)
                new_bands += [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
            if verbosity > 0:
                print("Log divided new bands", new_bands)
            bands = new_bands
        else:
            # divide each band into {divisions} evenly spaced (on a linear scale) subbands
            new_bands = []
            for band in bands:
                bounds = np.linspace(band[0], band[1],divisions+1)
                if verbosity > 1:
                    print("band, bounds", band, bounds)
                new_bands += [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
            if verbosity > 0:
                print("Linear Division new bands", new_bands)
            bands = new_bands
        assert len(bands) == num_bands
    # throw out any bands that have an upper bound less than min_freq
    bands = [band for band in bands if band[1] > min_freq]
    # throw out any bands that have a lower bound greater than fs
    bands = [band for band in bands if band[0] < fs//2]
    if verbosity > 0:
        print("bands", bands)
    return bands


def get_spectral_edges(psd, freqs, channels=CHANNELS, edge_increment=5, chan_axis=0, return_edge_names=True):
    """
    Compute the spectral edge at edge_increment increments
    Inputs:
        data: numpy array of shape (channels, freq_samples)
        channels: list of channel names
        fs: sampling frequency
        edge_increment: increment for the spectral edge
    Outputs:
        edges array of shape (channels, num_edges)
    """
    num_edges = int(100 / edge_increment)-1
    n_channels = psd.shape[chan_axis]
    if chan_axis == 0:
        edges = np.zeros((n_channels, num_edges))
    elif chan_axis == 1:
        edges = np.zeros((psd.shape[0], n_channels, num_edges))
    else:
        raise ValueError("chan_axis must be 0 or 1")

    dx = freqs[1] - freqs[0]

    edge_names = [(edge+1)*edge_increment / 100 for edge in range(num_edges)]
    if chan_axis == 0:
        for cdx, channel in enumerate(channels):
            total_cum_power = gp_integrate.cumulative_simpson(psd[cdx, :], dx=dx, initial=0) # https://stackoverflow.com/questions/18215163/cumulative-simpson-integration-with-scipy
            total_chan_power = total_cum_power[-1]
            # another approximation for integration of signal
            # total_simps_power = scipy.integrate.simps(pxs[channel,:], dx=dx) # close but not exact
            for edx in range(num_edges):
                edge_power = total_chan_power * edge_names[edx]
                edges[cdx, edx] = freqs[np.where(total_cum_power >= edge_power)[0][0]]
    elif chan_axis == 1:
        for ndx in range(psd.shape[0]):
            for cdx, channel in enumerate(channels):
                total_cum_power = gp_integrate.cumulative_simpson(psd[ndx, cdx, :], dx=dx, initial=0)
                total_chan_power = total_cum_power[-1]
                for edx in range(num_edges):
                    edge_power = total_chan_power * edge_names[edx]
                    edges[ndx, cdx, edx] = freqs[np.where(total_cum_power >= edge_power)[0][0]]
        
    if return_edge_names:
        return edges, edge_names
    else:
        return edges

def compute_mutual_information_matrix(data, discrete_features=False, verbosity=0):
    """
    Compute the mutual information matrix between columns in a given data array.

    Parameters:
    data (numpy.ndarray): Input data array with shape (n_samples, n_features).

    Returns:
    numpy.ndarray: Mutual information matrix with shape (n_features, n_features).
    """
    num_vars = data.shape[1]
    mi_matrix = np.zeros((num_vars, num_vars))
    if isinstance(discrete_features, np.ndarray):
        bool_mask_on = True
        assert discrete_features.shape == (num_vars,)
    else:
        bool_mask_on = False
    
    for j in range(num_vars):
        comptime = time.time()
        if verbosity > 0:
            print("Computing mutual information for variable", j)

        # check if discrete features is an array of booleans
        if bool_mask_on:
            if discrete_features[j]:
                mi_matrix[:, j] = mutual_info_classif(data[:, :], data[:, j], discrete_features=discrete_features)
            else:
                mi_matrix[:, j] = mutual_info_regression(data[:, :], data[:, j], discrete_features=discrete_features)
        else:
            mi_matrix[:, j] = mutual_info_regression(data[:, :], data[:, j], discrete_features=False)

        if verbosity > 0:
            print("Time to compute mutual information for variable", j, ":", time.time() - comptime)
    return mi_matrix

def find_low_variance_columns(df, threshold=0.0):
    """
    Find columns in a DataFrame with low variance.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    threshold (float): Threshold value to determine low variance. Default is 0.0.

    Returns:
    list: List of column names with low variance.
    """
    variances = df.var()
    constant_columns = variances[variances <= threshold].index.tolist()
    return constant_columns

def find_static_columns(df, num_unique_min=1):
    """
    Find columns in a DataFrame with only a single unique value.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    list: List of column names with only a single unique value.
    """
    static_columns = df.columns[df.nunique() < num_unique_min].tolist()
    return static_columns

def add_uncorrelated_features(top_features, correlated_pairs, threshold=0.1):
    """
    greedy approach where top_features is a list of ordered features by importance 
    correlated_pairs is a series of feature pairs and their mutual information
    threshold is the threshold for mutual information to be considered correlated
    """
    new_features = []
    correlated_features = set()
    for feat in top_features:
        for feature_pair, mi in correlated_pairs.items():
            if feat in feature_pair:
                if mi < threshold:
                    if feature_pair[0] not in top_features or feature_pair[1] not in top_features:
                        continue
                    if feat not in new_features and feat not in correlated_features:
                        new_features.append(feat)
                    if feature_pair[1] not in new_features and feature_pair[1] not in correlated_features:
                        new_features.append(feature_pair[1])
                    if feature_pair[0] not in new_features and feature_pair[0] not in correlated_features:
                        new_features.append(feature_pair[0])
                else:
                    correlated_features.add(feature_pair[1])
                    correlated_features.add(feature_pair[0])
    return new_features

def process_mi_X_y(mi_X, mi_y, feature_cols, num_top_features=10):
    top_mi_df = pd.DataFrame(mi_X, columns=feature_cols, index=feature_cols)
    top_mi_df["MI"] = mi_y

    arry = top_mi_df.values
    np.fill_diagonal(arry, 0)

    top_mi_no_diag = pd.DataFrame(arry, columns=top_mi_df.columns, index=top_mi_df.index)

    sorted_top_mi_df = top_mi_df.sort_values(by="MI", ascending=False, inplace=False)
    if num_top_features is not None:
        sorted_top_mi_df = sorted_top_mi_df.iloc[:num_top_features, :]
        top_mi_no_diag = top_mi_no_diag.iloc[:num_top_features, :]

    ordered_top_features = sorted_top_mi_df.index


    top_pairs = top_mi_no_diag.unstack().sort_values(ascending=True).drop_duplicates()
    selected_features = add_uncorrelated_features(ordered_top_features, top_pairs, threshold=0.1)
    print("Number of selected features: {}".format(len(selected_features)))
    print("Selected features: {}".format(selected_features))

    out_dict = {
        "top_mi_df": top_mi_df,
        "top_mi_no_diag": top_mi_no_diag,
        "top_pairs": top_pairs,
        "selected_features": selected_features
    }
    return out_dict

def get_y_from_df(df, label_dict=LABEL_DICT):
    return np.array([label_dict[int(ind)] for ind in df.index])

def drop_duplicate_columns(df):
    """Drops any columns with duplicate name and values from a dataframe.
    Keeps the first of such columns.

    Args:
    df: A Pandas DataFrame.

    Returns:
    A Pandas DataFrame with duplicate columns removed.
    """

    # Get the unique column names.
    og_df = df.copy()
    duplicate_columns = df.columns[df.columns.duplicated()].unique()
    seen_cols = set()
    for idx, column in enumerate(duplicate_columns):
        if column in seen_cols:
            continue
        sub_df = df[column].values
        # check if any column values are identical
        if len(sub_df.shape) <= 1:
            continue
        elif sub_df.shape[1] > 1:
            if all(np.allclose(sub_df[:, 0], sub_df[:, i]) for i in range(1, sub_df.shape[1])):
                # get the idx of the first column with this name
                duplicated_col_idx = df.columns.get_loc(column)
                col_mask = np.ones(len(df.columns), dtype=bool)
                col_mask[duplicated_col_idx] = False
                col_mask[np.where(df.columns == column)[0][0]] = True

                assert len(col_mask) - sum(col_mask) == sub_df.shape[1]-1
                # set diff to preserve the other columns not in the subset
                
                df = df.loc[:, col_mask]
            else:
                pass # not sure what to do if some dups are identical and some are not
        seen_cols.add(column)

    return df

def separate_feature_df(df, seps=['open', 'closed']):
    out_dict = {}
    for sep in seps:
        sepcols = [col for col in df.columns if sep in col]
        out_dict[sep] = df[sepcols]
    return out_dict

def print_nans(df, verbose=True):
    nan_indices = np.where(pd.isna(df))
    n_nans = len(nan_indices[0])
    print(f"Total NaNs: {n_nans}")
    if verbose:
        for row, col in zip(*nan_indices):
            print(f"NaN value at row {row} with index value {df.index[row]} in column {df.columns[col]}")

    
    return nan_indices
    
def print_infs(df):

    inf_indices = np.where(np.isinf(df))
    n_infs = len(inf_indices[0])
    print(f"Total Infs: {n_infs}")
    for row, col in zip(*inf_indices):
        print(f"Inf value at row {row} with index value {df.index[row]} in column {df.columns[col]}")
    return inf_indices

def get_binary_confusion_matrix(y_true, y_pred):
    """
    Returns a confusion matrix for the given true and predicted labels.
    """
    tps = np.sum((y_true == 1) & (y_pred == 1))
    fps = np.sum((y_true == 0) & (y_pred == 1))
    fns = np.sum((y_true == 1) & (y_pred == 0))
    tns = np.sum((y_true == 0) & (y_pred == 0))
    out_dict = {
        "TP": tps,
        "FP": fps,
        "FN": fns,
        "TN": tns
    }
    
    return out_dict


def load_df_to_from(df, load_to=None, load_from=None, random_load=False, num_load_subjs=None, base_folder='data/tables/'):
    annotations = ld.load_annotations(base_folder=base_folder)
    subjs = annotations['Study ID']
    if load_to is not None:
        subjs = subjs.iloc[:load_to]
        
    if load_from is not None:
        subjs = subjs.iloc[load_from:]

    if random_load:
        if num_load_subjs is None:
            num_load_subjs = len(subjs)
        subjs = subjs.sample(num_load_subjs)

    if isinstance(subjs.values[0], int):
        out_df = df.loc[[s for s in df.index if int(s) in subjs.values]]
    elif isinstance(subjs.values[0], str):
        out_df = df.loc[[s for s in df.index if str(s) in subjs.values]]
    else:
        out_df = df.loc[[s for s in df.index if str(s) in subjs.values or int(s) in subjs.values]]
    return out_df

# https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python/33532498#33532498
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def feature_df_to_regional_df(X_df: DataFrame, regional_mapping: Dict[str, List[str]], method: str ='mean', channels: List[str] = None, unique_features: List[str] = None) -> DataFrame:
    """
    Merges features in a dataframe into regional features based on a regional mapping.
    Inputs:
        X_df: DataFrame with features
        regional_mapping: dictionary with keys as region names and values as lists of channel names
        method: method to use to aggregate features in each region (mean, std, median, sum)
        channels (optional): list of channels to consider (default is all channels)
        unique_features (optional): list of unique features to consider (default is all unique features)

    Outputs:
        regional_df: DataFrame with regional features
    """
    valid_methods = ['mean', 'std', 'median', 'sum', 'skew', 'kurtosis', 'iqr', 'max', 'min']
    assert method in valid_methods or (method[0] == 'p' and method[1:].isdigit()), f"Method {method} not recognized"
    regional_df = pd.DataFrame()
    if channels is None:
        channels = mne.channels.make_standard_montage('standard_1020').ch_names
        channels = [ch for ch in channels if any([ch in col for col in X_df.columns])]
    if unique_features is None:
        unique_features = set()
        for col in X_df.columns:
            for ch in channels:
                if ch in col:
                    unique_features.add(col.replace(ch, ''))
    assert all([ch for ch in channels if any([ch in region for region in regional_mapping.values()])])
    for region, channels in regional_mapping.items():
        region_cols = [col for col in X_df.columns if any([ch in col for ch in channels])]
        for uf in unique_features:
            uf_cols = [col for col in region_cols if all([char in col for char in uf])]
            if len(uf_cols) == 0:
                continue
            region_feat = f'{region}_{uf}'
            if method == 'mean':
                regional_df[region_feat] = X_df[uf_cols].mean(axis=1)
            elif method == 'std':
                regional_df[region_feat] = X_df[uf_cols].std(axis=1)
            elif method == 'median':
                regional_df[region_feat] = X_df[uf_cols].median(axis=1)
            elif method == 'sum':
                regional_df[region_feat] = X_df[uf_cols].sum(axis=1)
            elif method == 'skew':
                regional_df[region_feat] = X_df[uf_cols].skew(axis=1)
            elif method == 'kurtosis':
                regional_df[region_feat] = X_df[uf_cols].kurtosis(axis=1)
            elif method == 'iqr':
                regional_df[region_feat] = X_df[uf_cols].quantile(0.75, axis=1) - X_df[uf_cols].quantile(0.25, axis=1)
            elif method == 'max':
                regional_df[region_feat] = X_df[uf_cols].max(axis=1)
            elif method == 'min':
                regional_df[region_feat] = X_df[uf_cols].min(axis=1)
            elif method[0] == 'p':
                percentile = int(method[1:])
                regional_df[region_feat] = X_df[uf_cols].quantile(percentile/100, axis=1)
            else:
                raise ValueError('Method {} not recognized'.format(method))
    return regional_df


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two Gaussian distributions.

    Parameters:
    - mu1: Mean of the first Gaussian distribution.
    - sigma1: Standard deviation of the first Gaussian distribution.
    - mu2: Mean of the second Gaussian distribution.
    - sigma2: Standard deviation of the second Gaussian distribution.

    Returns:
    - The KL divergence between the two distributions.
    """ 
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5 # KL = E[log(sigma2/sigma1) + 1/2 * [(x-mu2)/sigma2]^2 - (x-mu1)/sigma1]^2]

if __name__ == '__main__':
    pass