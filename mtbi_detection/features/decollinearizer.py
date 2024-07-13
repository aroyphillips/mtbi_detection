## Decollinarizer: A class to select top features and remove collinear features from a dataset in a greedy manner
import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
import timeit
import scipy
import itertools

import time
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from collections import defaultdict

import mtbi_detection.data.load_dataset as ld
import mtbi_detection.features.feature_utils as fu
from sklearn.linear_model import LogisticRegression


def calculate_kendall_tau_column(col1, col2):
    try:
        tau, _ = scipy.stats.kendalltau(col1, col2)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    time.sleep(0)
    return tau

def calculate_kendall_tau_parallel(X, num_processes, verbosity=0):
    """
    Calculate Kendall Tau correlation for each column in parallel.

    Args:
    - X: Input matrix (DataFrame or ndarray)
    - num_processes: Number of processes to use for parallelization

    Returns:
    - out_mat: Output matrix with Kendall Tau correlation between each pair of columns
    """

    if isinstance(X, pd.DataFrame):
        X = X.values

    num_columns = X.shape[1]
    if verbosity > 0:
        print(f"Calculating Kendall Tau correlation for {num_columns} columns...")
        starttime = time.time() 
    # use joblib to parallelize 
    results = Parallel(n_jobs=num_processes)(delayed(calculate_kendall_tau_column)(X[:, col1].copy(), X[:, col2].copy()) for col1, col2 in itertools.combinations(range(num_columns), 2))

    if verbosity > 1:
        print(f"Time to calculate Kendall Tau correlation for {num_columns} columns:", time.time() - starttime)
    out_mat = np.zeros((num_columns, num_columns))
    out_mat[np.triu_indices(num_columns, k=1)] = results
    out_mat += out_mat.T
    # set the diagonal to 1
    np.fill_diagonal(out_mat, 1.0)
    assert np.allclose(out_mat, out_mat.T), "out_mat should be symmetric"
    assert np.allclose(np.diag(out_mat), 1.0), "diagonal of out_mat should be 1.0"

    return out_mat

def compute_mi(data_col, data_mat, discrete_features, bool_mask_on, is_classif=True):
    """
    Given a column, compute the mutual information between that column and all other columns in the data array.
    inputs:
        data_col (n_samples, ): column 
        half_data_mat (n_samples, n_features): half of the data matrix
        discrete_features: boolean array indicating whether each feature is discrete or continuous
        bool_mask_on: whether to use the discrete_features mask
        is_classif: whether to use mutual_info_classif or mutual_info_regression
    outputs:
        mi (n_features, ): mutual information between column j and all other columns in data
    """
    try:
        if bool_mask_on:
            if is_classif:
                mi = mutual_info_classif(data_mat, data_col, discrete_features=discrete_features)
            else:
                mi = mutual_info_regression(data_mat, data_col, discrete_features=discrete_features)
        else:
            mi = mutual_info_regression(data_mat, data_col, discrete_features=False)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    time.sleep(0)
    return mi

def compute_mutual_information_matrix(data, n_jobs=10, discrete_features=False, verbosity=0):
    """
    Compute the mutual information matrix between columns in a given data array.

    Parameters:
    data (numpy.ndarray): Input data array with shape (n_samples, n_features).
    n_jobs (int): Number of jobs to run in parallel (default: None consumes all resources).
    discrete_features (numpy.ndarray): Boolean array indicating whether each feature is discrete or continuous (default: False).


    Returns:
    numpy.ndarray: Mutual information matrix with shape (n_features, n_features).
    """
    num_vars = data.shape[1]
    mi_matrix = np.zeros((num_vars, num_vars))
    if isinstance(discrete_features, np.ndarray):
        bool_mask_on = True
        assert discrete_features.shape == (num_vars,)
        is_classif = [True if feat else False for feat in discrete_features]
    else:
        bool_mask_on = False
        is_classif = [False for _ in range(num_vars)]

    # Use joblib to parallelize 
    ## minimize unnecessary computation by shrinking the data matrix on each iteration
    mi_values = Parallel(n_jobs=n_jobs)(delayed(compute_mi)(data[:, col].copy(), data[:, col:].copy(), discrete_features, bool_mask_on, is_classif=is_classif[col]) for col in range(num_vars))
    ## the output of this is a list of arrays [(num_vars,) (num_vars-1,) ... (1,))]
    mi_matrix = np.zeros((num_vars, num_vars))
    for idx, mi in enumerate(mi_values):
        mi_matrix[idx, idx:] = mi
        mi_matrix[idx:, idx] = mi

    return mi_matrix


def add_uncorrelated_features(top_features, correlated_pairs, threshold=0.1, verbosity=0):
    """
    greedy approach where top_features is a list of ordered features by importance 
    correlated_pairs is a series of feature pairs and their mutual information
    threshold is the threshold for mutual information to be considered correlated
    Input: 
        top_features: array like (including pandas index) of ordered features by importance
        correlated_pairs: pandas series of feature pairs and their mutual information
        threshold: threshold for mutual information to be considered correlated
        verbosity: verbosity level for printing
    Output:
        new_features: list of features that are uncorrelated with top_features
    """
    correlated_or_seen_features = defaultdict(list)
    new_features = []
    correlated_pairs_list = list(correlated_pairs.items())
    for idx, feat in enumerate(top_features):
        # find all pairs in correlated_pairs that contain feat
        if feat in correlated_or_seen_features.keys():
            if verbosity > 2:
                print(f"Feature {feat} already in correlated features list")
            continue
        if idx == 0:
            if verbosity > 2:
                print(f"Adding top feature {feat} to new features list")
            new_features.append(feat)
            correlated_or_seen_features[feat] = []
        
        # add the pairs above threshold to the correlated features set
        for idx, (feature_pair, mi) in enumerate(correlated_pairs_list):
            if verbosity > 4:
                print(f"Processing feature_pair {idx}/{len(correlated_pairs_list)}")
            if feat in feature_pair and mi >= threshold:
                if verbosity > 3:
                    print(f"Adding feature pair {feature_pair} to correlated features set due to feature {feat} with mi {mi}")
                correlated_or_seen_features[feature_pair[0]] = []
                correlated_or_seen_features[feature_pair[1]] = []
                time.sleep(0)
            else:
                unseen_feat = feature_pair[0] if feat == feature_pair[1] else feature_pair[1]
                if unseen_feat in correlated_or_seen_features.keys():
                    if verbosity > 2:
                        print(f"Feature {unseen_feat} already in correlated features list")
                    continue
                else: 
                    if verbosity > 2:
                        print(f"Adding feature {unseen_feat} to new features list")
        
                    new_features.append(unseen_feat)
                    correlated_or_seen_features[unseen_feat] = []

                    # unseen_feat_pairs = [(feature_pair, mi) for (feature_pair, mi) in correlated_pairs.items() if unseen_feat in feature_pair]
                    for (feature_pair, mi) in correlated_pairs_list[idx:]:
                        if unseen_feat in feature_pair and mi >= threshold:
                            if verbosity > 3:
                                print(f"Adding feature pair {feature_pair} to correlated features list due to feature {unseen_feat} with mi {mi}")
                            correlated_or_seen_features[feature_pair[0]] = []
                            correlated_or_seen_features[feature_pair[1]] = []
                time.sleep(0)

    if verbosity > 5:     
        print(f"Len new features: {len(new_features)}, len top features: {len(top_features)}, new features: {new_features}, top features: {top_features}")
    assert len(new_features) <= len(top_features), "new features should be less than or equal to top features but the following features are in new features but not in top features: {}".format(set(new_features) - set(top_features))
    return new_features

def process_mi_X_y(mi_X, mi_y, feature_cols, num_top_features=10, targ_threshold=None, feat_threshold=0.1, verbosity=0):
    """
    Given an array of relationships between the features and the target, and an array of relationships between the features,
    select the top features and remove for features with collinearity above the threshold.
    """
    st = time.time()
    if verbosity > 0:
        print(f"Processing {mi_X.shape} feature matrix and {mi_y.shape} target matrix...")

    top_mi_df = pd.DataFrame(mi_X, columns=feature_cols, index=feature_cols)
    top_mi_df["MI"] = mi_y

    arry = top_mi_df.values
    np.fill_diagonal(arry, 0)

    top_mi_no_diag = pd.DataFrame(arry, columns=top_mi_df.columns, index=top_mi_df.index)

    if verbosity > 0:
        print("Sorting top features...")
        sorttime = time.time()
    sorted_top_mi_df = top_mi_df.sort_values(by="MI", ascending=False, inplace=False)
    if verbosity > 1:
        print(f"Time to sort top features: {time.time() - sorttime}")
    if targ_threshold is not None:
        if verbosity > 0:
            print("Filtering top features by target threshold...")
            threshtime = time.time()
        sorted_top_mi_df = sorted_top_mi_df[sorted_top_mi_df["MI"] > targ_threshold]
        if verbosity > 1:
            print(f"Number of top features after filtering by target threshold: {len(sorted_top_mi_df)}")
            if verbosity > 3:
                print(f"Top features after filtering by target threshold: {sorted_top_mi_df.index}")
        if verbosity > 1:
            print(f"Time to filter top features by target threshold: {time.time() - threshtime}")
    if num_top_features is not None:
        if verbosity > 0:
            print("Selecting top features...")
            selecttime = time.time()
        sorted_top_mi_df = sorted_top_mi_df.iloc[:num_top_features, :]
        if verbosity > 1:
            print(f"Time to select top features: {time.time() - selecttime}")


    ordered_top_features = sorted_top_mi_df.index

    if verbosity > 1:
        print("Creating top pairs...")
        toptime = time.time()
    top_mi_no_diag = top_mi_no_diag.loc[ordered_top_features, ordered_top_features]
    # get the upper triangle of the matrix
    top_mi_no_diag = top_mi_no_diag.where(np.triu(np.ones(top_mi_no_diag.shape), k=1).astype(bool))
    top_pairs = top_mi_no_diag.unstack().sort_values(ascending=False).drop_duplicates().dropna()
    del top_mi_no_diag

    if verbosity > 2:
        print(f"Number of top pairs: {len(top_pairs)}")
        print(f"ordered_top_features shape {ordered_top_features.shape}")
        # print(f"top_mi_no_diag shape {top_mi_no_diag.shape}")
    if verbosity > 1:
        print(f"Time to create top pairs: {time.time() - toptime}")

    if verbosity > 0:
        print("Adding uncorrelated features...")
        addtime = time.time()
    selected_features = add_uncorrelated_features(ordered_top_features, top_pairs, threshold=feat_threshold, verbosity=verbosity-1)
    if verbosity > 1:
        print(f"Time to add uncorrelated features: {time.time() - addtime}")
    if verbosity > 0:
        print("Number of selected features: {}".format(len(selected_features)))
    if verbosity > 1:
        print("Selected features: {}".format(selected_features))

    out_dict = {
        # "top_mi_df": top_mi_df,
        # "top_mi_no_diag": top_mi_no_diag,
        # "top_pairs": top_pairs,
        "selected_features": selected_features
    }
    if verbosity > 0:
        print(f"Time to process feature matrix and target matrix: {time.time() - st}")
    return out_dict

class Decollinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_features=1, num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=10, verbosity=0, prune_method='pearson', targ_method='anova_pinv', label_type='classification'):
        """
        Initialize the Decollinarizer class with the following parameters
        Args:
            num_features: int (number of top features to prune from)
            targ_threshold: float (minimum threshold for mutual information between target and features)
            feat_threshold: float (maximum threshold for mutual information between features)
            discrete_features: bool or array (whether to treat features as discrete or continuous, if array, then True for discrete features and False for continuous features)
            cols: list (list of column names, necessary if not using a dataframe during fitting)
            n_jobs: int (number of jobs to run in parallel (default: None consumes all resources))
            verbosity: int (level of verbosity, 0 is silent, 1 prints after each fitting)
            prune_method: str (method to use for pruning features, options: 'mutual_information', 'pearson', 'spearman', 'kendall')
            targ_method: str (method to use for calculating mutual information between target and features, options: 'mutual_information', 'pearson', 'spearman', 'kendall', 'anova_pinv')
            label_type: str (type of target, options: 'classification', 'regression')

        Example usage:
        ```
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        deco = Decollinarizer(num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=10, verbosity=0, prune_method='pearson', targ_method='anova_pinv', label_type='classification')
        deco.fit(X, y)
        X_select= deco.transform(X)
        ```
        """
        self.num_features = num_features
        self.feature_indices_ = None
        self.feat_threshold = feat_threshold
        self.targ_threshold = targ_threshold
        self.discrete_features = discrete_features
        self.cols = cols
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.prune_method = prune_method
        self.targ_method = targ_method
        self.label_type = label_type
        self.min_features = min_features

        if self.feat_threshold > 1.0:
            self.skip_feat_threshold = True
        else:
            self.skip_feat_threshold = False

        if self.verbosity > 0:
            print(f"Decollinarizer initialized with num_features={self.num_features}, feat_threshold={self.feat_threshold}, targ_threshold={self.targ_threshold}, discrete_features={self.discrete_features}, cols={self.cols}, n_jobs={self.n_jobs}, verbosity={self.verbosity}")

    def fit(self, X, y=None):
        # check if X is a dataframe
        st = time.time()
        if not isinstance(X, pd.DataFrame):
            if self.cols is None:
                cols = [f"col_{i}" for i in range(X.shape[1])]
            else:
                if X.shape[1] != len(self.cols):
                    print(f"Number of columns in X ({X.shape[1]}) does not match number of columns in cols ({len(self.cols)}), using default column names.")
                    cols = [f"col_{i}" for i in range(X.shape[1])]
                else:
                    cols = self.cols
            X = pd.DataFrame(X, columns=cols)
            self.cols = cols

        if X.shape[1] < self.min_features:
            print(f"Number of possible features ({X.shape[1]}) is less than min_features ({self.min_features}), using ({X.shape[1]}) features")
            self.min_features = X.shape[1]

        mi_y = self._get_mi_y(X, y)
        # store only the top self.num_features features
        if self.num_features is not None:
            sorted_mi_y = np.argsort(mi_y)[::-1]
            mi_y = mi_y[sorted_mi_y[:self.num_features]]
            X = X.iloc[:, sorted_mi_y[:self.num_features]]
        assert X.shape[1] == len(mi_y), f"X shape {X.shape} does not match mi_y shape {mi_y.shape}" 
        
        if self.verbosity > 2:
            # print the size of the dataframe (in bytes) and the number of features
            print(f"X shape: {X.shape} ({X.values.nbytes / 1e9} GB), y shape: {y.shape} ({y.nbytes / 1e9} GB)")
    
        if not self.skip_feat_threshold:
            mi_X = self._compute_mi_X(X)
            mi_dict = process_mi_X_y(mi_X, mi_y, X.columns, num_top_features=self.num_features, targ_threshold=self.targ_threshold, feat_threshold=self.feat_threshold, verbosity=self.verbosity)
        else:
            mi_dict = {"selected_features": X.columns}
            mi_X = None

        if len(mi_dict["selected_features"]) < self.min_features:
            print(f"Number of selected features ({len(mi_dict['selected_features'])}) is less than min_features ({self.min_features}), adding {self.min_features - len(mi_dict['selected_features'])} features by default.")
            top_features_not_in_selected = X.columns[np.argsort(mi_y)[::-1]][~X.columns.isin(mi_dict["selected_features"])]
            top_features = mi_dict["selected_features"] + top_features_not_in_selected[:(self.min_features - len(mi_dict["selected_features"]))].tolist()
            mi_dict["selected_features"] = top_features
        assert len(mi_dict["selected_features"]) >= self.min_features, f"Number of selected features ({len(mi_dict['selected_features'])}) is less than min_features ({self.min_features})"
        
        self.feature_indices_ = [X.columns.get_loc(feat) for feat in mi_dict["selected_features"]]
        self.support_ = np.zeros(len(X.columns), dtype=bool)
        self.support_[self.feature_indices_] = True
        self.ranking_ = np.argsort(mi_y)[::-1]

        if self.verbosity > 0:
            print(f"Finished fitting Decollinarizer in {time.time() - st} seconds")
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'feature_indices_') ## NOTE: Not convinced why im using feature_indices instead of support_
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.feature_indices_]
        else:
            return X[:, self.feature_indices_]  
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self):
        check_is_fitted(self, 'feature_indices_')
        return self.feature_indices_    
    
    def _get_mi_y(self, X, y):
        """
        Given a dataframe X and a target y, compute the mutual information between X and y
        Inputs:
            X: dataframe of features (n_samples, n_features)
            y: target (n_samples,)
        Output:
            mi_y: mutual information between X and y (n_features,)
        """
        if self.verbosity > 0:
            print(f"Computing {self.targ_method} matrix between target and features...")
            computetime = time.time()
        if self.targ_method == 'mutual_information' and self.label_type=='classification':
            mi_y = mutual_info_classif(X.values, y, discrete_features=self.discrete_features)
        elif self.targ_method == 'mutual_information' and self.label_type=='regression':
            mi_y = mutual_info_regression(X.values, y, discrete_features=self.discrete_features)
        elif self.targ_method == 'pearson':
            mi_y = np.abs(np.corrcoef(X, y, rowvar=False)[:-1, -1])
        elif self.targ_method == 'spearman':
            mi_y = np.abs(X.corrwith(pd.Series(y), method='spearman').values)
        elif self.targ_method == 'kendall':
            mi_y = np.abs(X.corrwith(pd.Series(y), method='kendall').values)
        elif self.targ_method == 'anova_pinv':
            mi_y = fu.anova_pinv(X, y)
        # multivariate options
        elif self.targ_method == 'avg_mi_reg':
            mi_y = fu.avg_mi_reg(X, y)
        elif self.targ_method == 'avg_spearman':
            mi_y = fu.avg_spearman_corr(X, y)
        elif self.targ_method == 'avg_kendall':
            mi_y = fu.avg_kendall_corr(X, y)
        elif self.targ_method == 'avg_pearson':
            mi_y = fu.avg_pearson_corr(X, y)
        elif self.targ_method == 'max_mi_reg':
            mi_y = fu.max_mi_reg(X, y)
        elif self.targ_method == 'max_spearman':
            mi_y = fu.max_spearman_corr(X, y)
        elif self.targ_method == 'max_kendall':
            mi_y = fu.max_kendall_corr(X, y)
        elif self.targ_method == 'max_pearson':
            mi_y = fu.max_pearson_corr(X, y)
        else:
            raise ValueError(f"targ_method {self.targ_method} not recognized")

        if self.verbosity > 0:
            print(f"Time to compute {self.targ_method} matrix between target and features:", time.time() - computetime)
            print(f"Processing {self.prune_method} feature matrix and {self.targ_method} target matrix...")

        return mi_y
    
    def _compute_mi_X(self, X):
        if self.verbosity > 0:
            print(f"Computing {self.prune_method} matrix...")
            computetime = time.time()

        if self.prune_method == 'mutual_information':
            mi_X = compute_mutual_information_matrix(X.values, discrete_features=self.discrete_features, n_jobs=self.n_jobs, verbosity=self.verbosity)
        elif self.prune_method == 'pearson':
            mi_X = np.abs(np.corrcoef(X, rowvar=False)) # much faster than mutual information matrix
        elif self.prune_method == 'spearman':
            mi_X = np.abs(X.corr(method='spearman').values)
        elif self.prune_method == 'kendall':
            # mi_X = np.abs(X.corr(method='kendall').values)
            mi_X = calculate_kendall_tau_parallel(X, num_processes=self.n_jobs, verbosity=self.verbosity)
        else:
            raise ValueError(f"prune_method {self.prune_method} not recognized")
        if self.verbosity > 0:
            print(f"Time to compute {self.prune_method} matrix:", time.time() - computetime)
        if self.verbosity >2:
            print(f"mi_X shape with {self.prune_method}: {mi_X.shape} ({mi_X.nbytes / 1e9} GB)")
        return mi_X

  
    
### TESTS ###
# test get_mi_y and memory
def _test_get_mi_y():
    targ_methods = ['mutual_information', 'pearson', 'spearman', 'kendall', 'anova_pinv']
    label_types = ['classification', 'regression']
    n_jobs = [1, 2, 5, 10]
    times = np.zeros((len(targ_methods), len(label_types), len(n_jobs)))
    n_samples = 100
    n_features = 5000
    for tdx, targ_method in enumerate(targ_methods):
        for ldx, label_type in enumerate(label_types):
            for ndx, n_job in enumerate(n_jobs):
                st = time.time()

                print(f"Testing targ_method {targ_method} and label_type {label_type}")
                if label_type == 'classification':
                    y = np.random.randint(0, 2, size=n_samples)
                else:
                    y = np.random.rand(n_samples)
                deco = Decollinarizer(min_features=3, num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=n_job, verbosity=10, prune_method='pearson', targ_method=targ_method, label_type=label_type)
                X = np.random.rand(n_samples, n_features)
                X = pd.DataFrame(X)
                mi_y = deco._get_mi_y(X, y)
                assert mi_y.shape == (n_features,), f"mi_y shape {mi_y.shape} does not match number of features {n_features}"
                print(f"Passed targ_method {targ_method} and label_type {label_type} in {time.time() - st} seconds")
                times[tdx, ldx, ndx] = time.time() - st
    print("Passed all targ_methods and label_types")
    # print average times in each dimension
    avg_time_per_targ_method = np.mean(times, axis=1)
    avg_time_per_label_type = np.mean(times, axis=0)
    # make dataframe
    avg_time_targ_df = pd.DataFrame(avg_time_per_targ_method, index=targ_methods, columns=n_jobs)
    avg_time_label_df = pd.DataFrame(avg_time_per_label_type, index=label_types, columns=n_jobs)
    print("Average time per targ_method:")
    print(avg_time_targ_df)
    print("Average time per label_type:")
    print(avg_time_label_df)

    """
    1000 features, 1000 samples:
    n_jobs: 1         2         5         10
    mutual_information  3.932526  3.895281  3.893569  3.894939
    pearson             0.030069  0.028773  0.026410  0.026148
    spearman            0.996852  0.989144  0.976997  0.959864
    kendall             0.658808  0.664909  0.658148  0.676353
    anova_pinv          0.074289  0.068683  0.067094  0.067494
    
    100 samples, 1000 features:
                            1         2         5         10
    mutual_information  1.767056  1.755238  1.757120  1.757642
    pearson             0.012270  0.009435  0.008035  0.008010
    spearman            0.901474  0.877016  0.867222  0.854112
    kendall             0.560961  0.538049  0.540278  0.541776
    anova_pinv          0.023310  0.018290  0.017894  0.017715

    100 samples, 10000 features:
                        1         2         5         10
    mutual_information  1.767056  1.755238  1.757120  1.757642
    pearson             0.012270  0.009435  0.008035  0.008010
    spearman            0.901474  0.877016  0.867222  0.854112
    kendall             0.560961  0.538049  0.540278  0.541776
    anova_pinv          0.023310  0.018290  0.017894  0.017715
    100, 5000 features:
                    1         2         5         10
    mutual_information  8.989847  8.949269  8.981092  8.971698
    pearson             0.189510  0.188736  0.192353  0.187155
    spearman            4.278222  4.266773  4.251042  4.238678
    kendall             2.711382  2.702496  2.701907  2.704412
    anova_pinv          0.070226  0.063181  0.062789  0.062543
    """
    
    

def _test_get_mi_X():
    prune_methods = ['pearson', 'spearman'] # 'mutual_information',  'kendall']
    n_jobs = [1, 2, 5, 10, 50]
    times = np.zeros((len(prune_methods), len(n_jobs)))
    n_samples = 100
    n_features = 1000
    for pdx, prune_method in enumerate(prune_methods):
        for ndx, n_job in enumerate(n_jobs):
            print(f"Testing prune_method {prune_method}")
            st = time.time()
            deco = Decollinarizer(min_features=3, num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=n_job, verbosity=10, prune_method=prune_method, targ_method='anova_pinv', label_type='classification')
            X = np.random.rand(n_samples, n_features)
            X = pd.DataFrame(X)
            mi_X = deco._compute_mi_X(X)
            assert mi_X.shape == (n_features, n_features), f"mi_X shape {mi_X.shape} does not match X shape {X.shape}"
            # assert that it is symmetric
            assert np.allclose(mi_X, mi_X.T), "mi_X should be symmetric"
            print(f"Passed prune_method {prune_method} in {time.time() - st} seconds")
            times[pdx, ndx] = time.time() - st
    print("Passed all prune_methods")
    # print average times in each dimension
    time_df= pd.DataFrame(times, index=prune_methods, columns=n_jobs)
    print("Time per prune_method:")
    print(time_df)
    """
                            1   ...          50
    mutual_information  733.067922  ...  253.664270
    pearson               0.025699  ...    0.014338
    spearman              0.133133  ...    0.084778
    kendall             288.833874  ...  394.199695

                    1         2         5         10        50
    pearson   0.022718  0.015639  0.014272  0.012940  0.013809
    spearman  0.079133  0.077430  0.076127  0.075893  0.076371
    """


def _test_process_mi_X_y():
    n_samples = 100
    n_features = 1000
    X = np.random.rand(n_samples, n_features)
    X = pd.DataFrame(X)
    X.columns = [f"col_{i}" for i in range(n_features)]
    y = np.random.randint(0, 2, size=n_samples)
    deco = Decollinarizer(min_features=3, num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=10, verbosity=10, prune_method='pearson', targ_method='anova_pinv', label_type='classification')
    mi_X = deco._compute_mi_X(X)
    mi_y = deco._get_mi_y(X, y)
    mi_dict = process_mi_X_y(mi_X, mi_y, X.columns, num_top_features=10, feat_threshold=0.1)
    print("Passed process_mi_X_y")

def _test_fit():
    n_samples = 100
    n_features = 1000
    X = np.random.rand(n_samples, n_features)
    X = pd.DataFrame(X)
    X.columns = [f"col_{i}" for i in range(n_features)]
    y = np.random.randint(0, 2, size=n_samples)
    deco = Decollinarizer(min_features=3, num_features=10, targ_threshold=None, feat_threshold=0.1, discrete_features=False, cols=None, n_jobs=10, verbosity=10, prune_method='pearson', targ_method='anova_pinv', label_type='classification')
    deco.fit(X, y)
    print("Passed fit")

    deco.transform(X)
    print("Passed transform")

    deco.fit_transform(X, y)
    print("Passed fit_transform")

def _run_all_tests():
    _test_get_mi_y()
    _test_get_mi_X()
    _test_process_mi_X_y()
    _test_fit()

if __name__ == '__main__':
    _run_all_tests()
    # code to test mutual information in classification
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mi_time = timeit.timeit("compute_mutual_information_matrix(X_train.values)", number=100, globals=globals())
    print(f"Time taken to compute mutual information matrix: {mi_time}")


    mi_time = time.time()
    mi_X = compute_mutual_information_matrix(X_train.values)
    print("Time taken to compute MI: {}".format(time.time() - mi_time))
    mi_y = mutual_info_classif(X_train.values, y_train)

    mi_dict = process_mi_X_y(mi_X, mi_y, X_train.columns, num_top_features=10, feat_threshold=0.1)

    selected_features = mi_dict["selected_features"]

    X_train_mi = X_train.loc[:, selected_features]
    X_test_mi = X_test.loc[:, selected_features]

    lr = LinearRegression()
    lr.fit(X_train_mi, y_train)
    y_pred = lr.predict(X_test_mi)
    print("MSE: {}".format(mean_squared_error(y_test, y_pred)))

    # test pipeline
    pipe = Pipeline([
        ("select_mi", Decollinarizer(num_features=10, feat_threshold=0.5, cols=X_train_mi.columns, verbosity=0, prune_method='kendall', targ_method='kendall', n_jobs=64)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("MSE: {}".format(mean_squared_error(y_test, y_pred)))

    # try a cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    print("MSE: {}".format(np.mean(scores)))


