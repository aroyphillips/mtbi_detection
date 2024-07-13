import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import plotly
import shap
import mne

import mtbi_detection.modeling.final_ensemble_perturbations_resfit as fepr
import mtbi_detection.features.feature_utils as fu
import mtbi_detection.modeling.model_utils as mu
import mtbi_detection.modeling.model_analysis as ma
import mtbi_detection.modeling.model_selection as ms
import mtbi_detection.modeling.transfer_binary2regression as tbr
import scipy

import sklearn

CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

import matplotlib.font_manager as fm
def get_font_prop(fontsize=14, weight='normal'):
    font_path = '/home/ap60/inter.ttf'
    fontprop = fm.FontProperties(fname=font_path, size=fontsize, weight=weight)
    return fontprop
def extract_pred_results(pred_results, title='Base Predictions', scores=['MCC', 'Bal Acc.', 'Sensitivity', 'Specificity', 'ROC AUC'], figsize=(10, 6), ylim=[0,1]):
    pos_1_df = pred_results[[col for col in pred_results.columns if col.endswith('_1')]]

    all_models = [col.split('_')[0] for col in pos_1_df.columns]
    all_scores = {model: {} for model in all_models}
    y_true = fu.get_y_from_df(pos_1_df)
    for col in pos_1_df.columns:
        model = col.split('_')[0]
        rounded_values = pos_1_df[col].round().astype(int)
        for score in scores:
            if score == 'MCC':
                all_scores[model][score] = sklearn.metrics.matthews_corrcoef(y_true, rounded_values)
            elif score == 'Bal Acc.':
                all_scores[model][score] = sklearn.metrics.balanced_accuracy_score(y_true, rounded_values)
            elif score == 'Sensitivity':
                all_scores[model][score] = sklearn.metrics.recall_score(y_true, rounded_values)
            elif score == 'Specificity':
                all_scores[model][score] = sklearn.metrics.recall_score(y_true, rounded_values, pos_label=0)
            elif score == 'ROC AUC':
                all_scores[model][score] = sklearn.metrics.roc_auc_score(y_true, pos_1_df[col])
    all_scores_df = pd.DataFrame(all_scores)
    all_scores_df = all_scores_df.T
    return all_scores_df

def plot_holdout_perturbations(perturbation_df, title='Holdout Perturbations', score_names=['MCC', 'Balanced Accuracy', 'Sensitivity', 'Specificity', 'ROC AUC'], figsize=(10, 6), ylim=[0,1], model_names=['rf', 'lr', 'xgb'], fontsize=14, fontprop=None):

    # Reset the index
    df = perturbation_df.reset_index().rename(columns={'index': 'split'})

    # Melt the DataFrame to a long format
    df_melted = df.melt(id_vars='split', var_name='model_score', value_name='Score')

    # Extract model and score from model_score
    df_melted['model_name'] = [[mname for mname in model_names if mname in val][0] for val in df_melted['model_score']]
    df_melted['score_name'] = [[sname for sname in score_names if sname in val][0] for val in df_melted['model_score']]


    # Calculate the mean and standard error for each group
    df_grouped = df_melted.groupby(['model_name', 'score_name']).Score.agg(['mean', 'sem']).reset_index()

    # Create the barplot
    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x='model_name', y='Score', hue='score_name')
    # Add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=fontsize*.8)
    # set xticklabels
    xticklabels = plt.gca().get_xticklabels()
    for label in xticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize*.8)
    # set yticklabels
    yticklabels = plt.gca().get_yticklabels()
    for label in yticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize)
    # set the y-label
    plt.ylabel('', fontproperties=fontprop, fontsize=fontsize*.8)
    
    label_conversion = {'rf': 'Random Forest', 'lr': 'Logistic Regression', 'xgb': 'XGBoost'}
    plt.xlabel('', fontsize=fontsize, fontproperties=fontprop)
    plt.xticks(ticks=[0, 1, 2], labels=[label_conversion[mname] for mname in model_names])
    plt.title(title, fontsize=fontsize*1.2)# fontproperties=fontprop)
    plt.ylabel('Score', fontsize=fontsize*.8, fontproperties=fontprop)
    plt.ylim(ylim)
    plt.show()


def extract_perturbation_latex(perturbation_df, score_names=['MCC', 'Balanced Accuracy', 'Sensitivity', 'Specificity', 'ROC AUC'], model_names=['rf', 'lr', 'xgb']):
    # Reset the index
    df = perturbation_df.reset_index().rename(columns={'index': 'split'})

    # Melt the DataFrame to a long format
    df_melted = df.melt(id_vars='split', var_name='model_score', value_name='Score')

    # Extract model and score from model_score
    df_melted['model_name'] = [[mname for mname in model_names if mname in val][0] for val in df_melted['model_score']]
    df_melted['score_name'] = [[sname for sname in score_names if sname in val][0] for val in df_melted['model_score']]

    mean_pivot = df_melted.pivot_table(index='model_name', columns='score_name', values='Score', aggfunc='mean')
    std_pivot = df_melted.pivot_table(index='model_name', columns='score_name', values='Score', aggfunc='std')
    df_combined = mean_pivot.round(2).astype(str) + " (" + std_pivot.round(2).astype(str) + ")"
    df_combined = df_combined[score_names]
    model_conversion = {'rf': 'Random Forest', 'lr': 'Logistic Regression', 'xgb': 'XGBoost'}
    df_combined.index = [model_conversion[model] for model in df_combined.index]
    latex_table = df_combined.style.set_precision(2).to_latex()
    

    print(latex_table)
    return mean_pivot, std_pivot


def plot_base_model_cv(dset='eeg', which_results='csd_fecg', which_fs='best_train', results_df=None, loaded_model_data=None, base_avgs=None, results_table_path='../data/tables/', internal_folder='../data/internal/', tables_folder='../data/tables/', figsize=(8, 4), title='', sort_means=False, color_variances=False, fontsize=14, fontprop=None):
    """

    """
    if base_avgs is None:
        if loaded_model_data is None:
            if results_df is None:
                results_df = fepr.load_model_results(which_results=which_results, results_table_path=results_table_path, dataset=dset, which_fs=which_fs)
            loaded_model_data = fepr.load_model_data(results_df, which_results=which_results, internal_folder=internal_folder, tables_folder=tables_folder)
        print(f"Found base avgs")
        base_avgs = fepr.get_avg_model_best_estimators(loaded_model_data)
        
    df = pd.DataFrame(base_avgs['model_splits'])
    
    # Create a new DataFrame with all values from all models
    all_models_df = pd.DataFrame(df.values.flatten(), columns=['All Base Models'])

    # Concatenate the original DataFrame with the new DataFrame
    df_combined = pd.concat([df, all_models_df], axis=1)

    # Melt the DataFrame to a long format suitable for boxplot
    df_melted = df_combined.melt(var_name='Model', value_name='MCC Value')

    if sort_means:
        model_means = df_melted.groupby('Model')['MCC Value'].mean()
        
        df_melted['Model'] = df_melted['Model'].astype('category')
        df_melted['Model'].cat.reorder_categories(model_means.sort_values(ascending=False).index, inplace=True)


        # Normalize variances for color mapping
    plt.figure(figsize=figsize)
    if color_variances:
        model_vars = df_melted.groupby('Model')['MCC Value'].var()
        norm = mcolors.Normalize(vmin=model_vars.min(), vmax=model_vars.max())
        cmap = plt.cm.tab10
        m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Create a dictionary mapping models to colors based on variance
        model_colors = {model: m.to_rgba(var) for model, var in model_vars.items()}
        # else:
    #     model_colors = 'viridis'
        sns.boxplot(data=df_melted, y='Model', x='MCC Value', orient='h', palette=model_colors)
    else:
    # Create the boxplot
        sns.boxplot(data=df_melted, y='Model', x='MCC Value', orient='h')#, boxprops=dict(facecolor=(1, 1, 1, 0)))
    
    plt.xlabel('Matthews Correlation Coefficient', fontsize=fontsize*.9, fontproperties=fontprop)
    plt.ylabel('Score Value', fontsize=fontsize*.9, fontproperties=fontprop)
    plt.title(title, fontsize=fontsize, fontproperties=fontprop)
    # get all ticks and adjust the fontsize
    xticklabels = plt.gca().get_xticklabels()
    yticklabels = plt.gca().get_yticklabels()
    for label in xticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize*.8)

    for label in yticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize*.6)

    fig = plt.gcf()
    # plt.savefig(f'../figures/final_results/{dset}_base_dev_avg_{which_results}_{which_fs}_boxplot.svg')

    return fig

def plot_meta_dev_cv(fepouts, title='Development Set CV Scores for Each Metalearner'):
    splits = {'Random Forest': {}, 'Logistic Regression': {}, 'XGBoost': {}}
    for idx in range(5):
        splits['Random Forest'][f'split{idx}'] = fepouts['dev_cv_ensemble_split_dict']['cv_results'][f'split{idx}']['test_scores_rf_matthews_corrcoef']
        splits['Logistic Regression'][f'split{idx}'] = fepouts['dev_cv_ensemble_split_dict']['cv_results'][f'split{idx}']['test_scores_lr_matthews_corrcoef']
        splits['XGBoost'][f'split{idx}'] = fepouts['dev_cv_ensemble_split_dict']['cv_results'][f'split{idx}']['test_scores_xgb_matthews_corrcoef']
    splits_df = pd.DataFrame(splits)
    print(splits_df.mean())

    sns.boxplot(data=splits_df, orient='h')
    plt.xlabel('Matthews Correlation Coefficient')
    plt.ylabel('Model')
    plt.title(title)

    return plt.gcf()

def plot_ival_results_df(*dfs, suptitle='', titles=None, figsize=(10,6), x=['model_name'], ylim=[0,1], fontsize=14, fontprop=None):

    if len(dfs) == 1:
        assert len(x) == 1, f"x must be iterable of length 1, not {len(x)}"
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        df_melted = dfs[0].reset_index().melt(id_vars=x[0], var_name='Score Metric', value_name='Score')
        sns.barplot(data=df_melted, x=x[0], y='Score', hue='Score Metric', ax=ax)
        ax.set_title(titles)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=fontsize*.8)
        ax.set_ylim(ylim)
        # set the font properties
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        for label in xticklabels:
            label.set_fontproperties(fontprop)
            label.set_fontsize(fontsize*.8)
        for label in yticklabels:
            label.set_fontproperties(fontprop)
            label.set_fontsize(fontsize*.6)
        # set the y-label
        ax.set_ylabel('Score', fontproperties=fontprop, fontsize=fontsize*.8)
        # hide the x-label
        ax.set_xlabel('')


    else:
        fig, axs = plt.subplots(1, len(dfs), figsize=figsize, sharey=True)
        if titles is not None:
            assert len(titles) == len(dfs)
        for idx, (ax, df) in enumerate(zip(axs, dfs)):
            if len(x) == 1:
                x_val = x[0]
            else:
                assert len(x) == len(dfs)
                x_val = x[idx]
            df_melted = df.reset_index().melt(id_vars=x_val, var_name='Score Metric', value_name='Score')
            sns.barplot(data=df_melted, x=x_val, y='Score', hue='Score Metric', ax=ax)
            ax.set_title(titles[idx], fontsize=fontsize, fontproperties=fontprop)
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=fontsize*.8)
            ax.set_ylim(ylim)
            # set the font properties
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()
            for label in xticklabels:
                label.set_fontproperties(fontprop)
                label.set_fontsize(fontsize*.8)
            for label in yticklabels:
                label.set_fontproperties(fontprop)
                label.set_fontsize(fontsize*.6)
    plt.suptitle(suptitle, fontsize=fontsize*1.1, fontproperties=fontprop)
    plt.tight_layout()
    plt.show()

    # return the figure
    gcf = plt.gcf()
    return gcf
        
def extract_base_results(results_df, scores_of_interest=['mcc', 'ba', 'sensitivity', 'specificity', 'roc_auc'], which_split='ival'):
    """
    Given results df output of load_model_results, extract the base model results
    """
    score_df = results_df.copy(deep=True)
    score_df.index = results_df['model_name']
    score_df = score_df[[f'{which_split}_{s}' for s in scores_of_interest]]
    score_df.columns = scores_of_interest
    return score_df

def extract_metalearner_results(fitted_metamodels, unseen_pred_df, scores_of_interest=['mcc', 'ba', 'sensitivity', 'specificity', 'roc_auc'], print_results=True):
    """
    Given a dict of fitted metalearners and the unseen predictions, extract the results on the unseen data
    
    """
    metalearners = fitted_metamodels.keys()
    all_scores = {}
    for mlearner in metalearners:
        mdl = fitted_metamodels[mlearner]
        bin_scores = mu.score_binary_model(mdl, unseen_pred_df, fu.get_y_from_df(unseen_pred_df))
        if print_results:
            print(mlearner)
            mu.print_binary_scores(bin_scores)
        all_scores[mlearner] = bin_scores
    
    soi_df = pd.DataFrame([[all_scores[m][soi] for soi in scores_of_interest] for m in metalearners], columns=scores_of_interest, index=metalearners)
    return all_scores, soi_df


#### Feature analysis

def plot_base_model_weights(basemodel_dict, weights='normalized', figsize=(10, 6), title='Base Model Weights', fontsize=14, fontprop=None):
    """
    Given a dictionary of base model weights, plot the weights
    """
    fig, ax = plt.subplots(figsize=figsize)
    weight_df = pd.DataFrame.from_dict(basemodel_dict[weights], orient='index', columns=[f'{weights} weights'])
    weight_df = weight_df.sort_values(by=f'{weights} weights', ascending=False)
    sns.barplot(x=f'{weights} weights', y=weight_df.index, data=weight_df, ax=ax)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel('Feature')
    ax.set_xlabel('Weight')
    # set the font properties
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    for label in xticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize*.8)
    for label in yticklabels:
        label.set_fontproperties(fontprop)
        label.set_fontsize(fontsize*.6)
    plt.show()
    # return weight_df, fig


def feature_weight_analysis(metamodel, train_df, test_df, feat_method='inherent', results_table_path='data/tables/', which_results='csd_fecg', dataset='eeg', which_fs='recursive'):
    """
    Given a metal model and train_df / test_df, return the weights of the features
    """
    meta_weights = get_metamodel_weights(metamodel, train_df, test_df, feat_method=feat_method)

    basemodel_weight_dict = get_basemodel_weights(meta_weights, feat_method=feat_method,which_results=which_results, results_table_path=results_table_path, dataset=dataset, which_fs=which_fs)

    return meta_weights, basemodel_weight_dict

def get_basemodel_weights(meta_weights, feat_method='inherent',which_results='csd_fecg', results_table_path='data/tables/', dataset='eeg', which_fs='recursive'):
    """
    Given the meta_weights, return the weights of the base models
    args: meta_weights: dictionary of feature weights {basemodelname: weight} (output of get_metamodel_weights)
    returns: basemodel_weight_dict: dictionary of dictionary of feature weights (unweighted) {basemodelname: weight}
    """
    basemodel_weight_dict = {}
    basemodelnames = list(meta_weights.keys())
    results_df = fepr.load_model_results(which_results=which_results, results_table_path=results_table_path, dataset=dataset, which_fs=which_fs)
    for basemodelname in basemodelnames:
        loadname = [f for f in results_df['filename'] if basemodelname in f][0]
        model_cv, Xtrraw, Xtsraw = ms.load_model_data(loadname)
        model = model_cv.best_estimator_.named_steps['classifier']
        Xtr = ma.get_transformed_data(Xtrraw, model_cv, verbose=False)
        # Xunseen = load_unseen_data(json_filename, dset, Xtr.index, base_folder=tables_folder, internal_folder=internal_folder, which_results=which_results)
        if basemodelname.split('_')[0] == 'RandomForestClassifier':
            if feat_method == 'inherent':
                assert len(model.feature_importances_) == len(Xtr.columns), f"Length of model.feature_importances_ {len(model.feature_importances_)} != length of Xtr.columns {len(Xtr.columns)}"
                model_weights = {Xtr.columns[idx]: val for idx, val in enumerate(model.feature_importances_)}
                basemodel_weight_dict[basemodelname] = return_sorted_normalized_scaled_weights(model_weights, meta_weights[basemodelname])
            elif feat_method == 'shap':
                basemodel_weight_dict[basemodelname] = get_model_shap_weights(model, Xtr, meta_weights, basemodelname)
            else:
                raise ValueError(f"feat_method must be 'inherent' or 'shap', not {feat_method}")
            

        elif basemodelname.split('_')[0] == 'LogisticRegression':
            if feat_method == 'inherent':
                assert len(model.coef_[0]) == len(Xtr.columns), f"Length of model.coef_[0] {len(model.coef_[0])} != length of Xtr.columns {len(Xtr.columns)}"
                model_weights = {Xtr.columns[idx]: val for idx, val in enumerate(model.coef_[0])}
                basemodel_weight_dict[basemodelname] = return_sorted_normalized_scaled_weights(model_weights, meta_weights[basemodelname])
            elif feat_method == 'shap':
                basemodel_weight_dict[basemodelname] = get_model_shap_weights(model, Xtr, meta_weights, basemodelname)
            else:
                raise ValueError(f"feat_method must be 'inherent' or 'shap', not {feat_method}")
        
        elif basemodelname.split('_')[0] == 'XGBClassifier':
            if feat_method == 'inherent':
                xgb_gain_scores = model.get_booster().get_score(importance_type='gain')
                xgb_gain_keys = list(xgb_gain_scores.keys())
                xgb_cols = Xtr.columns[[int(col[1:]) for col in xgb_gain_keys]]
                assert len(xgb_cols) == len(xgb_gain_scores), f"Length of xgb_cols {len(xgb_cols)} != length of xgb_gain_scores {len(xgb_gain_scores)}"
                model_weights = {col: xgb_gain_scores[key] for col, key in zip(xgb_cols, xgb_gain_keys)}
                basemodel_weight_dict[basemodelname] = return_sorted_normalized_scaled_weights(model_weights, meta_weights[basemodelname])
            elif feat_method == 'shap':
                basemodel_weight_dict[basemodelname] = get_model_shap_weights(model, Xtr, meta_weights, basemodelname)
            else:
                raise ValueError(f"feat_method must be 'inherent' or 'shap', not {feat_method}")

        elif basemodelname.split('_')[0] == 'KNeighborsClassifier':
            explainer = shap.KernelExplainer(model.predict, Xtr)
            shap_values = explainer.shap_values(Xtr)
            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
            assert len(mean_abs_shap_values) == len(Xtr.columns), f"Length of mean_abs_shap_values {len(mean_abs_shap_values)} != length of Xtr.columns {len(Xtr.columns)}"
            model_weights = {Xtr.columns[idx]: val for idx, val in enumerate(mean_abs_shap_values)}
            basemodel_weight_dict[basemodelname] = return_sorted_normalized_scaled_weights(model_weights, meta_weights[basemodelname])
        elif basemodelname.split('_')[0] == 'GaussianNB':
            if feat_method == 'inherent':
                assert len(model.theta_[1]) == len(Xtr.columns), f"Length of model.theta_[1] {len(model.theta_[1])} != length of Xtr.columns {len(Xtr.columns)}"
                assert model.var_.shape==(2, len(Xtr.columns)), f"Shape of model.var_ {model.var_.shape} != (2, {len(Xtr.columns)})"
                gnb_means = model.theta_
                gnb_variances = model.var_
                # calculate ga
                kl_divergences = fu.kl_divergence_gaussian(gnb_means[0,:], np.sqrt(gnb_variances[0,:]), gnb_means[1,:], np.sqrt(gnb_variances[1,:]))
                model_weights = {Xtr.columns[idx]: val for idx, val in enumerate(kl_divergences)}

                basemodel_weight_dict[basemodelname] = return_sorted_normalized_scaled_weights(model_weights, meta_weights[basemodelname])
            elif feat_method == 'shap':
                basemodel_weight_dict[basemodelname] = get_model_shap_weights(model, Xtr, meta_weights, basemodelname)
        else:
            raise ValueError(f"basemodelname must be RandomForestClassifier, LogisticRegression, or XGBClassifier, not {basemodelname.split('_')[0]}")

        # if basemodel not in basemodel_weight_dict:
        #     basemodel_weight_dict[basemodel] = meta_weights[key]
        # else:
        #     basemodel_weight_dict[basemodel] += meta_weights[key]
    total_weights = {}
    for basemodel in basemodel_weight_dict.keys():
        for key in basemodel_weight_dict[basemodel]['scaled'].keys():
            if key not in total_weights:
                total_weights[key] = basemodel_weight_dict[basemodel]['scaled'][key]
            else:
                total_weights[key] += basemodel_weight_dict[basemodel]['scaled'][key]
    basemodel_weight_dict['Total'] = total_weights
    return basemodel_weight_dict

def get_metamodel_weights(metamodel, train_df, test_df, feat_method='inherent', plot=False):
    """
    Given a metal model and train_df / test_df, return the weights of the features
    args:
        metamodel trained on train_df and tested on test_df
        train_df: DataFrame used to train the metamodel
        test_df: DataFrame used to test the metamodel
        feat_method: 'inherent' or 'shap' # how to get the feature weights
        plot: whether to plot the feature weights
    returns:
        model_weights: dictionary of feature weights {basemodelname: weight}
            - normalized to sum to 1 and no sign for + or - weights
    """
    if metamodel.__class__.__name__ == 'XGBClassifier':
        if feat_method == 'inherent':
            xgb_gain_scores = metamodel.get_booster().get_score(importance_type='gain')
            sorted_xgb_gain_scores = dict(sorted(xgb_gain_scores.items(), key=lambda item: item[1], reverse=True))
            normalized_xgb_gain_scores = {key: value/sum(xgb_gain_scores.values()) for key, value in sorted_xgb_gain_scores.items()}
            combine_pos_neg_gain_scores = {}
            for key, value in normalized_xgb_gain_scores.items():
                new_key = '_'.join(key.split('_')[:-1])
                if new_key not in combine_pos_neg_gain_scores:
                    combine_pos_neg_gain_scores[new_key] = normalized_xgb_gain_scores[key]
                else:
                    combine_pos_neg_gain_scores[new_key] += normalized_xgb_gain_scores[key]
            model_weights = {**combine_pos_neg_gain_scores}
            model_pos_neg_weights = {**sorted_xgb_gain_scores}
        elif feat_method == 'shap':
            pass
        else:
            raise ValueError(f"feat_method must be 'inherent' or 'shap', not {feat_method}")
    elif metamodel.__class__.__name__ == 'RandomForestClassifier':
        if feat_method == 'inherent':
            model_pos_neg_weights = {train_df.columns[idx]: val for idx, val in enumerate(metamodel.feature_importances_)}
            sorted_neg_pos_weights = dict(sorted(model_pos_neg_weights.items(), key=lambda item: item[1], reverse=True))
            
            model_weights = {}
            for key, value in model_pos_neg_weights.items():
                new_key = '_'.join(key.split('_')[:-1])
                if new_key not in model_weights:
                    model_weights[new_key] = model_pos_neg_weights[key]
                else:
                    model_weights[new_key] += model_pos_neg_weights[key]
            sorted_model_weights = dict(sorted(model_weights.items(), key=lambda item: item[1], reverse=True))
            model_weights = {key: value/sum(sorted_model_weights.values()) for key, value in sorted_model_weights.items()}
            model_pos_neg_weights = {key: value/sum(sorted_neg_pos_weights.values()) for key, value in sorted_neg_pos_weights.items()}

    elif metamodel.__class__.__name__ == 'LogisticRegression':
        model_pos_neg_weights = {train_df.columns[idx]: val for idx, val in enumerate(metamodel.coef_[0])}
        model_weights = {}
        for key, value in model_pos_neg_weights.items():
            new_key = '_'.join(key.split('_')[:-1])
            if new_key not in model_weights:
                model_weights[new_key] = abs(model_pos_neg_weights[key])
            else:
                model_weights[new_key] += abs(model_pos_neg_weights[key])
        sorted_model_weights = dict(sorted(model_weights.items(), key=lambda item: item[1], reverse=True))
        model_weights = {key: value/sum(sorted_model_weights.values()) for key, value in sorted_model_weights.items()}
        sorted_neg_pos_weights = dict(sorted(model_pos_neg_weights.items(), key=lambda item: item[1], reverse=True))
        model_pos_neg_weights = {key: value/sum(np.abs(list(sorted_neg_pos_weights.values()))) for key, value in sorted_neg_pos_weights.items()}
    else:
        raise ValueError(f"metamodel must be XGBClassifier, RandomForestClassifier, or LogisticRegression, not {metamodel.__class__.name}")
    if plot:
        pos_neg_keys = [key for key in model_pos_neg_weights.keys()]
        new_pos_neg_keys = [key.split('_')[0] + '_' + key.split('_')[-1] for key in pos_neg_keys]
        pos_neg_weights = [value for value in model_pos_neg_weights.values()]
        sns.barplot(y=new_pos_neg_keys, x=pos_neg_weights)
        # plt.xticks(rotation=90)
        plt.title(f'Model weights for Metalearner: {metamodel.__class__.__name__}')
        plt.show()
    
    return model_weights

def get_model_shap_weights(model, Xtr, meta_weights=None, basemodelname=''):
    """
    Function to calculate and return model weights using SHAP values.
    Automatically decides whether to use KernelExplainer or TreeExplainer based on the model type.

    Parameters:
    - model: The trained model.
    - Xtr: Training data (features).
    - meta_weights: Optional. Meta weights for scaling the model weights.
    - basemodelname: Optional. The base model name for indexing in meta_weights.

    Returns:
    - A dictionary containing normalized, scaled, and raw model weights.
        - normalized means the weights are normalized to sum to 1.
        - scaled means the weights are scaled by the meta weight.
        - raw means the weights are the raw mean abs SHAP values.
    """
    # Decide which SHAP explainer to use based on the model type
    if any(tree_model in str(type(model)) for tree_model in ['XGB', 'LGBM', 'ctb', 'DecisionTree', 'RandomForest', 'ExtraTrees', 'Ada', 'GradientBoosting']):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, Xtr)

    # Calculate SHAP values
    shap_values = explainer.shap_values(Xtr)
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Ensure the lengths match
    assert len(mean_abs_shap_values) == len(Xtr.columns), f"Length of mean_abs_shap_values {len(mean_abs_shap_values)} != length of Xtr.columns {len(Xtr.columns)}"

    # Process SHAP values into model weights
    model_weights = {Xtr.columns[idx]: val for idx, val in enumerate(mean_abs_shap_values)}
    sorted_model_weights = dict(sorted(model_weights.items(), key=lambda item: item[1], reverse=True))
    normalized_model_weights = {key: value / sum(np.abs(list(sorted_model_weights.values()))) for key, value in sorted_model_weights.items()}

    # Apply meta weights if provided
    if meta_weights is not None and basemodelname in meta_weights:
        scaled_model_weights = {key: abs(value) * meta_weights[basemodelname] for key, value in normalized_model_weights.items()}
    else:
        scaled_model_weights = normalized_model_weights

    # Compile the weights into a dictionary
    basemodel_weight_dict = {
        'normalized': normalized_model_weights,
        'scaled': scaled_model_weights,
        'raw': model_weights
    }

    return basemodel_weight_dict

def return_sorted_normalized_scaled_weights(model_weights, meta_weight):
    """
    Given a dictionary of feature weights, return the normalized and scaled weights
    """
    sorted_model_weights = dict(sorted(model_weights.items(), key=lambda item: item[1], reverse=True))
    normalized_model_weights = {key: value / sum(np.abs(list(sorted_model_weights.values()))) for key, value in sorted_model_weights.items()}
    scaled_model_weights = {key: abs(value) * meta_weight for key, value in normalized_model_weights.items()}
    sorted_normalized_scaled_out_dict = {
        'raw': model_weights,
        'normalized': normalized_model_weights,
        'scaled': scaled_model_weights
    }
    return sorted_normalized_scaled_out_dict

def plot_band_topo(total_weights, channel_names=CHANNELS, title='Bands by feature weights', figsize=(10, 6), fontsize=14, fontprop=None, n_contours=0):

    # loaded_model_data = [ms.load_model_data(f) for f in total_weights.keys() if 'Total' != f]

    total_weight_df = pd.DataFrame.from_dict(total_weights, orient='index', columns=['Weight'])

    # basefp_df = pd.DataFrame(basefp, index=Xtr_proc.columns, columns=['Feature Importance'])
    band_by_region_count = ma.count_band_by_region(total_weight_df.index, feature_weights=total_weight_df.to_dict()['Weight'])
    cpr_fp = pd.DataFrame(band_by_region_count['bands_channels_weighted_count'])
    min_val= cpr_fp.min().min()
    max_val = cpr_fp.max().max()
    for bdx, band in enumerate(cpr_fp.columns):
        print(f"\nBand {band}")
        band_chs = cpr_fp.index
        band_wv = cpr_fp.iloc[:, bdx].values
        l_freq = band[0]
        h_freq = band[1]
        row = bdx//2
        col = bdx%2
        if h_freq>200:
            cbar =True
        else:
            cbar=False
        plot_topomap(band_wv, band_chs, title=f'{l_freq}-{h_freq} Hz', colorbar=cbar, fontsize=fontsize, clim=(min_val, max_val), cblabel='Weighted Importance of Feature in Model (a.u.)', n_contours=n_contours, figsize=figsize, ch_names=False)

    return band_by_region_count



def plot_channel_topo(total_weights, channel_names=CHANNELS, title='Channels by feature weights', figsize=(10, 6), fontsize=14, fontprop=None, normalize=True, n_contours=0, ch_names=False):
    """
    Given the total weights (basemodel_weight_Dict['Total'])
    Plot how often each channel shows up
    """

    total_weight_df = pd.DataFrame.from_dict(total_weights, orient='index', columns=['Weight'])

    count_dict = ma.count_channel_or_region(total_weight_df.index, feature_weights=total_weight_df.to_dict()['Weight'])

    fp_ch_counts = count_dict['channel_counts']['weighted_count']
    channels = [ch for ch in count_dict['channel_counts']['count'].keys() if ch not in ['T1', 'T2'] and count_dict['channel_counts']['count'][ch] > 0]
    if normalize:
        fp_wv = [fp_ch_counts[ch]/sum(np.abs(list(fp_ch_counts.values()))) for ch in channels]
    else:
        fp_wv = [fp_ch_counts[ch] for ch in channels]
    plot_topomap(fp_wv, channels, title=title, figsize=figsize, fontsize=fontsize, fontprop=fontprop, colorbar=True, cblabel='Weighted Importance of Channel in Model (a.u.)', n_contours=n_contours, ch_names=ch_names)

    return count_dict

def plot_topomap(eeg_powers, channel_names, title='', colorbar=False, cblabel='', cbar_orientation='vertical', cbar_norm=None, clim=None,n_contours=6, fig=None, ax=None, fontsize=12, fontprop=None, figsize=(10, 6), ch_names=False):
    assert len(eeg_powers) == len(channel_names), f"eeg powers shape {len(eeg_powers)} != len channel names {len(channel_names)}"
    assert type(channel_names[0]) == str, f"channel names must be strings, but got {type(channel_names[0])}"
    # Create a layout for the 21-channel EEG montage
    montage = mne.channels.make_standard_montage('standard_1020')

    valid_ch_names = [ch_name for ch_name in channel_names if ch_name in montage.ch_names]
    print(f"valid_ch_names: {valid_ch_names}")
    invalid_chidx = [ch_idx for ch_idx, ch_name in enumerate(channel_names) if ch_name not in montage.ch_names]
    eeg_power_valid = np.array([eeg_powers[ch_idx] for ch_idx in range(len(channel_names)) if ch_idx not in invalid_chidx]).reshape(-1, 1)
    # Create an info object with channel names and positions
    info = mne.create_info(ch_names=valid_ch_names, sfreq=1000, ch_types='eeg')
    info.set_montage(montage)

    ch_pos = montage.get_positions()['ch_pos']
    pos = np.stack([ch_pos[ch] for ch in valid_ch_names])
    x = pos[0, 0]
    y = pos[-1, 1]
    z = pos[:, -1].mean()
    radius = np.abs(pos[[2, 3], 0]).mean()

    # Create an empty RawArray object with EEG powers as data
    raw_array = mne.io.RawArray(data=eeg_power_valid, info=info)

    # Plot the topographic map
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if ch_names:
        names = valid_ch_names
    else:
        names = None

    top_im_cntr =  mne.viz.plot_topomap(raw_array.get_data()[:, 0], raw_array.info, axes=ax, contours=n_contours, names=names, show=False)
    ax.set_title(title, fontsize=fontsize, fontproperties=fontprop)

    if clim is not None:
        top_im_cntr[0].set_clim(clim)
    if colorbar:
        cmap = 'RdBu_r' if np.min(eeg_power_valid) < 0 else 'Reds'
        if cbar_norm is not None:
            cb1 = plt.colorbar(top_im_cntr[0], ax=ax, orientation=cbar_orientation, cmap=cmap, norm=mpl.colors.Normalize(vmin=cbar_norm[0], vmax=cbar_norm[1]))

        else:
            cb1 = plt.colorbar(top_im_cntr[0], ax=ax, orientation=cbar_orientation)
        cb1.set_label(cblabel, fontsize=fontsize*0.8, fontproperties=fontprop)


    if fig is not None:
        plt.show()
    return fig, ax, top_im_cntr

### SUBJECT ANALYSIS
import src.data.make_subject_subsets as mss
def metadata_analysis(predicted_diagnosis, true_diagnosis, groups, metainfo='imaging_abnormal'):
    

    assert len(predicted_diagnosis) == len(true_diagnosis)
    assert len(predicted_diagnosis) == len(groups)
    if metainfo == 'imaging_abnormal_plus_protocol_nonalternating':
        _, abnormal_subjects = mss.load_subset(basefolder='../data/internal/', return_subjects=True, verbose=False, subject_subset_method='imaging_abnormal')
        _, protocol_nonalternating_subjects = mss.load_subset(basefolder='../data/internal/', return_subjects=True, verbose=False, subject_subset_method='protocol_nonalternating')

        norm_nonalternating_group_idx = np.array([idx for idx, g in enumerate(groups) if g not in abnormal_subjects and g in protocol_nonalternating_subjects])
        abnorm_or_alternating_group_idx = np.array([idx for idx, g in enumerate(groups) if g in abnormal_subjects or g not in protocol_nonalternating_subjects])
        abnormal_group_idx = abnorm_or_alternating_group_idx
        normal_group_idx = np.array([idx for idx, g in enumerate(groups) if g not in abnorm_or_alternating_group_idx])
        abnormal_groups = [g for g in groups if g in abnormal_subjects or g not in protocol_nonalternating_subjects]
        normal_groups = [g for g in groups if g not in abnormal_subjects and g in protocol_nonalternating_subjects]
    else:
        _, subset_subjects = mss.load_subset(basefolder='../data/internal/', return_subjects=True, verbose=False, subject_subset_method=metainfo)
        abnormal_group_idx = np.array([idx for idx, g in enumerate(groups) if g in subset_subjects])
        normal_group_idx = np.array([idx for idx, g in enumerate(groups) if g not in subset_subjects])
        abnormal_groups = [g for g in groups if g in subset_subjects]
        normal_groups = [g for g in groups if g not in subset_subjects]
    assert all(np.array(groups)[abnormal_group_idx] == abnormal_groups)
    assert all(np.array(groups)[normal_group_idx] == normal_groups)

    abnormal_predictions = predicted_diagnosis[abnormal_group_idx]
    normal_predictions = predicted_diagnosis[normal_group_idx]

    abnormal_true = true_diagnosis[abnormal_group_idx]
    normal_true = true_diagnosis[normal_group_idx]



    print(f"{metainfo} subset results:")
    print(sklearn.metrics.classification_report(abnormal_true, abnormal_predictions))
    

    print(f"not {metainfo} subset results:")
    print(sklearn.metrics.classification_report(normal_true, normal_predictions))

    # plot a 2x1 confusion matrix
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(abnormal_true, abnormal_predictions), display_labels=['HC', 'mTBI']).plot(ax=axs[0])
    sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(normal_true, normal_predictions), display_labels=['HC', 'mTBI']).plot(ax=axs[1])
    # hide the colorbar
    axs[0].images[-1].colorbar.remove()
    axs[1].images[-1].colorbar.remove()
    axs[0].set_title(f"{metainfo} subset")
    axs[1].set_title(f"not {metainfo} subset")
    plt.show()

    # compute the mann-wh
    import scipy.stats  
    print(f"\nOverall Mann Whitney U test for {metainfo} subset vs Normal subset")
    print(scipy.stats.mannwhitneyu(abnormal_predictions==abnormal_true, normal_predictions==normal_true))

    pos_group_cls_abnormal = abnormal_predictions[abnormal_true == 1]
    neg_group_cls_abnormal = abnormal_predictions[abnormal_true == 0]

    pos_group_cls_normal = normal_predictions[normal_true == 1]
    neg_group_cls_normal = normal_predictions[normal_true == 0]

    print(f"\nMann Whitney U test for {metainfo} subset positive class vs negative class")
    print(scipy.stats.mannwhitneyu(pos_group_cls_abnormal, neg_group_cls_abnormal))
    print(f"\nMann Whitney U test for not {metainfo} subset positive class vs negative class")
    print(scipy.stats.mannwhitneyu(pos_group_cls_normal, neg_group_cls_normal))

    contingency_table = [[sum(abnormal_predictions==abnormal_true), sum(abnormal_predictions!=abnormal_true)],
                        [sum(normal_predictions==normal_true), sum(normal_predictions!=normal_true)]]
    print(f"Contengency table: {contingency_table}")
    chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table)
    print(f"\nChi2 contingency test for {metainfo} subset positive class vs negative class")
    print(f"Chi2 value: {chi2}, p-value: {p}, dof: {dof}, expected: {expected}")
    return abnormal_group_idx, normal_group_idx
###OLDS
def old_plot_results(results, cv_ensemble_split_dict=None, include_average=True, fontsize=20, figsize=(10, 4), title="Training Set CV Scores for Each Classifier"):
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

def pres_plot_holdout(soi_holdout_df):
    soi_select_holdout = soi_holdout_df.loc[['Meta XGBoost'], ['MCC', 'Balanced Accuracy', 'ROC AUC']]
    fontprop = get_font_prop(14)
    import seaborn as sns
    import seaborn.objects as so
    # fig, ax = plt.subplots(figsize=(10, 6))
    df_melted = soi_select_holdout.reset_index().melt(id_vars='model_name', var_name='Score Metric', value_name='Score')
    # sns.barplot(data=df_melted, ax=ax, x='model_name', y='Score', hue='Score Metric', dodge=True)
    sns.catplot(data=df_melted, x='model_name', y='Score', hue='Score Metric', kind='bar', height=5, aspect=2, legend=False)#, legend_out=True)
    plt.ylim(0, 1)
    # p = so.Plot(data=df_melted, x='model_name', y='Score').add(so.Bar(), so.Count(), so.Dodge(gap=0.1)).on(ax)
    # p.show()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop=get_font_prop(22))
    # hide xtick labels
    plt.xticks([0], [''])
    plt.xlabel('XGBoost Metalearner', fontproperties=get_font_prop(25))
    # plt.xlabel('')
    # increase ytick label size
    plt.ylabel('')
    yticks = plt.yticks(fontproperties=get_font_prop(16))
    # change the size of the figure
    plt.gcf().set_size_inches(6, 12)
    plt.show()

def get_best_model(ival_pred_df):
    ival_pos_pred_df = ival_pred_df[ival_pred_df.columns[ival_pred_df.columns.str.endswith('_1')]] >= 0.5
    ival_pos_pred_df = ival_pos_pred_df.astype(int)

    ival_y_test = fu.get_y_from_df(ival_pred_df)

    # choose the best model
    all_model_mccs_ival = [sklearn.metrics.matthews_corrcoef(ival_y_test, ival_pos_pred_df[col]) for col in ival_pos_pred_df.columns]

    best_model_idx = np.argmax(all_model_mccs_ival)

    best_model_name = ival_pos_pred_df.columns[best_model_idx]
    best_model_mcc = all_model_mccs_ival[best_model_idx]
    print(f"Best model: {best_model_name} with MCC: {best_model_mcc}")
    best_model = best_model_name[:-2]
    return best_model


def plot_regs(meta_holdout_df):
    meta_holdout_df.columns = ['ACE Total Score', 'Rivermead Total Score']
    holdout_true_y = tbr.get_reg_from_df(meta_holdout_df, questionnaires=True)
    holdout_true_y.columns = ['ACE Total Score', 'Rivermead Total Score']
    y_bin = fu.get_y_from_df(meta_holdout_df)
    for col in meta_holdout_df.columns:
        sns.regplot(x=holdout_true_y[col], y=meta_holdout_df[col], color=y_bin, label=col)
        # print the spearman p
        scorr, p_value = scipy.stats.spearmanr(holdout_true_y[col], meta_holdout_df[col])
        print(col, scorr, p_value)
        # plt.legend()
        plt.show()


def plot_regs(reg_preds_test, y_test, X_test):
    import matplotlib.patches as mpatches
    true_values = y_test['Score_Rivermead_Baseline']
    pred_values = reg_preds_test[:, [idx for idx, col in enumerate(y_train.columns) if 'Score_Rivermead_Baseline' in col][0]]
    import scipy.stats
    labels = fu.get_y_from_df(X_test).astype(str)
    labels[labels=='0'] = 'Controls'
    labels[labels=='1'] = 'mTBI'
    # sns regplot
    color_map = {'Controls': 'blue', 'mTBI': 'red'}
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.regplot(x=true_values, y=pred_values, scatter_kws={'color': [color_map[label] for label in labels]}, ax=ax)
    ax.set_ylabel('Predicted Rivermead Symptom Score', fontproperties=font_prop, fontsize=30)
    ax.set_xlabel('True Rivermead Score', fontproperties=font_prop, fontsize=30)
    # make the xticks font size 16 and inter
    plt.xticks(fontsize=14, fontproperties=font_prop)
    plt.yticks(fontsize=14, fontproperties=font_prop)

    # Create legend handles and labels
    handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]

    # Create legend from handles
    plt.legend(handles=handles, loc='upper left', fontsize=24)
    spearmanr, pval = scipy.stats.spearmanr(true_values, pred_values)
    print(f"Spearman correlation: {spearmanr}, p-value: {pval}")


def plot_confusion_matrix(y_true, y_preds, labels=['Controls', 'mTBI'], font_size=30, tick_label_size=20, title='', figsize=(15, 10), cblims=None):
    cm = sklearn.metrics.confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # font = {'weight' : 'bold', 'size' : font_size}
    # plt.rc('font', **font)

    # Create a discrete colormap

    if cblims is None:
        cblims = (np.min(cm), np.max(cm)+1)
    cmap = plt.cm.get_cmap('Blues', cblims[-1] - cblims[0])

    sns.heatmap(cm, annot=True, ax=ax, cmap=cmap, fmt='g', vmin=cblims[0], vmax=cblims[1], cbar_kws={"ticks":np.arange(cblims[0], cblims[1])}, annot_kws={"size": font_size})



    ax.set_xlabel('Predicted Diagnosis', fontproperties=get_font_prop(font_size*.8))
    ax.set_ylabel('Actual Diagnosis', fontproperties=get_font_prop(font_size*.8))
    ax.set_title(title, fontproperties=get_font_prop(font_size*1.2))
    ax.xaxis.set_ticklabels(labels, fontproperties=get_font_prop(tick_label_size))
    ax.yaxis.set_ticklabels(labels, fontproperties=get_font_prop(tick_label_size))
    for text in ax.texts:
        text.set_fontproperties(get_font_prop(font_size, weight='bold'))
    # change the colorbar fontproperties
    cbar = ax.collections[0].colorbar
    # make the cblimits go from cblims[0] to cblims[1]
    # cbar.set_clim(cblims[0], cblims[1])
    for ctick in cbar.ax.yaxis.get_ticklabels():
        ctick.set_fontproperties(get_font_prop(tick_label_size*.8))

    # add a label to the colorbar
    cbar.set_label('Number of samples', fontproperties=get_font_prop(tick_label_size))
    # add a little space between the label and the colorbar
    cbar.ax.yaxis.labelpad = 15
    
    ax.tick_params(axis='both', which='major')

    plt.show()

def find_confmat_groups(y_true, y_pred, groups):
    groups_correct = groups[y_true == y_pred]
    groups_incorrect = groups[y_true != y_pred]
    tp_groups = groups[(y_true == y_pred) & (y_true == 1)]
    tn_groups = groups[(y_true == y_pred) & (y_true == 0)]
    fp_groups = groups[(y_true != y_pred) & (y_true == 0) ]
    fn_groups = groups[(y_true != y_pred) & (y_true == 1)]
    out_dict = {'groups_correct': groups_correct, 'groups_incorrect': groups_incorrect, 'tp_groups': tp_groups, 'tn_groups': tn_groups, 'fp_groups': fp_groups, 'fn_groups': fn_groups}
    return out_dict 

def analyze_clf_by_bindemo(model_results_df, demogrs_df, bincol):
    """
    model_results_df: dataframe with columns y_true, y_pred, y_pred_proba0, y_pred_proba1 and index as subject ids
    demogrs_df: dataframe with demographic information and index as subject ids
    bincol: the binary column to split the groups by
    
    
    """
    pos_results_df = model_results_df.loc[[s for s in model_results_df.index if demogrs_df.loc[int(s)][bincol]]]
    neg_results_df = model_results_df.loc[[s for s in model_results_df.index if not demogrs_df.loc[int(s)][bincol]]]
    print(f"{bincol} classification report:\n", sklearn.metrics.classification_report(pos_results_df['y_true'], pos_results_df['y_pred']))
    print(f"Non-{bincol} classification report:\n", sklearn.metrics.classification_report(neg_results_df['y_true'], neg_results_df['y_pred']))

    # statistically compare the two groups using delong's test
    auc_pos = sklearn.metrics.roc_auc_score(pos_results_df['y_true'], pos_results_df['y_pred_proba1'])
    auc_neg = sklearn.metrics.roc_auc_score(neg_results_df['y_true'], neg_results_df['y_pred_proba1'])
    print("AUC for positive group:", auc_pos, "AUC for negative group:", auc_neg)
    # mann whitney u on the accuracies
    print(mannwhitneyu(pos_results_df['y_true']==pos_results_df['y_pred'], neg_results_df['y_true']==neg_results_df['y_pred']))
    return pos_results_df, neg_results_df
