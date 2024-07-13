# queue up the different experiments in a sequential manner to make use of the system resources
# doing it this way uses the preset argparse feature arguments
import os
import sys
import argparse
import glob
import psutil
import time
import numpy as np
import pandas as pd
import itertools
from inputimeout import inputimeout, TimeoutOccurred

DATASETS_NAME_DICT = {
    "eeg": "eeg",
    "ecg": "ecg",
    "sym": "symptoms",
    "sel": "selectsym",
}

MODEL_NAME_DICT = {
    "xgb": "XGBClassifier",
    "rf": "RandomForestClassifier",
    "knn": "KNeighborsClassifier",
    "gnb": "GaussianNB",
    "ada": "AdaBoost",
    "lre": "LogisticRegression",
}



DATAPATH = open('extracted_path.txt', 'r').read().strip() 
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')
RESULTS_SAVEPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'results', 'base_learners')

def check_if_previous_run(feat_shortcuts, model_shortcut, datapath=RESULTS_SAVEPATH, n_prior_files=None, verbose=True):
    """
    Check if there is a previous run for the given parameters:
    Returns: 
        - num_files: number of files that match the given parameters
        - has_more_files: whether there are more files than the given number of files: says whether the run is finished or not
    """

    file_format = "{model_name}_**_{dataset}_caf_kwargs.json" # fixes AdaBoost != AdaBoostClassifier and no longer need kcv
    
    all_possible_orderings = [list(perm) for perm in itertools.permutations(feat_shortcuts) if len(perm)==len(feat_shortcuts)]


    for ordering in all_possible_orderings:
        dataset = '-'.join([DATASETS_NAME_DICT[feat_shortcut] for feat_shortcut in ordering])
        file_format = file_format.format(
            model_name=MODEL_NAME_DICT[model_shortcut],
            dataset=dataset,
        )
        file_path = os.path.join(datapath, file_format)
        if verbose:
            print(f"Checking for files in {file_path}")
        files = glob.glob(file_path)
        num_files = len(files)
        has_files = num_files > 0
        if has_files:
            break

    n_prior_files = n_prior_files if n_prior_files is not None else 0
    has_more_files = num_files > n_prior_files

    return num_files, has_more_files

def get_avg_idle_cores(duration=30):
    """
    Sees how many cores are available to set a ceiling on n_jobs
    """
    print("Getting average idle cores")
    init_usage = np.array(psutil.cpu_percent(percpu=True))
    start_time = time.time()
    n_iters = 0
    while time.time() - start_time < duration:
        cpu_usage = np.array(psutil.cpu_percent(percpu=True))
        init_usage += cpu_usage
        n_iters += 1
        time.sleep(0)
    avg_usage = init_usage / n_iters
    n_idle_cores = sum(1 for i in avg_usage if i == 0.0)
    return avg_usage, n_idle_cores

def main(n_jobs=3, n_points=1, min_run_time=15, feat_shortcuts = ["eeg", "ecg", "sym", "sel"],\
         duration=30, hyper_cv=3, fs_cv=3, n_fs_repeats=2, n_iterations=100, n_hyper_repeats=2,\
            results_savepath=RESULTS_SAVEPATH, outbase=None, wait_time=15):

    """"
    Driver to run train_baselearners.py across all feature sets and base models
    """"
    print(f"Beginning to run {n_jobs} jobs in parallel")
    # call the python script with the correct arguments

    base_call = ("python3 src/models/new_split_faster_full_cv_options.py --scoring=mcc --method={ref_method} --n_jobs={n_jobs} --{filter_ecg} --{late_filter_ecg} "
                 "--hyper_cv={hyper_cv} --fs_cv={fs_cv} --n_fs_repeats={n_fs_repeats} --n_hyper_repeats={n_fs_repeats} --wrapper_method={wrapper_method} "
                 "--model_name={model_name} --n_iterations={n_iterations} --n_points={n_points} "
                 "--results_savepath={results_savepath} --which_features {which_features0} {which_features1} {which_features2} "
                 "{out_appdx}")
    n_total_runs = 0
    curr_run = 0
    model_shortcuts = ["xgb", "rf", "knn",  "gnb", "ada", "lre"] 
    for feat_shortcut in feat_shortcuts:
        for model_shortcut in model_shortcuts:
            if check_if_previous_run(feat_shortcut, model_shortcut, datapath=results_savepath)[1]:
                continue
            else:
                n_total_runs += 1
    print(f"Total number of runs: {n_total_runs}")
    max_n_jobs = np.copy(n_jobs)
    all_permutations = [list(perm) for perm in itertools.combination(feat_shortcuts) if not ('sel' in perm and 'sym' in perm)]
    for feat_shortcuts in all_permutations:
        for model_shortcut in model_shortcuts:
            cv_status = pd.read_csv("data/tables/cv_status.csv")
            if check_if_previous_run(feat_shortcut, model_shortcut, datapath=results_savepath)[1]: 
                continue
            else:
                curr_run += 1
                print(f"Running {feat_shortcut} {model_shortcut} ({curr_run}/{n_total_runs})")
                avg_usage, idle_cores = get_avg_idle_cores(duration=duration)
                print(f"Number of idle cores: {idle_cores}: {avg_usage}")

                if idle_cores < n_jobs:
                    try:
                        n_jobs = inputimeout(prompt=f"Number of idle cores ({idle_cores}) is less than the number of jobs ({max_n_jobs}). How many jobs do you want to run? ", timeout=300)
                        n_jobs = int(n_jobs)
                    except TimeoutOccurred:
                        print(f"Timeout occurred. Using default number of jobs: {n_jobs}")
                        # n_jobs = max_n_jobs if max_n_jobs is not None else n_points*6
                        n_jobs = max(int(idle_cores), int(max_n_jobs//2))
                if outbase is not None:
                    out_file = os.path.join(outbase, f"{feat_shortcut}_{model_shortcut}.out")
                    out_appdx = f"> {out_file}"
                else:
                    out_appdx = ''
               

                which_features = [DATASETS_NAME_DICT[feat_shortcut] for feat_shortcut in feat_shortcuts]
                which_features0 = ''
                which_features1 = ''
                which_features2 = ''
                if len(which_features) == 1:
                    which_features0 = which_features[0]
                elif len(which_features) == 2:
                    which_features0 = which_features[0]
                    which_features1 = which_features[1]
                elif len(which_features) == 3:
                    which_features0 = which_features[0]
                    which_features1 = which_features[1]
                    which_features2 = which_features[2]
                    
                
                call = base_call.format(n_jobs=n_jobs, 
                                        model_name=MODEL_NAME_DICT[model_shortcut], n_points=n_points, 
                                        which_features0=which_features0, which_features1=which_features1,
                                        which_features2=which_features2, out_file=out_file,
                                        results_savepath=results_savepath,
                                        hyper_cv=hyper_cv, fs_cv=fs_cv, n_fs_repeats=n_fs_repeats, 
                                        n_hyper_repeats=n_hyper_repeats, n_iterations=n_iterations)
                print(call)
                try:
                    uin = inputimeout("Is this correct? [y/n]: ", timeout=wait_time)
                except TimeoutOccurred:
                    uin = "y"
                st = time.time()
                if uin == "y":
                    print("Running")
                    os.system(call)
                else:
                    print("Exiting")
                    sys.exit(0)
                # wait until the job is done
                time.sleep(min_run_time*60)
                n_files, finished = check_if_previous_run(feat_shortcut, model_shortcut, skip_confirmation=True, n_prior_files=None,datapath=results_savepath)
                print(f"Number of files when finish={finished}: {n_files} for feat_shortcut={feat_shortcut}, model_shortcut={model_shortcut}, wrapper_shortcut={wrapper_shortcut}, search_shortcut={search_shortcut}")
                st = time.time()
                while not finished:
                    n_files, finished = check_if_previous_run(feat_shortcut, model_shortcut, skip_confirmation=True, n_prior_files=n_files,datapath=results_savepath, verbose=False)
                    # every 30 minutes print the status
                    if time.time() - st > 1800:
                        print(f"Still running {feat_shortcut} {model_shortcut} ({curr_run}/{n_total_runs}), files={n_files}")
                        st = time.time()
                    time.sleep(1)
                print(f"Finished in {time.time()-st}", feat_shortcut, model_shortcut)
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run my models sequentially')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of jobs to run in parallel')
    parser.add_argument('--n_points', type=int, default=2, help='Number of points to sample in the hyperparameter search')
    parser.add_argument('--hyper_cv', type=int, default=3, help='Number of cross validation folds for the hyperparameter search')
    parser.add_argument('--fs_cv', type=int, default=3, help='Number of cross validation folds for the feature selection')
    parser.add_argument('--n_fs_repeats', type=int, default=2, help='Number of times to repeat the feature selection')
    parser.add_argument('--n_hyper_repeats', type=int, default=2, help='Number of times to repeat the hyperparameter search')
    parser.add_argument('--n_iterations', type=int, default=100, help='Number of iterations for the hyperparameter search')
    parser.add_argument('--min_run_time', type=int, default=0, help='Minimum run time in minutes')
    parser.add_argument('--reverse_search_order', action=argparse.BooleanOptionalAction, default=False, help='Whether to reverse the search order')
    parser.add_argument('--feat_shortcuts', nargs='+', default=["eeg", "ecg", "sym", "sel"], help='Which features to run')
    parser.add_argument('--duration', type=int, default=30, help='Duration to check the number of idle cores')
    parser.add_argument('--results_savepath', type=str, default=RESULTS_SAVEPATH, help="The path to save the results of the grid search")
    parser.add_argument('--outbase', type=str, default=None, help="The path to save the output files")
    parser.add_argument('--search_shortcuts', nargs='+', default=["bay"], help="The search methods to use")
    parser.add_argument('--wait_time', type=int, default=15, help="The time to wait for user response (seconds)")

    args = parser.parse_args()

    main(**vars(args))
