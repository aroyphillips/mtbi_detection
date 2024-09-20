# queue up the different experiments in a sequential manner to make use of the system resources
# doing it this way uses the preset argparse feature arguments
import os
import sys
import argparse
import psutil
import time
import subprocess
import numpy as np
import itertools
import dotenv
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


DATAPATH = os.getenv('EXTRACTED_PATH')
LOCD_DATAPATH = os.getenv('OPEN_CLOSED_PATH')
# DATAPATH = open('extracted_path.txt', 'r').read().strip() 
# LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'features')
RESULTS_SAVEPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'results')
OUTBASE = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'outfiles', 'baselearners')

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

def generate_filtered_permutations(feat_shortcuts, exclusions=None):
    """
    Generates all combinations of elements from feat_shortcuts,
    
    Inputs:
        - feat_shortcuts: List of feature shortcuts.
        - exclusions: List of feature shortcuts to exclude from the permutations when they are present
    Outputs:
        - all_permutationsList of filtered permutations.
    """
    all_permutations = []
    for k in range(1, len(feat_shortcuts) + 1):
        for perm in itertools.combinations(feat_shortcuts, k):
            if exclusions is not None:
                if not all([excl in perm for excl in exclusions]):
                    all_permutations.append(list(perm))
            else:
                all_permutations.append(list(perm))
    return all_permutations


def main(n_jobs=3, n_points=1, min_run_time=15, feat_shortcuts = ["eeg", "ecg", "sym", "sel"],\
         duration=30, n_hyper_cv=3, n_fs_cv=3, n_fs_repeats=2, n_iterations=100, n_hyper_repeats=2,\
            results_savepath=RESULTS_SAVEPATH, outbase=None, wait_time=15):
    """
    Driver to run train_baselearners.py across all feature sets and base models
    """
    print(f"Beginning to run {n_jobs} jobs in parallel")
    # call the python script with the correct arguments
    if outbase is not None:
        os.makedirs(outbase, exist_ok=True)
        
    base_call = ("python3 mtbi_detection/modeling/train_baselearners.py --reference_method=CSD --n_jobs={n_jobs} "
                 "--n_hyper_cv={n_hyper_cv} --n_fs_cv={n_fs_cv} --n_fs_repeats={n_fs_repeats} --n_hyper_repeats={n_hyper_repeats} "
                 "--model_name={model_name} --n_iterations={n_iterations} --n_points={n_points} --skip_ui --verbosity=2 "
                 "--results_savepath={results_savepath} --which_features {which_features0} {which_features1} {which_features2} "
                 "{out_appdx}")
    n_total_runs = 0
    curr_run = 0
    model_shortcuts = ["xgb", "rf", "knn",  "gnb", "ada", "lre"] 
    all_permutations = generate_filtered_permutations(feat_shortcuts, exclusions=["sel", "sym"])
    n_total_runs = len(all_permutations) * len(model_shortcuts)
    print(f"Total number of runs: {n_total_runs}")
    max_n_jobs = np.copy(n_jobs)
    for feat_shortcuts in all_permutations:
        for model_shortcut in model_shortcuts:
            print(f"Running {feat_shortcuts} {model_shortcut} ({curr_run}/{n_total_runs-1})")
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
                out_file = os.path.join(outbase, f"{'-'.join(feat_shortcuts)}_{model_shortcut}_out.txt")
                out_appdx = f"&> {out_file}"
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
                                    which_features2=which_features2,
                                    results_savepath=results_savepath,
                                    n_hyper_cv=n_hyper_cv, n_fs_cv=n_fs_cv, n_fs_repeats=n_fs_repeats, 
                                    n_hyper_repeats=n_hyper_repeats, n_iterations=n_iterations, out_appdx=out_appdx)
            print(call)
            try:
                uin = inputimeout("Is this correct? [y/n]: ", timeout=wait_time)
            except TimeoutOccurred:
                uin = "y"
            st = time.time()
            if uin == "y":
                print("Running")
                subprocess.run(call, shell=True)
            else:
                print("Exiting")
                sys.exit(0)
            st = time.time()
            print(f"Finished in {time.time()-st} seconds ({curr_run}/{n_total_runs-1})")
            curr_run += 1

                        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run my models sequentially')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    parser.add_argument('--n_points', type=int, default=2, help='Number of points to sample in the hyperparameter search')
    parser.add_argument('--n_hyper_cv', type=int, default=3, help='Number of cross validation folds for the hyperparameter search')
    parser.add_argument('--n_fs_cv', type=int, default=3, help='Number of cross validation folds for the feature selection')
    parser.add_argument('--n_fs_repeats', type=int, default=2, help='Number of times to repeat the feature selection')
    parser.add_argument('--n_hyper_repeats', type=int, default=2, help='Number of times to repeat the hyperparameter search')
    parser.add_argument('--n_iterations', type=int, default=100, help='Number of iterations for the hyperparameter search')
    parser.add_argument('--min_run_time', type=int, default=0, help='Minimum run time in minutes')
    parser.add_argument('--feat_shortcuts', nargs='+', default=["eeg", "ecg", "sym", "sel"], help='Which features to run')
    parser.add_argument('--duration', type=int, default=2, help='Duration to check the number of idle cores')
    parser.add_argument('--results_savepath', type=str, default=RESULTS_SAVEPATH, help="The path to save the results of the grid search")
    parser.add_argument('--outbase', type=str, default=OUTBASE, help="The path to save the output files")
    parser.add_argument('--wait_time', type=int, default=3, help="The time to wait for user response (seconds)")

    args = parser.parse_args()

    main(**vars(args))
