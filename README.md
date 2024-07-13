# mtbi_detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for analyzing the acute mTBI data from Mission Connect Study uploaded to FITBIR: doi: 10.23718/FITBIR/1518866

Will be completed prior to publication.

Installation:
 ``git clone git@github.com:aroyphillips/mTBI_Classification.git``
 ``conda env create -f environment.yml``
 Note: the package scikit-optimize has been updated since numpy.int was deprecated. To fix this, locate usages in ~/./conda/pkgs/scikit-optimize*/space/transformers.py and ~/./conda/pkgs/scikit-optimize*/benchmarks/bench_ml.py replace np.int with int.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── internal       <- Contains internal information, i.e. dataset subject splits
│   └── tables         <- The annotations used to segment the data
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml   <- The yml file for reproducing the analysis environment, e.g.
│                         generated with `conda env export > environments.yml`
│
└── mtbi_detection                <- Source code for replication of this project.
    │
    ├── __init__.py    <- Makes mtbi_detection a Python module
    │
    ├── demo.py    <- code for new user to run once the FITBIR data is downloaded
    │
    ├── data           <- Scripts to download or generate data
    │   ├── extract_data.py <- Code to convert the FITBIR data to MNE format fif files
    │   ├── load_open_closed_data.py <- Code to segment the EEG files into eyes open / eyes closed segments
    │   └── filter_data.py, rereference_data.py, cleanpath.py, data_utils.py <- Scripts with helper functions to run the above two scripts
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── compute_all_features.py <- Extracts all sets of features
    │
    ├── modeling         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── train_base_models.py
    │   ├── train_meta_models.py
        └── evaluate_final_models.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

## System Specification:

The original work was performed on the following Dell Server running RedHat Enterprise Linux with the following specifications:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              128
On-line CPU(s) list: 0-127
Thread(s) per core:  2
Core(s) per socket:  32
Socket(s):           2
NUMA node(s):        2
Vendor ID:           AuthenticAMD
CPU family:          25
Model:               1
Model name:          AMD EPYC 7543 32-Core Processor
Stepping:            1
CPU MHz:             2794.804
BogoMIPS:            5589.60
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            512K
L3 cache:            32768K
NUMA node0 CPU(s):   0-31,64-95
NUMA node1 CPU(s):   32-63,96-127

Note: we are in the process of refactoring the codebase so that it may run on other machines. Please submit a Github Issue or contact Roy Phillips (ap60@rice.edu) with specific issues.
--------

