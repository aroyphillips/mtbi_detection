# mtbi_detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for analyzing the acute mTBI data from Mission Connect Study uploaded to FITBIR: doi: 10.23718/FITBIR/1518866

Will be completed prior to publication submission.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
    ├── tables         <- Metadata files, e.g. the segment annotations
    ├── tables         <- Internal files needed to run code (e.g. the scripts)
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for mtbi_detection
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mtbi_detection                <- Source code for use in this project.
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
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── train_base_models.py
    │   ├── train_meta_models.py
        └── evaluate_models.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

