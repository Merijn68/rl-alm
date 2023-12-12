RLALM
==============================

ALM Interest Rate Steering using Reinforcement Learning algorithms
This repo contains the source code and test results from the my Thesis on Reinforcement Learning in Asset and Liability Management.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── installation.txt   <- Installation instruction for Windows
    ├── data
    │   ├── model          <- Data from the trained models stored for analysis
    │   └── raw            <- Data downloaded from European Statistical Data Warehouse as Input for the model
    │
    ├── docs               <- Auto generated documentation from docstrings.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Containing the executed experiments during the research
    │
    ├── reports            <- Output of the reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── videos         <- Movie file showing optimal strategy
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported easier
    ├── src                <- Source code for use in this project.    
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   ├── tests          <- scripts used for unit testing
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations    
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

See below an example of the policy found by the SAC Algorythm to optimize funding.

<video src='https://github.com/Merijn68/rl-alm/tree/main/reports/videos/optimal_policy.mp4' width=180/>

