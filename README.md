# Critical automotive component failure classification

Vehicle sensors may reveal an engine, brake, or battery issue, but a high classification score on simulated data is not the same as an early warning before a real breakdown. This project compares several models on a supplied synthetic telemetry dataset and examines what their results do and do not show.

The workbook has 1,970 readings from 50 vehicles. Thirty-five readings are labelled with an engine, brake, or battery issue; those readings come from 25 vehicles. The three labels are combined into one binary outcome because each component has too few positive readings for a stable separate model comparison. All readings from a vehicle stay together during evaluation.

## Main results

The table reports pooled predictions for held-out vehicles, without the ABS fault indicator. AP measures how well a model ranks the rare labelled readings; F1 summarizes decisions at the fixed 0.5 threshold.

| Model | F1 | AP | Detected | False alerts | Missed |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4776 | 0.8427 | 32 | 67 | 3 |
| Random Forest | 0.9062 | 0.9631 | 29 | 0 | 6 |
| XGBoost, recorded run | 0.9714 | 0.9939 | 34 | 1 | 1 |
| Support Vector Machine | 0.9444 | 0.9896 | 34 | 3 | 1 |
| Gaussian Naive Bayes | 0.4960 | 0.8747 | 31 | 59 | 4 |
| Neural network, original inputs | 0.9315 | 0.9895 | 34 | 4 | 1 |
| Neural network, selected inputs | 0.9714 | 0.9909 | 34 | 1 | 1 |

Selecting inputs improved the neural network's F1 from 0.9315 to 0.9714; PCA reduced it to 0.8611. Random oversampling and SMOTENC helped some unweighted fits but did not improve on the strongest weighted XGBoost result in both AP and F1. Removing the ABS indicator did not erase the leading performance.

These are development results from one synthetic source, not measurements of warning time or reliability in a real fleet. Simple sensor cutoffs selected after all labels were known separate the observed component labels perfectly. The fitted models from the recorded XGBoost run were not saved, and a later reconstruction changed seven decisions. The original XGBoost row is therefore a recorded result rather than an exactly reproduced benchmark.

## Repository contents

    Resources/synthetic_telemetry_data.xlsx    Supplied synthetic workbook
    notebooks/                                   EDA, preparation, and initial modelling notebooks
    src/                                         Data, model, and diagnostic functions
    scripts/                                     Experiment and diagnostic entry points
    tests/                                       Grouping, sampling, and diagnostic checks
    outputs/tables/                              Preserved preparation and initial model tables
    results/extended/                            Selected model results and held-out predictions
    results/diagnostics/                         Selected error and feature diagnostics

Notebook outputs have been cleared; their code and explanations remain. Temporary build files, report drafts, and local environments are not part of this repository.

## Setup and checks

Python 3.12 is recommended. From the repository root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python scripts/verify_release.py
    python -m pytest -q

The verification command checks the workbook, vehicle folds, saved predictions, and comparison metrics without fitting a model.

## Reproduce the analysis

The notebooks show the exploratory analysis, preparation, and initial three-model study. To rerun the extended comparison and diagnostics from the supplied workbook and saved initial tables:

    python scripts/run_extended_experiments.py
    python scripts/summarize_extended_experiments.py
    python scripts/run_final_diagnostics.py

The extended run fits 24 configurations across five vehicle holdouts and can take substantial time. It writes new artifacts under outputs/experiments/ and outputs/diagnostics/; the selected result snapshots in results/ remain unchanged. Model scores can differ across software environments, particularly for XGBoost. The saved result snapshots document the original completed study.

## Data

The dataset was supplied for this project as synthetic vehicle telemetry. Its generation method, original public source, confirmed breakdown events, and warning horizon were not documented. The workbook is included for analysis reproducibility; no broader claim about its origin or suitability for real maintenance decisions is made.
