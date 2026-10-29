# Prediction and Maintenance of Critical Automotive Component Failure Using Machine Learning

A vehicle problem can interrupt a journey long before the next planned service. Sensor readings may offer clues about what is going wrong. The idea behind this project is to explore whether those readings can help identify an engine, brake or battery issue.

For this study, I compared six machine-learning approaches using a supplied synthetic dataset. The aim was to understand how well they recognise issue readings and what kinds of mistakes they make.

The dataset contains 1,970 readings from 50 vehicles. An issue label appears on just 35 readings. These readings come from 25 vehicles. Because each component has few examples, the three issue types are combined into one classification task.

## Results dashboard

The [interactive dashboard](https://prasadkolkar.github.io/critical-automotive-component-failure/dashboard.html) brings together the model results, confusion matrices and follow-up experiments. It also shows predictions for random readings from the saved evaluation results.

To view it offline, download [dashboard.html](dashboard.html) and open it in a browser.

## How the project is organised

The work is presented in three notebooks. They follow the same path as the report, starting with the data and ending with individual predictions.

| Notebook | What it covers |
| --- | --- |
| [Understanding the data](notebooks/01_understanding_the_data.ipynb) | Sensor patterns and data quality |
| [Comparing the models](notebooks/02_comparing_the_models.ipynb) | Six model families and the mistakes they make |
| [Improving and testing](notebooks/03_improving_and_testing.ipynb) | Follow-up experiments and predictions for random readings |

The [code walkthrough](docs/CODE_WALKTHROUGH.md) explains the choices behind the implementation. The notebooks contain the analysis code alongside its graphs and observations. Figures use Matplotlib and Seaborn.

## Making the comparison fair

Each vehicle contributes several readings. If some appear in training and others in testing, the model could benefit from already knowing that vehicle. Keeping whole vehicles together gives a more useful comparison.

In each round, 40 vehicles provide training data and ten are held out. Smaller groups within the training data are used to choose model settings. Preparation steps also learn from training readings only.

With so few issue examples, accuracy can give the wrong impression. Predicting every reading as ordinary would be 98.22% accurate while missing all 35 issues. The comparison therefore looks at both missed issues and false alerts.

Recall measures how many issues are found. Precision measures how often an alert matches an issue label. F1 brings those two measures together. Average precision looks at how well the model ranks issue readings across possible cutoffs.

## What the recorded study found

The table below uses the original saved predictions across all five vehicle groups. These models exclude the ABS fault indicator and use a decision cutoff of 0.5.

| Model | F1 | Average precision | Issues found | False alerts | Missed issues |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4776 | 0.8427 | 32 | 67 | 3 |
| Random Forest | 0.9062 | 0.9631 | 29 | 0 | 6 |
| XGBoost | 0.9714 | 0.9939 | 34 | 1 | 1 |
| SVM | 0.9444 | 0.9896 | 34 | 3 | 1 |
| Gaussian Naive Bayes | 0.4960 | 0.8747 | 31 | 59 | 4 |
| ANN with original inputs | 0.9315 | 0.9895 | 34 | 4 | 1 |
| ANN with selected sensors | 0.9714 | 0.9909 | 34 | 1 | 1 |

XGBoost and the ANN with selected sensors each found 34 of the 35 issue readings with one false alert. Their mistakes occurred on different readings. Other models showed a different balance between finding issues and raising unnecessary alerts.

Selecting sensors helped the ANN reduce its false alerts from four to one. PCA did not provide the same benefit. It lowered the ANN's F1 to 0.8611. The two sampling methods also failed to improve on recorded weighted XGBoost in both average precision and F1.

The original predictions remain in `results/recorded/`. Notebook 2 compares these with newly fitted models. The original fitted models are unavailable, and later Random Forest and XGBoost fits produced different predictions. Keeping both sets of results makes that difference visible.

## Trying individual readings

The final notebook makes the comparison easier to inspect. It picks three ordinary readings and three issue readings at random. Their sensor values appear beside the supplied labels and the models' predictions.

Running the example cell again picks a new sample. The models stay unchanged. Each reading uses a saved model that did not train on its vehicle. The ANN uses the selected sensors from the follow-up experiment.

## Running the notebooks

### Google Colab

Open a notebook in [Google Colab](https://colab.research.google.com/) and choose a Python CPU runtime. Run the cells in order and connect Google Drive when prompted. If setup updates a library that is already loaded, it pauses before the analysis. Choose **Runtime → Restart session**, then **Run all**.

The project folder is `MyDrive/Automotive_Component_Failure/`. It holds the dataset and saves models and results between sessions.

### Local Jupyter

The same notebooks can run locally with Python 3.12. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Each notebook has its own setup and data-loading cells. Local files follow the same layout as the Drive folder.

### After the first run

Training is needed the first time. Each completed model is then saved with its preparation steps. Later sessions load those files, so graphs and example predictions can be replayed without repeating training.

The dataset and vehicle groups stay fixed. The study also keeps the candidate settings and training seeds unchanged. Package versions are listed in `requirements.txt` and checked before saved models are reused.

## Project files

```text
notebooks/             Three analysis notebooks
dashboard.html         Interactive results dashboard
data/                  Dataset and vehicle groups
models/                Saved models
results/
  recorded/            Original study results
  tables/              Evaluation tables
  figures/             Analysis graphs
  predictions/         Held-out predictions
docs/                  Code walkthrough
reports/               Main and framework LaTeX packages
tests/                 Automated checks
environment.json       Package versions
requirements.txt       Project dependencies
```

Saved models and regenerated result files are excluded from Git. The notebooks include their recorded outputs. The two report packages are in `reports/`. Each ZIP contains the LaTeX source and figures for Overleaf.

The checks cover vehicle separation and the treatment of model inputs. They also check sampling and saved-model reuse:

```bash
python -m pytest -q
```

## What this tells us

The results show that several models can recognise the supplied issue labels. The data are synthetic, however, and there are few issue examples. A later analysis also found sensor cutoffs that separated the component labels perfectly after all labels were known.

This helps explain why the recorded scores are high. It leaves an important question open: would the same patterns provide a useful warning before a real breakdown? Answering that needs real vehicle histories linked to confirmed failures and a clearly defined warning period.
