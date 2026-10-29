# Code Walkthrough

The project begins with a simple question: can vehicle sensor readings help identify a component issue? The code follows that question through the data and model comparisons. It ends by examining predictions for individual readings.

Three notebooks present the work. They share the same dataset and vehicle groups, so the comparisons follow one consistent setup.

## Starting with the data

[Notebook 1](../notebooks/01_understanding_the_data.ipynb) takes a closer look at the workbook. There are 1,970 readings from 50 vehicles. Each reading contains sensor measurements and information about the vehicle's operation. The supplied labels identify engine, brake or battery issues.

Only 35 readings have an issue label. They come from 25 vehicles. This small number shaped the main modelling decision: combine the three issue types into one outcome.

A reading receives label **1** if any component issue is present. Otherwise, it receives label **0**. That gives 35 issue readings and 1,935 ordinary readings. The labels describe the supplied readings; they do not establish 35 independent real breakdowns.

The next step is to understand the measurements. Quality checks look for missing values and repeated records. They also highlight unusual values that need attention. The original readings are kept because the workbook does not provide a reliable way to correct them.

The graphs then make the data easier to explore. Histograms show where sensor values tend to fall. Boxplots compare readings with and without each component label. The heatmap looks at relationships between selected sensors.

Oil pressure stands out in the engine comparison. Brake-fluid pressure and battery voltage show similar differences for their corresponding labels. These observations provide a reason to test whether models can recognise the patterns on other vehicles.

## Moving from readings to models

[Notebook 2](../notebooks/02_comparing_the_models.ipynb) turns those observations into a model comparison. One important choice is deciding what the models should receive as input.

Failure date and failure type disclose the outcome. The original component flags do too. I left these fields out so the models would have to learn from the measurements. Vehicle ID is used to form groups, while the raw timestamp is excluded from the predictors.

The baseline uses 32 numerical measurements and brand. Brand is converted into indicator columns through one-hot encoding. Models that depend on feature scale also receive scaled numerical values. Tree models use the original scale.

All fitted preparation steps use training readings only. This keeps information from the held-out vehicles out of the fitting process.

### Why six model families?

Different approaches may recognise different patterns. A linear model gives a useful starting point. Trees can capture sensor cutoffs and interactions. The other approaches provide further ways to combine the same measurements.

| Model | Basic idea |
| --- | --- |
| Logistic Regression | Learns a linear decision rule |
| Random Forest | Combines decisions from several trees |
| XGBoost | Builds trees that refine earlier predictions |
| SVM | Learns a boundary between the classes |
| Gaussian Naive Bayes | Uses feature distributions within each class |
| ANN | Combines sensor inputs through a hidden layer |

The candidate settings follow the study. The ANN grid tries 16 or 32 hidden units, each paired with two regularization values.

### Testing on other vehicles

Each vehicle appears more than once in the workbook. Splitting individual readings at random could therefore place one vehicle in both training and testing. The comparison instead keeps whole vehicles together.

There are five fixed groups. Each round trains on 40 vehicles and holds out the remaining ten. Three smaller groups inside the training data help choose settings. Average precision determines the preferred setting. Ties are resolved using F1 and then recall. An exact tie keeps the original candidate order.

By the end, every reading has one prediction from a model that did not train on its vehicle.

### Understanding the results

Accuracy alone would hide the main challenge. Calling every reading ordinary would give 98.22% accuracy while finding none of the issues. The results therefore focus on the issues found and the false alerts raised.

Precision describes how often an alert matches an issue label. Recall describes how many issue readings are found. F1 combines both. Average precision examines how well the scores rank issue readings across possible cutoffs.

The confusion matrices make the counts visible. The precision–recall curves show how the balance changes as the cutoff moves. The class predictions use a fixed cutoff of 0.5.

### Keeping the fitted models

Each selected model is saved with its preparation steps and settings. The file also records which rows belonged to training and evaluation. These saved fits can then be used for later predictions.

This means that replaying a graph or choosing new examples does not require another training run. The data and settings are checked before reuse. Package versions are checked as well.

Notebook 2 also compares newly fitted models with the original saved study results. The two are kept separate because rebuilding a model can change its predictions.

## Looking for improvements

[Notebook 3](../notebooks/03_improving_and_testing.ipynb) follows up on the comparison. The main questions are whether fewer inputs help and whether more positive training examples improve the decisions.

### Feature selection and PCA

Feature selection keeps measurements that separate the training labels. The search tries ten or 20 numerical sensors. Brand is encoded separately.

PCA takes a different approach. It combines measurements into components that preserve overall variation. The search retains either 90% or 95% of that variation.

A rare issue pattern may contribute little to overall variation. That gives a reason to compare PCA with selection based on the labels. Both are tested against the original inputs for Logistic Regression and ANN.

### The ABS indicator

The workbook also contains an ABS fault indicator. Its timing relative to a later failure is unclear. Adding it to the inputs shows whether the model results depend on that field.

### More positive training examples

The small number of issue readings also led to an augmentation experiment. Random oversampling repeats existing positive readings. SMOTENC creates nearby training examples while treating brand as a category.

The engine, brake and battery patterns remain separate during sampling. An engine example is therefore not mixed with a battery example. Sampling uses training data only, while evaluation continues to use the original readings.

Three fixed sampler seeds show how the results vary. Weighted and unweighted models provide reference comparisons. The range bars in the graph show the lowest and highest scores across those seeds.

### Understanding what the models use

The notebook counts how often each sensor is selected. It also looks at individual errors and simple sensor cutoffs.

The feature-shuffling results come from the recorded XGBoost rebuild. The original fitted models are unavailable, so this analysis describes the rebuilt models. The descriptive cutoffs were chosen after every label was known. Their separation helps explain the dataset, but it is not an independent prediction test.

## Trying individual readings

The final example section makes the model outputs easier to inspect. It randomly picks three ordinary readings and three issue readings. Both classes are included so the rare issues appear in the demonstration.

The sensor values are shown beside the supplied label. Each model then returns a score and predicted class. The ANN uses the selected sensors from the follow-up experiment.

For each reading, the notebook loads the model from the correct evaluation fold. That model did not train on the reading's vehicle. Choosing another random sample changes the examples while leaving the fitted models unchanged.

These examples show individual decisions. The overall scores still use every held-out reading.

## What the comparison shows

In the recorded study, weighted XGBoost found 34 of the 35 issue readings with one false alert. The selected-sensor ANN matched these counts. Their errors occurred on different readings, which is why looking beyond the headline score matters.

Feature selection helped the ANN. PCA reduced its F1. Neither sampling method surpassed recorded weighted XGBoost in both average precision and F1.

The synthetic labels have strong associations with a few sensors. That may help explain the high recorded scores. The next question is whether such patterns would appear early enough to support real maintenance decisions. That requires confirmed failures and a known warning period.
