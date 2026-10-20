# Home Credit Default Risk Prediction

**A machine learning project predicting loan default risk, built as part of the bootcamp :**

This project estimates how likely a loan applicant is to have trouble repaying. The goal is to help Home Credit's underwriting and credit risk team decide which applications need a closer look. It was built as a team project for the neuefische Data Science bootcamp by **Narayani & Prithvi**.

## Project Overview

The **Home Credit Default Risk Prediction** project builds a machine learning tool that estimates how likely a loan is to default, to support better lending decisions. It uses the Home Credit dataset and covers data cleaning, exploratory data analysis, preprocessing, model building, tuning and evaluation.

Several classification models were compared on **ROC-AUC, recall, precision and F2-score**. Catching potential defaulters was the main priority. The final solution gives each application a predicted default probability. Applications at **42% or higher** are flagged for further review, which supports more consistent, risk-aware credit decisions.

## Business Problem

Home Credit is a consumer finance company focused on financial inclusion. Many of its applicants have little or no credit history, so a lending decision can go wrong in two ways:

| Mistake | Consequence |
|---|---|
| **Rejecting a good client** | Lost interest income and market share to competitors |
| **Approving a risky client** | Collection costs and write-offs, which are a direct financial loss |

**Objective:** when an application comes in, predict the probability (0–1) that the client will have payment difficulties. The underwriting team can then approve, decline or refer it for review.

This is a **binary classification** task. The target is `TARGET`: 1 means payment difficulty, 0 means the client repaid normally.

## Data

The data comes from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition on Kaggle.

| | |
|---|---|
| Training applications | 307,511 |
| Test applications | 48,744 |
| Features per application | 122 |
| Default rate | **8.07%** (about 1 in 12) |

Because only about 8% of applicants defaulted, the classes are **highly imbalanced**. A model that predicts "always repays" is 91.9% accurate but catches **0%** of defaulters, so accuracy is a misleading measure here.

### Key EDA Findings

- **External credit scores** (`EXT_SOURCE_1/2/3`) are the strongest predictors. The higher the score, the lower the default risk.
- **Age**: younger clients are slightly riskier than older ones. Most applicants are 30–40 years old.
- **Employment duration**: about 18% of applicants have an impossible value (about 1,000 years) in `DAYS_EMPLOYED`. This sentinel value is replaced with NaN before modelling.
- **Occupation type**: 31.3% of values are missing. Laborers are the largest occupation group (17.9% of applicants).

## Approach

1. **Data cleaning**: handle sentinel values, missing values and data-entry errors.
2. **EDA**: analyse the numeric and categorical features against the target.
3. **Preprocessing pipeline**: scikit-learn pipelines for imputation, scaling and encoding.
4. **Modelling**: a dummy baseline, then Logistic Regression, Decision Tree, Random Forest and XGBoost. Every model uses `class_weight="balanced"` (or its XGBoost equivalent) to handle the class imbalance. SMOTE oversampling was also tested.
5. **Hyperparameter tuning** with `RandomizedSearchCV` and cross-validation.
6. **Threshold analysis**: choose the decision threshold that maximises the F2-score.
7. **Feature importance** for the tree-based models.

### Evaluation Metrics

- **Recall** (primary): catch as many real defaulters as possible.
- **ROC-AUC / PR-AUC**: how well the model separates risky applicants from safe ones.
- **Precision**: avoid flagging too many reliable customers.
- **F2-score**: combines precision and recall, but counts recall twice as much as precision.

## Results

| Model | Accuracy | Recall | ROC-AUC | Precision |
|---|---|---|---|---|
| Baseline (always predicts "repays") | 0.91 | 0.00 | — | 0.00 |
| Logistic Regression (balanced, tuned) | 0.69 | 0.68 | 0.75 | 0.16 |
| Decision Tree (balanced, tuned) | 0.66 | 0.67 | 0.72 | 0.15 |
| Random Forest (balanced, tuned) | 0.70 | 0.62 | 0.72 | 0.16 |
| **XGBoost** | **0.69** | **0.69** | **0.76** | **0.17** |

### Final Model: XGBoost

- **ROC-AUC 0.76**, the best risk separation of all the models.
- **Recall 68.7%**: catches about 69% of actual defaulters.
- **F2-score 0.42** at the chosen threshold.
- **Precision 16.6%**: about 1 in 6 flagged applicants actually defaulted. That is twice the base rate, but the flagged group still needs manual review.

### How the Prediction Is Used

Every new application gets a default risk score. If the score is **42% or higher**, the application is **flagged for further review**. This threshold maximises the F2-score, so it favours finding defaulters (recall) over avoiding false alarms (precision).

> **Recommendation:** use the model to rank risk and support decisions, not to approve or reject loans automatically.

## Limitations & Next Steps

- **Low precision**: many flagged applicants are not actually risky.
- **Application data only**: the current model does not use historical credit behaviour.

Possible improvements:

- **Add historical data**: aggregate bureau, previous application, installment, credit card and POS/CASH data into customer-level features. This work has started in [09.Master_data.ipynb](09.Master_data.ipynb).
- **Explainability**: use SHAP values to explain each prediction to underwriters.
- **Cost-based threshold**: set the threshold from the real business cost of false positives and false negatives.
- **Product**: an underwriting dashboard that shows each applicant's risk score and the main factors behind it.

## Repository Structure

| File / Folder | Description |
|---|---|
| [04.eda-and-modeling-default-risk.ipynb](04.eda-and-modeling-default-risk.ipynb) | Main EDA and modelling notebook for Home Credit |
| [05.Presentation.pdf](05.Presentation.pdf) | Final stakeholder presentation |
| [06.eda_output.ipynb](06.eda_output.ipynb) | Modelling notebook with saved outputs, threshold analysis and SMOTE experiments |
| [07.eda_git.ipynb](07.eda_git.ipynb), [08.eda.ipynb](08.eda.ipynb) | Working versions of the EDA and modelling notebook |
| [09.Master_data.ipynb](09.Master_data.ipynb) | Merges the auxiliary tables (bureau, previous applications, etc.) into one master dataset |
| [data/](data/) | Datasets (not tracked by git) |
| [models/](models/) | Saved models |
| [archive/](archive/) | Bootcamp template files: project brief, Kanban board guide and the example coffee quality notebook |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Dependencies |

## Setup

**Requirements:** Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd ML_Project_Home_Credit_Default_Risk
uv sync
```

### Get the Data

Download the dataset from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data). Then place the files as follows:

- `application_train.csv` and `application_test.csv` → `data/`, used by the modelling notebooks
- all raw tables (`bureau.csv`, `previous_application.csv`, etc.) → `data/raw/`, used by `09.Master_data.ipynb`

### Run the Notebooks

```bash
code .
```

Open a notebook and select the `.venv` kernel that `uv sync` created.

**Main libraries:** pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, matplotlib, seaborn, plotly.

## Working with Notebooks in a Team

To avoid messy merge conflicts in `.ipynb` files, use `nbdime`:

```bash
uvx nbdime config-git --enable   # one-time setup
uvx nbdime mergetool             # resolve a conflict
```

It also helps to clear notebook outputs before committing (in VS Code: `Notebook: Clear All Outputs`).

## Authors

**Narayani & Prithvi**, neuefische Data Science Bootcamp, 2026

## License

See [LICENSE](LICENSE).
