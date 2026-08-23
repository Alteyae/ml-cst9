# ML CST9 — Introduction to Machine Learning

Course material for an introductory machine learning subject: a series of hands-on Jupyter notebooks covering the ML workflow end to end, from foundational concepts through data cleaning, feature engineering, and modelling.

## Contents

### Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_introduction_to_machine_learning.ipynb` | Runnable examples for every concept in Part 1: traditional programming vs. ML, AI/ML/DL, supervised (classification & regression) / unsupervised / reinforcement learning, and the full 7-step ML workflow. |
| 02 | `02_dataset_inspection_template.ipynb` | A reusable template: load any dataset, run the "first ten minutes" inspection habits, visualize it, and get a scored go/no-go verdict before committing to modelling it. |
| 03 | `03_modelling_template.ipynb` | The next step after inspection: feature engineering, train/test split, modelling, and evaluation — auto-adapts to classification or regression. |
| 04 | `04_preprocessing_demo.ipynb` | A fully worked, narrated walkthrough cleaning a deliberately broken dataset step by step (missing values, duplicates, outliers, wrong dtypes, leakage, class imbalance), with notes on how each problem shows up differently on other datasets. |
| 05 | `05_preprocessing_practice.ipynb` | The same exercise as notebook 04, but as a boilerplate with a different broken dataset for students to clean themselves, plus a self-grading check at the end. |
| 06 | `06_feature_engineering_demo.ipynb` | Encoding (one-hot, ordinal), scaling, date feature extraction, binning, and derived features — a worked demo followed by a practice section with a second dataset. |

Each numbered notebook builds on the last; templates (02, 03) and practice notebooks (05, 06's Part B) are meant to be copied and reused on a dataset of your own choosing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
python -m ipykernel install --user --name=ml-cst9 --display-name="Python 3 (ml-cst9)"
```

Open any notebook in Jupyter or VS Code and select the `ml-cst9` kernel.
