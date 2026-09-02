"""
Standalone cross-check for 07_linear_regression_lab.ipynb.

Recomputes the min-temp -> max-temp linear regression independently of
scikit-learn, using the closed-form least-squares formula directly with
NumPy, then compares the result against the notebook's saved output
(lab_results.json). This is deliberately a *different* method from the
notebook (no sklearn, no train/test split) -- the point is an independent
check, not a re-run of the same code.

Usage:
    python crosscheck_regression.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_FILE = "lab_results.json"
TOLERANCE = 0.05  # how close sklearn's and numpy's coefficients must be to "match"


def load_notebook_results(path: str) -> dict:
    results_path = Path(path)
    if not results_path.exists():
        print(f"Could not find {path!r}.")
        print("Run Section 8 of 07_linear_regression_lab.ipynb first to generate it.")
        sys.exit(1)
    with open(results_path) as f:
        return json.load(f)


def fit_line_by_hand(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares for y = coef * x + intercept, computed from
    the normal equations directly -- the same textbook formula behind
    every simple linear regression, just spelled out instead of hidden
    inside a library call."""
    x_mean = x.mean()
    y_mean = y.mean()
    coef = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - coef * x_mean
    return coef, intercept


def r_squared(x: np.ndarray, y: np.ndarray, coef: float, intercept: float) -> float:
    y_pred = coef * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main() -> None:
    notebook_results = load_notebook_results(RESULTS_FILE)

    file_path = notebook_results["file_path"]
    feature_col = notebook_results["feature_column"]
    target_col = notebook_results["target_column"]

    print(f"Loading {file_path!r} ...")
    df = pd.read_csv(file_path)

    for col in (feature_col, target_col):
        if col not in df.columns:
            print(f"Column {col!r} not found in {file_path!r}. Columns available: {list(df.columns)}")
            sys.exit(1)

    df = df[[feature_col, target_col]].dropna()
    x = df[feature_col].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    print(f"Using {len(df):,} rows (after dropping missing values).")

    correlation = np.corrcoef(x, y)[0, 1]
    numpy_coef, numpy_intercept = fit_line_by_hand(x, y)
    numpy_r2 = r_squared(x, y, numpy_coef, numpy_intercept)

    print("\n--- Independent check (NumPy, full dataset, no train/test split) ---")
    print(f"Correlation:        {correlation:.4f}")
    print(f"Coefficient:        {numpy_coef:.4f}")
    print(f"Intercept:          {numpy_intercept:.4f}")
    print(f"R^2 (on all data):  {numpy_r2:.4f}")
    print(f"Equation:           {target_col} = {numpy_coef:.4f} * {feature_col} + {numpy_intercept:.4f}")

    print("\n--- Notebook's result (scikit-learn, fit on the training split) ---")
    sklearn_coef = notebook_results["sklearn_coef"]
    sklearn_intercept = notebook_results["sklearn_intercept"]
    sklearn_r2 = notebook_results["sklearn_r2_test"]
    print(f"Coefficient:        {sklearn_coef:.4f}")
    print(f"Intercept:          {sklearn_intercept:.4f}")
    print(f"R^2 (on test set):  {sklearn_r2:.4f}")

    coef_diff = abs(numpy_coef - sklearn_coef)
    intercept_diff = abs(numpy_intercept - sklearn_intercept)

    print("\n--- Comparison ---")
    print(f"Coefficient difference:  {coef_diff:.4f}  (tolerance: {TOLERANCE})")
    print(f"Intercept difference:    {intercept_diff:.4f}  (tolerance: {TOLERANCE})")

    if coef_diff <= TOLERANCE and intercept_diff <= TOLERANCE:
        print("\nMATCH -- the notebook's regression agrees with the independent NumPy calculation.")
    else:
        print("\nMISMATCH -- results differ by more than the tolerance.")
        print("This can be expected if the notebook fit on a train/test split (different rows)")
        print("than this script's full-dataset fit -- some difference is normal. A LARGE")
        print("difference, though, is worth investigating: check you're using the same")
        print("feature/target columns, and that missing values were handled the same way.")


if __name__ == "__main__":
    main()
