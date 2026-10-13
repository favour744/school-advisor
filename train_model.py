"""Rebuild the model locally, using the same recipe as the Colab notebook.

Usage:
    python train_model.py                      # reads data/model_ready.csv
    python train_model.py --data other.csv --out model

Writes to the output folder:
    advisor_model.pkl   the fitted scikit-learn pipeline (best model by test MAE)
    course_info.csv     one row per course (category, units, level, ...) used by the API
    metrics.json        test-set scores, shown in the web page

You only need this if you want to retrain, or if the .pkl from Colab will not load
with the scikit-learn version installed here.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that must not be used as inputs: an id, the target, and two columns
# that are computed directly from the target (leakage).
DROP = ["student_id", "score", "grade", "grade_band"]
CAT_COLS = ["course_code", "category"]
DISPLAY_NAMES = {
    "Ridge (baseline)": "Ridge regression",
    "RandomForest": "Random forest",
    "HistGradBoost": "Gradient boosting",
}


def build_models(num_cols):
    def pre(scale):
        return ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
                ("num", StandardScaler() if scale else "passthrough", num_cols),
            ],
            sparse_threshold=0,  # never return a sparse matrix
        )

    return {
        "Ridge (baseline)": Pipeline([("pre", pre(True)), ("m", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(
            [("pre", pre(False)), ("m", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42))]
        ),
        "HistGradBoost": Pipeline([("pre", pre(False)), ("m", HistGradientBoostingRegressor(random_state=42))]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/model_ready.csv", type=Path)
    parser.add_argument("--out", default="model", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=DROP)
    y = df["score"]
    num_cols = [c for c in X.columns if c not in CAT_COLS]

    # Split by student so no student appears in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["student_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = build_models(num_cols)
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred),
        }

    print(f"scikit-learn {sklearn.__version__}")
    print(pd.DataFrame(results).T.round(3))

    best_name = min(results, key=lambda k: results[k]["mae"])
    best = models[best_name]
    print(f"\nBest model: {best_name}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, args.out / "advisor_model.pkl")

    course_info = (
        df.groupby("course_code")[["category", "units", "is_elective", "level", "semester", "has_prereqs"]]
        .first()
        .reset_index()
    )
    course_info.to_csv(args.out / "course_info.csv", index=False)

    metrics = {"model": DISPLAY_NAMES.get(best_name, best_name)}
    metrics.update({k: round(float(v), 3) for k, v in results[best_name].items()})
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved model, course table and metrics to {args.out}/")


if __name__ == "__main__":
    main()
