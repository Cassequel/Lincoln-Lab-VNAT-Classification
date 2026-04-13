"""
Training script for VNAT traffic classification.

Usage:
    python src/train.py --task category --model lgbm
    python src/train.py --task vpn     --model rf
    python src/train.py --task category --model all

Saves the best model to models/<task>_<model>.joblib.
Prints classification report and macro-F1 on the validation set.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/features.csv"
MODELS_DIR  = "models"
RANDOM_SEED = 42

RF_GRID = {
    "n_estimators":     [200, 400],
    "max_depth":        [None, 20],
    "min_samples_leaf": [1, 5],
}

LGBM_GRID = {
    "learning_rate":     [0.05, 0.1],
    "num_leaves":        [31, 63],
    "n_estimators":      [200, 400],
    "min_child_samples": [10, 30],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str):
    features = pd.read_csv(path)
    X = features.drop(columns=["is_vpn", "app", "category"])
    y_cat = features["category"]
    y_vpn = features["is_vpn"].astype(int)
    return X, y_cat, y_vpn


def split(X, y_cat, y_vpn):
    X_train, X_temp, yc_train, yc_temp, yv_train, yv_temp = train_test_split(
        X, y_cat, y_vpn, test_size=0.30, random_state=RANDOM_SEED, stratify=y_cat
    )
    X_val, X_test, yc_val, yc_test, yv_val, yv_test = train_test_split(
        X_temp, yc_temp, yv_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=yc_temp
    )
    return X_train, X_val, X_test, yc_train, yc_val, yc_test, yv_train, yv_val, yv_test


def class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def sample_weights(y, weight_dict):
    return y.map(weight_dict).values


def build_lr(cw):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight=cw,
                                   random_state=RANDOM_SEED, n_jobs=-1)),
    ])


def build_rf(cw):
    return GridSearchCV(
        RandomForestClassifier(class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1),
        RF_GRID, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1,
    )


def build_lgbm():
    return GridSearchCV(
        LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
        LGBM_GRID, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1,
    )


def evaluate(name, model, X_val, y_val):
    preds    = model.predict(X_val)
    macro_f1 = f1_score(y_val, preds, average="macro")
    print(f"\n=== {name}  (val macro-F1: {macro_f1:.4f}) ===")
    print(classification_report(y_val, preds))
    return macro_f1


def save(model, task, model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{task}_{model_name}.joblib")
    joblib.dump(model, path)
    print(f"Saved {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def train(task: str, model_name: str):
    print(f"Loading data from {DATA_PATH} ...")
    X, y_cat, y_vpn = load_data(DATA_PATH)
    X_train, X_val, X_test, yc_train, yc_val, yc_test, yv_train, yv_val, yv_test = split(X, y_cat, y_vpn)

    if task == "category":
        y_train, y_val_target = yc_train, yc_val
        cw = class_weights(y_train)
    else:
        y_train, y_val_target = yv_train, yv_val
        cw = "balanced"

    models_to_run = ["lr", "rf", "lgbm"] if model_name == "all" else [model_name]
    results = {}

    for m in models_to_run:
        print(f"\nTraining {m.upper()} for task={task} ...")
        if m == "lr":
            model = build_lr(cw)
            model.fit(X_train, y_train)
        elif m == "rf":
            model = build_rf(cw)
            model.fit(X_train, y_train)
            model = model.best_estimator_
        elif m == "lgbm":
            search = build_lgbm()
            sw = sample_weights(y_train, class_weights(y_train) if isinstance(cw, dict) else
                                {0: 1.0, 1: y_train.value_counts()[0] / y_train.value_counts()[1]})
            search.fit(X_train, y_train, sample_weight=sw)
            model = search.best_estimator_

        f1 = evaluate(m.upper(), model, X_val, y_val_target)
        results[m] = (f1, model)
        save(model, task, m)

    best_name = max(results, key=lambda k: results[k][0])
    best_f1, best_model = results[best_name]
    print(f"\nBest model: {best_name.upper()}  (val macro-F1: {best_f1:.4f})")
    save(best_model, task, "best")

    results_path = os.path.join(MODELS_DIR, f"{task}_results.json")
    with open(results_path, "w") as f:
        json.dump({k: v[0] for k, v in results.items()}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  choices=["category", "vpn"], default="category")
    parser.add_argument("--model", choices=["lr", "rf", "lgbm", "all"], default="lgbm")
    args = parser.parse_args()
    train(args.task, args.model)
