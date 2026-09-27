"""SIT720 mini research: six basic models, stacking, and a simpler proposal.

Run: python run_experiments.py
The dataset is included, so experiments need no internet connection.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import version

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hd-matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score,
                             confusion_matrix, matthews_corrcoef, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEEDS = list(range(42, 52))
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]

# 1. Set up the six models and the stack from the paper.


def make_models(seed):
    """All unspecified settings are ordinary, fixed choices, not tuned results."""
    classifiers = {
        "LR": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
        "DT": DecisionTreeClassifier(criterion="entropy", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini",
                                     max_features="sqrt", random_state=seed, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                             subsample=1.0, colsample_bytree=1.0,
                             reg_lambda=1.0, objective="binary:logistic",
                             eval_metric="logloss", tree_method="hist",
                             random_state=seed, n_jobs=1),
        "NB": GaussianNB(var_smoothing=1e-9),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2),
    }
    # Each learner gets its own scaler. Stacking clones these whole pipelines.
    models = {name: make_pipeline(StandardScaler(), model)
              for name, model in classifiers.items()}
    models["Stacking"] = StackingClassifier(
        estimators=list(models.items()),
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        stack_method="predict_proba", passthrough=False, n_jobs=1)
    return models


def make_proposed_model():
    """One understandable model with separate numeric and category handling."""
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])
    return make_pipeline(preprocessing, LogisticRegression(C=1.0, max_iter=2000))


def choose_threshold(model, X_train, y_train, seed):
    """Only training-fold predictions choose the threshold, never test labels."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Each row is predicted by a model trained on the other four folds.
    probabilities = cross_val_predict(model, X_train, y_train, cv=folds,
                                      method="predict_proba", n_jobs=1)[:, 1]
    rows = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.05), 2):
        prediction = (probabilities >= threshold).astype(int)
        rows.append({"threshold": float(threshold),
                     "f2": fbeta_score(y_train, prediction, beta=2),
                     "precision": precision_score(y_train, prediction, zero_division=0),
                     "recall": recall_score(y_train, prediction),
                     "distance_from_half": abs(float(threshold) - 0.5)})
    search = pd.DataFrame(rows)
    best = search.sort_values(["f2", "distance_from_half", "threshold"],
                              ascending=[False, True, True]).iloc[0]
    return float(best.threshold), search


def get_scores(y, probabilities, threshold=0.5):
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "auc": roc_auc_score(y, probabilities),
            "specificity": tn / (tn + fp),
            "mcc": matthews_corrcoef(y, prediction),
            "f2": fbeta_score(y, prediction, beta=2),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def record_result(rows, predictions, protocol, seed, name, y, prob, threshold):
    rows.append({"protocol": protocol, "seed": seed, "model": name,
                 "threshold": threshold, **get_scores(y, prob, threshold)})
    predictions.append(pd.DataFrame({"protocol": protocol, "seed": seed,
        "model": name, "record_id": y.index, "target": y.to_numpy(),
        "probability": prob, "threshold": threshold,
        "prediction": (prob >= threshold).astype(int)}))


def read_data():
    data = pd.read_csv(ROOT / "data" / "heart.csv")
    assert list(data.columns) == FEATURES + ["target"]
    assert data.shape == (1025, 14), "Use the included, unchanged dataset."
    assert data.isna().sum().sum() == 0, "Reconsider imputation if data changes."
    assert set(data.target.unique()) == {0, 1}
    # Never silently choose a label if identical predictors disagree.
    assert data.groupby(FEATURES).target.nunique().max() == 1
    return data


def save_published_results():
    # Direct transcription of the five required metrics in paper Table 11.
    # These are reference values, never substituted for experimental results.
    paper = pd.DataFrame([
        ["LR", .8439, .8196, .9090, .8620, .9204],
        ["DT", .9268, .9702, .8909, .9289, .9702],
        ["RF", .9268, .9130, .9545, .9333, .9734],
        ["XGB", .9073, .9174, .9090, .9132, .9830],
        ["NB", .8439, .8250, .9000, .8608, .9185],
        ["KNN", .8585, .8648, .8727, .8687, .9304],
        ["Stacking", .9853, 1.0000, .9727, .9861, .9880],
    ], columns=["model"] + METRICS)
    paper.to_csv(RESULTS / "published_table11.csv", index=False)
    return paper


def create_figures(scores, predictions, paper):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    raw = scores[scores.protocol == "paper_rows"].set_index("model")
    names = list(paper.model)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x - .18, paper.accuracy * 100, .36, label="Published Table 11")
    ax.bar(x + .18, raw.loc[names, "accuracy"] * 100, .36, label="Reproduced row split")
    ax.set(xticks=x, xticklabels=names, ylabel="Accuracy (%)", ylim=(0, 105))
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "paper_comparison.png", dpi=180); plt.close(fig)

    clean = scores[scores.protocol == "unique_rows"]
    selected = ["Stacking", "LR", "OneHot LR", "Proposed"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, metric in enumerate(["accuracy", "recall", "auc"]):
        values = clean.groupby("model")[metric].agg(["mean", "std"]).loc[selected]
        ax.bar(np.arange(4) + (j - 1) * .24, values["mean"], .24,
               yerr=values["std"], capsize=3, label=metric.upper())
    ax.set(xticks=range(4), xticklabels=selected, ylabel="Score (mean +/- SD)", ylim=(0, 1.12))
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "clean_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, protocol, title in zip(axes, ["paper_rows", "unique_rows"],
                                  ["Row split (seed 42)", "Unique records (seed 42)"]):
        subset = predictions[(predictions.protocol == protocol) & (predictions.seed == 42)]
        for name, group in subset.groupby("model", sort=False):
            if protocol == "unique_rows" and name not in selected:
                continue
            fpr, tpr, _ = roc_curve(group.target, group.probability)
            ax.plot(fpr, tpr, label=f"{name}: {roc_auc_score(group.target, group.probability):.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "roc_curves.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, name in zip(axes, ["Stacking", "OneHot LR", "Proposed"]):
        s = predictions[(predictions.protocol == "unique_rows") &
                        (predictions.seed == 42) & (predictions.model == name)]
        matrix = confusion_matrix(s.target, s.prediction, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=40)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > 20 else "black")
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1],
               xlabel="Predicted class", ylabel="Actual class")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrices.png", dpi=180); plt.close(fig)


def main():
    # 2. Read the original data and make a separate copy without duplicates.
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = read_data()
    clean = data.drop_duplicates(subset=FEATURES).copy()
    rows, predictions, splits, overlap, thresholds = [], [], [], [], []
    audit = {"rows": len(data), "unique_rows": len(clean),
             "extra_duplicate_rows": int(data.duplicated().sum()),
             "missing_cells": int(data.isna().sum().sum()),
             "class_counts_raw": data.target.value_counts().sort_index().to_dict(),
             "class_counts_unique": clean.target.value_counts().sort_index().to_dict(),
             "ca_4_rows": int((data.ca == 4).sum()),
             "thal_0_rows": int((data.thal == 0).sum()),
             "sha256": hashlib.sha256((ROOT / "data/heart.csv").read_bytes()).hexdigest()}

    # 3. Run the paper-style split, then repeat the clean comparison ten times.
    for protocol, frame, seeds in [("paper_rows", data, [42]), ("unique_rows", clean, SEEDS)]:
        X, y = frame[FEATURES], frame.target
        for seed in seeds:
            print(f"Running {protocol}, seed {seed}", flush=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed,
                stratify=y if protocol == "unique_rows" else None)
            train_keys = set(map(tuple, X_train.to_numpy()))
            seen = np.array([tuple(row) in train_keys for row in X_test.to_numpy()])
            overlap.append({"protocol": protocol, "seed": seed,
                            "train_rows": len(X_train), "test_rows": len(X_test),
                            "test_rows_seen_in_training": int(seen.sum()),
                            "test_positive": int(y_test.sum()),
                            "test_negative": int((y_test == 0).sum())})
            if protocol == "unique_rows":
                assert not seen.any(), "Duplicate leakage detected!"
            for part, ids in [("train", X_train.index), ("test", X_test.index)]:
                splits.append(pd.DataFrame({"protocol": protocol, "seed": seed,
                                            "part": part, "record_id": ids}))
            for name, model in make_models(seed).items():
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                record_result(rows, predictions, protocol, seed, name, y_test, prob, .5)
            if protocol == "unique_rows":
                model = make_proposed_model()
                threshold, search = choose_threshold(model, X_train, y_train, seed)
                thresholds.append(search.assign(seed=seed, selected=search.threshold == threshold))
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                for name, cutoff in [("OneHot LR", .5), ("Proposed", threshold)]:
                    record_result(rows, predictions, protocol, seed, name, y_test, prob, cutoff)

    # 4. Save the outputs so the report can be checked without retraining.
    scores = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True)
    scores.to_csv(RESULTS / "all_metrics.csv", index=False)
    pred.to_csv(RESULTS / "predictions.csv", index=False)
    pd.concat(splits).to_csv(RESULTS / "splits.csv", index=False)
    pd.DataFrame(overlap).to_csv(RESULTS / "split_audit.csv", index=False)
    pd.concat(thresholds).to_csv(RESULTS / "threshold_search.csv", index=False)
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, indent=2))
    paper = save_published_results()
    reproduced = scores[scores.protocol == "paper_rows"]
    reproduced.to_csv(RESULTS / "part1_metrics.csv", index=False)
    comparison = paper.merge(reproduced[["model"] + METRICS], on="model",
                             suffixes=("_paper", "_reproduced"))
    for metric in METRICS:
        comparison[metric + "_difference"] = comparison[metric + "_reproduced"] - comparison[metric + "_paper"]
    comparison.to_csv(RESULTS / "paper_comparison.csv", index=False)
    clean_scores = scores[scores.protocol == "unique_rows"]
    summary = clean_scores.groupby("model")[METRICS + ["specificity", "mcc", "f2"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(RESULTS / "part2_summary.csv")
    proposed = clean_scores[clean_scores.model == "Proposed"].set_index("seed")
    stacked = clean_scores[clean_scores.model == "Stacking"].set_index("seed")
    paired = []
    for metric in METRICS:
        delta = proposed[metric] - stacked[metric]
        paired.append({"metric": metric, "mean_difference": delta.mean(),
                       "sd_difference": delta.std(), "min_difference": delta.min(),
                       "max_difference": delta.max(), "wins": int((delta > 1e-12).sum()),
                       "ties": int((delta.abs() <= 1e-12).sum()), "losses": int((delta < -1e-12).sum())})
    pd.DataFrame(paired).to_csv(RESULTS / "paired_comparison.csv", index=False)
    import xgboost
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {**{p: version(p) for p in packages},
                                "xgboost": xgboost.__version__}, "seeds": SEEDS}
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2))
    create_figures(scores, pred, paper)
    print("\nPart 1 (scores from actual execution):")
    print(reproduced[["model"] + METRICS].round(4).to_string(index=False))
    print("\nPart 2 mean scores over ten duplicate-free splits:")
    print(summary[[m + "_mean" for m in METRICS]].round(4).to_string())
    print(f"\nSaved results in {RESULTS}")
    return scores


if __name__ == "__main__":
    main()
