"""Independently check saved scores, split boundaries and data provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def verify():
    data = pd.read_csv(ROOT / "data/heart.csv")
    features = list(data.columns[:-1])
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    assert hashlib.sha256((ROOT / "data/heart.csv").read_bytes()).hexdigest() == audit["sha256"]
    scores = pd.read_csv(ROOT / "results/all_metrics.csv")
    predictions = pd.read_csv(ROOT / "results/predictions.csv")
    splits = pd.read_csv(ROOT / "results/splits.csv")
    assert len(scores) == 97
    assert predictions.probability.between(0, 1).all()
    for row in scores.itertuples():
        part = predictions[(predictions.protocol == row.protocol) &
                           (predictions.seed == row.seed) & (predictions.model == row.model)]
        actual, predicted = part.target.to_numpy(), part.prediction.to_numpy()
        assert np.array_equal(predicted, (part.probability >= row.threshold).astype(int))
        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        assert (tp, tn, fp, fn) == (row.tp, row.tn, row.fp, row.fn)
        measured = [(tp+tn)/len(part), tp/(tp+fp) if tp+fp else 0,
                    tp/(tp+fn), 2*tp/(2*tp+fp+fn), roc_auc_score(actual, part.probability)]
        expected = [row.accuracy, row.precision, row.recall, row.f1, row.auc]
        np.testing.assert_allclose(measured, expected, atol=1e-12)
        s = splits[(splits.protocol == row.protocol) & (splits.seed == row.seed)]
        train_ids = set(s[s.part == "train"].record_id)
        test_ids = set(s[s.part == "test"].record_id)
        assert not train_ids & test_ids
        assert set(part.record_id) == test_ids
        assert np.array_equal(data.loc[part.record_id, "target"].to_numpy(), actual)
        if row.protocol == "unique_rows":
            train_records = set(map(tuple, data.loc[sorted(train_ids), features].to_numpy()))
            test_records = set(map(tuple, data.loc[sorted(test_ids), features].to_numpy()))
            assert not train_records & test_records
    # Independently recompute the summary numbers used in the report.
    summary = pd.read_csv(ROOT / "results/part2_summary.csv").set_index("model")
    for name, group in scores[scores.protocol == "unique_rows"].groupby("model"):
        assert len(group) == 10
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            np.testing.assert_allclose([group[metric].mean(), group[metric].std()],
                [summary.loc[name, metric + "_mean"], summary.loc[name, metric + "_std"]], atol=1e-12)
    message = f"PASS: {len(scores)} metric rows and {len(predictions)} predictions verified; clean splits have no exact-record overlap."
    print(message)
    return message


if __name__ == "__main__":
    verify()
