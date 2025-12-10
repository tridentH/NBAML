from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def train_logreg() -> str:
    print("=== NBAML: training logistic regression champion model ===")

    root = Path(__file__).resolve().parents[2]
    train_path = root / ".data" / "features" / "training_table.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training table not found: {train_path}")

    df = pd.read_csv(train_path)

    # Features and target
    feature_cols = [
        "mean_pt_diff_roll10",
        "mean_pts_roll10",
        "mean_opp_pts_roll10",
        "strength_score",
    ]

    X = df[feature_cols].values
    y = df["is_champion"].values

    # Simple train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df[["season", "team_abbreviation"]],
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Logistic regression with class_weight to handle rare champions
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nAccuracy:   {acc:.3f}")
    print(f"Log loss:   {ll:.3f}")
    print(f"ROC AUC:    {auc:.3f}")

    # Show top predicted champs in the test split
    meta_test = meta_test.copy()
    meta_test["true"] = y_test
    meta_test["pred_prob"] = y_prob
    top = meta_test.sort_values("pred_prob", ascending=False).head(10)

    print("\nTop 10 team-seasons by predicted champion probability (test split):")
    print(top)

    # Save model
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "logreg_champion_model.joblib"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
        },
        model_path,
    )

    print(f"\nSaved model to: {model_path}")
    return str(model_path)


if __name__ == "__main__":
    train_logreg()
