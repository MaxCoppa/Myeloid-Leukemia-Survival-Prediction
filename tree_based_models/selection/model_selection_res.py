import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from ..models import ResidualModel


def kfold_general_with_residuals(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    features_res: list[str],
    unique_id: str = "ROW_ID",
    feat_engineering=None,
    n_splits: int = 5,
    general_model_cls=None,
    general_params=None,
    residual_model_cls=None,
    residual_params=None,
    log: bool = False,
    log_note: str = None,
):
    """
    KFold cross-validation for the 'general + residual per ALLOCATION' approach.

    Uses ResidualModel class for fitting and predicting.
    """

    if not general_model_cls or not residual_model_cls:
        raise ValueError(
            "You must specify both general_model_cls and residual_model_cls."
        )

    # Ensure features_res ⊆ features
    missing = [f for f in features_res if f not in features]
    if missing:
        raise ValueError(
            f"Invalid features found (not in main feature list): {missing}"
        )

    # Split on unique IDs (e.g., TS)
    unique_vals = data[unique_id].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {"accuracy": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        train_vals = unique_vals[train_idx]
        test_vals = unique_vals[test_idx]

        train_df = data[data[unique_id].isin(train_vals)].copy()
        test_df = data[data[unique_id].isin(test_vals)].copy()

        # Optional feature engineering
        if feat_engineering:
            train_df = feat_engineering(train_df)
            test_df = feat_engineering(test_df)

        # --- Train model ---
        fold_model = ResidualModel(
            general_model_cls=general_model_cls,
            general_params=general_params,
            residual_model_cls=residual_model_cls,
            residual_params=residual_params,
        )
        fold_model.fit(train_df, target, features, features_res)

        # --- Predict ---
        y_pred = fold_model.predict(test_df, features, features_res)

        # --- Evaluate (sign accuracy) ---
        y_true_bin = (test_df[target] > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        metrics["accuracy"].append(acc)

        print(f"Fold {fold+1} - Acc: {acc*100:.2f}%")

    # Aggregate results
    accs = np.array(metrics["accuracy"])
    mean_acc, std_acc = accs.mean() * 100, accs.std() * 100
    min_acc, max_acc = accs.min() * 100, accs.max() * 100

    print(
        f"\nAccuracy: {mean_acc:.2f}% (± {std_acc:.2f}%) "
        f"[Min: {min_acc:.2f}% ; Max: {max_acc:.2f}%]"
    )

    # Logging results
    if log:
        logfile = "predictions/model_selection.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - {general_model_cls.__name__} + {residual_model_cls.__name__}: "
                f"Mean acc: {mean_acc:.2f}% | Std: {std_acc:.2f}% | "
                f"Min: {min_acc:.2f}% | Max: {max_acc:.2f}%{note_str}\n"
            )

    return metrics
