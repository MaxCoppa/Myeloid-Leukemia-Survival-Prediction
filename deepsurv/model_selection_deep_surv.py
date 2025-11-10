"""
DeepSurv K-Fold Model Selection (PyTorch)
-----------------------------------------
Perform k-fold cross-validation using DeepSurv neural Cox model.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from datetime import datetime

# Assuming DeepSurv + training utils are in deepsurv_pytorch.py
from .deep_surv import DeepSurv
from .train_deepsurv import train_deepsurv


def model_selection_using_kfold_deepsurv(
    data: pd.DataFrame,
    features: list[str],
    target: str = "OS_YEARS",
    status: str = "OS_STATUS",
    hidden_layers_sizes: list[int] = [64, 32],
    dropout: float = 0.3,
    n_splits: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
    l1_reg: float = 0.0,
    feat_engineering=None,
    unique_id: str = "ROW_ID",
    device: str = "cpu",
    log: bool = False,
    log_note: str = None,
):
    """
    Perform K-Fold cross-validation with DeepSurv (PyTorch).

    Args:
        data: DataFrame with features + survival targets.
        features: list of column names for X.
        target: survival time column.
        status: event indicator column (1 = event, 0 = censored).
        hidden_layers_sizes: list of hidden layer sizes.
        dropout: dropout probability.
        n_splits: number of CV folds.
        n_epochs: number of training epochs.
        lr: learning rate.
        l2_reg: L2 weight decay.
        l1_reg: L1 regularization strength.
        feat_engineering: optional callable to apply feature engineering.
        unique_id: column identifying unique subjects.
        device: 'cpu' or 'cuda'.
        log: whether to append results to log file.
        log_note: extra notes for logging.
    """

    unique_vals = data[unique_id].unique()
    metrics = {"cindex_train": [], "cindex_test": []}
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(kf.split(unique_vals)):
        print(f"\n===== Fold {i+1}/{n_splits} =====")

        train_ids = unique_vals[train_idx]
        test_ids = unique_vals[test_idx]

        train_mask = data[unique_id].isin(train_ids)
        test_mask = data[unique_id].isin(test_ids)

        df_train = data.loc[train_mask].copy()
        df_test = data.loc[test_mask].copy()

        # Optional feature engineering
        if feat_engineering:
            df_train = feat_engineering(df_train)
            df_test = feat_engineering(df_test)

        # Separate features and targets
        X_train = df_train[features]
        X_test = df_test[features]
        t_train = df_train[target].astype(np.float32)
        e_train = df_train[status].astype(np.float32)
        t_test = df_test[target].astype(np.float32)
        e_test = df_test[status].astype(np.float32)

        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize DeepSurv model
        n_in = X_train.shape[1]
        model = DeepSurv(
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            dropout=dropout,
            activation="relu",
        )

        # Train model
        hist = train_deepsurv(
            model,
            x_train=X_train,
            e_train=e_train.values,
            t_train=t_train.values,
            x_valid=X_test,
            e_valid=e_test.values,
            t_valid=t_test.values,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=l2_reg,
            l1_reg=l1_reg,
            device=device,
            verbose=False,
        )

        models.append(model)

        # Compute final metrics
        with torch.no_grad():
            risk_train = -(model.predict_risk(X_train))
            risk_test = -(model.predict_risk(X_test))
            cindex_train = concordance_index(t_train, risk_train, e_train)
            cindex_test = concordance_index(t_test, risk_test, e_test)

        metrics["cindex_train"].append(cindex_train)
        metrics["cindex_test"].append(cindex_test)

        print(
            f"Fold {i+1}: C-index (Train: {cindex_train:.4f} | Test: {cindex_test:.4f})"
        )

    # --- Aggregate results ---
    cindexes = np.array(metrics["cindex_test"])
    mean_c, std_c, min_c, max_c = (
        cindexes.mean(),
        cindexes.std(),
        cindexes.min(),
        cindexes.max(),
    )

    print(
        f"\nConcordance Index (Test): {mean_c:.4f} ± {std_c:.4f} "
        f"[Min: {min_c:.4f} ; Max: {max_c:.4f}]"
    )

    # --- Optional logging ---
    if log:
        logfile = "predictions/model_selection_deepsurv.log"
        note_str = f" | Note: {log_note}" if log_note else ""
        with open(logfile, "a") as f:
            f.write(
                f"{datetime.now()} - DeepSurv: Mean C-index: {mean_c:.4f} | Std: {std_c:.4f} | "
                f"Min: {min_c:.4f} | Max: {max_c:.4f}{note_str}\n"
            )

    return {
        "models": models,
        "metrics": metrics,
        "mean_cindex": mean_c,
        "std_cindex": std_c,
    }
