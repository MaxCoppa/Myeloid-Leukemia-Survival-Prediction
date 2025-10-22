import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ResidualModel(BaseEstimator):
    """
    General + Residual model wrapper.
    """

    def __init__(
        self, general_model_cls, general_params, residual_model_cls, residual_params
    ):
        self.general_model_cls = general_model_cls
        self.general_params = general_params
        self.residual_model_cls = residual_model_cls
        self.residual_params = residual_params

        self.general_model = None
        self.residual_models = {}

    def fit(
        self, data: pd.DataFrame, target: str, features: list, features_res: list
    ) -> None:
        """
        Fit general model and residual models per ALLOCATION.
        """
        X = data[features]
        y = data[target]

        # Train general model
        self.general_model = self.general_model_cls(**self.general_params)
        self.general_model.fit(X.drop(columns=["ALLOCATION"]), y)

        # Compute residuals
        data = data.copy()
        data["residuals"] = y - self.general_model.predict(
            X.drop(columns=["ALLOCATION"])
        )

        # Train residual models for each ALLOCATION group
        self.residual_models = {}
        for alloc, df_group in data.groupby("ALLOCATION"):
            model = self.residual_model_cls(**self.residual_params)
            model.fit(
                df_group[features_res].drop(columns=["ALLOCATION"]),
                df_group["residuals"],
            )
            self.residual_models[alloc] = model

    def predict(
        self, data: pd.DataFrame, features: list, features_res: list
    ) -> np.ndarray:
        """
        Predict using general + residual models.
        """
        X = data[features]

        # General predictions
        base_pred = self.general_model.predict(X.drop(columns=["ALLOCATION"]))
        corrections = np.zeros(len(X))

        # Add residual corrections
        for alloc, model in self.residual_models.items():
            mask = X["ALLOCATION"] == alloc
            if mask.any():
                X_group = X.loc[mask, features_res].drop(columns=["ALLOCATION"])
                corrections[mask] = model.predict(X_group)

        return base_pred + corrections
