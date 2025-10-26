import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


def get_param_space(model_name, trial):
    if model_name == "xgb":
        return {
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
    elif model_name == "lgbm":
        return {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        }
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
        }
    else:
        raise ValueError("Unknown model")


def make_datasets(model_name, X_train, y_train, X_val, y_val):
    if model_name == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        return dtrain, dval
    elif model_name == "lgbm":
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        return dtrain, dval
    elif model_name == "rf":
        return (X_train, y_train), (X_val, y_val)


def train_model(model_name, params, dtrain, dval):
    if model_name == "xgb":
        evals = [(dtrain, "train"), (dval, "valid")]
        return xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False,
        )
    elif model_name == "lgbm":
        return lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=200,
            early_stopping_rounds=20,
            verbose_eval=False,
        )
    elif model_name == "rf":
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(dtrain[0], dtrain[1])
        return m


def tune_model(model_name, X_train, y_train, X_val, y_val, n_trials=30):
    dtrain, dval = make_datasets(model_name, X_train, y_train, X_val, y_val)

    def objective(trial):
        params = get_param_space(model_name, trial)
        model = train_model(model_name, params, dtrain, dval)

        if model_name == "xgb":
            preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            y_true = y_val
        elif model_name == "lgbm":
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            y_true = y_val
        else:  # rf
            preds = model.predict(X_val)
            y_true = y_val

        return np.sqrt(mean_squared_error(y_true, preds))  # RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # --- print results ---
    print("\n===== Optuna Results =====")
    print(f"Best RMSE: {study.best_value:.5f}")
    print(f"Best Params: {study.best_params}")
    print("\nTop 5 Trials:")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params"))
    print(trials_df.sort_values("value").head(5))

    return study
