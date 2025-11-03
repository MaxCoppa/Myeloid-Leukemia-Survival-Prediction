# %%
import pandas as pd
import matplotlib.pyplot as plt
from tree_based_models import model_selection_using_kfold_surv, get_model
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis, IPCRidge, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

# %%
# Clinical Data
clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
clinical_test = pd.read_csv("data/X_test/clinical_test.csv")

# Molecular Data
molecular_train = pd.read_csv("data/X_train/molecular_train.csv")
molecular_test = pd.read_csv("data/X_test/molecular_test.csv")

target_train = pd.read_csv("data/X_train/target_train.csv")

# Preview the data
clinical_train.head()
# %%
train = pd.concat(
    [
        clinical_train.set_index("ID").select_dtypes(include="number"),
        target_train.set_index("ID"),
    ],
    axis=1,
)
train = train[~train["OS_YEARS"].isna()]
train.head()
# %%
molecular_train["LENGTH"] = molecular_train["END"] - molecular_train["START"]
# %%
tmp = molecular_train.groupby("ID").size().to_frame("Nmut")
length = molecular_train.groupby("ID")["LENGTH"].sum()
vaf = molecular_train.groupby("ID")["VAF"].sum()

train = train.merge(tmp, left_index=True, right_index=True, how="inner")
train = train.merge(vaf, left_index=True, right_index=True, how="inner")
train = train.merge(length, left_index=True, right_index=True, how="inner")

# %%
target = "OS_YEARS"
status = "OS_STATUS"
features = ["BM_BLAST", "WBC", "HB", "PLT", "Nmut", "VAF", "LENGTH"]

# %%
model_cls = GradientBoostingSurvivalAnalysis
model_params = {
    "n_estimators": 20,
    "max_depth": 5,
    "learning_rate": 0.2,
    "random_state": 42,
}
# %%
model_selection_using_kfold_surv(
    data=train.reset_index(),
    target=target,
    status=status,
    model_cls=model_cls,
    params=model_params,
    features=features,
    feat_engineering=None,
    unique_id="ID",
    plot_ft_importance=True,
    n_splits=6,
    log=False,
)

# %%
