# %%
import pandas as pd
import matplotlib.pyplot as plt
from tree_based_models import model_selection_using_kfold_surv, get_model
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis, IPCRidge, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from feature_engineering import (
    one_hot_aggregate,
    add_cytogenetic_features,
    create_molecular_feat,
)

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
        clinical_train.set_index("ID"),
        target_train.set_index("ID"),
    ],
    axis=1,
)
train = train[~train["OS_YEARS"].isna()]
train.head()


# %%
def feat_engineering(
    data: pd.DataFrame,
    molecular_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:

    ids_not_molecular = [
        id for id in data.index.unique() if id not in molecular_data["ID"].unique()
    ]

    not_molecular = data[data.index.isin(ids_not_molecular)]

    data = create_molecular_feat(data=data, molecular_data=molecular_data)

    data, col_clinical = add_cytogenetic_features(data)
    data, categories = one_hot_aggregate(molecular_data, data, "EFFECT")
    data, chromosomes = one_hot_aggregate(
        molecular_data, data, "CHR", fillna_value="no_chr"
    )
    data, genes = one_hot_aggregate(
        molecular_data, data, "GENE", fillna_value="no_gene"
    )

    new_feats = list(categories) + list(chromosomes) + list(genes) + list(col_clinical)

    not_molecular, col_clinical = add_cytogenetic_features(not_molecular)
    data = pd.concat([data, not_molecular])
    data[list(categories) + list(chromosomes) + list(genes)] = data[
        list(categories) + list(chromosomes) + list(genes)
    ].fillna(0)
    return data, new_feats


# %%
train, feats = feat_engineering(data=train, molecular_data=molecular_train)
train

# %%
target = "OS_YEARS"
status = "OS_STATUS"
features = ["BM_BLAST", "WBC", "HB", "PLT", "Nmut", "VAF"] + feats

# %%

best_params = {
    "n_estimators": 150,
    "max_depth": 15,
    "min_weight_fraction_leaf": 0.005,
    "random_state": 42,
}
model_cls = RandomSurvivalForest

model_params = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_weight_fraction_leaf": 0.005,
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
