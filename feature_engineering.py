# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree_based_models import model_selection_using_kfold, get_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing import Tuple
import re

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
    data: pd.DataFrame, molecular_data: pd.DataFrame
) -> tuple[pd.DataFrame, list]:

    data, _ = create_molecular_feat(data=data, molecular_data=molecular_data)
    data, col_clinical = add_cytogenetic_features(data)
    data, categories = one_hot_aggregate(molecular_data, data, "EFFECT")
    data, chromosomes = one_hot_aggregate(
        molecular_data, data, "CHR", fillna_value="no_chr"
    )
    data, genes = one_hot_aggregate(
        molecular_data, data, "GENE", fillna_value="no_gene"
    )

    new_feats = list(categories) + list(chromosomes) + list(genes) + list(col_clinical)

    return data, new_feats


# %%
train, feat_train = feat_engineering(data=train, molecular_data=molecular_train)
# %% Preprocessing Test Data
test = clinical_test.set_index("ID").copy()
not_molecular_test = clinical_test[
    clinical_test["ID"].isin(
        [
            id
            for id in clinical_test["ID"].unique()
            if id not in molecular_test["ID"].unique()
        ]
    )
].set_index("ID")

test, feat_test = feat_engineering(data=test, molecular_data=molecular_test)

feats = [ft for ft in feat_test if ft in feat_train]
# %%

not_molecular_test, col_clinical = add_cytogenetic_features(not_molecular_test)
test = pd.concat([test, not_molecular_test])
# %%
target = "OS_YEARS"
features = ["BM_BLAST", "WBC", "HB", "PLT", "Nmut", "VAF", "LENGTH"] + feats
model_type = "cat"

# %%
model_selection_using_kfold(
    data=train.reset_index(),
    target=target,
    features=features,
    model_type=model_type,
    feat_engineering=None,
    unique_id="ID",
    plot_ft_importance=True,
    n_splits=6,
    log=False,
    scale=True,
    n_importance=20,
)

# %% Fit model on the whole DataSet

model = get_model(model_type=model_type)
model.fit(train[features], train[target])

# %%
preds = -model.predict(test[features])

submission = pd.Series(preds, index=test.index, name="risk_score")
# %%
submission.to_csv("data/cat_feat_engineering_cypt.csv")

# %%
len(
    set(pd.read_csv("data/benchmark_submission.csv")["ID"]) & set(submission.index)
) / len(set(submission.index))
# %%
