# %%
import pandas as pd
import matplotlib.pyplot as plt
from deepsurv import model_selection_using_kfold_deepsurv
from feature_engineering import (
    one_hot_aggregate,
    add_cytogenetic_features,
    create_molecular_feat,
)

# %%
# === Load Data ===
clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
clinical_test = pd.read_csv("data/X_test/clinical_test.csv")
molecular_train = pd.read_csv("data/X_train/molecular_train.csv")
molecular_test = pd.read_csv("data/X_test/molecular_test.csv")
target_train = pd.read_csv("data/X_train/target_train.csv")

# Merge training data
train = pd.concat(
    [
        clinical_train.set_index("ID"),
        target_train.set_index("ID"),
    ],
    axis=1,
)
train = train[~train["OS_YEARS"].isna()]  # remove missing targets
train.head()


# %%
def feat_engineering(
    data: pd.DataFrame,
    molecular_data: pd.DataFrame,
    fill_not_molecular=False,
) -> tuple[pd.DataFrame, list]:
    """Apply domain-specific feature engineering for DeepSurv."""

    ids_not_molecular = [
        id for id in data.index.unique() if id not in molecular_data["ID"].unique()
    ]

    not_molecular = data[data.index.isin(ids_not_molecular)]

    data, molecular_feat = create_molecular_feat(
        data=data, molecular_data=molecular_data
    )

    data, col_clinical = add_cytogenetic_features(data)
    data, categories = one_hot_aggregate(molecular_data, data, "EFFECT")
    data, chromosomes = one_hot_aggregate(
        molecular_data, data, "CHR", fillna_value="no_chr"
    )
    data, genes = one_hot_aggregate(
        molecular_data, data, "GENE", fillna_value="no_gene"
    )

    new_feats = list(col_clinical) + list(categories) + list(chromosomes) + list(genes)

    not_molecular, col_clinical = add_cytogenetic_features(not_molecular)

    if fill_not_molecular:
        data = pd.concat([data, not_molecular])
        data[
            list(categories) + list(chromosomes) + list(genes) + list(molecular_feat)
        ] = data[
            list(categories) + list(chromosomes) + list(genes) + list(molecular_feat)
        ].fillna(
            0
        )

    return data, new_feats


# %%
# === Apply Feature Engineering ===
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

train, feat_train = feat_engineering(
    data=train, molecular_data=molecular_train, fill_not_molecular=True
)
test, feat_test = feat_engineering(
    data=test, molecular_data=molecular_test, fill_not_molecular=True
)

# Align train/test feature sets
feats = [ft for ft in feat_test if ft in feat_train]

# %%
# === Define Feature Columns ===
target = "OS_YEARS"
status = "OS_STATUS"
features = ["BM_BLAST", "WBC", "HB", "PLT", "Nmut", "VAF", "LENGTH"] + feats

# %%
# === Model Selection (DeepSurv) ===
results = model_selection_using_kfold_deepsurv(
    data=train.reset_index(),
    features=features,
    target=target,
    status=status,
    hidden_layers_sizes=[16, 8],
    dropout=0.4,
    n_splits=6,
    n_epochs=50,
    lr=1e-4,
    l2_reg=1e-5,
    unique_id="ID",
    log=False,
)

# %%
