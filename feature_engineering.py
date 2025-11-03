# %%

import pandas as pd
import matplotlib.pyplot as plt
from tree_based_models import model_selection_using_kfold, get_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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
molecular_train.nunique()

# %%
molecular_train.head()
# %%
molecular_train["LENGTH"] = molecular_train["END"] - molecular_train["START"]
# %%
tmp = molecular_train.groupby("ID").size().to_frame("Nmut")
length = molecular_train.groupby("ID")["LENGTH"].sum()
vaf = molecular_train.groupby("ID")["VAF"].sum()

train = train.merge(tmp, left_index=True, right_index=True, how="inner")
train = train.merge(vaf, left_index=True, right_index=True, how="inner")
train = train.merge(length, left_index=True, right_index=True, how="inner")

# %% One Hot Encoding Effect mutation

oht = OneHotEncoder(sparse_output=False)
encoding = oht.fit_transform(molecular_train[["EFFECT"]])
categories = oht.categories_

for i, cat in enumerate(categories[0]):
    molecular_train[cat] = encoding[:, i]

for cat in categories:
    new_col = molecular_train.groupby("ID")[cat].sum()
    train = train.merge(new_col, left_index=True, right_index=True, how="inner")

# %% One Hot Encoding Chromosom

oht = OneHotEncoder(sparse_output=False)
encoding = oht.fit_transform(molecular_train[["CHR"]].fillna("no_chr"))
chromosomes = oht.categories_

for i, cat in enumerate(chromosomes[0]):
    molecular_train[cat] = encoding[:, i]

for cat in chromosomes:
    new_col = molecular_train.groupby("ID")[cat].sum()
    train = train.merge(new_col, left_index=True, right_index=True, how="inner")

# %% One Hot Encoding Genes

oht = OneHotEncoder(sparse_output=False)
encoding = oht.fit_transform(molecular_train[["GENE"]].fillna("no_gene"))
genes = oht.categories_

for i, cat in enumerate(genes[0]):
    molecular_train[cat] = encoding[:, i]

for cat in genes:
    new_col = molecular_train.groupby("ID")[cat].sum()
    train = train.merge(new_col, left_index=True, right_index=True, how="inner")


# %%
target = "OS_YEARS"
features = (
    ["BM_BLAST", "WBC", "HB", "PLT", "Nmut", "VAF", "LENGTH"]
    + list(categories[0])
    + list(chromosomes[0])
    + list(genes[0])
)
model_type = "cat"


# features = ['PLT', 'VAF', 'HB', 'BM_BLAST', 'WBC', 'Nmut', 'TP53', 'LENGTH', 'non_synonymous_codon', 'SF3B1', '21', '17', 'stop_gained', '1', '20', 'NFE2', 'ASXL1', 'frameshift_variant', 'RUNX1', '5', '4', '2', '12', 'MLL', 'splice_site_variant']
# features = ['PLT', 'HB', 'BM_BLAST', 'VAF', 'TP53', 'WBC', 'non_synonymous_codon', 'Nmut', 'SF3B1', '21', '17', 'NFE2', 'LENGTH', 'stop_gained', 'MYC', '20', 'frameshift_variant', 'ASXL1', '12', '2']

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

# %%

# %%
