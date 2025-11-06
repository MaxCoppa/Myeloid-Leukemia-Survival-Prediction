import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
import re


def one_hot_aggregate(
    molecular_data: pd.DataFrame,
    data: pd.DataFrame,
    column: str,
    fillna_value: str | None = None,
) -> Tuple[pd.DataFrame, list]:
    """
    One-hot encode a categorical column in molecular_data, aggregate by 'ID', and merge with data.
    """
    molecular_data = molecular_data.copy()
    # Fill missing values if requested
    if fillna_value is not None:
        molecular_data = molecular_data.copy()
        molecular_data[column] = molecular_data[column].fillna(fillna_value)

    # Fit encoder
    oht = OneHotEncoder(sparse_output=False)
    encoded = oht.fit_transform(molecular_data[[column]])
    categories = oht.categories_[0]

    # Combine encoded columns with original molecular_data
    encoded_molecular_data = pd.DataFrame(
        encoded, columns=categories, index=molecular_data.index
    )
    molecular_data_encoded = pd.concat(
        [molecular_data[["ID"]], encoded_molecular_data], axis=1
    )

    # Aggregate all at once
    aggregated = molecular_data_encoded.groupby("ID").sum()
    # Merge once (vectorized, not in a loop)
    return (
        data.merge(aggregated, left_index=True, right_index=True, how="inner"),
        categories,
    )


def parse_iscn(iscn):
    feats = {
        "chromosome_count": np.nan,
        "sex": -1,
        "num_translocations": 0,
        "num_deletions": 0,
        "num_duplications": 0,
        "num_additions": 0,
        "num_monosomies": 0,
        "num_trisomies": 0,
        "abnormal_fraction": 0,
        # "bands_involved": "",
        "is_complex": False,
    }

    sex = {
        "XY": 0,
        "XX": 1,
    }
    if not isinstance(iscn, str):
        return feats

    s = iscn.lower().strip()

    # handle 'complex'
    if "complex" in s:
        feats["is_complex"] = True
        return feats

    # base chromosome count and sex
    base_match = re.match(r"(\d+),([xy]{1,2})", s)
    if base_match:
        feats["chromosome_count"] = int(base_match.group(1))
        feats["sex"] = sex.get(base_match.group(2).upper(), -1)

    # abnormal/normal clone fractions [n]/[m]
    counts = re.findall(r"\[(\d+)\]", s)
    if len(counts) >= 2:
        a, n = map(int, counts[:2])
        feats["abnormal_fraction"] = a / (a + n)

    # abnormality type counters
    feats["num_translocations"] = len(re.findall(r"t\(\d+;\d+\)", s))
    feats["num_deletions"] = len(re.findall(r"del\(\d+\)", s))
    feats["num_duplications"] = len(re.findall(r"dup\(\d+\)", s))
    feats["num_additions"] = len(re.findall(r"add\(\d+\)", s))

    # monosomy/trisomy shorthand (+7, -5, etc.)
    feats["num_monosomies"] = len(re.findall(r"-\d+", s))
    feats["num_trisomies"] = len(re.findall(r"\+\d+", s))

    # # collect bands (q12, p13, q22, etc.)
    # bands = re.findall(r'([pq]\d{1,2}(?:\.\d+)?)', s)
    # if bands:
    #     feats["bands_involved"] = ','.join(sorted(set(bands)))

    return feats


def add_cytogenetic_features(
    data: pd.DataFrame, column="CYTOGENETICS"
) -> tuple[pd.DataFrame, list]:
    """
    Apply ISCN parser and merge new features into the given DataFrame.
    """
    ids = data.index
    parsed = data[column].apply(parse_iscn)
    parsed_data = pd.DataFrame(parsed.tolist(), index=data.index)
    return pd.concat([data, parsed_data], axis=1), parsed_data.columns


def create_molecular_feat(
    data: pd.DataFrame, molecular_data: pd.DataFrame
) -> pd.DataFrame:

    molecular_data["LENGTH"] = molecular_data["END"] - molecular_data["START"]

    tmp = molecular_data.groupby("ID").size().to_frame("Nmut")
    length = molecular_data.groupby("ID")["LENGTH"].sum()
    vaf = molecular_data.groupby("ID")["VAF"].sum()

    data = data.merge(tmp, left_index=True, right_index=True, how="inner")
    data = data.merge(vaf, left_index=True, right_index=True, how="inner")
    data = data.merge(length, left_index=True, right_index=True, how="inner")

    return data
