# Myeloid Leukemia Survival Prediction - ENS Data Challenge  
## By QRT x Gustave Roussy

## Overview
This repository contains our team-of-two’s solution for the QRT x Gustave Roussy Data Challenge.  
The goal is to predict a **risk score** for overall survival (OS) in patients diagnosed with an adult subtype of myeloid leukemia.  
Models are evaluated using the **IPCW-C-index**, which measures how well risk predictions rank survival times under right-censoring.

This is a **work in progress**, so the repository is still evolving and not fully structured.

## Data
The dataset includes **3,323 training patients** and **1,193 test patients**, split into:
- **Clinical data** (one row per patient)  
- **Molecular data** (one row per somatic mutation)  

Training labels contain OS time and censoring status.

## Approach
We explored the data through:
- `data_vizualisation.ipynb`  
- `explo_surv_data.ipynb`

Feature engineering is implemented in `feature_engineering.py` and includes molecular aggregation, VAF statistics, cytogenetic encoding, and preprocessing of clinical variables.

Models tested in `myeloid_survival_prediction.ipynb`:
- CatBoost (non-censored)
- Random Survival Forest
- DeepSurv
- Ridge-penalized Cox model

Our best models (DeepSurv / RSF) currently reach **~0.72 IPCW-C-index**, outperforming the official benchmark (~0.68).  
Our work is still in progress: we need a deeper understanding of the feature space to engineer more meaningful representations, and we plan to experiment with additional modeling approaches soon.
