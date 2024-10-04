# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:03:40 2024

@author: dphilippus

This file generates trained model covariates.
"""

from NEWT import Watershed
from NEWT.coef_est import training_req_cols, preprocess

def build_training_data(data):
    """
    Prepare a training dataset by fitting watershed models.
    """
    if not all([col in data.columns for col in training_req_cols]):
        raise ValueError(f"Missing columns in input data; required: {training_req_cols}")
    coefs = data.groupby("id").apply(lambda x: 
        Watershed.from_data(x).coefs_to_df().drop(columns=["R2", "RMSE"]) if
        len(x[["day", "temperature"]].dropna()["day"].unique()) >= 181 else None,
        include_groups=False)
    coefs.index = coefs.index.get_level_values("id")
    covar = preprocess(data)
    return coefs.merge(covar, on="id")
