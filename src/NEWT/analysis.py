# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:03:02 2024

@author: dphilippus

This file contains analysis helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting as pdp
import os
rng = np.random.default_rng()

def perf_summary(data):
    """
    Summarize performance.  data just needs to have temperature and temp.mod.
    """
    return pd.DataFrame({
            "R2": [data["temperature"].corr(data["temp.mod"])**2],
            "RMSE": np.sqrt(np.mean((data["temp.mod"] - data["temperature"])**2)),
            "NSE": 1 - np.mean((data["temp.mod"] - data["temperature"])**2) / np.std(data["temperature"])**2,
            "Pbias": np.mean(data["temp.mod"] - data["temperature"]) / np.mean(data["temperature"])*100,
            "Bias": np.mean(data["temp.mod"] - data["temperature"]),
            "MaxMiss": data.assign(year=lambda x: x["date"].dt.year).groupby("year")[["temperature", "temp.mod"]].max().assign(maxmiss=lambda x: x["temperature"] - x["temp.mod"])["maxmiss"].max()
        })

def kfold(data, modbuilder, parallel=0, by="id", k=10, output=None, redo=False):
    """
    Run a k-fold cross-validation over the given data.  If k=1, run leave-one-
    out instead.  Return and save the results.

    Parameters
    ----------
    data : dataframe
        Contains id, date, (observed) temperature, any predictor columns, and
        [by] - the grouping variable.  Must not contain a GroupNo column.
    by : str
        Name of the grouping variable over which to cross-validate.
    modbuilder : function: dataframe -> (dataframe -> Watershed model)
        Function which prepares a coefficient model.  Accepts data, then
        returns a function which itself accepts predictor data and returns a
        prediction data frame.
    k : int
        Number of "folds".  k=1 will cause leave-one-out validation instead.
    output : str, filename
        File name for where to store raw results.
    redo : Bool
        If True, will rerun cross-validation regardless of whether it has
        already been done.  If False, will check if results already exist
        and just return those if that is the case.

    Returns
    -------
    Dataframe of raw cross-validation results.

    """
    if (not redo) and (output is not None) and os.path.exists(output):
        return pd.read_csv(output, dtype={"id": str}).\
            assign(date = lambda x: pd.to_datetime(x["date"]))
    groups = pd.DataFrame({by: data[by].unique()})
    if k==1:
        groups["GroupNo"] = np.arange(len(groups))
    else:
        # Randomly ordered indices, guaranteed at least floor(len/k) per index
        indices = np.arange(len(groups)) % k
        rng.shuffle(indices)
        groups["GroupNo"] = indices
    data = data.merge(groups, on=by)
    def rungrp(grp):
        (gn, df) = grp
        mod = modbuilder(data[data["GroupNo"] != gn])
        return pd.concat([mod(wsdat)
                          for _, wsdat in df.groupby("id")])
    result = pd.concat([rungrp(grp) for grp in data.groupby("GroupNo")])
    if output is not None:
        result.to_csv(output, index=False)
    return result
