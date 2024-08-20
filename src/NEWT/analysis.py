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
import scipy
rng = np.random.default_rng()

def convolve(series):
    # Run AT convolution on the provided series.
    conv = scipy.stats.lognorm.pdf(np.arange(0, 7), 1)
    return scipy.signal.fftconvolve(series,
                                    conv, mode="full")[:-(len(conv) - 1)]

def get_sensitivity(data):
    """
    Compute single-regression sensitivity of ST to AT anomaly.
    data must contain: anom_atmod, st_anom
    """
    x = data["anom_atmod"].to_numpy()
    x = np.array([np.ones(x.shape), x]).transpose()
    y = data["st_anom"].to_numpy()
    return np.linalg.lstsq(x, y, rcond=None)[0][1]

def get_sensitivities(data):
    """
    Compute the sensitivity of ST to AT anomaly, grouped by integer mean
    temperature (actemp).
    data must contain: anom_atmod, st_anom, actemp
    """
    return data.\
        assign(int_mean = lambda x: x["actemp"].apply(int)).\
        groupby("int_mean", as_index=False).\
        apply(get_sensitivity, include_groups=False).\
        rename(columns={None: "slope"})

def apply_cutoff(slopes, cut):
    """
    For a given slopes (sensitivities) dataset, compute correlation, min
    sensitivity, and max sensitivity below the specified cutoff.
    """
    blw = slopes[slopes["int_mean"] <= cut]
    return {
        "cutoff": cut,
        "corr": blw["int_mean"].corr(blw["slope"]),
        "min": blw["slope"].min(),
        "max": blw["slope"].max()
        }

def find_cutoff(sl):
    """
    sl: has int_mean, slope
    We want:
        Optimal cutoff point for corr., such that there are at least 3 values above and below the cutoff
        Sub-cutoff correlation, minimum, maximum
    """
    if len(sl) < 7:
        return {}
    means = sl["int_mean"].to_numpy()
    means.sort()
    min_cutoff = int(means[2]) + 1
    max_cutoff = int(means[-3])
    if max_cutoff <= min_cutoff:
        return apply_cutoff(sl, min_cutoff) | {"sensitivity": np.NaN}
    best_cutoff = None
    for cutoff in range(min_cutoff, max_cutoff + 1):
        fit = apply_cutoff(sl, cutoff)
        if best_cutoff is None or best_cutoff["corr"]**2 < fit["corr"]**2:
            best_cutoff = fit
    return best_cutoff
    

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
