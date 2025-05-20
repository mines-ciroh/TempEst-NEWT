# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:03:02 2024

@author: dphilippus

This file contains analysis helpers.
"""

import pandas as pd
import numpy as np
import scipy
from libschema.analysis import kfold, nse, perf_summary
rng = np.random.default_rng()

def convolve(series,
             conv=np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ])):
    """Implements anomaly smoothing for analysis use by convolution.
    

    Parameters
    ----------
    series : numeric array
        The series (typically, air temperature anomaly) to be smoothed.
    conv : numeric array
        The array to use for smoothing.  This is applies "backwards", so, for
        the final entry in the result, conv[0] is multiplied with series[-1],
        etc.

    Returns
    -------
    numeric array
        The smoothed/convolved result.

    """
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
    sens = np.linalg.lstsq(x, y, rcond=None)[0][1]
    # Apply some checks to make sure sensitivity is sane.  Things get... weird otherwise.
    if sens > 1:
        return 1.0
    if sens < 0:
        return 0.0
    return sens

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
        return None
    means = sl["int_mean"].to_numpy()
    means.sort()
    min_cutoff = int(means[2]) + 1
    max_cutoff = int(means[-3])
    if max_cutoff <= min_cutoff:
        return apply_cutoff(sl, min_cutoff) | {"sensitivity": np.NaN}
    best_cutoff = None
    for cutoff in range(min_cutoff, max_cutoff + 1):
        fit = apply_cutoff(sl, cutoff)
        if best_cutoff is None or best_cutoff["corr"] < fit["corr"]:
            best_cutoff = fit
    return best_cutoff


def anomalies(date, y):
    """
    Compute day-of-year anomalies for y.  Returns a Series with anomalies.
    """
    data = pd.DataFrame({"date": date,
                         "day": pd.to_datetime(date).dt.day_of_year,
                         "y": y})
    means = data.groupby("day", as_index=False)["y"].mean()
    data = data.merge(means, on="day", suffixes=["", "_mean"])
    data["anom"] = data["y"] - data["y_mean"]
    return data["anom"]


def circular_season(days, values):
    """
    Compute seasonality under circular statistics (Fisher 1993).

    Parameters
    ----------
    days : pandas Series
        Series of date string, Pandas datetime, or integer day of year.
    values : numeric pandas Series
        Whatever we're computing the seasonality of.

    Returns
    -------
    day_ctr : float
        Mean, or central, day of the quantity.
    I : float
        Seasonality index.
        
    Notes
    -----
    For angle from Jan. 1:
        S = sum(P*sin)
        C = sum(P*cos)
    Magnitude PR = sqrt(S^2 + C^2)
    Angle = atan(S/C); +180 if C <0 and +360 if S < 0.  This gives average
    time of ocurrence.
    Concentration in time (seasonality index, IS) = PR/P total.
    Fisher, N.I. (1993). Statistical Analysis of Circular Data. New York:
        Cambridge University Press.  Cited in
        Dingman, S. L. (2015). Physical Hydrology, 3rd ed.
        Long Grove: Waveland Press, Inc.
    """
    # First, make days into DOY integers.
    if days.dtype == 'int64' or days.dtype == 'int32':
        days = days.to_numpy()
    if days.dtype == 'O':  # string date
        days = pd.to_datetime(days).dt.day_of_year.to_numpy()
    else:
        days = days.dt.day_of_year.to_numpy()
    values = values.to_numpy()
    # Now let's do some math.
    dangle = (days - 1) * 2 * np.pi / 365
    dsin = np.sin(dangle)
    dcos = np.cos(dangle)
    S = (dsin * values).sum()
    C = (dcos * values).sum()
    PR = np.sqrt(S**2 + C**2)
    P = values.sum()
    I = PR/P
    # Compute offset
    offset = 0
    if S < 0:
        offset = 2*np.pi
    if C < 0:
        offset = np.pi
    # Get angle
    ctr = np.arctan(S/C) + offset
    day_ctr = ctr * 365 / (2 * np.pi) + 1
    return (day_ctr, I)
    

