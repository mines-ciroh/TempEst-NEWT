# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:27:59 2024

@author: dphilippus

Contains functions for generating static estimates.
"""

import pandas as pd
import numpy as np
import warnings


def fitrun_sd(data, name):
    (i, s, c) = fit_simple_daily(data, name)
    return simple_daily(name, i, s, c)


def simple_daily(name, intercept, ksin, kcos):
    """
    Construct a day-of-year timeseries (DataFrame of [day, name]) using a
    variable-phase sinusoid of y = a + b*sin(d) + c*cos(d)
    intercept = intercept, ksin = sine coef, kcos = cosine coef
    phase = atan(ksin/kcos)
    """
    days = np.arange(1, 367)
    day_rad = days * 2 * np.pi / 365
    return pd.DataFrame({
        "day": days,
        name: intercept + ksin*np.sin(day_rad) + kcos*np.cos(day_rad)
        })


def fit_simple_daily(data, name, to_df=False, crash=False):
    """
    Use OLS to fit simple-daily coefficients.  data must have day and name.
    """
    days = data["day"] * 2 * np.pi / 365
    y = data[name].to_numpy()
    x = np.array([np.ones(len(days)), np.sin(days), np.cos(days)]).transpose()
    try:
        sol = np.linalg.lstsq(x, y, None)[0]
    except Exception as e:
        if crash:
            raise e
        else:
            warnings.warn(f"fit_simple_daily: {e}")
            sol = np.array([y.mean(), 0, 0])
    if to_df:
        return pd.DataFrame({"intercept": [sol[0]], "ksin": sol[1], "kcos": sol[2]})
    else:
        return sol


def sd_2d_model(data, name, to_model=True):
    """
    Build a model to predict simple daily coefficients from latitude and
    elevation.  data must contain id, day, lat, elev, and name.
    """
    coefs = data.groupby("id").apply(
        lambda x: fit_simple_daily(x, name, True).assign(lat=x.loc["lat", 0],
                                                   elev=x.loc["elev", 0]))
    x = np.array([np.ones(len(coefs)),
                   coefs["lat"].to_numpy(),
                   coefs["elev"].to_numpy()]).transpose()
    ys = coefs[["intercept", "ksin", "kcos"]].to_numpy()
    sol = np.linalg.lstsq(x, ys, None)[0]
    if to_model:
        def predict(data):
            x = np.array([np.ones(len(data)),
                           data["lat"].to_numpy(),
                           data["elev"].to_numpy()]).transpose()
            res = x @ sol
            return pd.DataFrame(res, columns=["intercept", "ksin", "kcos"])
        return predict
    else:
        return pd.DataFrame(sol,
                            index=["1", "lat", "elev"],
                            columns=["intercept", "ksin", "kcos"])

