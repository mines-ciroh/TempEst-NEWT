# -*- coding: utf-8 -*-
"""
This file covers pre-specified modification engines for the Watershed class.
"""

import pandas as pd
import numpy as np
import rtseason as rts

season_limits = {
    "WinterDay": (0, 110),
    "SpringDay": (120, 180),
    "SummerDay": (200, 240),
    "FallDay": (300, 365)
    }

def trycatch(op):
    try:
        return op()
    except ValueError:
        return None

def makenp(data):
    data = data.to_numpy()
    return np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

def make_linear_season_engine(data, names, xs, start, until):
    """
    Build a linear seasonality engine.  `data` must have date, temperature,
    and all columns in xs.
    """
    data["day"] = data["date"].dt.day_of_year
    data["year"] = data["date"].dt.year
    ssn = data[["year", "day", "temperature"]].\
        groupby("year").\
            apply(lambda x:
                  trycatch(lambda: rts.ThreeSine.from_data(x).to_df()))
    reldata = (data[(data["day"] >= start) & (data["day"] <= until)] if
               until > start else
               data[(data["day"] >= start) | (data["day"] <= until)])
    weather = ssn.merge(
        reldata.
        groupby("year")[xs].mean(),
        on="year")
    sol = np.linalg.lstsq(makenp(weather[xs]), ssn[names].to_numpy(),
                          rcond=None)[0]
    return linear_season_engine(names, xs, sol, start, until)

def linear_season_engine(names, xs, coefficients, start, until):
    """
    names: seasonality columns of interest as list of colnames
        e.g. ["Intercept", "Amplitude", "SummerDay"]
    xs: names of predictors of interest  e.g. ["prcp"]
    coefficients: fitted coefficients as outputted from numpy.linalg.lstsq,
        i.e.: column corresponds to coefficient, row corresponds to predictor,
        first row corresponds to 1 (constant)
    start: day of year to compute means from
    until: day of year to compute means through
    """
    def f(seasonality, at_coef, vp_coef, dailies, history,
                         statics):
        history["year"] = history["date"].dt.year
        history = history[
            (history["year"] == history["year"].max())]
        history = (history[(history["day"] >= start) &
                          (history["day"] <= until)] if until > start else
                   history[(history["day"] >= start) |
                                     (history["day"] <= until)])
        varbs = makenp(history.groupby("year")[xs].mean())
        preds = pd.DataFrame(varbs @ coefficients, columns=names)
        # Make sure season timing isn't out of bounds
        for k in season_limits:
            if k in names:
                if preds[k].iloc[0] <= season_limits[k][0]:
                    preds[k] = season_limits[k][0] + 1
                if preds[k].iloc[0] >= season_limits[k][1]:
                    preds[k] = season_limits[k][1] - 1
        new_ssn = rts.ThreeSine.from_coefs(pd.concat([
            seasonality.to_df().drop(columns=names),
            preds
            ], axis=1))
        return (new_ssn, at_coef, vp_coef, dailies)
    return f

