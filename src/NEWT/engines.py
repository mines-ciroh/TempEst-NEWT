# -*- coding: utf-8 -*-
"""
This file covers pre-specified modification engines for the Watershed class.
"""

import pandas as pd
import numpy as np
import rtseason as rts
import NEWT.analysis as analysis

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

def make_threshold_engine(data, min_cor=0.7, zeroit=True):
    """
    Create a threshold sensitivity engine from the provided data.
    data must include actemp, st_anom, and anom_atmod.
    min_cor: minimum correlation to apply an engine at all.
    zeroit: passed on to engine.
    """
    sens = analysis.get_sensitivities(data)
    best_cut = analysis.find_cutoff(sens)  # cutoff, corr, min, max
    if best_cut["corr"] < min_cor:
        # act_cutoff == act_min: just returns current coefficient
        return threshold_sensitivity_engine(0, 0, 0, 0)
    # Recompute max to hold for all high-temp conditions
    true_max = sens[sens["int_mean"] > best_cut["cutoff"]]["slope"].mean()
    return threshold_sensitivity_engine(sens["int_mean"].min(),
                                        best_cut["min"],
                                        best_cut["cutoff"],
                                        true_max)

def threshold_sensitivity_engine(act_min, coef_min, act_cutoff, coef_max,
                                 zeroit=True):
    """
    Applies a linear-until-threshold-then-constant modification to the air
    temperature coefficient.
    act_min, coef_min = minimum annual cycle temperature and associated minimum
        sensitivity (AT coefficient).
    act_cutoff, coef_max = cutoff for linear/constant and associated constant
        sensitivity.
    zeroit = also set sensitivity to zero for sustained ~zero seasonal temperatures.
    """
    def f(seasonality, at_coef, vp_coef, dailies, history, statics):
        if act_cutoff == act_min:
            return (seasonality, at_coef, vp_coef, dailies)
        actemp = history["actemp"].to_numpy()
        lookback = 7 if len(actemp) > 7 else len(actemp)
        mean_ac = actemp[-lookback:].mean()
        if mean_ac >= act_cutoff:
            new_coef = coef_max
        else:
            position = (mean_ac - act_min) / (act_cutoff - act_min)
            new_coef = (coef_max - coef_min) * position + coef_min
        if zeroit and mean_ac < 0.1:
            new_coef = 0
        
        return (seasonality, new_coef, vp_coef, dailies)
    return f

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

