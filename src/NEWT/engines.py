# -*- coding: utf-8 -*-
"""
This file covers pre-specified modification engines for the Watershed class.
"""

import pandas as pd
import numpy as np
import rtseason as rts
import NEWT.analysis as analysis
import NEWT.coef_est as coef_est

class ModEngine(object):
    def apply(self, seasonality, at_coef, dailies, history, statics):
        raise NotImplementedError("ModEngine.apply")
        return (seasonality, at_coef, dailies)
    
    def coefficients(self):
        raise NotImplementedError("ModEngine.coefficients")
        return {}
    
    def from_data():
        raise NotImplementedError("ModEngine.from_data")
    
    def to_dict(self):
        raise NotImplementedError("ModEngine.to_dict")

    def from_dict(d):
        name = d["engine_name"]
        args = d["args"]
        engine = eval(name)
        return engine(*args)

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

class ClimateCoefficientEngine(ModEngine):
    required_columns = coef_est.req_cols
    required_columns_training = coef_est.training_req_cols
    
    def __init__(self, coef_model, years=0, recency=0):
        # If years is zero, then it will wait 10 years before starting to process.
        # Otherwise, it will wait until <years> have passed.
        self.coef_model = coef_model
        self.lookback = years
        self.recency = recency
        if self.lookback > 0:
            self.min_duration = self.lookback
        else:
            self.min_duration = 10
    
    def to_dict(self):
        return {
            "engine_name": "ClimateCoefficientEngine",
            "args": [self.coef_model, self.lookback, self.recency]}
    
    def from_data():
        raise NotImplementedError("ClimateCoefficientEngine.from_data cannot be implemented.  Use TempEst-NEXT to generate ClimateCoefficientEngines.")
    
    def coefficients(self):
        return {
            "climate_lookback": self.lookback,
            "climate_recency": self.recency
            }
    
    def apply(self, seasonality, at_coef, dailies, history, statics):
        history["year"] = history["date"].dt.year
        duration = history["year"].max() - history["year"].min()
        if duration >= self.min_duration:
            if self.lookback > 0:
                history = history[history["year"] > history["year"].max() - self.lookback]
            pdata = coef_est.preprocess(history)
            coefs = coef_est.predict_site_coefficients(self.coef_model,
                                                       pdata)
            seasonality = rts.ThreeSine(
                Intercept=coefs["Intercept"].iloc[0],
                Amplitude=coefs["Amplitude"].iloc[0],
                SpringSummer=coefs["SpringSummer"].iloc[0],
                FallWinter=coefs["FallWinter"].iloc[0],
                SpringDay=coefs["SpringDay"].iloc[0],
                SummerDay=coefs["SummerDay"].iloc[0],
                FallDay=coefs["FallDay"].iloc[0],
                WinterDay=coefs["WinterDay"].iloc[0]
            )
            n_dailies = history.groupby(["day"], as_index=False)["at"].mean().rename(columns={"at": "mean_tmax"})
            o_dailies = dailies[-dailies["day"].isin(n_dailies["day"])]
            dailies = pd.concat([n_dailies, o_dailies])
        return (seasonality, at_coef, dailies)
    

class ThresholdSensitivityEngine(ModEngine):
    def __init__(self, act_min, coef_min, act_cutoff, coef_max, zeroit=True):
        self.act_min = act_min
        self.coef_min = coef_min
        self.act_cutoff = act_cutoff
        self.coef_max = coef_max
        self.zeroit = zeroit

    def to_dict(self):
        return {
            "engine_name": "ThresholdSensitivityEngine",
            "args": [float(self.act_min), float(self.coef_min),
                     float(self.act_cutoff), float(self.coef_max),
                     self.zeroit]}

    def from_data(data, min_cor=0.7, zeroit=True, use_at=False):
        """
        Create a threshold sensitivity engine from the provided data.
        data must include actemp, st_anom, and anom_atmod.
        min_cor: minimum correlation to apply an engine at all.
        zeroit: passed on to engine.
        If use_at, try fitting against air temperature instead of mean stream
            temperature.  Data must include "tmax".
        """
        if use_at:  # terrible hack for testing purposes
            data = data.drop(columns="actemp").\
                rename(columns={"tmax": "actemp"})
        sens = analysis.get_sensitivities(data)
        best_cut = analysis.find_cutoff(sens)  # cutoff, corr, min, max
        if best_cut is None or best_cut["corr"] < min_cor:
            # act_cutoff == act_min: just returns current coefficient
            return ThresholdSensitivityEngine(-1, -1, -1, sens["slope"].mean())
        # Recompute max to hold for all high-temp conditions
        true_max = sens[sens["int_mean"] > best_cut["cutoff"]]["slope"].mean()
        act_min = sens["int_mean"].min()
        coef_min = best_cut["min"]
        act_cutoff = best_cut["cutoff"]
        coef_max = true_max
        return ThresholdSensitivityEngine(act_min, coef_min,
                                          act_cutoff, coef_max, zeroit)
    
    def apply(self, seasonality, at_coef, dailies, history, statics):
        if self.act_cutoff <= self.act_min:
            return (seasonality, at_coef, dailies)
        actemp = history["actemp"].to_numpy()
        lookback = 7 if len(actemp) > 7 else len(actemp)
        mean_ac = actemp[-lookback:].mean()
        if mean_ac >= self.act_cutoff:
            new_coef = self.coef_max
        else:
            position = (mean_ac - self.act_min) / (self.act_cutoff - self.act_min)
            new_coef = (self.coef_max - self.coef_min) * position + self.coef_min
        if self.zeroit and mean_ac < 0.1:
            new_coef = 0
        
        return (seasonality, new_coef, dailies)
    
    def coefficients(self):
        return {
            "threshold_act_min": self.act_min,
            "threshold_coef_min": self.coef_min,
            "threshold_act_cutoff": self.act_cutoff,
            "threshold_coef_max": self.coef_max
            }


class LinearSeasonEngine(ModEngine):
    def __init__(self, names, xs, coefs, start, until):
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
        self.names = names
        self.xs = xs
        self.coefs = coefs
        self.start = start
        self.until = until
        
    def to_dict(self):
        return {
            "engine_name": "LinearSeasonEngine",
            "args": [self.names, self.xs, self.coefs, self.start, self.until]}
    
    def from_data(data, names, xs, start, until):
        """
        Build a linear seasonality engine.  `data` must have date, temperature,
        and all columns in xs.
        """
        data["day"] = data["date"].dt.day_of_year
        data["year"] = data["date"].dt.year
        ssn = data[["year", "day", "temperature"]].\
            groupby("year").\
                apply(lambda x:
                      trycatch(lambda: rts.ThreeSine.from_data(x, warn=False).to_df()))
        reldata = (data[(data["day"] >= start) & (data["day"] <= until)] if
                   until > start else
                   data[(data["day"] >= start) | (data["day"] <= until)])
        weather = ssn.merge(
            reldata.
            groupby("year")[xs].mean(),
            on="year")
        sol = np.linalg.lstsq(makenp(weather[xs]), ssn[names].to_numpy(),
                              rcond=None)[0]
        return LinearSeasonEngine(names, xs, sol, start, until)
    
    def apply(self, seasonality, at_coef, dailies, history, statics):
        history["year"] = history["date"].dt.year
        history = history[
            (history["year"] == history["year"].max())]
        history = (history[(history["day"] >= self.start) &
                          (history["day"] <= self.until)] if self.until > self.start else
                   history[(history["day"] >= self.start) |
                                     (history["day"] <= self.until)])
        varbs = makenp(history.groupby("year")[self.xs].mean())
        preds = pd.DataFrame(varbs @ self.coefs, columns=self.names)
        # Make sure season timing isn't out of bounds
        for k in season_limits:
            if k in self.names:
                if preds[k].iloc[0] <= season_limits[k][0]:
                    preds[k] = season_limits[k][0] + 1
                if preds[k].iloc[0] >= season_limits[k][1]:
                    preds[k] = season_limits[k][1] - 1
        new_ssn = rts.ThreeSine.from_coefs(pd.concat([
            seasonality.to_df().drop(columns=self.names),
            preds
            ], axis=1))
        return (new_ssn, at_coef, dailies)
    
    def coefficients(self):
        coefs = {
            f"linear_ssn_{x}_to_{name}": self.coefs[i][j]
            for (i, x) in enumerate(self.xs)
            for (j, name) in enumerate(self.names)
            }
        return coefs | {
            "linear_ssn_start": self.start,
            "linear_ssn_until": self.until
            }
