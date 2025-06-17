# -*- coding: utf-8 -*-
"""
This file covers pre-specified modification engines for the Watershed class.
"""

import pandas as pd
import numpy as np
import rtseason as rts
import NEWT.analysis as analysis
from libschema.classes import ModEngine
from pygam import LinearGAM, s
from NEWT.watershed import Seasonality, Anomaly

class ClimateEngine(ModEngine):
    def __init__(self, coef_model):
        # coef_model should be: prediction data --> (Seasonality, Anomaly,
        # Periodics)
        self.coef_model = coef_model
    
    def apply(self, seasonality, anomaly, periodics, history):
        return self.coef_model(history)
    
    def to_dict(self):
        return {"climate_engine": True}


def wetDryHistory(data, month_range):
    data["year"] = data["date"].dt.year + (data["date"].dt.month > 9)
    data["tmax_early"] = data["tmax"] * (data["date"].dt.month.isin(month_range))  # try early-year conditions only
    data["prcp_early"] = data["prcp"] * (data["date"].dt.month.isin(month_range))
    var = ["tmax", "prcp", "tmax_early", "prcp_early"]
    means = data.groupby("year")[var].mean()
    return (means - data[var].mean()).merge(means, suffixes=["", "_base"], on="year")

class WetDryRunner(object):
    # Helper class for a pickle-able higher-order function.
    def __init__(self, gam, xvar):
        self.gam = gam
        self.xvar = xvar
    
    def apply(self, ssn, history):
        preds = history.copy()
        ssn_inp = ssn.to_dict()
        for v in ssn_inp:
            preds[v + "_ref"] = ssn_inp[v]
        return self.gam.predict(preds[self.xvar])[0]

class WetDryEngine(ModEngine):
    month_range = [12, 1, 2, 3, 4]
    var_sets = {
    	"Intercept": ['tmax', 'tmax_early'],
    	"Amplitude": ['Intercept_ref', 'FallWinter_ref', 'tmax', 'prcp', 'tmax_early', 'prcp_early', 'tmax_base', 'tmax_early_base'],
    	"WinterDay": ['Intercept_ref', 'Amplitude_ref', 'WinterDay_ref', 'tmax_early', 'tmax_base']
    }
    
    def __init__(self, models, month_range=None, var_sets=None):
        # models: dictionary of variable -> function(Seasonality, history -> Seasonality)
        self.models = models
        self.orig_ssn = None
        if month_range is not None:
            self.month_range = month_range
        if var_sets is not None:
            self.var_sets = var_sets

    def apply(self, seasonality, anomaly, periodics, history):
        ssn_dict = seasonality.to_dict()
        history = wetDryHistory(history, self.month_range)
        for var in self.models:
            ssn_dict[var] = self.models[var].apply(seasonality, history)
        new_ssn = Seasonality.from_dict(ssn_dict)
        return (new_ssn, anomaly, periodics)

    def from_data(coefs, year_coefs, data, month_range=None, var_sets=None):
        # Coefs: data frame with id and Seasonality terms
        # year_coefs: also with year column (water year)
        # data: date, tmax, prcp
        # month_range: override self.month_range. List of integer months.
        # var_sets: override self.var_sets. dictionary of coefficient -> predictor variables.
        if month_range is None:
            month_range = WetDryEngine.month_range
        if var_sets is None:
            var_sets = WetDryEngine.var_sets
        preds = data.groupby("id").apply(lambda x: wetDryHistory(x, month_range), include_groups=False)
        # reset_index is to preserve both id and year.
        inpdata = coefs.merge(year_coefs.reset_index(), on="id", suffixes=["_ref", ""]
                             ).set_index(["id", "year"]).merge(preds, on=["id", "year"]).dropna()
        models = {var: WetDryRunner(LinearGAM(sum([s(i) for i in range(1, len(var_sets[var]))], start=s(0)), lam=10
                                         ).fit(inpdata[var_sets[var]], inpdata[var]),
                                var_sets[var])
                  for var in var_sets}
        return WetDryEngine(models, month_range, var_sets)
