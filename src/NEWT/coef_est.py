# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:03:40 2024

@author: dphilippus

This file handles data preprocessing and coefficient estimation.
"""

from pygam import LinearGAM, s
import pandas as pd
import numpy as np
from NEWT import analysis, statics

# Used: ['slope', 'elev_min', 'elev', 'area', 'intercept', 'srad_sd', 'cold_prcp', 'prcp', 'prcp_sd', 'srad', 'water', 'wetland', 'developed', 'ssn_phi', 'Intercept', 'ice_snow', 'vp_sd', 'lat', 'tamp', 'frozen', 'lon', 'ssn_index', 'forest']

inp_cols = ["tmax", "prcp", "srad", "vp",
            "area", "elev_min", "elev", "slope",
            "forest", "wetland", "developed", "ice_snow", "water",
            "lat", "lon", "date", "day"]
req_cols = inp_cols + ["id"]
training_req_cols = req_cols + ["temperature"]

def ssn_df(data):
    ctr, I = analysis.circular_season(data["date"], data["prcp"])
    return pd.DataFrame({"ssn_phi": [ctr], "ssn_index": I})

def preprocess(data):
    """
    Convert raw input data into appropriate format, with all required covariates.
    """
    if not all([col in data.columns for col in req_cols]):
        missing = [col for col in req_cols if not col in data.columns]
        raise ValueError(f"Missing columns in input data; required: {req_cols}; missing: {missing}")
    data["frozen"] = data["tmax"] < 0    
    data["cold_prcp"] = data["prcp"] * data["frozen"]
    predictors = data.groupby("id", as_index=False)[
        req_cols + ["frozen", "cold_prcp"]].mean().merge(
        data.groupby("id", as_index=False)[["prcp", "srad", "vp"]].std(),
        on="id", suffixes=["", "_sd"]).merge(
            data.groupby("id", as_index=False).apply(ssn_df, include_groups=False),
            on="id").merge(
                data.groupby("id", as_index=False).apply(
                    lambda x: statics.fit_simple_daily(x, "tmax", True).\
                        assign(tamp = lambda x: np.sqrt(x["ksin"]**2 + x["kcos"]**2))),
                on="id"
                )
    return predictors

var_sets = [
    {"name": "Intercept",
     "vars": ['intercept', 'cold_prcp', 'frozen', 'area', 'elev_min'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4), "lam": 10},
    {"name": "Amplitude",
     "vars": ['frozen', 'water', 'forest', 'area', 'elev', 'elev_min', 'prcp_sd', 'vp_sd', 'ssn_phi', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 50},
    {"name": "SpringSummer",
     "vars": ['intercept', 'prcp', 'cold_prcp', 'frozen', 'water', 'elev', 'elev_min', 'prcp_sd', 'srad_sd', 'vp_sd'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 10},
    {"name": "FallWinter",
     "vars": ['intercept', 'frozen', 'srad', 'water', 'forest', 'elev', 'slope', 'lat', 'lon', 'vp_sd', 'ssn_phi', 'ssn_index', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 10},
    {"name": "SpringDay",
     "vars": ['intercept', 'elev', 'lat', 'lon', 'ssn_phi', 'ssn_index', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6), "lam": 10},
    {"name": "SummerDay",
     "vars": ['prcp', 'cold_prcp', 'frozen', 'wetland', 'ice_snow', 'elev', 'lat', 'lon', 'prcp_sd', 'srad_sd', 'vp_sd', 'ssn_index', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 10},
    {"name": "FallDay",
     "vars": ['srad', 'water', 'developed', 'forest', 'wetland', 'ice_snow', 'elev_min', 'lat', 'lon'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 10},
    {"name": "WinterDay",
     "vars": ['frozen', 'water', 'forest', 'wetland', 'srad_sd', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5), "lam": 10},
    {"name": "threshold_coef_max",
     "vars": ['Intercept', 'intercept', 'frozen', 'srad', 'water', 'area', 'lat', 'lon', 'ssn_index'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 10},
    {"name": "threshold_coef_min",
     "vars": ['intercept', 'frozen', 'srad', 'elev', 'tamp'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4), "lam": 10},
    {"name": "threshold_act_cutoff",
     "vars": ['frozen', 'elev_min', 'lat', 'lon', 'srad_sd'],
     "eq": s(0) + s(1) + s(2) + s(3) + s(4), "lam": 10},
]

coef_names = ["Intercept", "Amplitude", "SpringSummer", "FallWinter", "SpringDay", "SummerDay", "FallDay", "WinterDay",
         "threshold_coef_max", "threshold_coef_min", "threshold_act_cutoff"
        ]


def build_model_from_data(tr_data):
    """
    Prepares a coefficient estimation model from the provided training data.
    """
    vars_local = var_sets.copy()
    for vs in vars_local:
        cd = tr_data if not vs["name"] in ["threshold_coef_min", "threshold_act_cutoff"] else \
            tr_data[tr_data["threshold_act_cutoff"] > -1]
        vs["gam"] = LinearGAM(vs["eq"], lam=vs["lam"]).fit(cd[vs["vars"]], cd[vs["name"]])
    return vars_local


def predict_site_coefficients(model, data):
    """
    Predicts model coefficients using the provided (pre-processed) data for
    a specific site.
    """
    predictor = lambda cols, gam, ws: gam.predict(ws[cols])[0]
    statics = data
    for vs in model:  # Essential: sensitivity stuff is LAST - it uses Intercept, Amplitude
        statics[vs["name"]] = predictor(vs["vars"], vs["gam"], statics)
    return statics[coef_names]


def predict_all_coefficients(model, data):
    """
    Predicts model coefficients for all sites.
    """
    keepll = "lat" in data.columns and "lon" in data.columns
    keep = data[["id", "elev", "lat", "lon"]] if keepll else data[["id", "elev"]]
    coefs = data.groupby("id").apply(
        lambda x: predict_site_coefficients(model, x),
        include_groups=False)
    return coefs.droplevel(1).merge(keep, how="left", on="id")
