# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:03:40 2024

@author: dphilippus

This file handles data preprocessing and coefficient estimation.
"""

from pygam import LinearGAM, s, l, te
import pandas as pd
import numpy as np
from NEWT import analysis, statics

# Used: ['slope', 'elev_min', 'elev', 'area', 'intercept', 'srad_sd', 'cold_prcp', 'prcp', 'prcp_sd', 'srad', 'water', 'wetland', 'developed', 'ssn_phi', 'Intercept', 'ice_snow', 'vp_sd', 'lat', 'tamp', 'frozen', 'lon', 'ssn_index', 'forest']

inp_cols = ["tmax", "prcp", "srad", "vp",
            "area", "elev_min", "elev", "slope",
            "forest", "wetland", "developed", "ice_snow", "water",
            "canopy", "flowdir",
            "lat", "lon", "date", "day"]
req_cols = inp_cols + ["id"]
training_req_cols = req_cols + ["temperature"]

def ssn_df(col):
    def f(data):
        ctr, I = analysis.circular_season(data["date"], data[col])
        return pd.DataFrame({col + "_phi": [ctr], col + "_index": I})
    return f

def preprocess(data, allow_no_id=True):
    """
    Convert raw input data into appropriate format, with all required covariates.
    """
    if not "id" in data.columns and allow_no_id:
        data["id"] = "null"
    if not all([col in data.columns for col in req_cols]):
        missing = [col for col in req_cols if not col in data.columns]
        raise ValueError(f"Missing columns in input data; required: {req_cols}; missing: {missing}")
    data["frozen"] = data["tmax"] < 0    
    data["cold_prcp"] = data["prcp"] * data["frozen"]
    predictors = data.groupby("id", as_index=False)[
        inp_cols + ["frozen", "cold_prcp"]].mean().merge(
        data.groupby("id", as_index=False)[["prcp", "srad", "vp"]].std(),
        on="id", suffixes=["", "_sd"]).merge(
            # Why different grouping?  apply was dropping id
            data.groupby("id").apply(ssn_df("prcp"), include_groups=False).reset_index(),
            on="id").merge(
                data.groupby("id").apply(ssn_df("tmax"), include_groups=False).reset_index(),
                on="id"
                )
    return predictors

var_sets = [
    {"name": "PCA0", "vars": ['tmax', 'vp', 'elev_min', 'forest', 'wetland', 'water', 'cold_prcp', 'vp_sd', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 58},
    {"name": "PCA1", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'forest', 'wetland', 'ice_snow', 'water', 'vp_sd', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + te(10, 11), "lam": 20},
    {"name": "PCA2", "vars": ['tmax', 'elev_min', 'elev', 'forest', 'developed', 'ice_snow', 'water', 'canopy', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 120},
    {"name": "PCA3", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'wetland', 'developed', 'cold_prcp', 'prcp_phi', 'prcp_index', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + te(9, 10), "lam": 333},
    {"name": "PCA4", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'frozen', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 100},
    {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'forest', 'wetland', 'developed', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + te(11, 12), "lam": 100},
    {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'canopy', 'frozen', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 100},
    {"name": "PCA7", "vars": ['tmax', 'area', 'elev', 'developed', 'vp_sd', 'tmax_phi', 'frozen', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + te(6, 7), "lam": 37},
    {"name": "PCA8", "vars": ['tmax', 'area', 'elev', 'forest', 'wetland', 'developed', 'water', 'canopy', 'cold_prcp', 'vp_sd', 'prcp_phi', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 949},
]

# var_sets = [
#     {"name": "PCA0", "vars": ['tmax', 'vp', 'elev_min', 'forest', 'wetland', 'water', 'cold_prcp', 'vp_sd', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8), "lam": 58},
#     {"name": "PCA1", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'slope', 'forest', 'wetland', 'ice_snow', 'water', 'frozen', 'vp_sd', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 20},
#     {"name": "PCA2", "vars": ['tmax', 'elev_min', 'elev', 'forest', 'developed', 'ice_snow', 'water', 'canopy', 'cold_prcp', 'vp_sd', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 120},
#     {"name": "PCA3", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'wetland', 'developed', 'frozen', 'cold_prcp', 'prcp_phi', 'prcp_index', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10), "lam": 333},
#     {"name": "PCA4", "vars": ['tmax', 'prcp', 'vp', 'area', 'elev_min', 'elev', 'slope', 'wetland', 'ice_snow', 'water', 'frozen', 'prcp_phi', 'prcp_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 100},
#     {"name": "PCA5", "vars": ['tmax', 'vp', 'elev_min', 'elev', 'forest', 'wetland', 'developed', 'frozen', 'vp_sd', 'prcp_phi', 'prcp_index', 'tmax_phi', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12), "lam": 100},
#     {"name": "PCA6", "vars": ['tmax', 'prcp', 'vp', 'elev_min', 'elev', 'slope', 'water', 'canopy', 'frozen', 'prcp_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9), "lam": 100},
#     {"name": "PCA7", "vars": ['tmax', 'area', 'elev', 'developed', 'frozen', 'vp_sd', 'tmax_phi', 'tmax_index'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7), "lam": 37},
#     {"name": "PCA8", "vars": ['tmax', 'area', 'elev', 'forest', 'wetland', 'developed', 'water', 'canopy', 'cold_prcp', 'vp_sd', 'prcp_phi', 'tmax_phi'], "eq": s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11), "lam": 949},
# ]


coef_names = ["PCA" + str(i) for i in range(len(var_sets))]

col_order = ['Intercept', 'Amplitude', 'FallDay', 'WinterDay', 'SpringDay',
       'SummerDay', 'SpringSummer', 'FallWinter', 'at_coef']
pca_components = np.array([[-0.3635536 , -0.36253655,  0.18978457, -0.4401217 , -0.24528224,
         0.21700805,  0.38140546, -0.3270315 , -0.38339944],
       [ 0.38491387, -0.29840726,  0.46560378,  0.12243858,  0.37358605,
         0.43343509, -0.16075314, -0.381675  ,  0.17907109],
       [ 0.21515426,  0.42568867,  0.14601904, -0.32076195, -0.53292759,
         0.31257888,  0.12651994, -0.02602839,  0.49988585],
       [-0.13209679,  0.1202627 , -0.36689606,  0.14785714,  0.42261061,
         0.43907167,  0.63080335,  0.10279969,  0.17818129],
       [-0.07619769, -0.26799642,  0.40814928,  0.20224315, -0.03685127,
        -0.52885627,  0.4819317 ,  0.06431149,  0.44434513],
       [-0.22202771,  0.01563487, -0.361087  ,  0.40879354, -0.24807321,
        -0.06783746, -0.08212493, -0.7319151 ,  0.21065805],
       [ 0.35877293, -0.61655974, -0.53625048, -0.33023217, -0.08368292,
        -0.06183943, -0.03335436,  0.0757308 ,  0.27720677],
       [-0.47126455,  0.1246728 ,  0.0134238 , -0.52115533,  0.45438736,
        -0.15212244, -0.28078702, -0.14495279,  0.40139952],
       [-0.5000185 , -0.34072077,  0.08090875,  0.27907185, -0.24756717,
         0.40409877, -0.3095345 ,  0.4117611 ,  0.24396599]])
offset = pd.Series([12.740452585874603, 8.88225678976997, 326.75461454940285, 66.92073832790444, 154.07057546145495, 217.70684039087948, 0.7392294811845936, 1.39835333783497, 0.6101532876129115],
                   index=col_order)
scale = pd.Series([3.9935065545919537, 2.7232799140867754, 15.606734293670241, 32.47165509597058, 21.642058805538785, 17.322166212360017, 0.9551794545253263, 0.9203378017283892, 0.1661541252404621],
                  index=col_order)



def build_model_from_data(tr_data):
    """
    Prepares a coefficient estimation model from the provided training data.  Training data is assumed to have coefficients listed in col_order,
    which will be converted through PCA.
    """
    vars_local = var_sets.copy()
    X = tr_data.drop(columns=col_order)
    Y = tr_data[["id"] + col_order].set_index("id")
    Y = (Y - offset) / scale  # normalize scale
    Y = Y @ np.transpose(pca_components)
    Y.columns = coef_names
    for vs in vars_local:
        vs["gam"] = LinearGAM(vs["eq"], lam=vs["lam"]).fit(X[vs["vars"]], Y[vs["name"]])
    return vars_local


def predict_site_coefficients(model, data):
    """
    Predicts model coefficients using the provided (pre-processed) data for
    a specific site.  Then invert PCA to produce NEWT coefficients.
    """
    predictor = lambda cols, gam, ws: gam.predict(ws[cols])[0]
    pcaed = {}
    for vs in model:
        pcaed[vs["name"]] = predictor(vs["vars"], vs["gam"], data)
    pcaed = pd.DataFrame(pcaed, index=[0])[coef_names]  # ensure correct order
    inv = pcaed @ pca_components
    inv.columns = col_order
    return inv * scale + offset


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
