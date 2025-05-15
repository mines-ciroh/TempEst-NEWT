#!/usr/bin/env python
# coding: utf-8

# # Equifinality Testing
# 
# Something that might add a lot of noise to coefficient estimation is equifinality in site-specific models. This Notebook analyzes the actual impact of such equifinality.
# 
# The basic procedure for a given watershed:
# 
# - Fit a calibrated model to the full timeseries. (We're interested in equally-performant coefficient sets, not in cal/val uncertainty, so we don't use a validation period.)
# - Evaluate performance.
# - Now, generate N random parameter sets, uniformly distributed. See which have less than an x% RMSE penalty and identify a behavioral envelope (think GLUE).
# - Analyze distributions.
# 
# The immediate result, for $G$ gages and $p$ parameters, is a $(G\cdot N)\times p$ matrix of parameter sets mapped to a vector of $G\cdot N$ percentage RMSE penalties and a matrix of $G\times p$ original (fitted) parameter sets. This can be analyzed in a variety of ways; two major ones come to mind:
# 
# - 1:1 plots showing whether the behavioral envelope varies with the value of the coefficient.
# - Histograms of "coefficient elasticity of RMSE" by site, identifying how variable the sensitivity is.
# 
# Another interesting question will be whether any parameter sets actually have a *negative* penalty, i.e., outperform the fitted coefficient set. (This is a distinct behavior from a, strictly speaking, calibrated model, since our site parameters aren't actually calibrated but rather explicitly computed from the data.)
# 
# For this test, we disable the anomaly GAM because setting it up for the randomized models would be logistically annoying.

# # Setup

# In[41]:


import NEWT
from rtseason import ThreeSine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import scipy
from scipy.stats.qmc import LatinHypercube
import sys
sns.set_context("paper")
bp = "/scratch/dphilippus/notebooks/next_validation/"


# In[2]:


dev_data = pd.read_csv(bp + "DevDataBuffers.csv", dtype={"id": "str"}, parse_dates=["date"])[
    ["id", "date", "day", "temperature", "tmax"]
]
dev_data = dev_data.loc[(dev_data["temperature"] > -0.5) & (dev_data["temperature"] < 40)]


# # Reference Performance

# In[3]:


def perf_wcoef(site):
    model = NEWT.Watershed.from_data(site, use_anomgam=False)
    if model is None:
        return None
    pred = model.run_series(site)
    perf = NEWT.analysis.perf_summary(pred)["RMSE"]
    coefs = model.coefs_to_df().drop(columns=["R2", "RMSE"])
    print("|", end="")
    return pd.concat([coefs, perf], axis=1)


# In[4]:


with warnings.catch_warnings(action="ignore"):
    reference = dev_data.groupby("id").apply(perf_wcoef, include_groups=False).droplevel(1)


# In[5]:


reference


# # Randomized Parameters
# 
# We'll just generate one set of 10,000 for all sites, just to make things simple. Each parameter is allowed to be anywhere in the observed range (above).

# ## Generate Samples

# In[46]:


rawmat = pd.DataFrame(LatinHypercube(9).random(n=10000), columns=reference.columns[:9])


# In[47]:


mins = reference[rawmat.columns].min()
maxes = reference[rawmat.columns].max()
ranges = maxes - mins
rpar = rawmat * ranges + mins


# In[48]:


rpar


# ## Evaluate Performance
# 
# To avoid an excessively large result, for each site, we only keep the 1000 rows with the best performance. This will result in approx. 1M rows total, instead of the unmanageable 10M resulting from storing everything.
# 
# The first site, with 10k iterations, took about 200 seconds, or 0.02 seconds per run. That's fast individually, but too many to be manageable in a Jupyter notebook.

# In[9]:


def random_perf(params=rpar):
    def inner(site):
        daily = site.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        pser = pd.Series([
            NEWT.analysis.perf_summary(
                NEWT.Watershed(
                    seasonality=ThreeSine(
                        Intercept=prow.Intercept,
                        Amplitude=prow.Amplitude,
                        SpringSummer=prow.SpringSummer,
                        FallWinter=prow.FallWinter,
                        SpringDay=prow.SpringDay,
                        SummerDay=prow.SummerDay,
                        FallDay=prow.FallDay,
                        WinterDay=prow.WinterDay
                    ),
                    at_coef=prow.at_coef,
                    at_day=daily
                ).run_series(site)
            )["RMSE"].iloc[0]
            for prow in params.itertuples()
        ]).rename("RMSE")
        print("|", end="")
        return pd.concat([params, pser], axis=1).sort_values("RMSE").head(10)
    return inner


# In[36]:


outfile = bp + f"equif/EquifinalMatrix_{sys.argv[1]}.csv"
with warnings.catch_warnings(action="ignore"):
    start = time.time()
    rpdata = dev_data.loc[dev_data["id"].isin(reference.index)
                         ].groupby("id").apply(random_perf(), include_groups=False)
    print(f"Runtime: {(time.time() - start)/60: .02f} minutes for {len(rpar)} iterations over {len(reference)} sites")
    rpdata.to_csv(outfile)


# In[ ]:




