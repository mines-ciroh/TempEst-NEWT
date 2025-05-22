"""
This file defines a watershed class for provided coefficients.  It simply implements prediction logic.
"""

import rtseason as rts
import scipy
import pandas as pd
import numpy as np
import pygam
from libschema import SCHEMA
import libschema.classes as classes

def anomilize(data, obs=True):
    data["day"] = data["date"].dt.day_of_year
    if obs:
        data = data.merge(rts.ThreeSine.from_data(data[["day", "temperature"]]
                                                  ).generate_ts(), on="day")  # adds `actemp`
        data["st_anom"] = data["temperature"] - data["actemp"]
    # data["stm_anom"] = data["temperature.max"] - data["actemp"]
    data = data.merge(data.groupby("day")["tmax"].mean().rename("tmax_day"), on="day")
    data["at_anom"] = data["tmax"] - data["tmax_day"]
    return data

logalot = False

class Seasonality(classes.Seasonality):
    def __init__(self, ssn: rts.ThreeSine):
        self.ssn = ssn
        self.timeseries = ssn.generate_ts()
    
    def apply(self, period):
        return self.timeseries.iloc[period - 1]["actemp"]
    
    def apply_vec(self, period_array):
        return self.timeseries.iloc[period_array - 1]["actemp"].to_numpy()
    
    def from_dict(d):
        return Seasonality(rts.ThreeSine(**d))
    
    def to_dict(self):
        return self.ssn.to_dict()
    
class Anomaly(classes.Anomaly):
    def __init__(self, at_coef, anomgam=None, quantiles=None, anomnoise=0,
                 conv=np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ])):
        if anomgam is None and quantiles is not None:
            raise ValueError("Anomaly.__init__: If `quantiles` is set, `anomgam` must also be set.")
        self.at_coef = at_coef
        self.anomgam = anomgam
        self.anomnoise = anomnoise
        self.conv = conv
        if isinstance(quantiles, int):
            step = 1/(quantiles + 1)
            # Rounding mitigates floating point errors
            rounder = max(2, int(np.log(quantiles) / np.log(10)))
            self.quantiles = np.round(np.arange(step, 1, step), rounder)
        else:
            self.quantiles = quantiles
    
    def apply_vec(self, periodic, period, anom_history):
        temp_hist = anom_history["tmax"].to_numpy()
        temp_anom = scipy.signal.fftconvolve(
            temp_hist, self.conv, mode="full"
            )[:-(len(self.conv) - 1)] * self.at_coef
        if self.anomgam is not None:
            temp_anom = self.anomgam.predict(
                pd.DataFrame({"actemp": periodic,
                             "at_anom": temp_anom[-len(periodic):]})
                )
        # Prevent 0-degree predictions
        temp_anom[temp_anom < -periodic] = -periodic[temp_anom < -periodic]
        if self.quantiles is not None:
            anomq = self.anomgam.confidence_intervals(
                pd.DataFrame({"actemp": periodic, "at_anom": temp_anom}),
                quantiles=self.quantiles)
            for (i, q) in enumerate(self.quantiles):
                anom = anomq[:,i]
                anomq[anom < -periodic,i] = -(periodic[anom < -periodic])
            return anomq
        return temp_anom
    
    def apply(self, periodic, period, anom_history):
        res = self.apply_vec(np.array([periodic]), np.array([period]), anom_history)
        if self.quantiles is None:
            return res[0]  # just a vector
        return res[0,:]  # first row
    
    def to_dict(self):
        return {
            "at_coef": self.at_coef,
            "anomgam": self.anomgam,
            "quantiles": self.quantiles,
            "anomnoise": self.anomnoise,
            "conv": self.conv}
    
    def from_dict(d):
        return Anomaly(**d)
        

class Watershed(SCHEMA):
    basic_histcol = ["date", "day", "tmax"]
    def __init__(self,
                 seasonality: Seasonality | dict,
                 anomaly: Anomaly | dict,
                 at_day: pd.DataFrame,
                 engines: list[tuple[int, classes.ModEngine]],
                 extra_columns=[],
                 logfile=None):
        """
        seasonality: a Seasonaly object or dictionary of {Intercept, Amplitude,
                     SpringSummer, FallWinter, SpringDay, SummerDay, FallDay,
                     WinterDay}.
        anomaly: an Anomaly object or dictionary of {at_coef} with optional
                 {anomgam, quantiles, anomnoise, conv}.
        at_day: data frame of [day, mean_tmax] or [period, tmax].
        engines: list of [(frequency, classes.ModEngine)]
            Modification engines to apply at specified frequencies.
        extra_columns: names of added columns to include in model
            history (e.g., for use by modification engines).  All columns
            specified must be provided for each step or specified through
            setters.
        """
        if type(seasonality) == dict:
            seasonality = Seasonality(**seasonality)
        if type(anomaly) == dict:
            anomaly = Anomaly(**anomaly)
        at_day = at_day.rename(columns={"day": "period",
                                        "mean_tmax": "tmax"})
        super().__init__(seasonality,
                       anomaly,
                       at_day,
                       engines,
                       self.basic_histcol + extra_columns,
                       max_period=365,
                       window=6,
                       logfile=logfile)
        if anomaly.quantiles is not None:
            for qn in anomaly.quantiles:
                self.history[f"output_{qn}"] = []
    
    def coefs_to_df(self):
        return pd.DataFrame([self.to_dict()])

    def run_step(self, inputs=None, period=None):
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        step = super().run_step(inputs, period)
        # If there are quantiles, then the prediction is a vector, not a single value.
        # In that case, history["output"] just got a vector appended, and the
        # step above is a vector.
        # Returning the vector is reasonable, but we need to fix the history.
        if self.anomaly.quantiles is not None:
            for (i, qn) in enumerate(self.anomaly.quantiles):
                self.history[f"output_{qn}"].append(step[i])
            # Use the mean for the single-output column.
            self.history["output"][-1] = np.mean(self.history["output"][-1])
        return step

    def run_series(self, data, context=True):
        """
        Run a full timeseries at once.
        data must have columns date (as an actual date type), tmax.
        Will be returned with new columns day, actemp, anom, temp.mod
        This runs things all at once, so it's much faster, but only works without engines.
        If engines are present, it switches to an incremental run.
        Returns the predicted array (if context=False) or the data with an added prediction
        column (if context=True).
        """
        data["day"] = data["date"].dt.day_of_year
        (result, outdata) = super().run_series(data, "date", period_col="day")
        if self.anomaly.quantiles is None:
            if context:
                return outdata
            return result
        # If we have quantiles, then outdata has a nonsensical prediction
        # column, and result is a matrix, not a column. Returning result
        # as-is is reasonable, but we should add names to it.
        names = [f"prediction_{qn}" for qn in self.anomaly.quantiles]
        result = pd.DataFrame(result, columns=names)
        outdata = pd.concat([outdata.drop(columns=["prediction"]),
                             result])
        if context:
            return outdata
        return result

    def from_data(data,
                  anomgam=None,
                  at_conv=np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ]),
                  extra_columns=[],
                  use_anomgam=True,
                  quantiles=None,
                  **kwargs):
        """
        Train a watershed model from data.
        Requires columns date, temperature, tmax
        """
        data["day"] = data["date"].dt.day_of_year
        if len(data["day"].unique()) < 181:
            return None
        anoms = anomilize(data)
        at_day = data.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        ssn = rts.ThreeSine.from_data(data[["day", "temperature"]])
        anoms["anom_atmod"] = scipy.signal.fftconvolve(anoms["at_anom"],
                                                       at_conv, mode="full")[:-(len(at_conv) - 1)]
        sol = np.linalg.lstsq(np.array(anoms[["anom_atmod"
                                              # ,"anom_hummod"
                                              ]]), anoms["st_anom"].to_numpy().transpose(), rcond=None)[0]
        at_coef = sol[0]
        anomnoise = 0
        if anomgam is None and use_anomgam:
            X = anoms[["actemp", "anom_atmod"]].copy()
            X["anom_atmod"] *= at_coef
            y = anoms["st_anom"]
            anomgam = pygam.LinearGAM(pygam.te(0, 1)).fit(X, y)
            anomnoise = np.sqrt(np.mean((anomgam.predict(X) - y)**2))
        return Watershed(Seasonality(ssn),
                         Anomaly(at_coef, anomgam, quantiles, anomnoise, at_conv),
                         at_day,
                         [],
                         extra_columns,
                         **kwargs)
