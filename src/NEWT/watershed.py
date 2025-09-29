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
from yaml import load, Loader

def anomilize(data: pd.DataFrame, obs=True) -> pd.DataFrame:
    """
    Add anomaly information, comparing the air and optionally water temperature
    timeseries to their observed baselines.

    Parameters
    ----------
    data : DataFrame
        The input dataframe. Columns: date, tmax. Also temperature if obs is True.
    obs : Bool, optional
        Whether the data includes observed temperature. The default is True.

    Returns
    -------
    data : DataFrame
        Data frame with added columns: day, at_day and at_anom (air temperature),
        optionally actemp and st_anom (water temperature).

    """
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
    def __init__(self, ssn: rts.ThreeSine, quantiles=None):
        self.ssn = ssn
        self.timeseries = ssn.generate_ts()
        self.quantiles = quantiles
    
    def apply(self, period: int):
        res = self.timeseries.iloc[period - 1]["actemp"]
        if self.quantiles is None:
            return res
        # So the dimensions agree
        return np.array([res] * self.quantiles)
    
    def apply_vec(self, period_array: np.array):
        res = self.timeseries.iloc[period_array - 1]["actemp"].to_numpy()
        if self.quantiles is None:
            return res
        # So the dimensions agree
        return np.tile(np.array([res]).transpose(), (1, self.quantiles))
    
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
        if self.quantiles is not None:
            # Make sure exactly 0 or exactly 1 aren't included.
            corrector = lambda c: 0.001 if c==0 else 0.999 if c==1 else c
            self.quantiles = [corrector(c) for c in self.quantiles]
    
    def apply_vec(self, periodic, period, anom_history):
        if self.quantiles is not None:
            periodic = periodic[:,0]  # it will be a matrix
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
        if self.quantiles is not None:
            periodic = periodic[0]  # it will be a vector
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
    """
    Parameters
    ----------
    seasonality : Seasonality | dict
        a Seasonaly object or dictionary of {Intercept, Amplitude,
        SpringSummer, FallWinter, SpringDay, SummerDay, FallDay,
        WinterDay}.
    anomaly : Anomaly | dict
        an Anomaly object or dictionary of {at_coef} with optional
        {anomgam, quantiles, anomnoise, conv}.
    at_day : pd.DataFrame
        data frame of [day, mean_tmax] or [period, tmax].
    engines : list[tuple[int, classes.ModEngine]]
        Modification engines to apply at specified frequencies.
    extra_columns : list of strings, optional
        history (e.g., for use by modification engines).  All columns
        specified must be provided for each step or specified through
        setters. The default is [].
    logfile : str, optional
        Where to store logs, if anywhere. The default is None.
    """
    basic_histcol = ["date", "day", "tmax"]
    def __init__(self,
                 seasonality: Seasonality | dict,
                 anomaly: Anomaly | dict,
                 at_day: pd.DataFrame,
                 engines: list[tuple[int, classes.ModEngine]],
                 extra_columns: list[str]=[],
                 logfile: str=None,
                 **kwargs):
        """
        Initialize a Watershed object.

        Parameters
        ----------
        seasonality : Seasonality | dict
            a Seasonaly object or dictionary of {Intercept, Amplitude,
            SpringSummer, FallWinter, SpringDay, SummerDay, FallDay,
            WinterDay}.
        anomaly : Anomaly | dict
            an Anomaly object or dictionary of {at_coef} with optional
            {anomgam, quantiles, anomnoise, conv}.
        at_day : pd.DataFrame
            data frame of [day, mean_tmax] or [period, tmax].
        engines : list[tuple[int, classes.ModEngine]]
            Modification engines to apply at specified frequencies.
        extra_columns : list of strings, optional
            history (e.g., for use by modification engines).  All columns
            specified must be provided for each step or specified through
            setters. The default is [].
        logfile : str, optional
            Where to store logs, if anywhere. The default is None.

        Returns
        -------
        None.

        """
        if type(seasonality) == dict:
            seasonality = Seasonality(**seasonality)
        if type(anomaly) == dict:
            anomaly = Anomaly(**anomaly)
        if anomaly.quantiles is not None:
            seasonality.quantiles = len(anomaly.quantiles)
        at_day = at_day.rename(columns={"day": "period",
                                        "mean_tmax": "tmax"})
        # Exclude all specified arguments from kwargs to avoid duplicates
        for nm in ["seasonality", "anomaly", "periodics", "engines",
                   "columns", "max_period", "window", "logfile"]:
            if nm in kwargs:
                kwargs.pop(nm)
        self.log("Initializing")
        super().__init__(seasonality,
                       anomaly,
                       at_day,
                       engines,
                       self.basic_histcol + extra_columns,
                       max_period=365,
                       window=6,
                       logfile=logfile,
                       **kwargs)

    def init_with_schema_names(**kwargs):
        # Switch variable names to work with LibSCHEMA args.
        kwargs["at_day"] = kwargs["periodics"]
        return Watershed(**kwargs)
    
    def from_file(filename: str):
        # Watershed uses different arguments from SCHEMA, but they are align-able.
        with open(filename) as f:
            coefs = load(f, Loader)
        return Watershed.init_with_schema_names(**coefs)
    
    def initialize_run(self, period: int):
        self.log("Initializing run")
        super().initialize_run(period)
        if self.anomaly.quantiles is not None:
            for qn in self.anomaly.quantiles:
                self.history[f"output_{qn}"] = []
        self.log("Initialized run")
    
    def coefs_to_df(self):
        return pd.DataFrame([self.to_dict()])

    def run_step(self, inputs=None, period=None):
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        self.log("Running step")
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
        self.log("Ran step")
        return step

    def run_series(self, data, context=True):
        """
        Run a full timeseries at once.
        data must have columns date (as an actual date type), day, tmax.
        Will be returned with new columns day, actemp, anom, temp.mod
        This runs things all at once, so it's much faster, but only works without engines.
        If engines are present, it switches to an incremental run.
        Returns the predicted array (if context=False) or the data with an added prediction
        column (if context=True).
        """
        data["day"] = data["date"].dt.day_of_year
        if self.anomaly.quantiles is None:
            return super().run_series(data, "date", period_col="day", context=context)
        # If we have quantiles, then outdata has a nonsensical prediction
        # column, and result is a matrix, not a column. Returning result
        # as-is is reasonable, but we should add names to it.
        result = super().run_series(data, "date", period_col="day", context=False)
        names = [f"prediction_{qn}" for qn in self.anomaly.quantiles]
        result = pd.DataFrame(result, columns=names)
        if context:
            data.index = range(len(data))
            result.index = range(len(result))
            return pd.concat([data, result], axis=1)
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
