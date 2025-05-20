"""
This file defines a watershed class for provided coefficients.  It simply implements prediction logic.
"""

import rtseason as rts
import scipy
import pandas as pd
import numpy as np
from yaml import load, dump, Loader
from datetime import timedelta
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
    def __init__(self, sensitivity, anomgam=None, quantiles=None, anomnoise=0,
                 conv=np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ])):
        if anomgam is None and quantiles is not None:
            raise ValueError("Anomaly.__init__: If `quantiles` is set, `anomgam` must also be set.")
        self.sensitivity = sensitivity
        self.anomgam = anomgam
        self.conv = conv
        if isinstance(quantiles, int):
            step = 1/(quantiles + 1)
            # Rounding mitigates floating point errors
            rounder = max(2, int(np.log(quantiles) / np.log(10)))
            self.quantiles = np.round(np.arange(step, 1, step), rounder)
        else:
            self.quantiles = quantiles
    
    def apply(self, periodic, period, anom_history):
        temp_hist = anom_history["tmax"].to_numpy()
        

class Watershed(SCHEMA):
    basic_histcol = ["date", "day", "at", "actemp", "anom", "temp.mod"]
    # TODO: refactor init to use anomaly arguments, not just anomaly object.
    def __init__(self, seasonality, anomaly, at_day,
                extra_history_columns=[],
                logfile=None):
        """
        seasonality: a three-sine seasonality object
        at_coef: air temperature anomaly coefficient
        at_day: data frame of [day, mean_tmax]
        at_conv: air temperature anomaly convolution
        engines and periods: modification engines and recurrence periods in days
            which update model coefficients.
            dynamic_engine is for short-term adjustments at daily or near-daily
                resolution, for example to adjust sensitivity.
            year_engine (which recurs at a set day-of-year, not frequency)
                is for annual adjustments, like changing seasonality for
                a wet/dry year.
            climate_engine is for updating watershed coefficients for
                climate (atmospheric conditions).  It may also accept
                a learning rate, or rate to approach atmospheric-driven
                coefficients vs static coefficients, and a recency weight,
                which drives how responsive the watershed is to changing
                atmospheric conditions.
            An engine must implement the engines.ModEngine class.
        extra_history_columns: names of added columns to include in model
            history (e.g., for use by modification engines).  All columns
            specified must be provided for each step or specified through
            setters.
        anomgam: a trained GAM that takes seasonal temperature and
            (smoothed + weighted) anomaly to predict adjusted ST anomaly.
            If not provided, adjustment will not be applied.
        anomnoise: a float quantifying anomaly prediction noisiness.
        intervals: a list of quantiles to predict instead of a single value.
            Alternatively, an integer number of quantiles to predict.  An odd number
            is recommended to include 0.5 (median).
            This applies to anomaly only, NOT seasonality.  The model must
            be using an anomgam.
        """
        # TODO: not done refactoring this.
        super.__init__(self, seasonality, anomaly, ...)
        self.seasonality = seasonality
        self.anomaly = anomaly
        self.dailies = at_day
        self.period = 0
        self.date = None
        self.histcol = [c for c in extra_history_columns if not c in
                        self.basic_histcol]
        self.logfile = logfile
        quantiles = anomaly.quantiles
        if quantiles is not None:
            for nm in ["anom", "temp.mod"]:
                for q in quantiles:
                    self.histcol.append(
                        f"{nm}_{q}"
                    )
    
    def coefs_to_df(self):
        base = self.seasonality.to_df().assign(
            at_coef = self.at_coef
            )
        for engine in [self.dynamic_engine, self.year_engine,
                       self.climate_engine]:
            if engine is not None:
                coefs = engine.coefficients()
                for k in coefs:
                    base[k] = coefs[k]
        return base
    
    def log(self, text, reset=False):
        if self.logfile is not None:
            with open(self.logfile, "w" if reset else "a") as f:
                f.write(text + "\n")
    
    def get_history(self):
        return pd.DataFrame(self.history)

    # BMI: Getters and setters
    def get_at(self):
        return self.at
    def set_at(self, at):
        self.at = at
    def get_st(self):
        return self.temperature
    def get_date(self):
        return self.date
    def set_date(self, date):
        self.date = pd.to_datetime(date)
    def set_extra(self, key, value):
        self.extras[key] = value
    def get_extras(self):
        return self.extras

    def step(self, date=None, at=None, extras=None):
        # TODO: take this out, but work in the anomaly function first.
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        if logalot:
            self.log("Began step")
        for k in self.histcol:
            if (not k in extras) and (self.extras[k] is None):
                raise ValueError(f"In step, must provide all specified extra data. Missing: {k}")
        if extras is not None:
            self.extras = extras
        self.date = self.date + timedelta(1) if date is None else pd.to_datetime(date)
        self.doy = self.date.day_of_year
        today = self.dailies[self.dailies["day"] == self.doy]
        at = at if at is not None else self.at
        # prcp = prcp if prcp is not None else self.prcp
        # swe = swe if swe is not None else self.swe
        # srad = srad if srad is not None else self.srad
        # "logs" allow efficient processing without having to grab the whole
        # history.
        self.at_log.append(at - today["mean_tmax"].iloc[0])
        # Cut off beginning of logs if they're too long.
        if len(self.at_log) > len(self.at_conv):
            self.at_log = self.at_log[-len(self.at_conv):]
        # Now, build the prediction
        ssn = self.ssn_timeseries["actemp"][
            self.ssn_timeseries["day"] == self.doy].iloc[0]
        at_anom = np.convolve(self.at_log, self.at_conv, mode="valid")[-1]
        anom = at_anom * self.at_coef
        if self.anomgam is not None:
            anom = self.anomgam.predict(np.array([[ssn, anom]]))[0]
        if self.quantiles is not None:
            # Function uncertainty
            anomq = self.anomgam.confidence_intervals(np.array([[ssn, anom]]),
                                                      quantiles=self.quantiles)[0,:]
            # Noise
            anomq += scipy.stats.norm.ppf(self.quantiles) * self.anomnoise
            anomq[anomq < -ssn] = -ssn
            predq = anomq + ssn
            for (nm, vals) in [("anom", anomq), ("temp.mod", predq)]:
                for (i, q) in enumerate(self.quantiles):
                    self.history[f"{nm}_{q}"].append(vals[i])
        pred = ssn + anom
        pred = pred if pred >= 0 else 0
        self.temperature = pred
        # Update history
        self.history["date"].append(self.date)
        self.history["day"].append(self.doy)
        self.history["at"].append(at)
        self.history["actemp"].append(ssn)
        self.history["anom"].append(anom)
        self.history["temp.mod"].append(pred)            
        for k in self.histcol:
            self.history[k].append(self.extras[k])
        # self.history["swe"].append(swe)
        # self.history["prcp"].append(prcp)
        # self.history["srad"].append(srad)
        # Run triggers
        self.period += 1
        if (self.climate_engine is not None and
            self.period % self.climate_period == 0):
            self.trigger_engine(self.climate_engine)
        if (self.year_engine is not None and
            self.doy == self.year_doy):
            self.trigger_engine(self.year_engine)
        if (self.dynamic_engine is not None and 
            self.period % self.dynamic_period == 0):
            self.trigger_engine(self.dynamic_engine)
        self.timestep += 86400  # seconds per day
        if logalot:
            self.log(f"Concluded step; predicted ST: {self.temperature}")
        # Result
        if self.quantiles is None:
            return pred
        else:
            return predq
    
    def run_series_incremental(self, data):
        """
        Run a full timeseries at once, but internally use the stepwise approach.
        data must have columns date (as an actual date type), tmax.
        Will be returned with columns date, day, at, actemp, anom, temp.mod
        """
        self.initialize_run()
        for row in data.itertuples():
            extras = {k: getattr(row, k) for k in self.histcol}
            yield self.step(row.date, row.tmax, extras)
        

    def run_series(self, data):
        """
        Run a full timeseries at once.
        data must have columns date (as an actual date type), tmax.
        Will be returned with new columns day, actemp, anom, temp.mod
        This runs things all at once, so it's much faster, but ignores engines.
        """
        # _ = list(self.run_series_incremental(data))
        # res = self.get_history()[["date", "actemp", "anom", "temp.mod"]]
        # return data.merge(res, on="date")
        data = data.copy()
        data["day"] = data["date"].dt.day_of_year
        anoms = anomilize(data, obs=False)
        data = data.merge(self.ssn_timeseries[["day", "actemp"]], on="day", how="left").merge(anoms[["date", "at_anom"]], on="date", how="left")
        data["at_anom"] = scipy.signal.fftconvolve(data["at_anom"], self.at_conv, mode="full")[:-(len(self.at_conv) - 1)] * self.at_coef
        if self.anomgam is not None:
            data["anom"] = self.anomgam.predict(data[["actemp", "at_anom"]])
        else:
            data["anom"] = data["at_anom"]
        data["temp.mod"] = data["actemp"] + data["anom"]
        data.loc[data["temp.mod"] < 0, "temp.mod"] = 0
        if self.quantiles is not None:
            anomq = self.anomgam.confidence_intervals(data[["actemp", "at_anom"]],
                                                      quantiles=self.quantiles)
            for (i, q) in enumerate(self.quantiles):
                anomcol = f"anom_{q}"
                predcol = f"temp.mod_{q}"
                ssn = data["actemp"]
                anom = anomq[:,i]
                anom[anom < -ssn] = -(ssn[anom < -ssn])
                data[anomcol] = anom
                data[predcol] = ssn + anom
        return data.drop(columns=["at_anom"])

    def from_data(data,
                  anomgam=None,
                  at_conv=np.array([0.132, 0.401, 0.162, 0.119, 0.056, 0.13 ]),
                  extra_history_columns=[],
                  use_anomgam=True,
                  **kwargs):
        """
        Train a watershed model from data.
        Requires columns date, temperature, tmax
        threshold_engine: fit threshold sensitivity engine?
        lin_ssn: fit linear seasonality engine?
        names, xs, start, until: passed to linear seasonality trainer
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
        return Watershed(ssn, at_coef, at_day, at_conv,
                         extra_history_columns=extra_history_columns,
                         anomgam=anomgam, anomnoise=anomnoise, **kwargs)
