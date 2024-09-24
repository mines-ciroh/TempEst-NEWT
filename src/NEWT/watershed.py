"""
This file defines a watershed class for provided coefficients.  It simply implements prediction logic.
"""

import rtseason as rts
import scipy
import pandas as pd
import numpy as np
from yaml import load, dump, Loader
from datetime import timedelta
from NEWT import engines, analysis
from NEWT.engines import ModEngine

def anomilize(data):
    data["day"] = data["date"].dt.day_of_year
    data = data.merge(rts.ThreeSine.from_data(data[["day", "temperature"]]
                                              ).generate_ts(), on="day")  # adds `actemp`
    data["st_anom"] = data["temperature"] - data["actemp"]
    # data["stm_anom"] = data["temperature.max"] - data["actemp"]
    data = data.merge(data.groupby("day")["tmax"].mean().rename("tmax_day"), on="day")
    data["at_anom"] = data["tmax"] - data["tmax_day"]
    return data


def ws_to_data(ssn, date, at_coef, at_day, at_conv, dyn_eng, dyn_period,
               year_eng, year_period, climate_eng, climate_period,
               ext_hist, history):
    """
    Converts watershed data to a file in a consistent format.
    """
    ssn = {k: float(v[0]) for k, v in ssn.to_df().items()
           if not k in ["R2", "RMSE"]}
    atd_str = at_day[["day", "mean_tmax"]].to_dict()
    at_coef = float(at_coef)
    at_conv = at_conv.tolist()
    data = {
        "seasonality": ssn,
        "date": str(date),
        "at_coef": at_coef,
        "at_conv": at_conv,
        "at_day": atd_str,
        "history": history.to_dict(),
        "dynamic_engine": dyn_eng.to_dict() if dyn_eng is not None else None,
        "dynamic_period": dyn_period,
        "year_engine": year_eng.to_dict() if year_eng is not None else None,
        "year_period": year_period,
        "climate_engine": climate_eng.to_dict() if climate_eng is not None else None,
        "climate_period": climate_period,
        "ext_hist_col": ext_hist
        }
    return data


def ws_from_data(coefs):
    """
    Generates watershed from coefficients dictionary.
    """
    ws = Watershed(rts.ThreeSine.from_coefs(pd.DataFrame(coefs["seasonality"], index=[0])),
                   at_coef=coefs["at_coef"],
                   at_day=pd.DataFrame(coefs["at_day"]),
                   at_conv=np.array(coefs["at_conv"]),
                   dynamic_engine=ModEngine.from_dict(coefs["dynamic_engine"]) if coefs["dynamic_engine"] is not None else None,
                   dynamic_period=coefs["dynamic_period"],
                   year_engine=ModEngine.from_dict(coefs["year_engine"]) if coefs["year_engine"] is not None else None,
                   year_doy=coefs["year_period"],
                   climate_engine=ModEngine.from_dict(coefs["climate_engine"]) if coefs["climate_engine"] is not None else None,
                   climate_period=coefs["climate_period"],
                   extra_history_columns=coefs["ext_hist_col"]
                   )
    ws.history = pd.DataFrame(coefs["history"])
    ws.date = np.datetime64(coefs["date"])
    return ws
    


class Watershed(object):
    def __init__(self, seasonality, at_coef, at_day,
                at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                dynamic_engine=None,
                dynamic_period=1,
                year_engine=None,
                year_doy=180,
                climate_engine=None,
                climate_period=365,
                extra_history_columns=[]):
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
        """
        self.seasonality = seasonality
        self.ssn_timeseries = seasonality.generate_ts()  # day, actemp
        self.at_coef = at_coef
        self.dailies = at_day
        self.at_conv = at_conv
        self.statics = {
            "seasonality": seasonality,
            "at_coef": at_coef,
            "dailies": self.dailies.copy()
            }
        self.dynamic_engine = dynamic_engine
        self.dynamic_period = dynamic_period
        self.year_engine = year_engine
        self.year_doy = year_doy
        self.climate_engine = climate_engine
        self.climate_period = climate_period
        self.period = 0
        self.date = None
        self.histcol = extra_history_columns
    
    def from_file(filename, init=False, estimator=None):
        with open(filename) as f:
            coefs = load(f, Loader)
        ws = ws_from_data(coefs)
        if init:
            ws.initialize_run(coefs["date"])
        return ws
    
    def to_file(self, filename):
        data = ws_to_data(self.seasonality, self.date, self.at_coef,
                          self.dailies, self.at_conv, self.dynamic_engine,
                          self.dynamic_period, self.year_engine,
                          self.year_doy, self.climate_engine,
                          self.climate_period, self.histcol,
                          self.get_history())
        with open(filename, "w") as f:
            dump(data, f)
    
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

    def initialize_run(self, start=None):
        # Logs allow efficient handling of a rolling anomaly
        self.at_log = []
        self.at = None
        self.temperature = None
        self.timestep = 0
        self.date = start
        self.extras = {x: None for x in self.histcol}
        self.history = {
            "date": [],
            "day": [],
            "at": [],
            "actemp": [],
            "anom": [],
            "temp.mod": []
            } | {x: [] for x in self.histcol}
    
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
        self.date = date
    def set_extra(self, key, value):
        self.extras[key] = value
    def get_extras(self):
        return self.extras
        
    def trigger_engine(self, engine):
        """
        For dynamic modification of model coefficients, we may trigger a
        modification engine.  Triggering itself is handled in the stepper
        function.

        Here, we update coefficients as specified by the engine, then
        regenerate anything as necessary.  The engine may modify seasonality
        coefficients and sensitivity coefficients, as well as seasonality
        baselines.
        """
        blend_window = 60
        (self.seasonality, self.at_coef, self.dailies) =\
            engine.apply(self.seasonality, self.at_coef, self.dailies,
                    self.get_history(), self.statics)
        # We want to smoothly blend it in, rather than an abrupt jump
        ssnts = self.ssn_timeseries.rename(columns={"actemp": "oldtemp"}).merge(
            self.seasonality.generate_ts(), on="day"
            )
        ssnts["weights"] = 1.0
        window_overlap = self.doy + blend_window > 366
        if window_overlap:
            pos = (ssnts["day"] >= self.doy) | (ssnts["day"] < (self.doy + blend_window) % 366)
        else:
            pos = (ssnts["day"] >= self.doy) & (ssnts["day"] < self.doy + blend_window)
        ssnts.loc[pos,
                          "weights"] =\
            np.arange(blend_window) / blend_window
        ssnts["actemp"] = (ssnts["actemp"] * ssnts["weights"] +
                            ssnts["oldtemp"] * (1-ssnts["weights"]))
        self.ssn_timeseries = ssnts.drop(columns=["oldtemp", "weights"])
    
    def reset_coefs(self):
        """
        Reset coefficients, removing effects of any modification engines.
        """
        self.seasonality = self.statics["seasonality"]
        self.at_coef = self.statics["at_coef"]
        self.dailies = self.statics["dailies"].copy()
        self.ssn_timeseries = self.seasonality.generate_ts()

    def step(self, date=None, at=None, extras=None):
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        for k in self.histcol:
            if (not k in extras) and (self.extras[k] is None):
                raise ValueError(f"In step, must provide all specified extra data. Missing: {k}")
        if extras is not None:
            self.extras = extras
        self.date = self.date + timedelta(1) if date is None else date
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
        # Result
        return pred
    
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
        """
        _ = list(self.run_series_incremental(data))
        res = self.get_history()[["date", "actemp", "anom", "temp.mod"]]
        return data.merge(res, on="date")

    def from_data(data,
                  threshold_engine=True,
                  lin_ssn=False,
                  names=["Intercept", "Amplitude", "SummerDay"],
                  xs=["at"],
                  start=330,
                  until=30,
                  at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                  extra_history_columns=[]):
        """
        Train a watershed model from data.
        Requires columns date, temperature, tmax
        threshold_engine: fit threshold sensitivity engine?
        lin_ssn: fit linear seasonality engine?
        names, xs, start, until: passed to linear seasonality trainer
        """
        linear_ssn = (
            engines.LinearSeasonEngine.from_data(
                data.rename(columns={"tmax": "at"}),
                                      names, xs, start, until) if
            lin_ssn else None)
        data["day"] = data["date"].dt.day_of_year
        anoms = anomilize(data)
        at_day = data.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        ssn = rts.ThreeSine.from_data(data[["day", "temperature"]])
        uncal_prd = Watershed(ssn, 0, at_day,
                              year_engine=linear_ssn, year_doy=until,
                              extra_history_columns=extra_history_columns).\
            run_series(data) if lin_ssn else None
        anoms = anoms.drop(columns=["temperature", "st_anom"]).merge(
            uncal_prd[["date", "temp.mod", "temperature"]], on="date").assign(
                st_anom = lambda x: x["temperature"] - x["temp.mod"]) if lin_ssn else anoms
        anoms["anom_atmod"] = scipy.signal.fftconvolve(anoms["at_anom"],
                                                       at_conv, mode="full")[:-(len(at_conv) - 1)]
        thres_eng = engines.ThresholdSensitivityEngine.from_data(
            anoms, use_at=False) if threshold_engine else None
        sol = np.linalg.lstsq(np.array(anoms[["anom_atmod"
                                              # ,"anom_hummod"
                                              ]]), anoms["st_anom"].to_numpy().transpose(), rcond=None)[0]
        at_coef = sol[0]
        return Watershed(ssn, at_coef, at_day, at_conv,
                         year_engine=linear_ssn, year_doy=until,
                         dynamic_engine=thres_eng, dynamic_period=7,
                         extra_history_columns=extra_history_columns)
