"""
This file defines a watershed class for provided coefficients.  It simply implements prediction logic.
"""

import rtseason as rts
import scipy
import pandas as pd
import numpy as np
from datetime import timedelta

def anomilize(data):
    data["day"] = data["date"].dt.day_of_year
    data = data.merge(rts.ThreeSine.from_data(data).generate_ts(), on="day")  # adds `actemp`
    data["st_anom"] = data["temperature"] - data["actemp"]
    # data["stm_anom"] = data["temperature.max"] - data["actemp"]
    data = data.merge(data.groupby("day")["tmax"].mean().rename("tmax_day"), on="day")
    data["at_anom"] = data["tmax"] - data["tmax_day"]
    data = data.merge(data.groupby("day")["vp"].mean().rename("vp_day"), on="day")
    data["vp_anom"] = data["vp"] - data["vp_day"]
    return data

def trycatch(op):
    try:
        return op()
    except ValueError:
        return None

def makenp(data):
    data = data.to_numpy()
    return np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

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
    weather = ssn.merge(
        data[data["day"] >= start & data["day"] <= until].
        groupby("year")[xs].mean(),
        on="year")
    sol = np.linalg.lstsq(makenp(weather[xs]), makenp(ssn[names]),
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
        history = history[
            history["date"].dt.year == history["date"].dt.year.max() &
            history["day"] >= start &
            history["day"] <= until
            ]
        varbs = makenp(history[xs])
        preds = pd.DataFrame(varbs @ coefficients, columns=names)
        new_ssn = rts.ThreeSine.from_coefs(pd.concat([
            seasonality.to_df().drop(columns=names),
            preds
            ], axis=1))
        return (new_ssn, at_coef, vp_coef, dailies)
    return f


class Watershed(object):
    def __init__(self, seasonality, at_coef, vp_coef, at_day, vp_day,
                at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                vp_conv=[1, 1],
                dynamic_engine=None,
                dynamic_period=1,
                year_engine=None,
                year_doy=180,
                climate_engine=None,
                climate_period=365,
                climate_learnrate=0,
                climate_recency=0):
        """
        seasonality: a three-sine seasonality object
        at_coef: air temperature anomaly coefficient
        vp_coef: vapor pressure anomaly coefficient
        at_day: data frame of [day, mean_tmax]
        vp_day: data frame of [day, mean_vp]
        at_conv, vp_conv: air temperature and vapor pressure anomaly convolutions
        engines and periods: functions (engines) and recurrence periods in days
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
            An engine function must accept seasonality, at_coef, vp_coef,
                dailies, history (data frame), statics;
                and return seasonality, at_coef, vp_coef, dailies
        """
        self.seasonality = seasonality
        self.ssn_timeseries = seasonality.generate_ts()  # day, actemp
        self.at_coef = at_coef
        self.vp_coef = vp_coef
        self.dailies = at_day.merge(vp_day, on="day")
        self.at_conv = at_conv
        self.vp_conv = vp_conv
        self.statics = {
            "seasonality": seasonality,
            "at_coef": at_coef,
            "vp_coef": vp_coef,
            "dailies": self.dailies.copy()
            }
        self.dynamic_engine = dynamic_engine
        self.dynamic_period = dynamic_period
        self.year_engine = year_engine
        self.year_doy = year_doy
        self.climate_engine = climate_engine
        self.climate_period = climate_period
        self.climate_learnrate = climate_learnrate
        self.climate_recency = climate_recency
        self.period = 0

    def initialize_run(self):
        # Logs allow efficient handling of a rolling anomaly
        self.at_log = []
        self.vp_log = []
        self.history = {
            "date": [],
            "day": [],
            "at": [],
            "vp": [],
            "actemp": [],
            "anom": [],
            "temp.mod": []
            }
    
    def get_history(self):
        return pd.DataFrame(self.history)

    # BMI: Getters and setters
    def get_at(self):
        return self.at
    def set_at(self, at):
        self.at = at
    def get_vp(self):
        return self.vp
    def set_vp(self, vp):
        self.vp = vp
    def get_st(self):
        return self.temperature
    def get_date(self):
        return self.date
    def set_date(self, date):
        self.date = date
        
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
        (self.seasonality, self.at_coef, self.vp_coef, self.dailies) =\
            engine(self.seasonality, self.at_coef, self.vp_coef, self.dailies,
                   self.get_history(), self.statics)
        self.ssn_timeseries = self.seasonality.generate_ts()
    
    def reset_coefs(self):
        """
        Reset coefficients, removing effects of any modification engines.
        """
        self.seasonality = self.statics["seasonality"]
        self.at_coef = self.statics["at_coef"]
        self.vp_coef = self.statics["vp_coef"]
        self.dailies = self.statics["dailies"].copy()
        self.ssn_timeseries = self.seasonality.generate_ts()

    def step(self, date=None, at=None, vp=None):
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        self.date = date if date is not None else self.date + timedelta(1)
        self.doy = self.date.day_of_year
        today = self.dailies[self.dailies["day"] == self.doy]
        at = at if at is not None else self.at
        vp = vp if vp is not None else self.vp
        # "logs" allow efficient processing without having to grab the whole
        # history.
        self.at_log.append(at - today["mean_tmax"].iloc[0])
        self.vp_log.append(vp - today["mean_vp"].iloc[0])
        # Cut off beginning of logs if they're too long.
        if len(self.at_log) > len(self.at_conv):
            self.at_log = self.at_log[-len(self.at_conv):]
        if len(self.vp_log) > len(self.vp_conv):
            self.vp_log = self.vp_log[-len(self.vp_conv):]
        # Now, build the prediction
        ssn = self.ssn_timeseries["actemp"][
            self.ssn_timeseries["day"] == self.doy].iloc[0]
        at_anom = np.convolve(self.at_log, self.at_conv, mode="valid")[-1]
        vp_anom = np.convolve(self.vp_log, self.vp_conv, mode="valid")[-1]
        anom = at_anom * self.at_coef + vp_anom * self.vp_coef
        pred = ssn + anom
        self.temperature = pred
        # Update history
        self.history["date"].append(self.date)
        self.history["day"].append(self.doy)
        self.history["at"].append(at)
        self.history["vp"].append(vp)
        self.history["actemp"].append(ssn)
        self.history["anom"].append(anom)
        self.history["temp.mod"].append(pred)
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
        # Result
        return pred
    
    def run_series_incremental(self, data):
        """
        Run a full timeseries at once, but internally use the stepwise approach.
        data must have columns date (as an actual date type), tmax, vp.
        Will be returned with columns date, day, at, vp, actemp, anom, temp.mod
        """
        self.initialize_run()
        for row in data.itertuples():
            yield self.step(row.date, row.tmax, row.vp)
        

    def run_series(self, data):
        """
        Run a full timeseries at once.
        data must have columns date (as an actual date type), tmax, vp.
        Will be returned with new columns day, actemp, anom, temp.mod
        """
        self.run_series_incremental(data)
        res = self.get_history()[["date", "day", "actemp", "anom", "temp.mod"]]
        return data.merge(res, on="date")

    def from_data(data,
                  lin_ssn=True,
                  names=["Intercept", "Amplitude", "SummerDay"],
                  xs=["prcp"],
                  start=1,
                  until=180,
                  at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                  vp_conv=[1, 1]):
        """
        Train a watershed model from data.
        Requires columns date, temperature, tmax, vp
        lin_ssn: fit linear seasonality engine?
        names, xs, start, until: passed to linear seasonality trainer
        """
        linear_ssn = (
            make_linear_season_engine(data, names, xs, start, until) if lin_ssn
            else None)
        anoms = anomilize(data)
        at_day = data.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        vp_day = data.groupby(["day"], as_index=False)["vp"].mean().rename(columns={"vp": "mean_vp"})
        ssn = rts.ThreeSine.from_data(data)
        anoms["anom_atmod"] = scipy.signal.fftconvolve(anoms["at_anom"], at_conv, mode="full")[:-(len(at_conv) - 1)]
        anoms["anom_hummod"] = scipy.signal.fftconvolve(anoms["vp_anom"], vp_conv, mode="full")[:-(len(vp_conv) - 1)]
        sol = np.linalg.lstsq(np.array(anoms[["anom_atmod", "anom_hummod"]]), anoms["st_anom"].to_numpy().transpose(), rcond=None)[0]
        at_coef = sol[0]
        vp_coef = sol[1]
        return Watershed(ssn, at_coef, vp_coef, at_day, vp_day, at_conv, vp_conv,
                         year_engine=linear_ssn, year_doy=until)
