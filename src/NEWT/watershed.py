"""
This file defines a watershed class for provided coefficients.  It simply implements prediction logic.
"""

import rtseason as rts
import scipy
import pandas as pd
import numpy as np

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


class Watershed(object):
    def __init__(self, seasonality, at_coef, vp_coef, at_day, vp_day,
                at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                vp_conv=[1, 1]):
        """
            seasonality: a three-sine seasonality object
            at_coef: air temperature anomaly coefficient
            vp_coef: vapor pressure anomaly coefficient
            at_day: data frame of [day, mean_tmax]
            vp_day: data frame of [day, mean_vp]
            at_conv, vp_conv: air temperature and vapor pressure anomaly convolutions
        """
        self.seasonality = seasonality
        self.ssn_timeseries = seasonality.generate_ts()  # day, actemp
        self.at_coef = at_coef
        self.vp_coef = vp_coef
        self.dailies = at_day.merge(vp_day, on="day")
        self.at_conv = at_conv
        self.vp_conv = vp_conv

    def initialize_run(self, start_day):
        # Logs allow efficient handling of a rolling anomaly
        self.at_log = []
        self.vp_log = []
        self.date = start_day

    # TODO: implement incremental run.
    def run_series(self, data):
        """
        Run a full timeseries at once.
        data must have columns date (as an actual date type), tmax, vp.
        Will be returned with new columns day, actemp, anom, temp.mod
        """
        data["day"] = data["date"].dt.day_of_year
        data = data.merge(self.ssn_timeseries, on="day")
        anoms = data.merge(self.dailies, on="day")
        anoms["at_anom"] = anoms["tmax"] - anoms["mean_tmax"]
        anoms["vp_anom"] = anoms["vp"] - anoms["mean_vp"]
        anoms["anom"] = scipy.signal.fftconvolve(anoms["at_anom"], self.at_conv, mode="full")[:-(len(self.at_conv) - 1)] * self.at_coef +\
                        scipy.signal.fftconvolve(anoms["vp_anom"], self.vp_conv, mode="full")[:-(len(self.vp_conv) - 1)] * self.vp_coef
        data = data.merge(anoms[["date", "anom"]], on="date")
        data["temp.mod"] = data["actemp"] + data["anom"]
        return data

    def from_data(data,
                  at_conv=scipy.stats.lognorm.pdf(np.arange(0, 7), 1),
                  vp_conv=[1, 1]):
        """
        Train a watershed model from data.
        Requires columns date, temperature, tmax, vp
        """
        anoms = anomilize(data)
        # TODO: these are full timeseries length...
        # at_day = anoms[["day", "tmax_day"]].rename(columns={"tmax_day": "mean_tmax"})
        at_day = data.groupby(["day"], as_index=False)["tmax"].mean().rename(columns={"tmax": "mean_tmax"})
        vp_day = data.groupby(["day"], as_index=False)["vp"].mean().rename(columns={"vp": "mean_vp"})
        ssn = rts.ThreeSine.from_data(data)
        anoms["anom_atmod"] = scipy.signal.fftconvolve(anoms["at_anom"], at_conv, mode="full")[:-(len(at_conv) - 1)]
        anoms["anom_hummod"] = scipy.signal.fftconvolve(anoms["vp_anom"], vp_conv, mode="full")[:-(len(vp_conv) - 1)]
        sol = np.linalg.lstsq(np.array(anoms[["anom_atmod", "anom_hummod"]]), anoms["st_anom"].to_numpy().transpose(), rcond=None)[0]
        at_coef = sol[0]
        vp_coef = sol[1]
        return Watershed(ssn, at_coef, vp_coef, at_day, vp_day, at_conv, vp_conv)
