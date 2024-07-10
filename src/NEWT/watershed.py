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

    def step(self, date, at, vp):
        """
        Run a single step, incrementally.  Updates history and returns
        today's prediction.
        """
        self.doy = date.day_of_year
        self.date = date
        today = self.dailies[self.dailies["day"] == self.doy]
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
        # Update history
        self.history["date"].append(self.date)
        self.history["day"].append(self.doy)
        self.history["at"].append(at)
        self.history["vp"].append(vp)
        self.history["actemp"].append(ssn)
        self.history["anom"].append(anom)
        self.history["temp.mod"].append(pred)
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
