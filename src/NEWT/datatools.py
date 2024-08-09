"""
This file covers automated data retrieval tools for ease-of-use.
"""

import pandas as pd
import numpy as np
import dataretrieval.nwis as nwis
import pynldas2 as nldas
import s3fs
import zarr
import xarray as xr
import rioxarray as rio
import cartopy.crs as ccrs
import metpy
from pynhd import NLDI

# Direct copy-paste from https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_HowToDownload.html
hrrr_projection = ccrs.LambertConformal(central_longitude=262.5, 
                                       central_latitude=38.5, 
                                       standard_parallels=(38.5, 38.5),
                                        globe=ccrs.Globe(semimajor_axis=6371229,
                                                         semiminor_axis=6371229))

def get_shape_usgs(gid):
    # Retrieve basin shapefile for a given USGS gage.
    nldi = NLDI()
    return nldi.get_basins(gid)


def get_nldas(shape, name, new_name, start, end, operator=lambda x: x.max()):
    """
    Retrieve and process NLDAS data, by date, for a given basin (shape).
    shape: basin geometry
    name: NLDAS variable name, e.g. "temp", "prcp", "humidity"
    new_name: Output variable name for consistency, e.g. "tmax"
    start: start date, YYYY-MM-DD
    end: end date
    operator: how to summarize hours-to-days, e.g. max for temp -> tmax, sum for prcp

    Returns a data frame of date,new_name.
    """
    offset = 273 if name == "temp" else 0  # K/C
    rt = nldas.get_bygeom(shape, start, end, 4326, variables=name).\
        groupby("time").map(lambda x: x.mean()).to_dataframe()
    return operator(rt.assign(date=lambda x: x.index.date).groupby("date")[[name]]).rename(columns={name: new_name}) - offset


def get_full_forecast(date, var="TMP"):
    """
    Retrieve a full forecast run as an xarray.
    date: YYYYMMDD, using the 06z run (which should roughly correspond to day-of in US timezones)
    var: TMP, etc.  Variable to retrieve.

    Returns an xarray dataset for full CONUS, every hour for 48 hours.

    See https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_HowToDownload.html
    """
    offset = 273 if var == "TMP" else 0
    url_base = "s3://hrrrzarr/"
    tmpn = f"sfc/{date}/{date}_06z_fcst.zarr/surface/{var}"
    group_url = url_base + tmpn
    subgroup_url = group_url + "/surface"
    fs = s3fs.S3FileSystem(anon=True)
    grid = zarr.open(s3fs.S3Map(url_base, s3=fs))["grid/HRRR_chunk_index.zarr"]
    ds = xr.open_mfdataset([s3fs.S3Map(u, s3=fs) for u in [group_url, subgroup_url]], engine="zarr")
    ds["time"] = (ds["time"] - np.timedelta64(6, 'h')).astype("datetime64[D]")  # approximate, but gets it to the right day-ish
    
    ds = ds.rename(projection_x_coordinate="x", projection_y_coordinate="y").\
        metpy.assign_crs(hrrr_projection.to_cf()).\
        metpy.assign_latitude_longitude()    
    return ds[var] - offset


def get_daily_forecasts(date, operator=lambda x: x.max(), var="TMP"):
    """
    Retrieve full forecast run, then returns daily summaries.

    :47 because the last hour is two days out
    """
    ds = get_full_forecast(date, var)
    return operator(ds[:47,:,:].groupby("time"))


def forecast_watershed_clip(forecast, basin):
    """
    Clip forecast output to a watershed.
    """
    return forecast.rio.write_crs(hrrr_projection).rio.clip(basin.geometry.values, basin.crs)


def forecast_areal_summary(clipped_fcst, new_name, operator=lambda x: x.mean()):
    """
    Generate areal summary of a selected forecast, grouped by time.
    """
    summary = clipped_fcst.groupby("time").map(operator).to_pandas().rename(new_name)
    return pd.DataFrame({"date": summary.index, new_name: summary})

