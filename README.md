"She turned me into a NEWT! I got better... Then I moved on to the NEXT big thing."

# TempEst-NEWT

TempEst-NEWT (stream temperature estimation: near-term expected watershed temperatures) is a statistical model to hindcast and forecast daily stream temperature for watersheds using air temperature and vapor pressure only.  It's essentially a complicated air-temperature regression, but a high-performing one, with a median validation-period RMSE of 1.4 C, R2 of 0.95, and bias of 0.003%.

NEWT is principally developed as the calibrated component of TempEst-NEXT, which couples NEWT with coefficient estimation for ungaged-watershed forecasting, but it is also useful as a standalone, calibrated model and is therefore made available as its own package.  NEWT and NEXT are part of the TempEst family of models, including [TempEst 1](https://github.com/river-tempest/tempest) for remote sensing-based monthly mean temperatures and [TempEst 2](https://github.com/mines-ciroh/TempEst2) for remote sensing-based daily mean and maximum temperatures, both in ungaged watersheds.  TempEst 2 and NEXT have similar functionality, but TempEst-NEXT is capable of forecasting and disturbance modeling, while TempEst 2 is only for historical estimation (but is much faster and uses less data).  TempEst 1/2 are intended for fast analyses of large-domain historical patterns, while NEWT/NEXT are more focused on in-depth analysis of changes over time, as well as forecasting.

TempEst-NEWT is implemented using [libSCHEMA](https://mines-ciroh.github.io/libSCHEMA/).

## Quick Start

### Installation

Install from PyPI: `pip install tempest-newt`.  The installed module is called NEWT: `import NEWT`.  The main model class is `NEWT.Watershed`.

### Data Preparation

To generate predictions with a NEWT model with specified coefficients (`Watershed.run_series(data)`), the requirement is a data frame with columns `date` (a Pandas datetime), `day` (Julian day), and `tmax` (daily max air temperature, averaged over the watershed). To train a model from data (`Watershed.from_data(data)`), there also needs to be a `temperature` (observed daily stream water temperature) column. Performance results are for daily mean, but this also works with daily max or min.

### Model Execution

The standard approach to running the model is `Watershed.run_series(data)`, which executes the model for the full provided timeseries and adds a `prediction` column. If modification engines are not in use, this runs a fast single-pass approach. Otherwise, it runs step-by-step, applying modification engines when needed. You can also manually run step-by-step using `.run_step(inputs)` - a single step - or `.run_series_incremental(data)`, which forces the full series to be run step-by-step.

An optional argument to `run_series` is `context`. If True (default), it adds a `prediction` column, as described. Otherwise, it just returns the array of predictions.

## Capabilities Overview

The default NEWT configuration runs a stationary seasonality plus weather model, a direct implementation of the SCHEMA (Seasonal Conditions Historical Expectation with Modeled daily Anomaly) approach developed for TempEst 2. However, you can also specify "modification engines", which activate at specified intervals to modify coefficients based on model history. This allows accounting for disturbances, droughts, climate change, etc.

### Using Modification Engines

## Science Overview

ssn + anom, etc

## Citation

...
