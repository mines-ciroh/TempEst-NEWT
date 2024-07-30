"She turned me into a NEWT! I got better... Then I moved on to the NEXT big thing."

# TempEst-NEWT

TempEst-NEWT (stream temperature estimation: near-term expected watershed temperatures) is a statistical model to hindcast and forecast daily stream temperature for watersheds using air temperature and vapor pressure only.  It's essentially a complicated air-temperature regression, but a very high-performing one, with a median validation-period RMSE of 1.4 C, R2 of 0.96, and bias of -0.18%.

NEWT is principally developed as the calibrated component of TempEst-NEXT, which couples NEWT with coefficient estimation for ungaged-watershed forecasting, but it is also useful as a standalone, calibrated model and is therefore made available as its own package.  NEWT and NEXT are part of the TempEst family of models, including [TempEst 1](https://github.com/river-tempest/tempest) for remote sensing-based monthly mean temperatures and TempEst 2 (link) for remote sensing-based daily mean and maximum temperatures, both in ungaged watersheds.  TempEst 2 and NEXT have similar functionality, but TempEst-NEXT is capable of forecasting and disturbance modeling, while TempEst 2 is only for historical estimation (but is much faster and uses less data).  TempEst 1/2 are intended for fast analyses of large-domain historical patterns, while NEWT/NEXT are more focused on in-depth analysis of changes over time, as well as forecasting.

## Quick Start

### Installation

Install from PyPI: `pip install tempest-newt`.

### Data Preparation

### Model Execution

## Capabilities Overview

The default NEWT configuration runs a stationary seasonality plus weather model, a direct implementation of the SCHEMA (Seasonal Conditions Historical Estimation with Modeled daily Anomaly) approach developed for TempEst 2.  However, you can also specify "dynamic", "yearly" and "climate" modification engines, which support non-stationarity by allowing the model to update itself over time.  Modification engines receive all model input data (up to the time of activation) and can arbitrarily modify model coefficients.  Dynamic and climate engines both activate every N days (intended for short and long periods, respectively), and can be used to track changing climate conditions or account for something like a dry month.  Yearly engines activate on a specific day-of-year (Julian day), and are used to account for something like a dry winter.

...

## Science Overview

ssn + anom, etc

## Citation

...
