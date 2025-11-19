"She turned me into a NEWT! I got better... Then I moved on to the NEXT big thing."

# TempEst-NEWT

TempEst-NEWT (stream temperature estimation: near-term expected watershed temperatures) is a statistical model to hindcast and forecast daily stream temperature for watersheds using air temperature and vapor pressure only.  It's essentially a complicated air-temperature regression, but a high-performing one, with a median validation-period RMSE of 1.4 C, R2 of 0.95, and bias of 0.003%.

NEWT is principally developed as the calibrated component of [TempEst-NEXT](https://github.com/mines-ciroh/TempEst-NEXT), which couples NEWT with coefficient estimation for ungaged-watershed forecasting, but it is also useful as a standalone, calibrated model and is therefore made available as its own package.  NEWT and NEXT are part of the TempEst family of models, including [TempEst 1](https://github.com/river-tempest/tempest) for remote sensing-based monthly mean temperatures and [TempEst 2](https://github.com/mines-ciroh/TempEst2) for remote sensing-based daily mean and maximum temperatures, both in ungaged watersheds.  TempEst 2 and NEXT have similar functionality, but TempEst-NEXT is capable of forecasting and disturbance modeling, while TempEst 2 is only for historical estimation (but is much faster and uses less data).  TempEst 1/2 are intended for fast analyses of large-domain historical patterns, while NEWT/NEXT are more focused on in-depth analysis of changes over time, as well as forecasting.

TempEst-NEWT is implemented using [libSCHEMA](https://rivertempest.org/libschema/readme.html).

## Quick Start

### Installation

Install from PyPI: `pip install tempest-newt`.  The installed module is called NEWT: `import NEWT`.  The main model class is `NEWT.Watershed` ([documentation](https://rivertempest.org/newt/NEWT.html#NEWT.watershed.Watershed)).

Dependencies: pandas, numpy (>=2), rtseason (>=1.1.3), bmipy, scipy, pygam (>=0.10), libschema (>=0.1.4), pyyaml

### Data Preparation

To generate predictions with a NEWT model with specified coefficients (`Watershed.run_series(data)`), the requirement is a data frame with columns `date` (a Pandas datetime), `day` (Julian day), and `tmax` (daily max air temperature, averaged over the watershed). To train a model from data (`Watershed.from_data(data)`), there also needs to be a `temperature` (observed daily stream water temperature) column. Performance results are for daily mean, but this also works with daily max or min. Extra data columns may be required depending on the use of modification engines (see [LibSCHEMA docs](https://rivertempest.org/libschema/libschema.html#libschema.model.SCHEMA)).

### Model Execution

The standard approach to running the model is `Watershed.run_series(data)`, which executes the model for the full provided timeseries and adds a `prediction` column. If modification engines are not in use, this runs a fast single-pass approach. Otherwise, it runs step-by-step, applying modification engines when needed. You can also manually run step-by-step using `.run_step(inputs)` - a single step - or `.run_series_incremental(data)`, which forces the full series to be run step-by-step.

An optional argument to `run_series` is `context`. If True (default), it adds a `prediction` column, as described. Otherwise, it just returns the array of predictions.

Additional information can be found in the documentation for the `SCHEMA` class in [LibSCHEMA](https://rivertempest.org/libschema/libschema.html#libschema.model.SCHEMA).

Example:

```
from NEWT import Watershed
import pandas as pd

# `temperature` column included to demonstrate building a model from data. Not required for prediction.
input_data = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02", ...]),
                            "day": [1, 2, 3, ..., 365], "tmax": [-3.1, 0, 5.6, 2, ...]},
                            "temperature": [1.1, 0.5, 0.6, 1.3, ...]})

# Usually, the provided coefficients would be fitted or estimated, but you can also manually specify them, as here.
model = Watershed(seasonality={"Intercept": 10, "Amplitude": 8, "SpringSummer": 1.2, "FallWinter": 0.5,
                                "SpringDay": 150, "SummerDay": 220, "FallDay": 330, "WinterDay": 30},
                  anomaly={"at_coef": 0.3},
                  at_day=pd.DataFrame({"day": [1, 2, 3, ..., 365], "mean_tmax": [5.0, 4.9, 5.0, 4.7, 4.5, ...]}),
                  engines=[])
# OR (more common), with `temperature` column included
model = Watershed.from_data(input_data)

# either way...
prediction = model.run_series(input_data)
```

## Detailed Documentation

See the [documentation](https://rivertempest.org/newt/NEWT.html) at RiverTempest.org. Since a NEWT model is a LibSCHEMA model, much of the documentation about running NEWT models is under the [libSCHEMA documentation](https://rivertempest.org/libschema/libschema.html).

## Capabilities Overview

The default NEWT configuration runs a stationary seasonality plus weather model, a direct implementation of the SCHEMA (Seasonal Conditions Historical Expectation with Modeled daily Anomaly) approach developed for TempEst 2. However, you can also specify "modification engines", which activate at specified intervals to modify coefficients based on model history. This allows accounting for disturbances, droughts, climate change, etc. These are simply passed as an argument to initializing the watershed model; see the reference linked above.

## Citation

If you use TempEst-NEWT in your research, please cite:

Philippus, Corona and Hogue. "Daily Stream Water Temperature Modeling and Forecasting for Ungaged Watersheds at the CONUS Scale." In review.
