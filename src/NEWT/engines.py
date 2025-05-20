# -*- coding: utf-8 -*-
"""
This file covers pre-specified modification engines for the Watershed class.
"""

import pandas as pd
import numpy as np
import rtseason as rts
import NEWT.analysis as analysis
from libschema.classes import ModEngine

class ClimateEngine(ModEngine):
    def __init__(self, coef_model):
        self.coef_model = coef_model
    
    def apply(self, seasonality, anomaly, periodics, history):
        return (seasonality, anomaly, periodics)


class WetDryEngine(ModEngine):
    pass
