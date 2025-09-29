# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:34:31 2024

@author: dphilippus

BMI implementation of the Watershed class.  Referred to as NEXT in reference
to the parent library (with coefficient estimation), but the actual BMI part
is the watershed model NEWT.
"""

from NEWT import Watershed
from bmipy import Bmi
from libschema.bmi import SchemaBmi
import numpy as np

logalot = False

class NextBmi(SchemaBmi):
    def __init__(self):
        super().__init__(
            name="TempEst-NEXT",
            inputs=("land_surface_air__temperature",),
            input_map={"land_surface_air__temperature": "tmax"},
            input_units=["Celsius"],
            output="channel_water__temperature",
            output_units="Celsius"
        )

    def initialize(self, filename):
        super().initialize(Watershed, filename)

    def update(self):
        self._values["day"] = self._model.period
        super().update()
        raise NotImplementedError("get_grid_nodes_per_face")
        
