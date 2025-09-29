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
    
    

class NextOldBmi(Bmi):
    """
    BMI implementation for TempEst-NEXT.
    Example: https://github.com/csdms/bmi-example-python/blob/master/heat/bmi_heat.py
    """
    
    _name = "TempEst-NEXT"
    _input_var_names = ("land_surface_air__temperature",
                        # "air_water~vapor__partial_pressure", # I guess?
                        )
    _output_var_names = ("channel_water__temperature",)
    # Convenience
    swt = _output_var_names[0]
    at = _input_var_names[0]
    def __init__(self):
        self._model = None
        self._values = {}
        self._var_units = {}
        self._var_loc = {}
        self._grids = {}
        self._grid_type = {}
        self._start_time = 0.0
        self._end_time = np.finfo("float").max
        self._time_units = "s"  # required, unfortunately
    
    def initialize(self, filename):
        """
        Initialize the model.  Filename points to input file.
        """
        if logalot:
            print("Initializing BMI model")
        self._model = Watershed.from_file(filename, True)
        # self._model.initialize_run()
        try:
            self._values = {self.swt: [self._model.get_st, None],
                            self.at: [self._model.get_at, self._model.set_at],
                            # self.vp: [self._model.get_vp, self._model.set_vp]
                            }
            self._vptrs = {self.swt: np.array([self._model.temperature]),
                            self.at: np.array([self._model.at])}
            self._var_units = {self.swt: "Celsius",
                               self.at: "Celsius",
                               # self.vp: "Pa"
                               }
            self._var_loc = {self.swt: "node"}
            self._grids = self.swt
            self._grid_type = "scalar"
            self._timestep = 0.0
            self._temps = []
            if logalot:
                self._model.log("Finished BMI initialization")
        except Exception as e:
            self._model.log(f"Error in initialization: {e}")
    
    def update(self):
        # self._model.step()
        try:
            for k in self._vptrs:
                setter = self._values[k][1]
                if setter is not None:
                    setter(self._vptrs[k][0])
            self._timestep += 3600
            if self._timestep % 86400 < 1:
                self._model.step()
            for k in self._vptrs:
                self._vptrs[k][0] = self._values[k][0]()
        except Exception as e:
            self._model.log(f"Error in update step: {e}")
    
    def update_until(self, time):
        while self._timestep < time:
            self.update()
    
    def finalize(self):
        self._model.get_history().to_csv("newt_bmi_run_history.csv", index=False)
        self._model = None
    
    def get_component_name(self):
        return self._name
    
    def get_input_item_count(self):
        return len(self._input_var_names)
    
    def get_output_item_count(self):
        return len(self._output_var_names)
    
    def get_input_var_names(self):
        return self._input_var_names
    
    def get_output_var_names(self):
        return self._output_var_names
    
    def get_var_grid(self, name):
        return 0
    
    def get_var_type(self, name):
        return "float"
    
    def get_var_units(self, name):
        return self._var_units[name]
    
    def get_var_itemsize(self, name):
        return 8
    
    def get_var_nbytes(self, name):
        try:
            return self.get_value_ptr(name).nbytes
        except Exception as e:
            self._model.log(f"Error in get_value_ptr: {e}")
    
    def get_var_location(self, name):
        return "node"
    
    def get_current_time(self):
        return self._timestep
    
    def get_start_time(self):
        return 0.0
    
    def get_end_time(self):
        return self._end_time
    
    def get_time_units(self):
        return self._time_units
    
    def get_time_step(self):
        return 3600.0
    
    def get_value_ptr(self, name):
        return self._vptrs[name]
    
    def get_value(self, name, dest):
        dest[:] = np.array(self.get_value_ptr(name))
        return dest
    
    def get_value_at_indices(self, name, dest, inds):
        return self.get_value(name, dest)
    
    def set_value(self, name, src):
        val = src[0]
        if name == self.at:
            self._temps.append(val)
            val = max(self._temps[-24:])  # 24-hour max temperature
        if name != self.swt:
            self._values[name][1](val)
        else:
            self._model.log("Warning: tried to set stream temperature.")
        self._vptrs[name][0] = self._values[name][0]()
        
    def set_value_at_indices(self, name, inds, src):
        self.set_value(name, src)
        
    def get_grid_type(self, grid):
        return "scalar"
    
    def get_grid_rank(self, grid):
        return 1
    
    def get_grid_size(self, grid):
        return 1
    
    def get_grid_shape(self, grid, shape):
        shape[:] = np.array([1])
        return np.array([1])
    
    def get_grid_spacing(self, grid, spacing):
        raise NotImplementedError("get_grid_spacing")
    
    def get_grid_origin(self, grid, origin):
        raise NotImplementedError("get_grid_origin")
        
    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")
    
    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")
    
    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")
    
    def get_grid_node_count(self, grid):
        return 1
    
    def get_grid_edge_count(self, grid):
        return 0
    
    def get_grid_face_count(self, grid):
        return 0
    
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")
        
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")
        
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
        
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")
        
