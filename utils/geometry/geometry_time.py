from deepxde.geometry import GeometryXTime
import numpy as np

class ModifiedGeometryXTime(GeometryXTime):
    def __init__(self, geometry, timedomain):
        # Call the parent class constructor
        super().__init__(geometry, timedomain)

    def boundary_tangential_1(self, x):
        # Assuming the base geometry has a method 'boundary_tangential'
        _n = self.geometry.boundary_tangential_1(x[:, :-1])
        return np.hstack([_n, np.zeros((len(_n), 1))])
    
    def boundary_tangential_2(self, x):
        # Assuming the base geometry has a method 'boundary_tangential'
        _n = self.geometry.boundary_tangential_2(x[:, :-1])
        return np.hstack([_n, np.zeros((len(_n), 1))])