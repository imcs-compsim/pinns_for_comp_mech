"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf

import sys
from pathlib import Path
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import momentum_2d 
from elasticity_postprocessing import meshGeometry, postProcess

geom_rectangle = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 1])
geom_disk = dde.geometry.Disk([1, 1], 1)
geom = dde.geometry.csg.CSGIntersection(geom1=geom_rectangle, geom2=geom_disk)


def boundary_upper(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def fun_middle_point(x):
    return 0

n_mid_points = 20
middle_points_u = np.vstack((np.full(n_mid_points, 1),np.linspace(0, 1, num=n_mid_points))).T
middle_points_u = middle_points_u[[0,-1]]
middle_points_u_2 = middle_points_u[[0]]

observe_u  = dde.PointSetBC(middle_points_u, fun_middle_point(middle_points_u), component=0)
observe_v  = dde.PointSetBC(middle_points_u_2, fun_middle_point(middle_points_u_2), component=1)

bc1 = dde.DirichletBC(geom, lambda _: 0.0, boundary_upper, component=0) # fixed in x direction
bc2 = dde.DirichletBC(geom, lambda _: -1.0, boundary_upper, component=1) # apply disp in y direction

data = dde.data.PDE(
    geom,
    momentum_2d,
    [bc1, bc2, observe_u, observe_v],
    num_domain=400,
    num_boundary=80,
    anchors=middle_points_u,
    num_test=100,
)

# two inputs x and y, output is ux and uy 
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1, display_every=1000)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################
X, triangles = meshGeometry(geom, n_boundary=130, max_mesh_area=0.01, boundary_distribution="Sobol")

postProcess(model, X, triangles, output_name="displacement")