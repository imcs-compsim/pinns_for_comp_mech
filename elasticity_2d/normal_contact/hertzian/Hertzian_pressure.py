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

from utils.elasticity.elasticity_utils import stress_plane_strain, momentum_2d 
from utils.geometry.geometry_utils import calculate_boundary_normals
from utils.postprocess.elasticity_postprocessing import meshGeometry, postProcess

geom_rectangle = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 1])
geom_disk = dde.geometry.Disk([1, 1], 1)
geom = dde.geometry.csg.CSGIntersection(geom1=geom_rectangle, geom2=geom_disk)

def boundary_upper(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def boundary_circle(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - [1,1], axis=-1), 1) and ~np.isclose(x[0],0) and ~np.isclose(x[0],2) 

pressure = 2

def pressure_x(x, y, X):    
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure*normals[:,0:1]

def pressure_y(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure*normals[:,1:2]

def observations(x):
    return 0

bc1 = dde.DirichletBC(geom, lambda _: 0.0, boundary_upper, component=0) # fixed in x direction
bc2 = dde.DirichletBC(geom, lambda _: 0.0, boundary_upper, component=1) # fixed in y direction
bc3 = dde.OperatorBC(geom, pressure_x, boundary_circle)
bc4 = dde.OperatorBC(geom, pressure_y, boundary_circle)

p = np.vstack((np.array([0,1]),np.array([2,1])))

observe_u  = dde.PointSetBC(p, observations(p), component=0)
observe_v  = dde.PointSetBC(p, observations(p), component=1)

data = dde.data.PDE(
    geom,
    momentum_2d,
    [bc1, bc2, bc3, bc4, observe_u, observe_v],
    num_domain=1024,
    num_boundary=256,
    num_test=100,
    anchors=p
    #train_distribution = "Sobol",
)

# two inputs x and y, output is ux and uy 
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1,1,100,100,1,1,1,1])
losshistory, train_state = model.train(epochs=1, display_every=1000)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, triangles = meshGeometry(geom, n_boundary=130, max_mesh_area=0.01, boundary_distribution="Sobol")

postProcess(model, X, triangles, output_name="displacement")
