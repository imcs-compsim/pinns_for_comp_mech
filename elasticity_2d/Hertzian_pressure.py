"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
import os

from .. utils.elasticity_utils import stress_plane_strain, momentum_2d 
from .. utils.geometry_utils import calculate_boundary_normals, polar_transformation_2d

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
losshistory, train_state = model.train(epochs=3000, display_every=1000)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################
X = geom.random_points(600, random="Sobol")
boun = geom.uniform_boundary_points(100)
X = np.vstack((X,boun))

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
sigma_rr, sigma_theta, sigma_rtheta, theta_radian = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triang = tri.Triangulation(x, y)
dol_triangles = triang.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
cell_types = np.ones(dol_triangles.shape[0])*5

file_path = os.path.join(os.getcwd(),"Hertzian_pressure")

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress_polar" : combined_stress_polar, "stress": combined_stress})
