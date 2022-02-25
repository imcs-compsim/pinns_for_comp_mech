"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import matplotlib.tri as tri
from deepxde import utils
from pyevtk.hl import unstructuredGridToVTK

from .. utils.deep_utils import stress_plane_strain, calculate_boundary_normals, polar_transformation_2d

def pde(x, y):    
    # calculate strain terms (kinematics, small strain theory)

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    # governing equation
    sigma_xx_x = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
    sigma_yy_y = dde.grad.jacobian(sigma_yy, x, i=0, j=1)
    sigma_xy_x = dde.grad.jacobian(sigma_xy, x, i=0, j=0)
    sigma_xy_y = dde.grad.jacobian(sigma_xy, x, i=0, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x

    return [momentum_x, momentum_y]

radius_inner = 1
center_inner = [0,0]
radius_outer = 2
center_outer = [0,0]

geom_disk_1 = dde.geometry.Disk(center_inner, radius_inner)
geom_disk_2 = dde.geometry.Disk(center_outer, radius_outer)
geom = dde.geometry.csg.CSGDifference(geom1=geom_disk_2, geom2=geom_disk_1)

pressure_inlet = 1
pressure_outlet = 2

def pressure_inner_x(x, y, X):    
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure_inlet*normals[:,0:1]

def pressure_outer_x(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure_outlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure_inlet*normals[:,1:2]

def pressure_outer_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure_outlet*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.OperatorBC(geom, pressure_outer_x, boundary_outer)
bc4 = dde.OperatorBC(geom, pressure_outer_y, boundary_outer)

data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=600,
    num_boundary=120,
    num_test=120,
    train_distribution = "uniform"
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
X = geom.random_points(600, random="Sobol")
boun = geom.uniform_boundary_points(100)
X = np.vstack((X,boun))

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triang = tri.Triangulation(x, y)

#masking off the unwanted triangles
condition = np.isclose(np.sqrt((x[triang.triangles]**2+y[triang.triangles]**2)),np.array([1, 1, 1]))
condition = ~np.all(condition, axis=1)

dol_triangles = triang.triangles[condition]
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
cell_types = np.ones(dol_triangles.shape[0])*5

file_path = "/home/a11btasa/git_repos/phd_materials/pinns/Lame_problem/lame_problem_pressure_test"

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp, "stress" : combined_stress_polar})

