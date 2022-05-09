import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters, zero_neumman_plane_stress_x, zero_neumman_plane_stress_y
from geometry_utils import calculate_boundary_normals
from custom_geometry import GmshGeometry2D
from gmsh_models import Block_2D
import elasticity_utils


'''
The correct order for the normals --> 1 2 1 1

Reference solution:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.504.4507&rep=rep1&type=pdf
'''

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
block_2d = Block_2D(coord_left_corner=[0,-0.5], coord_right_corner=[4,0.5], mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry2D(gmsh_model)

l = block_2d.coord_right_corner[0] -block_2d.coord_left_corner[0]
h = block_2d.coord_right_corner[1] -block_2d.coord_left_corner[1]

shear_y = 0.01
Inertia = 1/12*h**3

# change global variables in elasticity_utils
e_1 = 2000
nu_1 = 0.3
elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
elasticity_utils.shear = e_1/(2*(1+nu_1))
# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

def neumann_x(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    return sigma_xx[cond]

def neumann_y(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]
    
    y_loc = x[:,1:2][cond]
    
    return sigma_yx_n_x + shear_y/(2*Inertia)*(y_loc - h/2)*(y_loc + h/2)*(-1)#*normals[:,0:1]


nu, lame, shear, e_modul = problem_parameters()

def fun_u_x(x):
    x_loc = x[:,0:1]
    y_loc = x[:,1:2]
    u_x_analy = shear_y*y_loc/(6*e_modul*Inertia)*(2+nu)*(y_loc**2-h**2/4)

    return u_x_analy

def fun_u_y(x):
    x_loc = x[:,0:1]
    y_loc = x[:,1:2]
    u_y_analy = -shear_y/(6*e_modul*Inertia)*(3*nu*y_loc**2*l)

    return u_y_analy

def top_bottom(x, on_boundary):
    points_top = np.logical_and(np.isclose(x[1],h/2),~np.isclose(x[0],l))
    points_bottom = np.logical_and(np.isclose(x[1],-h/2),~np.isclose(x[0],l))
    
    return on_boundary and np.logical_or(points_top, points_bottom)

def left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def right(x, on_boundary):
    return on_boundary and np.isclose(x[0],l)

bc1 = dde.DirichletBC(geom, fun_u_x, left, component=0)
bc2 = dde.DirichletBC(geom, fun_u_y, left, component=1)
bc3 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, top_bottom)
bc4 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, top_bottom)
bc5 = dde.OperatorBC(geom, neumann_x, right)
bc6 = dde.OperatorBC(geom, neumann_y, right)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc3, bc4, bc5, bc6],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    x_loc = x[:, 0:1]
    return tf.concat([ u*0.01, v*0.001], axis=1)

# in case hard Dirichlet is desired (no scaling!! so it must be tested)
# def output_transform(x, y):
#     x_loc = x[:,0:1]
#     y_loc = x[:,1:2]
#     u_x_analy = y[:,0:1]*shear_y*y_loc/(6*e_modul*Inertia)*((6*l-3*x_loc)*x_loc + (2+nu)*(y_loc**2-h**2/4))
#     u_y_analy = -y[:,1:2]*shear_y/(6*e_modul*Inertia)*(3*nu*y_loc**2*(l-x_loc) + (4+5*nu)*h**2*x_loc/4 + (3*l-x_loc)*x_loc**2)
#     return tf.concat([ u_x_analy, u_y_analy], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights=[1,1,1e5,1e7,1,1,1,1]
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)

losshistory, train_state = model.train(epochs=5000, display_every=200)
model.compile("L-BFGS",loss_weights=loss_weights)
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_stress)

nu, lame, shear, e_modul = problem_parameters()

x = X[:,0:1]
y = X[:,1:2]
Inertia = 1/12*h**3
u_x_analy = shear_y*y/(6*e_modul*Inertia)*((6*l-3*x)*x + (2+nu)*(y**2-h**2/4))
u_y_analy = -shear_y/(6*e_modul*Inertia)*(3*nu*y**2*(l-x) + (4+5*nu)*h**2*x/4 + (3*l-x)*x**2)

sigma_xx_analy = shear_y*(l-x)*y/Inertia
sigma_yy_analy = np.zeros(sigma_xx_analy.shape[0])
sigma_xy_analy = shear_y/(2*Inertia)*(y - h/2)*(y + h/2)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_disp_analy = tuple(np.vstack((u_x_analy.flatten(),u_y_analy.flatten(),np.zeros(u_x_analy.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_analy = tuple(np.vstack((sigma_xx_analy.flatten(),sigma_yy_analy.flatten(), sigma_xy_analy.flatten())))

error_x = abs(np.array(displacement[:,0].tolist()) - u_x_analy.flatten())
error_y =  abs(np.array(displacement[:,1].tolist()) - u_y_analy.flatten())
combined_error = tuple(np.vstack((error_x, error_y,np.zeros(error_x.shape[0]))))


file_path = os.path.join(os.getcwd(), "Beam2D")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                    cell_types, pointData = { "displacement" : combined_disp,
                    "disp_analy": combined_disp_analy, "stress" : combined_stress,
                    "stress_analy": combined_stress_analy,
                    "error":combined_error})


