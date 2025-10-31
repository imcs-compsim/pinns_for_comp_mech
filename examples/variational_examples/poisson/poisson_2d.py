import deepxde as dde
import numpy as np
import os
from deepxde.backend import torch
from pyevtk.hl import unstructuredGridToVTK
import time
from pathlib import Path
import matplotlib.pyplot as plt
from deepxde.icbc.boundary_conditions import npfunc_range_autocache
from deepxde import utils as deepxde_utils
from deepxde import backend as bkd

from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.vpinns.quad_rule import get_test_function_properties

from compsim_pinns.vpinns.v_pde import VariationalPDE
import gmsh


omega = 8 * np.pi
amp = 1
r1 = 80

def u_exact(x):
    x_coord = x[:,0:1]
    y_coord = x[:,1:2]
    return 2*(1+y_coord)/((3+x_coord)**2 + (1+y_coord)**2)

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
coord_left_corner=[-1,-1]
coord_right_corner=[1,1]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.2, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=2, ngp=3) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 10
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")


# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]

geom = GmshGeometryElement(gmsh_model, 
                           dimension=2,
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature=weight_quadrature, 
                           test_function=test_function, 
                           test_function_derivative=test_function_derivative, 
                           n_test_func=n_test_func,
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list)


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    x = deepxde_utils.to_numpy(x) #convert to numpy
    
    return -dy_xx  #- np.pi ** 2 * torch.sin(np.pi * x)

def weak_form(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)[beg:]
    du_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)[beg:]
    
    vx_x = g_test_function_derivative[:,0:1]
    vy_y = g_test_function_derivative[:,1:2]
    
    vx = g_test_function[:,0:1]
    vy = g_test_function[:,1:2]
    
    weighted_residual = -g_weights[:,0:1]*g_weights[:,1:2]*(du_x*vx_x*vy + du_y*vy_y*vx)*g_jacobian
    return bkd.reshape(weighted_residual, (n_e, n_gp))

def boundary(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, u_exact, boundary, component=0)

n_dummy = 1
weak = True

if weak:
    data = VariationalPDE(geom, 
                        weak_form, 
                        bc, 
                        num_domain=n_dummy, 
                        num_boundary=n_dummy, 
                        solution=u_exact
                        )
else:
    data = dde.data.PDE(geom, 
                        pde, 
                        bc,
                        num_domain=n_dummy, 
                        num_boundary=n_dummy, 
                        solution=u_exact, 
                        num_test=n_dummy)

layer_size = [2] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

model.compile("adam", lr=0.001, loss=mean_squared_error, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

model.compile("L-BFGS", loss=mean_squared_error)
losshistory, train_state = model.train(display_every=200)

################ Post-processing ################
gmsh.clear()
gmsh.finalize()

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[-1,-1]
coord_right_corner=[1,1]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
geom = GmshGeometryElement(gmsh_model, dimension=2, only_get_mesh=True)

X, offset, cell_types, elements = geom.get_mesh()

u_pred = model.predict(X)
u_act = u_exact(X)
error = np.abs(u_pred - u_act)

combined_disp_pred = tuple(np.vstack((u_pred.flatten(), np.zeros(u_pred.shape[0]), np.zeros(u_pred.shape[0]))))
combined_disp_act = tuple(np.vstack((u_act.flatten(), np.zeros(u_act.shape[0]), np.zeros(u_act.shape[0]))))
combined_error = tuple(np.vstack((error.flatten(), np.zeros(error.shape[0]), np.zeros(error.shape[0]))))


file_path = os.path.join(os.getcwd(), "2D_poisson_variational")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros_like(y)

unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                      cell_types, pointData = {"disp_pred" : combined_disp_pred, 
                                               "disp_act" : combined_disp_act,
                                               "error": combined_error})