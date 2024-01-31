import deepxde as dde
import numpy as np

import matplotlib.pyplot as plt

from deepxde import utils as deepxde_utils
from deepxde import backend as bkd

from utils.geometry.gmsh_models import Line_1D
from utils.geometry.custom_geometry import GmshGeometryElement

from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.vpinns.quad_rule import get_test_function_properties

from utils.vpinns.v_pde import VariationalPDE

import tensorflow as tf
import torch

'''
Solving 1D poisson's equation via VPINNS
'''

omega = 8 * np.pi
amp = 1
r1 = 80

def u_exact(x):
    utemp = 0.1 * np.sin(omega * x) + np.tanh(r1 * x)
    return amp * utemp

def func(x):
    gtemp = -0.1 * (omega ** 2) * np.sin(omega * x) - (2 * r1 ** 2) * (
        np.tanh(r1 * x)
    ) / ((np.cosh(r1 * x)) ** 2)
    return -amp * gtemp

def cosh(x):
    bkd_name = bkd.get_preferred_backend()
    if (bkd_name == "tensorflow") or (bkd_name == "tensorflow.compat.v1"):
        return tf.math.cosh(x)
    elif bkd_name == "pytorch":
        return torch.cosh(x) 

def func_tensor(x): 
    gtemp = -0.1 * (omega ** 2) * bkd.sin(omega * x) - (2 * r1 ** 2) * (
        bkd.tanh(r1 * x)
    ) / ((cosh(r1 * x)) ** 2)
    return -amp * gtemp


quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=1, ngp=40) # gauss_legendre
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 15
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1}
coord_left = -1
coord_right = 1
# create a line element
Element_1d = Line_1D(coord_left, coord_right, mesh_size=0.1, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = Element_1d.generateGmshModel(visualize_mesh=False)

geom = GmshGeometryElement(gmsh_model, 1, coord_quadrature, weight_quadrature, test_function, test_function_derivative, n_test_func)

def reshape_tensor(input_tensor):
    return bkd.reshape(input_tensor, (input_tensor.shape[0]*input_tensor.shape[1],input_tensor.shape[2]))

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    x = deepxde_utils.to_numpy(x) #convert to numpy
    f_x = bkd.as_tensor(func(x))
    f_x.requires_grad_()
    
    return -dy_xx - f_x #- np.pi ** 2 * torch.sin(np.pi * x)
    # Use torch.sin for backend pytorch
    # return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x)
    # Use paddle.sin for backend paddle
    # return -dy_xx - np.pi ** 2 * paddle.sin(np.pi * x)

def weak_form(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    dy_xx = dde.grad.hessian(outputs, inputs)
    residual = -(dy_xx[beg:] + func_tensor(inputs)[beg:])*g_test_function
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    weighted_residual = g_weights*residual*g_jacobian
    return bkd.reshape(weighted_residual, (n_e, n_gp))
    # Use torch.sin for backend pytorch
    # return -dy_xx - np.pi ** 2 * torch.sin(np.pi * x)
    # Use paddle.sin for backend paddle
    # return -dy_xx - np.pi ** 2 * paddle.sin(np.pi * x)

def boundary(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, u_exact, boundary)

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

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

model.compile("adam", lr=0.001, loss=mean_squared_error, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=25000)

model.compile("L-BFGS", loss=mean_squared_error)
losshistory, train_state = model.train(display_every=200)

x_input = geom.mapped_coordinates

pred = model.predict(x_input)
act = u_exact(x_input)

plt.scatter(x_input, pred, color="r")
plt.plot(x_input, act)
plt.savefig("1D_poisson_variational")
plt.show()

