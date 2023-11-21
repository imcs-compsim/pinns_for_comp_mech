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

from utils.geometry.gmsh_models import Line_1D
from utils.geometry.custom_geometry import GmshGeometry1D

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain
from utils.contact_mech.contact_utils import zero_tangential_traction, positive_normal_gap_sign, negative_normal_traction_sign, zero_complimentary
from utils.contact_mech.contact_utils import positive_normal_gap_adopted_sigmoid, negative_normal_traction_adopted_sigmoid
from utils.contact_mech.contact_utils import zero_complementarity_function_based_popp, zero_complementarity_function_based_fisher_burmeister
from utils.elasticity import elasticity_utils
from utils.contact_mech import contact_utils

#####
from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.vpinns.quad_rule import get_test_function_properties, modified_legendre, modified_legendre_2, modified_legendre_derivative, modified_legendre_derivative_2

from utils.vpinns.v_pde import VariationalPDE

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


quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=1, ngp=20) # gauss_legendre
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 30
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1}
coord_left = -1
coord_right = 1
# create a line element
Element_1d = Line_1D(coord_left, coord_right, mesh_size=0.1, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = Element_1d.generateGmshModel(visualize_mesh=False)

geom = GmshGeometry1D(gmsh_model, coord_left, coord_right, coord_quadrature, weight_quadrature, test_function, test_function_derivative, n_test_func, func)


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
    
def weak_form(x, y):
    dy_xx = dde.grad.hessian(y, x)
    # Use tf.sin for backend tensorflow.compat.v1 or tensorflow
    return -dy_xx
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

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

# Optional: Save the model during training.
# checkpointer = dde.callbacks.ModelCheckpoint(
#     "model/model", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = dde.callbacks.MovieDumper(
#     "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# losshistory, train_state = model.train(iterations=10000, callbacks=[checkpointer, movie])

x_input = geom.mapped_coordinates

pred = model.predict(x_input)
act = u_exact(x_input)

plt.scatter(x_input, pred, color="r")
plt.plot(x_input, act)
plt.show()

#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# # Optional: Restore the saved model with the smallest training loss
# # model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
# # Plot PDE residual
# x = geom.uniform_points(1000, True)
# y = model.predict(x, operator=pde)
# plt.figure()
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("PDE residual")
# plt.show()

