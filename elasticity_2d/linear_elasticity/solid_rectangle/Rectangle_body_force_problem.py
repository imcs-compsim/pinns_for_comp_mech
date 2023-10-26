"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.interpolate
import os

from utils.elasticity.elasticity_utils import (
    stress_plane_strain,
    problem_parameters,
    momentum_2d,
)
from utils.postprocess.elasticity_postprocessing import meshGeometry, postProcess

"""
This script is used to create the PINN model of 2D Elasticity example. The example is taken from
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics with the following link
https://www.semanticscholar.org/paper/A-physics-informed-deep-learning-framework-for-and-Haghighat-Raissi/e420b8cd519909b4298b16d1a46fbd015c86fc4e
"""


def pde(x, y):
    # Use common linear elasticity momentum equation
    [mom_x, mom_y] = momentum_2d(x, y)

    nu, lame, shear, e_modul = problem_parameters()
    Q_param = 4

    # Extract spatial coordinates x_s and y_s from the network inputs x
    x_s = x[:, 0:1]
    y_s = x[:, 1:2]

    # body forces
    f_x = lame * (
        4 * np.pi**2 * tf.cos(2 * np.pi * x_s) * tf.sin(np.pi * y_s)
        - np.pi * tf.cos(np.pi * x_s) * Q_param * y_s**3
    ) + shear * (
        9 * np.pi**2 * tf.cos(2 * np.pi * x_s) * tf.sin(np.pi * y_s)
        - np.pi * tf.cos(np.pi * x_s) * Q_param * y_s**3
    )
    f_y = lame * (
        -3 * tf.sin(np.pi * x_s) * Q_param * y_s**2
        + 2 * np.pi**2 * tf.sin(2 * np.pi * x_s) * tf.cos(np.pi * y_s)
    ) + shear * (
        -6 * tf.sin(np.pi * x_s) * Q_param * y_s**2
        + 2 * np.pi**2 * tf.sin(2 * np.pi * x_s) * tf.cos(np.pi * y_s)
        + np.pi**2 * tf.sin(np.pi * x_s) * Q_param * y_s**4 / 4
    )

    momentum_x = mom_x + f_x
    momentum_y = mom_y + f_y

    return [momentum_x, momentum_y]


def fun_sigma_xx(x, y, X):
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x, y)

    return sigma_xx


def fun_sigma_yy(x, y, X):
    nu, lame, shear, e_modul = problem_parameters()
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x, y)

    return sigma_yy - (lame + 2 * nu) * 4 * tf.sin(np.pi * x[:, 0:1])


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


def boundary_t(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)


def func(x):
    return -(x**6) / 90 + x**5 / 15 + -(x**4) / 6 + x**3 / 6 - x / 18


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

bc1 = dde.DirichletBC(geom, lambda _: 0, boundary_l, component=1)
bc2 = dde.OperatorBC(geom, fun_sigma_xx, boundary_l)
bc3 = dde.DirichletBC(geom, lambda _: 0, boundary_r, component=1)
bc4 = dde.OperatorBC(geom, fun_sigma_xx, boundary_r)
bc5 = dde.DirichletBC(geom, lambda _: 0, boundary_b, component=0)
bc6 = dde.DirichletBC(geom, lambda _: 0, boundary_b, component=1)
bc7 = dde.DirichletBC(geom, lambda _: 0, boundary_t, component=0)
bc8 = dde.OperatorBC(geom, fun_sigma_yy, boundary_t)


data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8],
    num_domain=200,
    num_boundary=50,
    solution=func,
    num_test=100,
)
# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=1, display_every=1000)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, triangles = meshGeometry(
    geom, n_boundary=100, max_mesh_area=0.01, boundary_distribution="uniform"
)

postProcess(
    model,
    X,
    triangles,
    output_name="displacement",
    operator=stress_plane_strain,
    operator_name="stress",
)

# The rest is time taking, so use exit()
exit()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X = geom.random_points(10000)
# if the uniform boundary points are possible (for some geometries not possible),
# otherwise comment to next two lines
boun = geom.uniform_boundary_points(100)  # comment this
X = np.vstack((X, boun))  # comment this

output = model.predict(X)

u_anal = np.cos(2 * np.pi * X[:, 0:1]) * np.sin(np.pi * X[:, 1:2])
v_anal = np.sin(np.pi * X[:, 0:1]) * 4 * X[:, 1:2] ** 4 / 4

u_pred = output[:, 0]
v_pred = output[:, 1]

res_u_analy = np.hstack(
    [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), u_anal.reshape(-1, 1)]
)
res_v_analy = np.hstack(
    [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), v_anal.reshape(-1, 1)]
)

res_u_pred = np.hstack(
    [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), u_pred.reshape(-1, 1)]
)
res_v_pred = np.hstack(
    [X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), v_pred.reshape(-1, 1)]
)

residum_u = u_pred.reshape(-1, 1) - u_anal.reshape(-1, 1)
residum_u = np.hstack([X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), residum_u])

residum_v = v_pred.reshape(-1, 1) - v_anal.reshape(-1, 1)
residum_v = np.hstack([X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), residum_v])

# ------------------------------------------------------------------------------------
#####################################################################################
##################### True displacement in u direction ##############################
#####################################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_analy_u.png")
x, y, z = res_u_analy[:, 0], res_u_analy[:, 1], res_u_analy[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

f = plt.figure(1)
plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
plt.clf()

# -----------------------------------------------------------------------------------------
##########################################################################################
##################### Predicted displacement in u direction ##############################
##########################################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_pred_u.png")
x, y, z = res_u_pred[:, 0], res_u_pred[:, 1], res_u_pred[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
plt.clf()

# ---------------------------------------------------------------------------
############################################################################
##################### Residuum in u direction ##############################
############################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_resid_u.png")
x, y, z = residum_u[:, 0], residum_u[:, 1], residum_u[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
plt.clf()
# ------------------------------------------------------------------------------------
#####################################################################################
##################### True displacement in v direction ##############################
#####################################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_analy_v.png")
x, y, z = res_v_analy[:, 0], res_v_analy[:, 1], res_v_analy[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

f = plt.figure(1)
plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
plt.clf()

# -----------------------------------------------------------------------------------------
##########################################################################################
##################### Predicted displacement in v direction ##############################
##########################################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_pred_v.png")
x, y, z = res_v_pred[:, 0], res_v_pred[:, 1], res_v_pred[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
plt.clf()

# ---------------------------------------------------------------------------
############################################################################
##################### Residuum in v direction ##############################
############################################################################
filename = os.path.join(os.getcwd(), "2d_elasticity_results/2d_resid_v.png")
x, y, z = residum_v[:, 0], residum_v[:, 1], residum_v[:, 2]

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
zi = rbf(xi, yi)

plt.imshow(
    zi,
    vmin=z.min(),
    vmax=z.max(),
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
)
plt.scatter(x, y, c=z)
plt.colorbar()
plt.savefig(filename)
