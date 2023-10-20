"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf

"""
This script is used to create the PINN model of clamped beam
see the manuscript for the example, Section 4, Figure 4.2 and 4.3, Deep Learning in Computational Mechanics
"""


def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


p = lambda x: 1
EI_material = lambda x: 1


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + p(x)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def func(x):
    return -p(x) * x**2 / (24 * 1) * ((x - 1) ** 2)


geom = dde.geometry.Interval(0, 1)

bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.NeumannBC(geom, lambda x: 0, boundary_r)


data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=20,
    num_boundary=2,
    solution=func,
    num_test=100,
)

layer_size = [1] + [30] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=40000, display_every=1000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)

"""
fname="case_1"
loss_fname = fname + "_" + "loss.dat"
train_fname = fname + "_" + "train.dat"
test_fname = fname + "_" + "test.dat"
dde.saveplot(losshistory, train_state, issave=True, isplot=True, plot_name=fname,
loss_fname=loss_fname, train_fname=train_fname, test_fname=test_fname, 
output_dir="/home/a11btasa/git_repos/deepxde_2/beam_results")
"""
