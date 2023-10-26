#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves a space-time model problem based on the heat equation.
Follows Section 3.3.1 in Compressible Flow Simulation with Space-Time FE

Created on Wed Nov 17 13:27:23 2021

@author: maxvondanwitz
"""
import numpy as np

import deepxde as dde
from deepxde.backend import tf

from postProcessModel import compareModelPredictionAndAnalyticalSolution


### Model problem
# PDE
def pde(x, y, k):
    """
    Expresses the PDE residual of the heat equation.
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)  # dy_i / dx_j
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # d^2y / dx_i dx_j
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)  # d^2y / dx_i dx_j
    return dy_t - k * (dy_xx + dy_yy)


def diffusionCoeff(x):
    """
    Provides the scalar diffusion coefficient for the PDE residual of the heat
    equation. For the dde.pde interface, the scalar is wrapped in a function
    that returns an numpy array.
    """
    return np.array([[0.1]])


# Initial condition
def initial_condition(x):
    """
    Evaluates the initial condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the space coordinate.
    """
    x_1 = x[:, 0:1]
    x_2 = x[:, 1:2]
    return tf.cos(np.pi * x_1) * tf.cos(np.pi * x_2)


# Boundary condition
def boundary_condition(x):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """
    # x_t = x[:,1:2]
    # k = diffusionCoeff(x=0)[0]
    # return -tf.exp(-k*(np.pi)**2*x_t)
    return 0.0


# Analytical solution
def analytical_solution(x, y, t, k):
    """
    Returns the exact solution of the model problem at a point identified by
    its x-, y-, and t-coordinates for given k.

    Parameters
    ----------
    x : x-coordinate
    y : y-coordinate
    t : time-coordinate
    k : diffusion coefficient
    """

    return np.exp(-2 * k * np.pi**2 * t) * np.cos(np.pi * x) * np.cos(np.pi * y)


# Computational domain
x_min = -0.5
x_max = 0.5
y_min = -0.5
y_max = 0.5
t_min = 0
t_max = 2

spaceDomain = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
timeDomain = dde.geometry.TimeDomain(t_min, t_max)
spaceTimeDomain = dde.geometry.GeometryXTime(spaceDomain, timeDomain)


# Why do we define these functions? TimePDE seems to provide alreaddy a
# boolean that indicates whether a point is on the boundary.
def boundary_space(x, on_boundary):
    return on_boundary


def boundary_initial(x, on_initial):
    return on_initial


# Boundary and initial conditions
bc = dde.DirichletBC(spaceTimeDomain, boundary_condition, boundary_space)
ic = dde.IC(spaceTimeDomain, initial_condition, boundary_initial)

# Number of residual points (points where loss functions are evaluated.)
points_on_domain = 1000
points_on_boundary = 1000
points_on_initial = 100
points_for_testing = 1000

# Define the PDE problem:
data = dde.data.TimePDE(
    spaceTimeDomain,
    pde,
    [bc, ic],
    num_domain=points_on_domain,
    num_boundary=points_on_boundary,
    num_initial=points_on_initial,
    train_distribution="LHS",
    num_test=points_for_testing,
    auxiliary_var_function=diffusionCoeff,
)


# ... and configure an appropriate network.
# input layer must have the size of pde domain, here x,y,t ==> 3
# output layer must have the number of pde dofs, here u (scalar) ==> 1
net = dde.nn.FNN([3] + [20] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# First guess on some scaling of the individual terms in the loss function
# ToDo: Can we derive a physics-informed scaling of these terms?
lw = [1, 100, 100]

# Build the model:
# Creates a graph???
model.compile("adam", lr=1e-3, loss_weights=lw)


# and train the model
# Determines weights??
losshistory, train_state = model.train(epochs=10000)

# # Plot/print the results
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

compareModelPredictionAndAnalyticalSolution(model, analytical_solution)
