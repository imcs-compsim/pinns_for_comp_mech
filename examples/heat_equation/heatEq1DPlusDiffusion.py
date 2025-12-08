#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves a space-time model problem based on the heat equation. The parameter


Follows Section 3.3.1 in Compressible Flow Simulation with Space-Time FE

Created on Wed Nov 17 13:27:23 2021

@author: maxvondanwitz
"""
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
import numpy as np

import deepxde as dde
from deepxde.backend import tf


from postProcessModelParameter import compareModelPredictionAndAnalyticalSolution

### Model problem

# Computational domain
x_min = -1.0
x_max = 1.0
k_min = 0.1
k_max = 0.5
t_min = 0
t_max = 0.5

spaceDomain = dde.geometry.Rectangle([x_min, k_min], [x_max, k_max])
timeDomain = dde.geometry.TimeDomain(t_min, t_max)
spaceTimeDomain = dde.geometry.GeometryXTime(spaceDomain, timeDomain)


# PDE
def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    du_t = dde.grad.jacobian(y, x, i=0, j=2) # dy_i / dx_j
    du_xx = dde.grad.hessian(y, x, i=0, j=0) # d^2y / dx_i dx_j

    # In this test case, the diffusion coefficient, k, is an input to the
    # neural network, namely
    k = x[:,1:2]

    return du_t - k * du_xx


# Initial condition
def initial_condition(x):
    """
    Evaluates the initial condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the space coordinate.
    """
    x_1 = x[:,0:1]
    return tf.cos(np.pi*x_1)

# Boundary condition
def boundary_condition(x):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """
    x_t = x[:,2:3]
    k = x[:,1:2]
    return -tf.exp(-k*(np.pi)**2*x_t)

def pde_on_param_boundary(x, y, X):
    """
    Expresses the PDE residual of the heat equation.
    """
    du_t = dde.grad.jacobian(y, x, i=0, j=2) # dy_i / dx_j
    du_xx = dde.grad.hessian(y, x, i=0, j=0) # d^2y / dx_i dx_j

    # In this test case, the diffusion coefficient, k, is an input to the
    # neural network, namely
    k = x[:,1:2]

    return du_t - k * du_xx

# Analytical solution
def analytical_solution(x, k, t):
    """
    Returns the exact solution of the model problem at a point identified by
    its x-, k-, and t-coordinate.

    Parameters
    ----------
    x : x-coordinate
    t : time-coordinate
    k : diffusion coefficient
    """

    return np.exp(-k*np.pi**2*t) * np.cos(np.pi*x)



# TimePDE provides alreaddy a boolean that indicates whether a point is on
# the boundary. The additional checks can be used to select specific parts of
# the boundary.
def boundary_space(x, on_boundary):
    return on_boundary and (np.isclose(x[0], x_min) or np.isclose(x[0], x_max))

def boundary_param(x, on_boundary):
    return on_boundary and (np.isclose(x[1], k_min) or np.isclose(x[1], k_max))

def boundary_initial(x, on_initial):
    return on_initial

# Boundary and initial conditions
bc1 = dde.DirichletBC(spaceTimeDomain, boundary_condition, boundary_space)
bc2 = dde.OperatorBC(spaceTimeDomain, pde_on_param_boundary, boundary_param)
ic = dde.IC(spaceTimeDomain, initial_condition , boundary_initial)

# Number of residual points (points where loss functions are evaluated.)
points_on_domain = 100
points_on_boundary = 200
points_on_initial = 100
points_for_testing = 100

# Define the PDE problem:
data = dde.data.TimePDE(spaceTimeDomain, pde, [bc1, bc2, ic],
                        num_domain = points_on_domain,
                        num_boundary = points_on_boundary,
                        num_initial = points_on_initial,
                        train_distribution = "uniform",
                        num_test = points_for_testing)


# ... and configure an appropriate network.
# input layer must have the size of pde domain, here x,k,t ==> 3
# output layer must have the number of pde dofs, here u (scalar) ==> 1
net = dde.nn.FNN([3] + [20] * 4 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# First guess on some scaling of the individual terms in the loss function
# ToDo: Can we derive a physics-informed scaling of these terms?
lw = [1, 1, 1, 1]

# Build the model:
# Creates a graph???
model.compile("adam", lr=1e-3, loss_weights=lw)

# and train the model
# Determines weights??
losshistory, train_state = model.train(epochs=2000)


model.compile("L-BFGS",loss_weights=lw)
losshistory, train_state = model.train()

# # Plot/print the results
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

compareModelPredictionAndAnalyticalSolution(model, analytical_solution)
