#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves a space-time model problem based on the heat equation.
Follows Section 3.3.1 in Compressible Flow Simulation with Space-Time FE

Created on Wed Nov 17 13:27:23 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import torch

def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    # Diffusion coefficient 
    k = 0.1
    
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - k * dy_xx

# Initial condition
def initial_condition(x):
    """
    Evaluates the initial condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the space coordinate.
    """
    x_s = torch.tensor(x[:,0:1])
    return torch.cos(np.pi*x_s)

# Boundary condition
def boundary_condition(x):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """
    x_t = torch.tensor(x[:,1:2])
    k = 0.1
    
    return -torch.exp(-k*(np.pi)**2*x_t)


# Analytical solution
def analytical_solution(x, t, k):
    """
    Returns the exact solution of the model problem at a point identified by
    its x- and t-coordinates for given k.

    Parameters
    ----------
    x : x-coordinate
    t : time-coordinate
    k : diffusion coefficient
    """

    return np.exp(-k*np.pi**2*t) * np.cos(np.pi*x)

def postProcess(model):
    '''
    Performs heat equation specific post-processing of a trained model.

    Parameters
    ----------
    X : trained deepxde model

    '''
    import os, sys
    from pathlib import Path
    path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
    sys.path.append(path_utils)
    from exportVtk import meshGeometry, solutionFieldOnMeshToVtk

    geom = model.data.geom

    X, triangles = meshGeometry(geom, numberOfPointsOnBoundary=20)

    temperature = model.predict(X)

    pointData = { "temperature" : temperature.flatten()}

    file_path = os.path.join(os.getcwd(),"heatEquation2D")

    solutionFieldOnMeshToVtk(X, triangles, pointData, file_path)


# Computational domain
xmin = -1
xmax = 1
tmin = 0
tmax = 2

spaceDomain = dde.geometry.Interval(xmin, xmax)
timeDomain = dde.geometry.TimeDomain(tmin, tmax)
spaceTimeDomain = dde.geometry.GeometryXTime(spaceDomain, timeDomain)

# Why do we define these functions. TimePDE seems to provide alreaddy a
# boolean that indicates whether a point is on the boundary.
def boundary_space(x, on_boundary):
    return on_boundary

def boundary_initial(x, on_initial):
    return on_initial

# Boundary and initial conditions
bc = dde.DirichletBC(spaceTimeDomain, boundary_condition, boundary_space)
ic = dde.IC(spaceTimeDomain, initial_condition , boundary_initial)

# First guess on some scaling of the individual terms in the loss function
# ToDo: Can we derive a physics-informed scaling of these terms?
lw = [1, 100, 100]

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(spaceTimeDomain, pde, [bc, ic], num_domain=250,
                        num_boundary=32, num_initial=16, num_test=254,
                        # auxiliary_var_function=diffusionCoeff
                        )

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
model.compile("adam", lr=1e-3, loss_weights=lw)
losshistory, train_state = model.train(epochs=5000)

# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


model.compile("L-BFGS")
losshistory, train_state = model.train()
# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# postProcess(model)


# Define some query points on our compuational domain.
# Number of points in each dimension:
x_dim, t_dim = (21, 26)

# Bounds of 'x' and 't':
x_min, t_min = (xmin, tmin)
x_max, t_max = (xmax, tmax)

# Create tensors:
t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T

# Compute and plot the exact solution for these query points
k = 0.1
usol = analytical_solution(xx, tt, k)
plt.scatter(xx,tt,c=usol)
plt.show()

# Plot model prediction.
y_pred = model.predict(X).reshape(t_dim, x_dim)
plt.scatter(xx,tt,c=y_pred)
plt.xlabel('x')
plt.ylabel('t')
ax = plt.gca()
ax.set_aspect('equal','box')
#plt.colorbar(cax=ax)
plt.savefig('heatEqPred.pdf')
plt.show()