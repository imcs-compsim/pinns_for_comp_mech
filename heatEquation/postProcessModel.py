#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides some post-processing functionality to compare PINN predictions
with analytical solutions.

Created on Wed Dec 17 13:27:23 2021

@author: maxvondanwitz
"""

import matplotlib.pyplot as plt
import numpy as np

# Plotting
def plotSideBySide(xx, yy, usol, y_pred):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="3d"))

    # Analytical solution
    ax1.scatter(xx, yy, usol)
    
    # Model prediction
    ax2.scatter(xx, yy, y_pred)
    
    plt.show()
    
def compareModelPredictionAndAnalyticalSolution(model, analytical_solution):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    x : x passed to this function by the dde.pde is the NN input. Therefore,
        we must first extract the time coordinate.
    """

    # Retrieve geometry information from model
    x_min = model.data.geom.geometry.xmin[0]
    y_min = model.data.geom.geometry.xmin[1]
    
    x_max = model.data.geom.geometry.xmax[0]
    y_max = model.data.geom.geometry.xmax[1]
    
    t_min = model.data.geom.timedomain.t0
    t_max = model.data.geom.timedomain.t1
    
    # Estimate a decent resolution based on the number of residual points in the domain.
    dim = round(2*model.data.num_domain**(1./3))
    x_dim, y_dim, t_dim = (dim, dim, dim)
    
    # Create tensors:
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    y = np.linspace(y_min, y_max, num=y_dim).reshape(y_dim, 1)
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)

    xx, yy = np.meshgrid(x, y)
    X = np.vstack((np.ravel(xx), np.ravel(yy))).T

    for tau in t:

        # Analytical solution
        usol = analytical_solution(xx, yy, tau, model.data.auxiliary_var_fn(1))
       
        # Model prediction
        T = np.array([tau]*x_dim*y_dim)
        XT = np.hstack((X,T))
        y_pred = model.predict(XT).reshape(y_dim, x_dim)
        
        plotSideBySide(xx, yy, usol, y_pred)
