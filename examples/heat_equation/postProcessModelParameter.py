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
def plotSideBySide(xx, yy, tau, uLeft, uRight):
    """
    Plots two solution fields side by side.

    Parameters
    ----------
    xx : x-coordinates generated with meshgrid
    yy : y-coordinates generated with meshgrid
    uLeft: solution field to be displayed on the left
    uRight: solution field to be displayed on the right
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="3d"))

    # Analytical solution
    ax1.scatter(xx, yy, uLeft)
    ax1.set_xlabel('x')
    ax1.set_ylabel('k')
    ax1.set_zlabel('u(x,t,k)')
    ax1.set_title('Analytical solution')
    
    # Model prediction
    ax2.scatter(xx, yy, uRight)
    ax2.set_xlabel('x')
    ax2.set_ylabel('k')
    ax2.set_zlabel('u(x,t,k)')
    ax2.set_title('and model prediction at t = '+str(tau))
    
    plt.show()
    
def compareModelPredictionAndAnalyticalSolution(model, analytical_solution):
    """
    Evaluates the boundary condition.

    Parameters
    ----------
    model : trained dde model
    analytical_solution : function to retrieve analaytical solution
    """

    # Retrieve geometry information from model
    x_min = model.data.geom.geometry.xmin[0]
    k_min = model.data.geom.geometry.xmin[1]
    
    x_max = model.data.geom.geometry.xmax[0]
    k_max = model.data.geom.geometry.xmax[1]
    
    t_min = model.data.geom.timedomain.t0
    t_max = model.data.geom.timedomain.t1
    
    # Estimate a decent resolution based on the number of residual points in the domain.
    dim = round(2*model.data.num_domain**(1./3))
    x_dim, k_dim, t_dim = (dim, dim, dim)
    
    # Create tensors:
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    y = np.linspace(k_min, k_max, num=k_dim).reshape(k_dim, 1)
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)

    xx, yy = np.meshgrid(x, y)
    X = np.vstack((np.ravel(xx), np.ravel(yy))).T

    for tau in t:

        # Analytical solution
        usol = analytical_solution(xx, yy, tau)
       
        # Model prediction
        T = np.array([tau]*x_dim*k_dim)
        XT = np.hstack((X,T))
        y_pred = model.predict(XT).reshape(k_dim, x_dim)
        
        plotSideBySide(xx, yy, tau, usol, y_pred)
