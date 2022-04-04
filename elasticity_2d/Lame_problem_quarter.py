#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves the quarter Lame problem.

  * * * * * *
  *           *
  *             *
  *               *     
     *             *   
       *            * 
        *            *    
  y      *           *
  |__x   * * * * * * *
     

Dirichlet BCs:

u(x=0,y) = 0
v(x,y=0) = 0

where u represents the displacement in x direction, while v represents the displacement in y direction. 

In this problem set the material properties as follows:
    - lame : 1153.846
    - shear: 769.23

which will lead Young's modulus: 2000 and Poisson's coeff: 0.3. In this example, the Dirichlet boundary conditions are enforced hardly by choosing a surrogate model as follows:

u_s = u_p*x
v_s = v_p*y

where u_p and v_p are the network predictions.   


The problem definition and analytical solution:
https://par.nsf.gov/servlets/purl/10100420

@author: tsahin
"""
from cgi import test
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import matplotlib.tri as tri
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters
from geometry_utils import calculate_boundary_normals, polar_transformation_2d
import elasticity_utils

# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23

radius_inner = 1
center_inner = [0,0]
radius_outer = 2
center_outer = [0,0]

# First create two cylinders and subtract the small one from the large one. Then create a rectangle and intersect it with the region which is left.
geom_disk_1 = dde.geometry.Disk(center_inner, radius_inner)
geom_disk_2 = dde.geometry.Disk(center_outer, radius_outer)
geom_disk = dde.geometry.csg.CSGDifference(geom1=geom_disk_2, geom2=geom_disk_1)
geom_rect = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 2])

geom = dde.geometry.csg.CSGIntersection(geom1=geom_disk, geom2=geom_rect)

# The applied pressure 
pressure_inlet = 1

def pressure_inner_x(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure_inlet*normals[:,0:1]

def pressure_inner_y(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure_inlet*normals[:,1:2]

def traction_outer_x(x, y, X):
    '''
    Represents the x component of the zero traction
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y

def traction_outer_y(x, y, X):
    '''
    Represents the y component of the zero traction
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, traction_outer_x, boundary_outer)
bc6 = dde.OperatorBC(geom, traction_outer_y, boundary_outer)

data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc5, bc6],
    num_domain=1500,
    num_boundary=500,
    num_test=500,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    return tf.concat([ u*x*0.001, v*y*0.001], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1,1,1,1,1,1])

losshistory, train_state = model.train(epochs=2000, display_every=100)
#model.compile("L-BFGS", loss_weights=[1,1,1,1,1,1])
#model.train()


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''

    nu,lame,shear,e_modul = problem_parameters()
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])

    dr2 = (radius_outer**2 - radius_inner**2)

    sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
    sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2
    u_rad = radius_inner**2*pressure_inlet*r/(e_modul*(radius_outer**2-radius_inner**2))*(1-nu+(radius_outer/r)**2*(1+nu))

    r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
    disps = model.predict(r_x)
    u_pred, v_pred = disps[:,0:1], disps[:,1:2]
    u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
    sigma_xx, sigma_yy, sigma_xy = model.predict(r_x, operator=stress_plane_stress)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, r_x)

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    axs[0].plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta}$")
    axs[0].plot(r/radius_inner, sigma_theta/radius_inner, label = r"Predicted $\sigma_{\theta}$")
    axs[0].set(ylabel="Normalized stress", xlabel = "r/a")
    axs[1].plot(r/radius_inner, u_rad/radius_inner, label = r"Analytical $u_r$")
    axs[1].plot(r/radius_inner, u_rad_pred/radius_inner, label = r"Predicted $u_r$")
    axs[1].set(ylabel="Normalized radial displacement", xlabel = "r/a")
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()
    fig.tight_layout()

    plt.savefig("Lame_quarter_e_2000_hard")
    plt.show()

X = geom.random_points(600, random="Sobol")
boun = geom.random_boundary_points(100, random="Sobol")
X = np.vstack((X,boun))
X_corners = np.array([[radius_inner, 0],[radius_outer, 0],[0, radius_inner],[0, radius_outer]])
X = np.vstack((X,X_corners))

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_stress)
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triang = tri.Triangulation(x, y)

#masking off the unwanted triangles
condition = np.isclose(np.sqrt((x[triang.triangles]**2+y[triang.triangles]**2)),np.array([1, 1, 1]))
condition = ~np.all(condition, axis=1)

dol_triangles = triang.triangles[condition]
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
cell_types = np.ones(dol_triangles.shape[0])*5

file_path = os.path.join(os.getcwd(), "Lame_quarter_e_2000_hard")

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar})

compareModelPredictionAndAnalyticalSolution(model)