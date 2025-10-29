#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves the quarter Lame problem.

  * * * * * *
  *            *
  *              *
  *                *     
     *              *   
       *             * 
        *             *    
  y      *            *
  |__x   * * * * * * **
  -------| --> R_i
  ---------------------| -->R_o     
 
Dirichlet BCs:

u_x(x=0,y) = 0
u_y(x,y=0) = 0

where u_x represents the displacement in x direction, while u_y represents the displacement in y direction. 

Neumann boundary conditions (in polar coordinates)
P(r=R_i,\theta) = 1 

In this problem set the material properties as follows:
    - lame : 1153.846
    - shear: 769.23

which will lead Young's modulus: 2000 and Poisson's coeff: 0.3. In this example, the Dirichlet boundary conditions are enforced hardly by choosing a surrogate model as follows:

u_s = u_x*x
v_s = u_y*y

where u_x and u_y are the network predictions.   


The problem definition and analytical solution:
https://par.nsf.gov/servlets/purl/10100420

@author: tsahin
"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.tri as tri
import deepxde.backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from utils.elasticity.elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters, zero_neumman_plane_stress_x, zero_neumman_plane_stress_y, stress_to_traction_2d
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.elasticity import elasticity_utils

# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23

# geometrical parameters
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
elasticity_utils.geom = geom

# Inner pressure
pressure_inlet = 1

def pressure_inner_x(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx + pressure_inlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty + pressure_inlet*normals[:,1:2]


def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

soft_dirichlet = True # enforce the Dirichlet BC softly

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
if soft_dirichlet:
    bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
    bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, boundary_outer)
bc6 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, boundary_outer)
bc7 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, boundary_bottom)
bc8 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, boundary_left)

data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8], # remove bc3 and bc4, if you want to enforce Dirichlet BC hardly
    num_domain=1500,
    num_boundary=500,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform_hard(x, y):
    """
    Enforces the Dirichlet BCs in a hard way.

    u_x = u_x * x
    u_y = u_y * y
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    return bkd.concat((u*x, v*y), axis=1)

def output_transform_hard_scaled(x, y):
    """
    Enforces the Dirichlet BCs in a hard way and scale them.

    u_x = u_x * x * 0.001
    u_y = u_y * y * 0.001
    """

    u = y[:, 0:1]
    v = y[:, 1:2]
    return bkd.concat((u*x*0.001, v*y*0.001), axis=1)

def output_transform_scaled(x, y):
    """
    Scale the network output:

    u_x = u_x * 0.001
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    return bkd.concat((u*0.001, v*0.001), axis=1)

# two inputs x and y, two outputs ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

output_scaling = True

if soft_dirichlet:
    if output_scaling:
        net.apply_output_transform(output_transform_scaled)
else:
    if output_scaling:
        net.apply_output_transform(output_transform_hard_scaled)
    else:
        net.apply_output_transform(output_transform_hard)

loss_scaling = True

if loss_scaling:
    loss_weights = [1,1,1,1,1e6,1e6,1,1,1,1]
else:
    if not soft_dirichlet:
        loss_weights = [1,1,1,1,1,1,1,1]
    else:
        loss_weights = [1,1,1,1,1,1,1,1,1,1]

model = dde.Model(data, net)
# train adam
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=4000, display_every=200)

vtu_and_plot_name = "Lame_quarter_e_2000_soft_scaled_weighted"

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions vs the predictions. 
    '''

    nu,_,_,e_modul = problem_parameters()
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])

    dr2 = (radius_outer**2 - radius_inner**2)

    sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
    sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2
    u_rad_analytical = radius_inner**2*pressure_inlet*r/(e_modul*(radius_outer**2-radius_inner**2))*(1-nu+(radius_outer/r)**2*(1+nu))

    r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
    disps = model.predict(r_x)
    u_pred, v_pred = disps[:,0:1], disps[:,1:2]
    u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
    sigma_xx, sigma_yy, sigma_xy = model.predict(r_x, operator=stress_plane_stress)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, r_x)

    err_norm_disp = np.sqrt(np.sum((u_rad_pred.flatten()-u_rad_analytical.flatten())**2))
    ex_norm_disp = np.sqrt(np.sum(u_rad_analytical.flatten()**2))
    rel_err_l2_disp = err_norm_disp/ex_norm_disp
    print("Relative L2 error for displacement: ", rel_err_l2_disp)

    err_norm_stress = np.sqrt(np.sum((sigma_rr_analytical-sigma_rr.flatten())**2+(sigma_theta_analytical-sigma_theta.flatten())**2))
    ex_norm_stress = np.sqrt(np.sum(sigma_rr_analytical**2+sigma_theta_analytical**2))
    rel_err_l2_stress = err_norm_stress/ex_norm_stress
    print("Relative L2 error for stress: ", rel_err_l2_stress)

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    axs[0].plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta}$")
    axs[0].plot(r/radius_inner, sigma_theta/radius_inner, label = r"Predicted $\sigma_{\theta}$")
    axs[0].set(ylabel="Normalized radial stress", xlabel = r"r/$R_i$")
    axs[1].plot(r/radius_inner, u_rad_analytical/radius_inner, label = r"Analytical $u_r$")
    axs[1].plot(r/radius_inner, u_rad_pred/radius_inner, label = r"Predicted $u_r$")
    axs[1].set(ylabel="Normalized radial displacement", xlabel = r"r/$R_i$")
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()
    fig.tight_layout()

    plt.savefig(vtu_and_plot_name)
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

file_path = os.path.join(os.getcwd(), vtu_and_plot_name)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar})

compareModelPredictionAndAnalyticalSolution(model)