import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import problem_parameters, pde_mixed_plane_stress
from geometry_utils import calculate_boundary_normals
from custom_geometry import GmshGeometry2D
from gmsh_models import Block_2D

'''
Single patch-test for testing contact conditions. It is a simple block under compression. Check problem_figures/Contact_patch.png for details.

@author: tsahin
'''

## Generate block ##
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[0.,0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.025, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# The applied pressure 
ext_traction = -0.1

# contact conditions 
distance = 0

def fun_gap(x, y, X):

    gap_y = x[:,1:2] + y[:,1:2] + distance

    normals, cond = calculate_boundary_normals(X,geom)

    gn = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))

    return (1-tf.math.sign(gn))*gn

def normal_traction(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_nx_nx = sigma_xx[cond]*normals[:,0:1]*normals[:,0:1]
    sigma_xy_ny_nx = sigma_xy[cond]*normals[:,1:2]*normals[:,0:1]
    sigma_yx_nx_ny = sigma_xy[cond]*normals[:,0:1]*normals[:,1:2]
    sigma_yy_ny_ny = sigma_yy[cond]*normals[:,1:2]*normals[:,1:2]

    Pn = sigma_xx_nx_nx+sigma_xy_ny_nx+sigma_yx_nx_ny+sigma_yy_ny_ny

    return (1+tf.math.sign(Pn))*Pn

def tangential_traction(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_nx_ny = sigma_xx[cond]*normals[:,0:1]*normals[:,1:2]
    sigma_xy_ny_ny = sigma_xy[cond]*normals[:,1:2]*normals[:,1:2]
    sigma_yx_nx_nx = sigma_xy[cond]*normals[:,0:1]*normals[:,0:1]
    sigma_yy_nx_ny = sigma_yy[cond]*normals[:,0:1]*normals[:,1:2]

    Tt = -sigma_xx_nx_ny-sigma_xy_ny_ny+sigma_yx_nx_nx+sigma_yy_nx_ny

    return Tt

def complimentary(x,y,X):
    
    normals, cond = calculate_boundary_normals(X,geom)

    gap_y = x[:,1:2] + y[:,1:2] + distance
    gn = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    sigma_xx_nx_nx = sigma_xx[cond]*normals[:,0:1]*normals[:,0:1]
    sigma_xy_ny_nx = sigma_xy[cond]*normals[:,1:2]*normals[:,0:1]
    sigma_yx_nx_ny = sigma_xy[cond]*normals[:,0:1]*normals[:,1:2]
    sigma_yy_ny_ny = sigma_yy[cond]*normals[:,1:2]*normals[:,1:2]

    Pn = sigma_xx_nx_nx+sigma_xy_ny_nx+sigma_yx_nx_ny+sigma_yy_ny_ny

    return gn*Pn

def external_pressure(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]
    
    traction_y = sigma_yx_n_x + sigma_yy_n_y

    return traction_y + ext_traction*normals[:,0:1]

def boundary_contact(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

bc_gn = dde.OperatorBC(geom, fun_gap, boundary_contact)
bc_Pn = dde.OperatorBC(geom, normal_traction, boundary_contact)
bc_complimentary = dde.OperatorBC(geom, complimentary, boundary_contact)
bc_Tt = dde.OperatorBC(geom, tangential_traction, boundary_contact)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_stress,
    [bc_gn, bc_Pn, bc_complimentary, bc_Tt],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return tf.concat([u*(x_loc),v, sigma_xx*(l_beam-x_loc), ext_traction + sigma_yy*(h_beam-y_loc),sigma_xy*(l_beam-x_loc)*(x_loc)*(h_beam-y_loc)], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
activation = "elu"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=2000, display_every=100)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################


X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

output = model.predict(X)
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]

theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
theta_radian = (theta-90)*np.pi/180
theta_radian = theta_radian
sigma_rr = ext_traction/2 + ext_traction/2*np.cos(2*theta_radian)
sigma_theta = ext_traction/2 - ext_traction/2*np.cos(2*theta_radian)
sigma_rtheta =  -ext_traction/2*np.sin(2*theta_radian)
sigma_combined_radial = np.hstack((sigma_rr,sigma_theta,sigma_rtheta))

k = (3-nu)/(1+nu)
r = np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
u_r = ext_traction/(4*shear)*r*((k-1)/2+np.cos(2*theta_radian))
u_theta = -ext_traction/(4*shear)*r*np.sin(2*theta_radian)
u_x = u_r*np.cos(theta_radian) - u_theta*np.sin(theta_radian)
u_y = u_r*np.sin(theta_radian) + u_theta*np.cos(theta_radian)
u_x_temp = u_x
u_x = u_y
u_y = u_x_temp
u_x = u_x*-1
u_combined = np.hstack((u_x,u_y))

theta_radian = theta_radian.flatten()
A = [[np.cos(theta_radian)**2, np.sin(theta_radian)**2, 2*np.sin(theta_radian)*np.cos(theta_radian)],[np.sin(theta_radian)**2, np.cos(theta_radian)**2, -2*np.sin(theta_radian)*np.cos(theta_radian)],[-np.sin(theta_radian)*np.cos(theta_radian), np.sin(theta_radian)*np.cos(theta_radian), np.cos(theta_radian)**2-np.sin(theta_radian)**2]]
A = np.array(A)

sigma_analytical = np.zeros((len(sigma_rr),3))

for i in range(len(sigma_rr)):
    sigma_analytical[i:i+1,:] = np.matmul(np.linalg.inv(A[:,:,i]),sigma_combined_radial.T[:,i:i+1]).T
sigma_analytical_temp = sigma_analytical.copy()
sigma_analytical[:,0] = sigma_analytical_temp[:,1]
sigma_analytical[:,1] = sigma_analytical_temp[:,0]

error_x = abs(np.array(output[:,0].tolist()) - u_x.flatten())
error_y =  abs(np.array(output[:,1].tolist()) - u_y.flatten())
combined_error_disp = tuple(np.vstack((error_x, error_y,np.zeros(error_x.shape[0]))))

error_x = abs(np.array(output[:,2].tolist()) - sigma_analytical[:,0].flatten())
error_y =  abs(np.array(output[:,3].tolist()) - sigma_analytical[:,1].flatten())
error_z =  abs(np.array(output[:,4].tolist()) - sigma_analytical[:,2].flatten())
combined_error_stress = tuple(np.vstack((error_x, error_y,error_z)))

combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_analytical = tuple(np.vstack((np.array(sigma_analytical[:,0].flatten().tolist()),np.array(sigma_analytical[:,1].flatten().tolist()),np.array(sigma_analytical[:,2].flatten().tolist()))))
combined_disp_analytical = tuple(np.vstack((np.array(u_combined[:,0].flatten().tolist()),np.array(u_combined[:,1].flatten().tolist()),np.zeros(u_combined[:,1].shape[0]))))

file_path = os.path.join(os.getcwd(), "Patch")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
                                               ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})