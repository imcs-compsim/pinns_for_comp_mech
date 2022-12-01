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

from custom_geometry import GmshGeometry2D
from gmsh_models import QuarterDisc
from elasticity_utils import problem_parameters, pde_mixed_plane_strain
from geometry_utils import calculate_boundary_normals, polar_transformation_2d
import elasticity_utils

#dde.config.set_default_float("float64")

'''
@author: tsahin
'''
#dde.config.real.set_float64()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=265, refine_times=1, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# # change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu_analy,lame_analy,shear_analy,e_modul_analy = problem_parameters() # with dimensions, will be used for analytical solution
# This will lead to e_modul_analy=200 and nu_analy=0.3

# The applied pressure 
ext_traction = -0.5

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

distance = 0

def fun_gap(x, y, X):

    gap_y = x[:,1:2] + y[:,1:2] + radius + distance

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

    gap_y = x[:,1:2] + y[:,1:2] + radius +  distance
    gn = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    sigma_xx_nx_nx = sigma_xx[cond]*normals[:,0:1]*normals[:,0:1]
    sigma_xy_ny_nx = sigma_xy[cond]*normals[:,1:2]*normals[:,0:1]
    sigma_yx_nx_ny = sigma_xy[cond]*normals[:,0:1]*normals[:,1:2]
    sigma_yy_ny_ny = sigma_yy[cond]*normals[:,1:2]*normals[:,1:2]

    Pn = sigma_xx_nx_nx+sigma_xy_ny_nx+sigma_yx_nx_ny+sigma_yy_ny_ny

    return gn*Pn

def zero_traction_x(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]
    
    traction_x = sigma_xx_n_x + sigma_xy_n_y

    return traction_x

def zero_traction_y(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]
    
    traction_y = sigma_yx_n_x + sigma_yy_n_y

    return traction_y


# def boundary_circle(x, on_boundary):
#     return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius)

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

ext_data = geom.random_boundary_points(1) #np.isclose(callback_pts[:,1],0)
ext_data = ext_data[np.isclose(np.linalg.norm(ext_data - center, axis=-1), radius)]#,callback_pts[:,0]>=x_loc_partition)
ext_data = ext_data[np.isclose(ext_data[:,1],-radius)]
observe_u  = dde.PointSetBC(ext_data, np.zeros(1).reshape(-1,1), component=1)

bc1 = dde.OperatorBC(geom, zero_traction_x, boundary_circle_not_contact)
bc2 = dde.OperatorBC(geom, zero_traction_y, boundary_circle_not_contact)

bc3 = dde.OperatorBC(geom, fun_gap, boundary_circle_contact)
bc4 = dde.OperatorBC(geom, normal_traction, boundary_circle_contact)
bc5 = dde.OperatorBC(geom, tangential_traction, boundary_circle_contact)
bc6 = dde.OperatorBC(geom, complimentary, boundary_circle_contact)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    [bc1,bc2,bc3,bc4,bc5,bc6,observe_u], #,bc3,bc4,bc5,bc6, observe_u
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
    anchors=ext_data
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    #return tf.concat([u*(-x_loc), ext_dips + v*(-y_loc), sigma_xx, sigma_yy, sigma_xy*(x_loc)*(y_loc)], axis=1)
    return tf.concat([u*(-x_loc)/e_modul_analy, v/e_modul_analy, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

loss_weights = [1e0,1e0,1e0,1e0,1e0,1e0,1e0,1e4,1e0,1e0,1e2,1e0]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=2000, display_every=100) 

model.compile("L-BFGS-B", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=None, refine_times=None, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)
elasticity_utils.geom = geom

X, offset, cell_types, dol_triangles = geom.get_mesh()

output = model.predict(X)
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)


combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

file_path = os.path.join(os.getcwd(), "Hertzian_normal_contact")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar})