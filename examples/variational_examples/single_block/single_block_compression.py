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

from compsim_pinns.elasticity.elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters, lin_iso_elasticity_plane_stress
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement
from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.elasticity import elasticity_utils

import gmsh
from deepxde import backend as bkd

from compsim_pinns.vpinns.v_pde import VariationalPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.vpinns.quad_rule import get_test_function_properties


'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

Reference for PINNs formulation:
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
coord_left_corner=[-0,-0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.25, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=2, ngp=3) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 10
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]

geom = GmshGeometryElement(gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           test_function=test_function, 
                           test_function_derivative=test_function_derivative, 
                           n_test_func=n_test_func,
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list)


# The applied pressure 
pressure = 0.1

def constitutive_law(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation in case of plane strain

    Parameters
    ----------
    x : tensor
        the input arguments
    y: tensor
        the network output

    Returns
    -------
    momentum_x, momentum_y, term_x, term_y, term_xy: tensor
        momentum_x, momentum_y: momentum terms based on derivatives of predicted stresses
        term_x, term_y, term_xy: difference between predicted stresses and calculated stresses in X, Y and XY direction
    '''
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [term_x, term_y, term_xy]

residual_form = "1"

def weak_form_x(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    
    if residual_form == "1":
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        sigma_xx_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
        sigma_xy_y = dde.grad.jacobian(outputs, inputs, i=4, j=1)
        
        residual_x = vx*vy*(sigma_xx_x[beg:] + sigma_xy_y[beg:])
        
    elif residual_form == "2":
        sigma_xx = outputs[:, 2:3]
        sigma_xy = outputs[:, 4:5]
        
        vx_x = g_test_function_derivative[:,0:1]
        vy_y = g_test_function_derivative[:,1:2]
        
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        residual_x = -(sigma_xx[beg:]*vx_x*vy + sigma_xy[beg:]*vx*vy_y)
    
    weighted_residual_x = g_weights[:,0:1]*g_weights[:,1:2]*(residual_x)*g_jacobian
    
    return bkd.reshape(weighted_residual_x, (n_e, n_gp))

def weak_form_y(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    
    if residual_form == "1":
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        sigma_yy_y = dde.grad.jacobian(outputs, inputs, i=3, j=1)
        sigma_xy_x = dde.grad.jacobian(outputs, inputs, i=4, j=0)
        
        residual_y = vx*vy*(sigma_yy_y[beg:] + sigma_xy_x[beg:])
    
    elif residual_form == "2":
        sigma_yy = outputs[:, 3:4]
        sigma_xy = outputs[:, 4:5]
        
        vx_x = g_test_function_derivative[:,0:1]
        vy_y = g_test_function_derivative[:,1:2]
        
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        residual_y = -(sigma_xy[beg:]*vx_x*vy + sigma_yy[beg:]*vx*vy_y)
    
    weighted_residual_y = g_weights[:,0:1]*g_weights[:,1:2]*(residual_y)*g_jacobian
    
    return bkd.reshape(weighted_residual_y, (n_e, n_gp))

def zero_neumann_x(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]
    
    traction_x = sigma_xx_n_x + sigma_xy_n_y

    return traction_x

def zero_neumann_y(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]
    
    traction_y = sigma_yx_n_x + sigma_yy_n_y

    return traction_y

def pressure_x(x, y, X):
    '''
    Represents the x component of the applied pressure
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure*normals[:,0:1]

n_dummy = 1
data = VariationalPDE(
    geom,
    [weak_form_x,weak_form_y],
    [],
    constitutive_law,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
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
    
    #return tf.concat([u*(x_loc+l_beam/2),v*(y_loc+h_beam/2), pressure + sigma_xx*(l_beam/2-x_loc), sigma_yy*(h_beam/2-y_loc),sigma_xy*(l_beam/2-x_loc)*(x_loc+l_beam/2)*(h_beam/2-y_loc)*(y_loc+h_beam/2)], axis=1)
    return bkd.concat([u*(x_loc),v*(y_loc), pressure + sigma_xx*(l_beam-x_loc), sigma_yy*(h_beam-y_loc),sigma_xy*(l_beam-x_loc)*(x_loc)*(h_beam-y_loc)*(y_loc)], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
activation = "elu"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

model.compile("adam", lr=0.001, loss=mean_squared_error)
losshistory, train_state = model.train(epochs=2000, display_every=100)

model.compile("L-BFGS", loss=mean_squared_error)
losshistory, train_state = model.train(display_every=200)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################


################ Post-processing ################
gmsh.clear()
gmsh.finalize()

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[0,0]
coord_right_corner=[1,1]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
geom = GmshGeometryElement(gmsh_model, dimension=2, only_get_mesh=True)

X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

output = model.predict(X)
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]

theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
theta_radian = theta*np.pi/180
theta_radian = theta_radian
sigma_rr = pressure/2 + pressure/2*np.cos(2*theta_radian)
sigma_theta = pressure/2 - pressure/2*np.cos(2*theta_radian)
sigma_rtheta =  -pressure/2*np.sin(2*theta_radian)
sigma_combined_radial = np.hstack((sigma_rr,sigma_theta,sigma_rtheta))

k = (3-nu)/(1+nu)
r = np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
u_r = pressure/(4*shear)*r*((k-1)/2+np.cos(2*theta_radian))
u_theta = -pressure/(4*shear)*r*np.sin(2*theta_radian)
u_x = u_r*np.cos(theta_radian) - u_theta*np.sin(theta_radian)
u_y = u_r*np.sin(theta_radian) + u_theta*np.cos(theta_radian)
u_combined = np.hstack((u_x,u_y))


theta_radian = theta_radian.flatten()
A = [[np.cos(theta_radian)**2, np.sin(theta_radian)**2, 2*np.sin(theta_radian)*np.cos(theta_radian)],[np.sin(theta_radian)**2, np.cos(theta_radian)**2, -2*np.sin(theta_radian)*np.cos(theta_radian)],[-np.sin(theta_radian)*np.cos(theta_radian), np.sin(theta_radian)*np.cos(theta_radian), np.cos(theta_radian)**2-np.sin(theta_radian)**2]]
A = np.array(A)

sigma_analytical = np.zeros((len(sigma_rr),3))

for i in range(len(sigma_rr)):
    sigma_analytical[i:i+1,:] = np.matmul(np.linalg.inv(A[:,:,i]),sigma_combined_radial.T[:,i:i+1]).T

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

file_path = os.path.join(os.getcwd(), "patch_test_weak")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
                                               ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})