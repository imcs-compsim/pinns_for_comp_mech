import deepxde as dde
import numpy as np
import os
import sys
from pathlib import Path
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, elastic_strain_2d, stress_plane_strain
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.elasticity import elasticity_utils

import gmsh
from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

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
coord_left_corner=[0,0.]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]

geom = GmshGeometryElementDeepEnergy(gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list)


# The applied pressure
pressure = 0.1
nu,lame,shear,e_modul = problem_parameters()
applied_disp_y = -pressure/e_modul*(1-nu**2)*1

residual_form = "1"

def potential_energy_x(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights):
    
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(inputs,outputs)
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(inputs,outputs)
    
    potential_energy = 1/2*(sigma_xx[beg:]*eps_xx[beg:] + sigma_xy[beg:]*eps_xy[beg:])
    
    total_potential_energy = g_weights[:,0:1]*g_weights[:,1:2]*(potential_energy)*g_jacobian
    
    return bkd.reshape(total_potential_energy, (n_e, n_gp))

def potential_energy_y(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights):
    
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(inputs,outputs)
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(inputs,outputs)
    
    potential_energy = 1/2*(sigma_yy[beg:]*eps_yy[beg:] + sigma_xy[beg:]*eps_xy[beg:])
    
    total_potential_energy = g_weights[:,0:1]*g_weights[:,1:2]*(potential_energy)*g_jacobian
    
    return bkd.reshape(total_potential_energy, (n_e, n_gp))

def points_at_top(x, on_boundary):
    points_top = np.isclose(x[1],h_beam)
    
    return on_boundary and points_top

def points_at_bottom(x, on_boundary):
    points_bottom = np.isclose(x[1],0)
    
    return on_boundary and points_bottom

bc_u_y = dde.DirichletBC(geom, lambda _: applied_disp_y, points_at_top, component=1)
# bc_u_y = dde.DirichletBC(geom, lambda _: 0, points_at_bottom, component=1)

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    [potential_energy_x, potential_energy_y],
    [bc_u_y],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(x_loc),v*(y_loc)], axis=1)
    #return bkd.concat([u*(x_loc),v*(1-y_loc)+applied_disp_y], axis=1)    

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
loss_weights=[1,1,1e3]

model.compile("adam", lr=0.001, loss_weights=loss_weights)
losshistory, train_state = model.train(epochs=1000, display_every=100)

model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=200)

# ###################################################################################
# ############################## VISUALIZATION PARTS ################################
# ###################################################################################


# ################ Post-processing ################
# gmsh.clear()
# gmsh.finalize()

# # Define GMSH and geometry parameters
# gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
# coord_left_corner=[0,0]
# coord_right_corner=[1,1]

# # create a block
# block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

# # generate gmsh model
# gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
# geom = GmshGeometryElementDeepEnergy(gmsh_model, dimension=2, only_get_mesh=True)

# X, offset, cell_types, dol_triangles = geom.get_mesh()
# nu,lame,shear,e_modul = problem_parameters()

# output = model.predict(X)
# u_pred, v_pred = output[:,0], output[:,1]
# sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)

# theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
# theta_radian = theta*np.pi/180
# theta_radian = theta_radian
# sigma_rr = pressure/2 + pressure/2*np.cos(2*theta_radian)
# sigma_theta = pressure/2 - pressure/2*np.cos(2*theta_radian)
# sigma_rtheta =  -pressure/2*np.sin(2*theta_radian)
# sigma_combined_radial = np.hstack((sigma_rr,sigma_theta,sigma_rtheta))

# k = (3-nu)/(1+nu)
# r = np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
# u_r = pressure/(4*shear)*r*((k-1)/2+np.cos(2*theta_radian))
# u_theta = -pressure/(4*shear)*r*np.sin(2*theta_radian)
# u_x = u_r*np.cos(theta_radian) - u_theta*np.sin(theta_radian)
# u_y = u_r*np.sin(theta_radian) + u_theta*np.cos(theta_radian)
# u_combined = np.hstack((u_x,u_y))


# theta_radian = theta_radian.flatten()
# A = [[np.cos(theta_radian)**2, np.sin(theta_radian)**2, 2*np.sin(theta_radian)*np.cos(theta_radian)],[np.sin(theta_radian)**2, np.cos(theta_radian)**2, -2*np.sin(theta_radian)*np.cos(theta_radian)],[-np.sin(theta_radian)*np.cos(theta_radian), np.sin(theta_radian)*np.cos(theta_radian), np.cos(theta_radian)**2-np.sin(theta_radian)**2]]
# A = np.array(A)

# sigma_analytical = np.zeros((len(sigma_rr),3))

# for i in range(len(sigma_rr)):
#     sigma_analytical[i:i+1,:] = np.matmul(np.linalg.inv(A[:,:,i]),sigma_combined_radial.T[:,i:i+1]).T

# error_x = abs(np.array(output[:,0].tolist()) - u_x.flatten())
# error_y =  abs(np.array(output[:,1].tolist()) - u_y.flatten())
# combined_error_disp = tuple(np.vstack((error_x, error_y,np.zeros(error_x.shape[0]))))

# error_x = abs(sigma_xx.flatten() - sigma_analytical[:,0].flatten())
# error_y = abs(sigma_yy.flatten() - sigma_analytical[:,1].flatten())
# error_z = abs(sigma_xy.flatten() - sigma_analytical[:,2].flatten())
# combined_error_stress = tuple(np.vstack((error_x, error_y,error_z)))

# combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
# combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
# combined_stress_analytical = tuple(np.vstack((np.array(sigma_analytical[:,0].flatten().tolist()),np.array(sigma_analytical[:,1].flatten().tolist()),np.array(sigma_analytical[:,2].flatten().tolist()))))
# combined_disp_analytical = tuple(np.vstack((np.array(u_combined[:,0].flatten().tolist()),np.array(u_combined[:,1].flatten().tolist()),np.zeros(u_combined[:,1].shape[0]))))

# file_path = os.path.join(os.getcwd(), "deep_energy_patch")

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)

# #np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
#                                                ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})

X, offset, cell_types, dol_triangles = geom.get_mesh()
nu,lame,shear,e_modul = problem_parameters()

# start_time_calc = time.time()
output = model.predict(X)
# end_time_calc = time.time()
# final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# print(final_time)

u_x_pred, u_y_pred = output[:,0], output[:,1]
u_pred, v_pred = output[:,0], output[:,1]
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)

u_x_analytical = pressure/e_modul*nu*(1+nu)*X[:,0]
u_y_analytical = -pressure/e_modul*(1-nu**2)*X[:,1]
s_xx_analytical = np.zeros(X.shape[0])
s_yy_analytical = -pressure*np.ones(X.shape[0])
s_xy_analytical = np.zeros(X.shape[0])

error_u_x = abs(u_x_pred - u_x_analytical)
error_u_y = abs(u_y_pred - u_y_analytical)
combined_error_disp = tuple(np.vstack((error_u_x, error_u_y, np.zeros(error_u_x.shape[0]))))

error_s_xx = abs(sigma_xx.flatten() - s_xx_analytical)
error_s_yy = abs(sigma_yy.flatten() - s_yy_analytical)
error_s_xy = abs(sigma_xy.flatten() - s_xy_analytical)
combined_error_stress = tuple(np.vstack((error_s_xx, error_s_yy, error_s_xy)))

combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
combined_stress = tuple(np.vstack((sigma_xx.flatten(), sigma_yy.flatten(), sigma_xy.flatten())))
combined_disp_analytical = tuple(np.vstack((u_x_analytical, u_y_analytical, np.zeros(u_x_analytical.shape[0]))))
combined_stress_analytical = tuple(np.vstack((s_xx_analytical, s_yy_analytical, s_xy_analytical)))


file_path = os.path.join(os.getcwd(), "deep_energy_patch")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

#np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "analy_stress" : combined_stress_analytical, "analy_disp" : combined_disp_analytical
                                               ,"error_disp":combined_error_disp, "error_stress":combined_error_stress})

# Calculate l2-error
u_combined_pred = np.asarray(combined_disp).T
s_combined_pred = np.asarray(combined_stress).T
u_combined_analytical = np.asarray(combined_disp_analytical).T
s_combined_analytical = np.asarray(combined_stress_analytical).T

rel_err_l2_disp = np.linalg.norm(u_combined_pred - u_combined_analytical) / np.linalg.norm(u_combined_analytical)
print("Relative L2 error for disp: ", rel_err_l2_disp)
rel_err_l2_stress = np.linalg.norm(s_combined_pred - s_combined_analytical) / np.linalg.norm(s_combined_analytical)
print("Relative L2 error for stress: ", rel_err_l2_stress)