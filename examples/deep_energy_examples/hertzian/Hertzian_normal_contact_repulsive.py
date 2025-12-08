# Set the backend as tensorflow.compat.v1 before importing DeepXDE
'''
@author: tsahin
'''
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path
import matplotlib.tri as tri
import pandas as pd
from deepxde.backend import tf

from compsim_pinns.geometry.geometry_utils import polar_transformation_2d
from compsim_pinns.elasticity import elasticity_utils

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, elastic_strain_2d, stress_plane_strain, problem_parameters
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy

from compsim_pinns.geometry.gmsh_models import QuarterDisc

from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.04, angle=255, refine_times=10, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]

def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

def on_top(x):
    return np.isclose(x[1],0)

boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                          {"boundary_function" : on_top, "tag" : "on_top"},]

quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=1, ngp=6) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)

# # change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters() # with dimensions, will be used for analytical solution

# The applied pressure 
ext_traction = 0.5

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

def potential_energy(X, 
                     inputs, 
                     outputs, 
                     beg_pde, 
                     beg_boundary, 
                     n_e, 
                     n_gp, 
                     n_e_boundary, 
                     n_gp_boundary, 
                     jacobian_t, 
                     global_element_weights_t, 
                     mapped_normal_boundary_t, 
                     jacobian_boundary_t, 
                     global_weights_boundary_t,
                     boundary_selection_tag):
    
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(inputs,outputs)
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(inputs,outputs)
    
    # get the internal energy
    internal_energy_density = 1/2*(sigma_xx[beg_pde:beg_boundary]*eps_xx[beg_pde:beg_boundary] + 
                            sigma_yy[beg_pde:beg_boundary]*eps_yy[beg_pde:beg_boundary] + 
                          2*sigma_xy[beg_pde:beg_boundary]*eps_xy[beg_pde:beg_boundary])
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density)*jacobian_t
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    
    external_force_density = -ext_traction*u_y
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    
    gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
    gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
    # eta=1e4
    ### repulsive force
    r_0 = 1e4
    psi_0 = 1e3
    # exp type energy
    Erepulsive = psi_0 * tf.exp(-r_0 * tf.math.abs(gap_n))
    contact_work = global_weights_boundary_t[cond]*(Erepulsive)*jacobian_boundary_t[cond]
    
    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points  
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # contact_work_reshaped = bkd.sum(bkd.reshape(contact_work, (n_e_boundary_contact, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) - bkd.reduce_sum(external_work_reshaped) + bkd.reduce_sum(contact_work_reshaped)
    
    return [internal_energy, -external_work, contact_work]


def points_at_top(x):
    cond_points_top = np.isclose(x, 0)
    return cond_points_top

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(-x_loc)/e_modul, v/e_modul], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=2000, display_every=100)

model.compile("L-BFGS")
model.train_step.optimizer_kwargs["options"]['maxiter']=1000
losshistory, train_state = model.train(display_every=200)

# # post

# X, offset, cell_types, dol_triangles = geom.get_mesh()
# nu,lame,shear,e_modul = problem_parameters()

# # start_time_calc = time.time()
# output = model.predict(X)
# # end_time_calc = time.time()
# # final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# # print(final_time)

# u_x_pred, u_y_pred = output[:,0], output[:,1]
# u_pred, v_pred = output[:,0], output[:,1]
# sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)


# combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
# combined_stress = tuple(np.vstack((sigma_xx.flatten(), sigma_yy.flatten(), sigma_xy.flatten())))

# file_path = os.path.join(os.getcwd(), "deep_energy_hertzian")

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)

# #np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress})


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

fem_path = str(Path(__file__).parent.parent.parent)+"/elasticity_2d/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
df = pd.read_csv(fem_path)
fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
fem_results = fem_results.to_numpy()
node_coords_xy = fem_results[:,0:2]
displacement_fem = fem_results[:,2:4]
stress_fem = fem_results[:,4:7]

X = node_coords_xy
x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triangles = tri.Triangulation(x, y)

# # predictions
# start_time_calc = time.time()
output = model.predict(X)
# end_time_calc = time.time()
# final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# print(final_time)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = model.predict(X, operator=stress_plane_strain)
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))
combined_stress_polar_pred = tuple(np.vstack((np.array(sigma_rr_pred.tolist()),np.array(sigma_theta_pred.tolist()),np.array(sigma_rtheta_pred.tolist()))))

# fem
u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]
sigma_rr_fem, sigma_theta_fem, sigma_rtheta_fem = polar_transformation_2d(sigma_xx_fem, sigma_yy_fem, sigma_xy_fem, X)

combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))
combined_stress_polar_fem = tuple(np.vstack((np.array(sigma_rr_fem.tolist()),np.array(sigma_theta_fem.tolist()),np.array(sigma_rtheta_fem.tolist()))))

# error
error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

error_stress_x = abs(np.array(sigma_xx_pred.flatten().tolist()) - sigma_xx_fem.flatten())
error_stress_y =  abs(np.array(sigma_yy_pred.flatten().tolist()) - sigma_yy_fem.flatten())
error_stress_xy =  abs(np.array(sigma_xy_pred.flatten().tolist()) - sigma_xy_fem.flatten())
combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))

error_polar_stress_x = abs(np.array(sigma_rr_pred.flatten().tolist()) - sigma_rr_fem.flatten())
error_polar_stress_y =  abs(np.array(sigma_theta_pred.flatten().tolist()) - sigma_theta_fem.flatten())
error_polar_stress_xy =  abs(np.array(sigma_rtheta_pred.flatten().tolist()) - sigma_rtheta_fem.flatten())
combined_error_polar_stress = tuple(np.vstack((error_polar_stress_x, error_polar_stress_y, error_polar_stress_xy)))

file_path = os.path.join(os.getcwd(), "deep_energy_hertzian_normal_contact_repulsive")

dol_triangles = triangles.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = {"displacement_pred" : combined_disp_pred,
                                               "displacement_fem" : combined_disp_fem,
                                               "stress_pred" : combined_stress_pred, 
                                               "stress_fem" : combined_stress_fem, 
                                               "polar_stress_pred" : combined_stress_polar_pred, 
                                               "polar_stress_fem" : combined_stress_polar_fem, 
                                               "error_disp" : combined_error_disp, 
                                               "error_stress" : combined_error_stress, 
                                               "error_polar_stress" : combined_error_polar_stress
                                            })