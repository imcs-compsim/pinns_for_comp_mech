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

from compsim_pinns.geometry.gmsh_models import RingHalf
from compsim_pinns.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, first_piola_stress_tensor_2D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from compsim_pinns.contact_mech import contact_utils

from compsim_pinns.postprocess.custom_callbacks import EpochTracker, SaveModelVTU

from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

radius = 1
center = [0,0]

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = RingHalf(center=[0,0,0], inner_radius=0.9, outer_radius=1, mesh_size=0.025, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_3"]
revert_normal_dir_list = [1,1,2,1]

def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius)

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
# change global variables in elasticity_utils
hyperelasticity_utils.e_modul = 10
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

applied_disp_y = -0.8

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

projection_plane = {"y" : -1} # projection plane formula
contact_utils.projection_plane = projection_plane

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
    
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
    
    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    
    # gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
    # gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
    eta=3e4
    contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
    
    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points  
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # contact_work_reshaped = bkd.sum(bkd.reshape(contact_work, (n_e_boundary_contact, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) - bkd.reduce_sum(external_work_reshaped) + bkd.reduce_sum(contact_work_reshaped)
    
    return [internal_energy, contact_work]

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None
)

# stabilzation model epoch
stabilization_model_epoch = None
epochs = 50000
steps = 10

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    if model.data.current_epoch is not None:
        if stabilization_model_epoch is not None:
            current_epoch = model.data.current_epoch - stabilization_model_epoch
        else:
            current_epoch = model.data.current_epoch

        step_size = epochs / steps  # e.g., 10
        current_step = int(current_epoch // step_size)
        displacement_chunk = (current_step + 1) * applied_disp_y / steps
    else:
        displacement_chunk = 0
        
    # tf.math.abs(x_loc)
    return bkd.concat([u*(-y_loc)/e_modul, v*(-y_loc)/e_modul+displacement_chunk], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
file_path = os.path.join(os.getcwd(), "deep_energy_contact_ring_instability_half")
epoch_tracker = EpochTracker()
model_saver_incremental = SaveModelVTU(period=int(epochs/steps), filename=file_path)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=epochs, callbacks=[epoch_tracker, model_saver_incremental], display_every=100)

# # # post

# # X, offset, cell_types, dol_triangles = geom.get_mesh()
# # nu,lame,shear,e_modul = problem_parameters()

# # # start_time_calc = time.time()
# # output = model.predict(X)
# # # end_time_calc = time.time()
# # # final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# # # print(final_time)

# # u_x_pred, u_y_pred = output[:,0], output[:,1]
# # u_pred, v_pred = output[:,0], output[:,1]
# # sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)


# # combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
# # combined_stress = tuple(np.vstack((sigma_xx.flatten(), sigma_yy.flatten(), sigma_xy.flatten())))

# # file_path = os.path.join(os.getcwd(), "deep_energy_hertzian")

# # x = X[:,0].flatten()
# # y = X[:,1].flatten()
# # z = np.zeros(y.shape)

# # #np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

# # unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
# #                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress})


# ###################################################################################
# ############################## VISUALIZATION PARTS ################################
# ###################################################################################

# def polar_transformation_2d_tensor(T_xx, T_yy, T_xy, T_yx, X):
#     '''
#     Transforms a general 2nd-order 2D tensor (not necessarily symmetric) from Cartesian to polar coordinates.

#     Parameters
#     ----------
#     X : numpy array, shape (N, 2)
#         Coordinates of points.
#     T_xx, T_yy, T_xy, T_yx : numpy arrays
#         Components of the tensor in Cartesian coordinates.

#     Returns
#     -------
#     T_rr, T_rtheta, T_thetar, T_thetatheta : numpy arrays
#         Components of the tensor in polar coordinates.
#     '''

#     theta = np.arctan2(X[:, 1], X[:, 0])  # in radians
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)

#     # Rotation matrix components
#     Q11 = cos_theta.reshape(-1,1)
#     Q12 = sin_theta.reshape(-1,1)
#     Q21 = -sin_theta.reshape(-1,1)
#     Q22 = cos_theta.reshape(-1,1)

#     # Perform the transformation using Einstein summation convention
#     # T'_ij = Q_ip Q_jq T_pq
#     T_rr = Q11 * (Q11 * T_xx + Q12 * T_yx) + Q12 * (Q11 * T_xy + Q12 * T_yy)
#     T_rtheta = Q11 * (Q21 * T_xx + Q22 * T_yx) + Q12 * (Q21 * T_xy + Q22 * T_yy)
#     T_thetar = Q21 * (Q11 * T_xx + Q12 * T_yx) + Q22 * (Q11 * T_xy + Q12 * T_yy)
#     T_thetatheta = Q21 * (Q21 * T_xx + Q22 * T_yx) + Q22 * (Q21 * T_xy + Q22 * T_yy)

#     return T_rr.astype(np.float32), T_rtheta.astype(np.float32), T_thetar.astype(np.float32), T_thetatheta.astype(np.float32)

# X, offset, cell_types, dol_triangles = geom.get_mesh()

# # start_time_calc = time.time()
# output = model.predict(X)
# # end_time_calc = time.time()
# # final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# # print(final_time)

# u_x_pred, u_y_pred = output[:,0], output[:,1]
# sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(X, operator=cauchy_stress_2D)

# combined_disp = tuple(np.vstack((u_x_pred, u_y_pred, np.zeros(u_x_pred.shape[0]))))
# combined_stress = tuple(np.vstack((sigma_xx.flatten(), sigma_yy.flatten(), sigma_xy.flatten())))

# file_path = os.path.join(os.getcwd(), f"deep_energy_contact_ring_instability_half_p_{abs(applied_disp_y)}")

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)

# #np.savetxt("Lame_inverse_large", X=np.hstack((X,output[:,0:2])))

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress})