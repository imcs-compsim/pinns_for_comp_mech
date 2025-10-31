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
from compsim_pinns.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, first_piola_stress_tensor_2D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from compsim_pinns.contact_mech import contact_utils

from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

from compsim_pinns.postprocess.custom_callbacks import EpochTracker, SaveModelVTU

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.04, angle=225, refine_times=10, gmsh_options=gmsh_options)

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
                           boundary_selection_map=boundary_selection_map,
                           lagrange_method=True)

# change global variables in elasticity_utils
hyperelasticity_utils.e_modul = 50
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

projection_plane = {"y" : -1} # projection plane formula
contact_utils.projection_plane = projection_plane

# The applied pressure 
ext_traction = 5
epochs = 5000
steps = 10

# stabilzation model epoch
stabilization_model_epoch = None

increment_tracker = steps*[True]

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
                     boundary_selection_tag,
                     lagrange_parameter_boundary):
    
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    if model.data.current_epoch is not None:
        # if model.data.current_epoch == 0:
        #     shear_load_chunk = 0
        #     print(shear_load_chunk)
        # else:
        # current_epoch = model.data.current_epoch
        # chunk = current_epoch//((epochs+1)/steps)
        # shear_load_chunk = (chunk + 1)*shear_load/steps
        if stabilization_model_epoch is not None:
            current_epoch = model.data.current_epoch - stabilization_model_epoch
        else:
            current_epoch = model.data.current_epoch
        step_size = epochs / steps  # e.g., 10
        current_step = int(current_epoch // step_size)
        step_load = (current_step + 1) * ext_traction / steps
        #print(shear_load_chunk)
        # if (current_epoch % 2) == 0:
        #     print(shear_load_chunk)
        #if current_epoch//
    else:
        step_load = 0
    
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    
    external_force_density = -step_load*u_y
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)

    eta=3e4
    contact_force_density = 1/(2*eta)*bkd.relu(-lagrange_parameter_boundary[cond]-eta*gap_n)*bkd.relu(-lagrange_parameter_boundary[cond]-eta*gap_n)
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
    
    # update lambda
    if model.data.current_epoch is not None:
        if not(current_epoch == 0) and not(current_epoch == 1):
            # previous_step = int((current_epoch-1) // step_size)
            if ((current_epoch +1 ) % step_size)== 0:
                # The first increment is for the test data, however we dont want to to that
                if increment_tracker[current_step]:
                    lagrange_parameter_boundary[cond] = -eta*bkd.relu(-gap_n)
                    #print("update")
                if increment_tracker[current_step] == True:
                   increment_tracker[current_step] = False 
            
    return [internal_energy, -external_work, contact_work], lagrange_parameter_boundary

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

# model.compile("adam", lr=0.001)
# losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)

file_path = os.path.join(os.getcwd(), "deep_energy_hertzian_normal_contact_nonlinear_lagrange")
epoch_tracker = EpochTracker()
model_saver_incremental = SaveModelVTU(op=cauchy_stress_2D, period=500, stabilization_epoch=stabilization_model_epoch, filename=file_path)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=epochs, callbacks=[epoch_tracker, model_saver_incremental], display_every=100)

# # model.compile("L-BFGS")
# # model.train_step.optimizer_kwargs["options"]['maxiter']=1000
# # losshistory, train_state = model.train(display_every=200)

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

# fem_path = str(Path(__file__).parent.parent.parent)+"/elasticity_2d/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
# df = pd.read_csv(fem_path)
# fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
# fem_results = fem_results.to_numpy()
# node_coords_xy = fem_results[:,0:2]
# displacement_fem = fem_results[:,2:4]
# stress_fem = fem_results[:,4:7]

# X = node_coords_xy
# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)
# triangles = tri.Triangulation(x, y)

# # # predictions
# # start_time_calc = time.time()
# output = model.predict(X)
# # end_time_calc = time.time()
# # final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# # print(final_time)

# u_pred, v_pred = output[:,0], output[:,1]
# sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, sigma_yx_pred = model.predict(X, operator=cauchy_stress_2D)
# sigma_rr_pred, sigma_rtheta_pred, sigma_theta_r_pred, sigma_theta_pred = polar_transformation_2d_tensor(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, sigma_yx_pred, X)

# combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
# combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))
# combined_stress_polar_pred = tuple(np.vstack((np.array(sigma_rr_pred.flatten().tolist()),np.array(sigma_theta_pred.flatten().tolist()),np.array(sigma_rtheta_pred.flatten().tolist()))))

# # fem
# u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
# sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]
# sigma_rr_fem, sigma_rtheta_fem, sigma_theta_r_fem, sigma_theta_fem= polar_transformation_2d_tensor(sigma_xx_fem, sigma_yy_fem, sigma_xy_fem, sigma_xy_fem, X)

# combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
# combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))
# combined_stress_polar_fem = tuple(np.vstack((np.array(sigma_rr_fem.flatten().tolist()),np.array(sigma_theta_fem.flatten().tolist()),np.array(sigma_rtheta_fem.flatten().tolist()))))

# # error
# error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
# error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
# combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

# error_stress_x = abs(np.array(sigma_xx_pred.flatten().tolist()) - sigma_xx_fem.flatten())
# error_stress_y =  abs(np.array(sigma_yy_pred.flatten().tolist()) - sigma_yy_fem.flatten())
# error_stress_xy =  abs(np.array(sigma_xy_pred.flatten().tolist()) - sigma_xy_fem.flatten())
# combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))

# error_polar_stress_x = abs(np.array(sigma_rr_pred.flatten().tolist()) - sigma_rr_fem.flatten())
# error_polar_stress_y =  abs(np.array(sigma_theta_pred.flatten().tolist()) - sigma_theta_fem.flatten())
# error_polar_stress_xy =  abs(np.array(sigma_rtheta_pred.flatten().tolist()) - sigma_rtheta_fem.flatten())
# combined_error_polar_stress = tuple(np.vstack((error_polar_stress_x, error_polar_stress_y, error_polar_stress_xy)))

# file_path = os.path.join(os.getcwd(), "deep_energy_hertzian_normal_contact_nonlinear")

# dol_triangles = triangles.triangles
# offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
# cell_types = np.ones(dol_triangles.shape[0])*5

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = {"displacement_pred" : combined_disp_pred,
#                                                "displacement_fem" : combined_disp_fem,
#                                                "stress_pred" : combined_stress_pred, 
#                                                "stress_fem" : combined_stress_fem, 
#                                                "polar_stress_pred" : combined_stress_polar_pred, 
#                                                "polar_stress_fem" : combined_stress_polar_fem, 
#                                                "error_disp" : combined_error_disp, 
#                                                "error_stress" : combined_error_stress, 
#                                                "error_polar_stress" : combined_error_polar_stress
#                                             })