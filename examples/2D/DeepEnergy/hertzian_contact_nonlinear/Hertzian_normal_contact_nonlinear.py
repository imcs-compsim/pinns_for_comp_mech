### Quarter disc hertzian contact problem using the Deep Energy Method (DEM)
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
from deepxde import backend as bkd
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import QuarterDisc
from utils.elasticity.elasticity_utils import problem_parameters, elastic_strain_2d, stress_plane_strain
from utils.geometry.geometry_utils import polar_transformation_2d
from utils.elasticity import elasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, first_piola_stress_tensor_2D
from utils.hyperelasticity import hyperelasticity_utils
from utils.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from utils.contact_mech import contact_utils
from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.deep_energy.deep_pde import DeepEnergyPDE

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = False

## Create geometry
# Dimensions of disk
radius = 1
center = [0,0]
# Create the quarter disk using gmsh
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.04, angle=225, refine_times=10, gmsh_options=gmsh_options)
gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)
# Modifications to define a proper outer normal
revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
# Define boundary selection map
def on_boundary_circle_contact(x):
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)
def on_top(x):
    return np.isclose(x[1],0)
def points_at_top(x):
    cond_points_top = np.isclose(x, 0)
    return cond_points_top
boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                          {"boundary_function" : on_top, "tag" : "on_top"},]
# Define quadrature rule for interior
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()
# Define quadrature rule for boundary
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=1, ngp=6) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()
# Create geom object
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
# Define geometric parameters
projection_plane = {"y" : -1} # projection plane formula

## Adjust global definitions
# Material parameters
hyperelasticity_utils.e_modul = 50
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

# Communicate parameters to dependencies
elasticity_utils.geom = geom
contact_utils.projection_plane = projection_plane

## Define BCs
# Applied pressure 
ext_traction = 5

## Define energy potentials (internal energy, external work and contact work)
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
    
    ## Internal energy
    # Get internal energy density
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)
    # Compute internal energy
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t

    ## External work
    # Select the points where external force is applied
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:,1:2][beg_boundary:][cond]
    # Get external work density
    external_force_density = -ext_traction*u_y
    # Compute external work
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]

    ## Contact work
    # Select the points on the boundary
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    # Compute boundary gap
    gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
    # Get contact force density
    eta=3e4
    contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
    # Compute contact work
    contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
    
    return [internal_energy, -external_work, contact_work]



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
    '''
    Enforce the following conditions in a hard way
            u(x=0)=0
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(-x_loc)/e_modul, v/e_modul], axis=1)

## Define the neural network
layer_size = [2] + [50] * 5 + [2] # 2 inputs: x, y, 5 hidden layers with 50 neurons each, 2 outputs: ux, uy
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent)
simulation_case = f"herztian_contact_nonlinear"
adam_iterations = 5000

if not restore_pretrained_model:
    start_time_train = time.time()

    model.compile("adam", lr=0.001) # No adjustment of loss weights
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)
    end_time_adam_train = time.time()

    model.compile("L-BFGS-B") # No adjustment of loss weights
    dde.optimizers.config.set_LBFGS_options(maxiter=1000)
    end_time_LBFGS_compile = time.time()
    losshistory, train_state = model.train(display_every=200, model_save_path=f"{model_path}/{simulation_case}")

    end_time_train = time.time()
    time_train = f"Total compilation and training time: {(end_time_train - start_time_train):.3f} seconds"
    print(time_train)

    # Retrieve the total number of iterations at the end of training
    n_iterations = train_state.step
    
    # Print times to output file
    with open(f"{model_path}/{simulation_case}-{n_iterations}_times.txt", "w") as text_file:
        print(f"Compilation and training times in [s]", file=text_file)
        print(f"Adam compilation:    {(end_time_adam_compile - start_time_train):6.3f}", file=text_file)
        print(f"Adam training:       {(end_time_adam_train - end_time_adam_compile):6.3f}", file=text_file)
        print(f"L-BFGS compilation:  {(end_time_LBFGS_compile - end_time_adam_train):6.3f}", file=text_file)
        print(f"L-BFGS training:     {(end_time_train - end_time_LBFGS_compile):6.3f}", file=text_file)
        print(f"Total:               {(end_time_train - start_time_train):6.3f}", file=text_file)

    # Save results
    dde.saveplot(
        losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
        loss_fname=f"{simulation_case}-{n_iterations}_loss.dat", 
        train_fname=f"{simulation_case}-{n_iterations}_train.dat", 
        test_fname=f"{simulation_case}-{n_iterations}_test.dat"
    )

else:
    n_iterations = 17938
    model_restore_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}.ckpt"
    model_loss_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}_loss.dat"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

## Create a comparison with FEM results
# Load the FEM results

# X, offset, cell_types, dol_triangles = geom.get_mesh()

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

# # predictions
# start_time_calc = time.time()
# output = model.predict(X)
# end_time_calc = time.time()
# final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
# print(final_time)

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