import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import deepxde as dde
from deepxde import backend as bkd
from pathlib import Path
import pyvista as pv
import time
dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)

import torch
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''
@author: svoelkl

Incremental bending beam test for EBE-PINNs vs. FEM convergence study.
'''
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import Block_2D_square
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_2d, compute_elastic_properties, cauchy_stress_2D, first_piola_stress_tensor_2D, cauchy_stress_2D_mixed_formulation
from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.postprocess.custom_callbacks import LossPlateauStopping, WeightsBiasPlateauStopping

time_dict = {"meshing":[],
             "element_information":[],
             "setup":[],
             "simulation_compiling_adam":[],
             "simulation_training_adam":[],
             "simulation_compiling_lbfgs":[],
             "simulation_training_lbfgs":[],
             "simulation_prediction":[],
             "total":[]}
time_dict["total"].append(time.time())
time_dict["meshing"].append(time.time())

coords_lower_left_corner = [0,-1]
coords_upper_right_corner = [20,1]
mesh_size = 0.25 # default 0.25

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
block_2d = Block_2D_square(coord_left_corner=coords_lower_left_corner, coord_right_corner=coords_upper_right_corner, mesh_size=mesh_size, gmsh_options=gmsh_options)
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())
domain_dimension = 2
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = domain_dimension - 1
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=4) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

l_beam = block_2d.coord_right_corner[0] -block_2d.coord_left_corner[0]
h_beam = block_2d.coord_right_corner[1] -block_2d.coord_left_corner[1]

def boundary_right(x):
    return np.isclose(x[0],l_beam)

boundary_selection_map = [{"boundary_function" : boundary_right, "tag" : "boundary_right"}]

revert_curve_list = []
revert_normal_dir_list = [2,2,1,2] # all in x-direction: [2,2,1,2]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=domain_dimension,
                           coord_quadrature=coord_quadrature,
                           weight_quadrature= weight_quadrature,
                           revert_curve_list=revert_curve_list,
                           revert_normal_dir_list=revert_normal_dir_list,
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           boundary_dim=boundary_dimension,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)
time_dict["element_information"].append(time.time())
time_dict["setup"].append(time.time())

# change global variables in elasticity_utils
hyperelasticity_utils.lame = 2.78
hyperelasticity_utils.shear = 4.17
hyperelasticity_utils.stress_state = "plane_strain"
nu,lame,shear,e_modul = compute_elastic_properties()

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

    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs[:,0:2])

    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["boundary_right"]

    u_y = outputs[:,1:2][beg_boundary:][cond]

    external_force_density = -shear_load*u_y
    external_work = global_weights_boundary_t[cond]*(external_force_density)*jacobian_boundary_t[cond]
    ####################################################################################################################
    # get the difference between computed stress from predicted displacement and network prediction of stress
    firstpiola_xx_u, firstpiola_yy_u, firstpiola_xy_u, firstpiola_yx_u = first_piola_stress_tensor_2D(inputs, outputs[:,0:2])
    residuum_firstpiolafirstpiola = (firstpiola_xx_u - outputs[:,2:3]) ** 2 + (firstpiola_xy_u - outputs[:,3:4]) ** 2 + (firstpiola_yx_u - outputs[:,4:5]) ** 2 + (firstpiola_yy_u - outputs[:,5:6]) ** 2
    consistency_domain = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*(residuum_firstpiolafirstpiola[beg_pde:beg_boundary])*jacobian_t
    consistency_boundary = global_weights_boundary_t*(residuum_firstpiolafirstpiola[beg_boundary:])*jacobian_boundary_t
    consistency = bkd.concat([consistency_domain, consistency_boundary], axis=0)
    _, _, T_xy, T_yx = cauchy_stress_2D_mixed_formulation(inputs, outputs)
    symmetry = (T_xy - T_yx) ** 2
    divergence = (dde.grad.jacobian(outputs, inputs, i=2, j=0) + dde.grad.jacobian(outputs, inputs, i=3, j=1)) ** 2 + (dde.grad.jacobian(outputs, inputs, i=4, j=0) + dde.grad.jacobian(outputs, inputs, i=5, j=1)) ** 2
    return [internal_energy, -external_work, consistency, symmetry, divergence]

n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def input_transform(x):
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]

    return bkd.concat([2 / h_beam * x_loc , 2 / l_beam * y_loc], axis=1)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]

    P_xx = y[:, 2:3]
    P_xy = y[:, 3:4]
    P_yx = y[:, 4:5]
    P_yy = y[:, 5:6]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]

    return bkd.concat([x_loc/l_beam * h_beam * u , x_loc/l_beam * h_beam * v, (l_beam - x_loc) / l_beam * e_modul * P_xx, 2 / h_beam * (h_beam/2 - y_loc) * 2 / h_beam * (h_beam/2 + y_loc) * e_modul * P_xy, -shear_load + (l_beam - x_loc) / l_beam * e_modul * P_yx, 2 / h_beam * (h_beam/2 - y_loc) * 2 / h_beam * (h_beam/2 + y_loc) * e_modul * P_yy], axis=1)

# 2 inputs, 6 outputs for 2D: u_x, u_y, P_xx, P_xy, P_yx, P_yy
layer_size = [2] + [50] * 5 + [6]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# Model parameters
# The applied pressure 
steps = 1
max_shear_load = 1e-2
model_path = str(Path(__file__).parent)
simulation_case = f"Beam_under_shear_load_mixed_formulation"
learning_rate_adam = 1E-3
learning_rate_total_decay = 1E-1
adam_iterations = 50000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 0
earlystopping = True
earlystopping_choice = "weightsbiases" # "loss" or "weightsbiases"
time_dict["setup"].append(time.time())
# # plot the different shapes over load steps
# shape_points_x = int((coords_upper_right_corner[0]-coords_lower_left_corner[0]) / mesh_size)
# shape_points_y = int((coords_upper_right_corner[1]-coords_lower_left_corner[1]) / mesh_size)
# edge_space_x = np.linspace(coords_lower_left_corner[0], coords_upper_right_corner[0], shape_points_x+1)
# edge_space_y = np.linspace(coords_lower_left_corner[1], coords_upper_right_corner[1], shape_points_y+1)
# coords_edge_points = np.unique(np.vstack([np.stack(np.meshgrid(edge_space_x, [coords_lower_left_corner[1]]), -1).reshape(-1, 2),
#                                             np.stack(np.meshgrid(edge_space_x, [coords_upper_right_corner[1]]), -1).reshape(-1, 2),
#                                             np.stack(np.meshgrid([coords_lower_left_corner[0]], edge_space_y), -1).reshape(-1, 2),
#                                             np.stack(np.meshgrid([coords_upper_right_corner[0]], edge_space_y), -1).reshape(-1, 2)]), axis=0)
# coords_corners = np.array((coords_upper_right_corner,[coords_lower_left_corner[0],coords_upper_right_corner[1]],coords_lower_left_corner,[coords_upper_right_corner[0],coords_lower_left_corner[1]]))
# trajectory_edge_points = np.empty((steps+1,coords_edge_points.shape[0],domain_dimension))
# trajectory_edge_points[0,:,:] = coords_edge_points
# trajectory_corners = np.empty((steps+1,coords_corners.shape[0],domain_dimension))
# trajectory_corners[0,:,:] = coords_corners
# # plot errors over load steps
rel_err_l2_disp = []
rel_err_l2_stress = []
l2_iteration = []

if earlystopping:
    if earlystopping_choice == "loss":
        early = LossPlateauStopping(patience=500, min_delta=1e-5)
    elif earlystopping_choice == "weightsbiases":
        early = WeightsBiasPlateauStopping(patience=500, min_delta=1e-5, norm_choice="fro")
    else:
        raise ValueError("The specified stopping choice is not implemented or correct.")

# Weights
residuum_penalty = 1E-3
symmetry_penalty = 1E-4
divergence_penalty = 1E-5
loss_weights = [1,1,residuum_penalty,symmetry_penalty,divergence_penalty] # internal_energy, external_energy, residuum_penalty, symmetry_penalty

# Incremental loop
for i in range(steps):
    shear_load = max_shear_load/steps*(i+1)
    print(f"\nTraining for a shear load of {shear_load}.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile("adam", loss_weights=loss_weights, lr=learning_rate_adam, decay=("exponential", exponential_decay))
    time_dict["simulation_compiling_adam"].append(time.time())
    time_dict["simulation_training_adam"].append(time.time())
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100, callbacks=[early for _ in [1] if earlystopping])
    time_dict["simulation_training_adam"].append(time.time())

    if lbfgs_iterations>0:
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
        model.compile("L-BFGS", loss_weights=loss_weights)
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        time_dict["simulation_training_lbfgs"].append(time.time())
        losshistory, train_state = model.train(display_every=1000)
        time_dict["simulation_training_lbfgs"].append(time.time())

    # Save results
    time_dict["simulation_prediction"].append(time.time())
    points, _, cell_types, elements = geom.get_mesh()
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    n_points = points.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, np.c_[points, np.zeros((n_points, 1))])
    output = model.predict(points)
    displacement_pred = np.column_stack((output[:,0:1], output[:,1:2]))
    sigma_xx, sigma_yy, sigma_xy, _ = model.predict(points, operator=cauchy_stress_2D_mixed_formulation)
    cauchy_stress_pred = np.column_stack((sigma_xx, sigma_yy, sigma_xy))
    grid.point_data["pred_displacement"] = np.c_[displacement_pred, np.zeros((n_points, 1))]
    grid.point_data["pred_cauchy_stress"] = np.column_stack((cauchy_stress_pred[:, 0], cauchy_stress_pred[:, 1], np.zeros((n_points, 1)), cauchy_stress_pred[:, 2], np.zeros((n_points, 1)), np.zeros((n_points, 1))))

    ## Compare with FEM reference
    if int(round(shear_load * 1E3)) in set(range(1, 11)):
        fem_path = str(Path(__file__).parent.parent)
        fem_reference = pv.read(fem_path+f"/fem_reference/fem_reference_2d_bending_beam_{int(round(shear_load * 1E3)):02}.vtu")
        points_fem = fem_reference.points
        displacement_fem = fem_reference.point_data["displacement"]
        cauchy_stress_fem = fem_reference.point_data["nodal_cauchy_stresses_xyz"]

        # Compute values on FEM nodes
        pred_on_fem_mesh = model.predict(points_fem[:,0:2])
        displacement_pred_on_fem_mesh = pred_on_fem_mesh[:,0:2]
        sigma_xx_pred_on_fem_mesh, sigma_yy_pred_on_fem_mesh, sigma_xy_pred_on_fem_mesh, sigma_yx_pred_on_fem_mesh = model.predict(points_fem[:,0:2], operator=cauchy_stress_2D_mixed_formulation)
        cauchy_stress_pred_on_fem_mesh = np.column_stack((sigma_xx_pred_on_fem_mesh, sigma_yy_pred_on_fem_mesh, np.zeros_like(sigma_xx_pred_on_fem_mesh), sigma_xy_pred_on_fem_mesh, np.zeros_like(sigma_xx_pred_on_fem_mesh), np.zeros_like(sigma_xx_pred_on_fem_mesh)))
        tensor_cauchy_stress_pred_on_fem_mesh = np.transpose(np.array([[sigma_xx_pred_on_fem_mesh.flatten(), sigma_xy_pred_on_fem_mesh.flatten()],
                                                                       [sigma_yx_pred_on_fem_mesh.flatten(), sigma_yy_pred_on_fem_mesh.flatten()]]),(2,0,1))
        tensor_cauchy_stress_fem = np.array([[cauchy_stress_fem[:,0], cauchy_stress_fem[:,3],
                                              cauchy_stress_fem[:,3], cauchy_stress_fem[:,1]]]).T.reshape(-1,2,2)

        # Compute L2-error
        volume_integral = fem_reference.copy()
        volume_integral.point_data["squared_error_disp"] = np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem, axis=1) ** 2
        volume_integral.point_data["squared_disp"] = np.linalg.norm(displacement_fem, axis=1) ** 2
        volume_integral.point_data["squared_error_stress"] = np.linalg.norm(tensor_cauchy_stress_pred_on_fem_mesh - tensor_cauchy_stress_fem, axis=(1,2), ord="fro") ** 2
        volume_integral.point_data["squared_stress"] = np.linalg.norm(tensor_cauchy_stress_fem, axis=(1,2), ord="fro") ** 2
        volume_integral = volume_integral.integrate_data()
        l2_iteration.append(train_state.step)
        rel_err_l2_disp.append(np.sqrt(volume_integral.point_data["squared_error_disp"][0] / volume_integral.point_data["squared_disp"][0]))
        print(f"Relative L2 error for displacement:   {rel_err_l2_disp[-1]}")
        rel_err_l2_stress.append(np.sqrt(volume_integral.point_data["squared_error_stress"][0] / volume_integral.point_data["squared_stress"][0]))
        print(f"Relative L2 error for stress:         {rel_err_l2_stress[-1]}")

        # Compute mean absolute error
        print(f"Mean absolute error for displacement: {np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem)/len(displacement_fem)}")
        print(f"Mean absolute error for stress:       {np.mean(np.linalg.norm(tensor_cauchy_stress_pred_on_fem_mesh - tensor_cauchy_stress_fem, axis=(1,2), ord="fro"))}")

        # Create output with relative pointwise errors
        fem_reference.point_data["displacement_prediction"] = np.hstack((displacement_pred_on_fem_mesh, np.zeros_like(displacement_pred_on_fem_mesh[:,0:1])))
        fem_reference.point_data["cauchy_stresses_prediction"] = cauchy_stress_pred_on_fem_mesh
        fem_reference.point_data["absolute_displacement_error"] = np.hstack((abs(displacement_pred_on_fem_mesh - displacement_fem), np.zeros_like(displacement_pred_on_fem_mesh[:,0:1])))
        fem_reference.point_data["absolute_cauchy_stress_error"] = abs(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem)
        fem_reference.point_data["relative_displacement_error"] = np.divide(np.abs(displacement_pred_on_fem_mesh - displacement_fem), np.abs(displacement_fem), out=np.zeros_like(displacement_fem, dtype=float), where=displacement_fem!=0)
        fem_reference.point_data["relative_cauchy_stress_error"] = np.divide(np.abs(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem), np.abs(cauchy_stress_fem), out=np.zeros_like(cauchy_stress_fem, dtype=float), where=cauchy_stress_fem!=0)
        file_path_fem_compare = os.path.join(model_path, f"{simulation_case}_fem_compare_{int(round(shear_load * 1E3)):02}")
        fem_reference.save(f"{file_path_fem_compare}.vtu")

    # Predict shape of the beam in each time step
    # trajectory_edge_points[i+1,:,:] = coords_edge_points + model.predict(coords_edge_points)
    # trajectory_corners[i+1,:,:] = coords_corners + model.predict(coords_corners)

    # Save results
    file_path = os.path.join(model_path, f"{simulation_case}_{int(shear_load * 1E3):02}")
    grid.save(f"{file_path}.vtu")
    time_dict["simulation_prediction"].append(time.time())

model.save(f"{model_path}/{simulation_case}")
dde.saveplot(
    losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
    loss_fname=f"{simulation_case}-{train_state.step}_loss.dat", 
    train_fname=f"{simulation_case}-{train_state.step}_train.dat", 
    test_fname=f"{simulation_case}-{train_state.step}_test.dat"
)

## Plot energy
fig1, ax1 = plt.subplots(1,2,figsize=(20,8))
ax1[0].plot(losshistory.steps, [loss[0] for loss in losshistory.loss_train], label="Internal energy", marker="x")
ax1[0].plot(losshistory.steps, [loss[1] for loss in losshistory.loss_train], label="External work", marker="x")
ax1[0].plot(losshistory.steps, [sum(losses) for losses in losshistory.loss_train], label="Total energy", marker="x")
ax1[0].set_xlabel("Iterations", size=17)
ax1[0].set_ylabel("Energy", size=17)
ax1[0].tick_params(axis="both", labelsize=15)
ax1[0].legend(fontsize=17)
ax1[0].grid()

ax1[1].plot(losshistory.steps, [abs(loss[0]) for loss in losshistory.loss_train], label="Internal energy", marker="x")
ax1[1].plot(losshistory.steps, [abs(loss[1]) for loss in losshistory.loss_train], label="External work", marker="x")
ax1[1].plot(losshistory.steps, [abs(sum(losses)) for losses in losshistory.loss_train], label="Total energy", marker="x")
ax1[1].set_xlabel("Iterations", size=17)
ax1[1].set_ylabel("Energy", size=17)
ax1[1].set_yscale("log")
ax1[1].tick_params(axis="both", labelsize=15)
ax1[1].legend(fontsize=17)
ax1[1].grid()
plt.tight_layout()
fig1.savefig(f"{model_path}/{simulation_case}-{train_state.step}_loss_plot.png", dpi=300)

## Plot beam outline and trajectory over load steps
# fig2, ax2 = plt.subplots(1,2,figsize=(20,8))
# cmap = plt.colormaps[("cool")]
# colors = cmap(np.linspace(0, 1, steps+1))
# center = coords_edge_points.mean(axis=0)
# angles = np.arctan2(coords_edge_points[:, 1] - center[1], coords_edge_points[:, 0] - center[0])
# sort_idx = np.argsort(angles)
# trajectory_edge_points_sorted = trajectory_edge_points[:, sort_idx, :]
# for t, P in enumerate(trajectory_edge_points_sorted):
#     P = np.vstack([P, P[0]])
#     ax2[0].plot(P[:,0], P[:,1], color=colors[t], lw=2, label=f"Shear load of {t/steps*max_shear_load:1.3f}", marker=".")
# ax2[0].set_xlabel("$x$", size=17)
# ax2[0].set_ylabel("$y$", size=17)
# ax2[0].tick_params(axis="both", labelsize=15)
# ax2[0].grid()
# ax2[0].legend()
# ax2[1].plot(trajectory_corners[:,0,0], trajectory_corners[:,0,1], color="b", lw=2, label="Trajectory of upper right corner", marker="x")
# ax2[1].plot(trajectory_corners[:,3,0], trajectory_corners[:,3,1], color="r", lw=2, label="Trajectory of lower right corner", marker="x")
# ax2[1].set_xlabel("$x$", size=17)
# ax2[1].set_ylabel("$y$", size=17)
# ax2[1].grid()
# ax2[1].legend()
# plt.tight_layout()
# fig2.savefig(f"{model_path}/{simulation_case}-{train_state.step}_edge_trajectory.png", dpi=300)

# Output trajectory points
# np.savez(f"{model_path}/{simulation_case}_edge_trajectory_meshsize_{2/mesh_size:03.0f}.npz", x=trajectory_edge_points_sorted, y=trajectory_corners)
np.savez(f"{model_path}/{simulation_case}_l2_errors.npz", x=rel_err_l2_disp, y=rel_err_l2_stress)

time_dict["total"].append(time.time())
# Print times to output file
with open(f"{model_path}/{simulation_case}-{train_state.step}_times.txt", "w") as text_file:
    print(f"Compilation and training times in       [s]", file=text_file)
    print(f"==============================================", file=text_file)
    print(f"Meshing:                              {(time_dict["meshing"][1] - time_dict["meshing"][0]):8.3f}", file=text_file)
    print(f"Building element information:         {(time_dict["element_information"][1] - time_dict["element_information"][0]):8.3f}", file=text_file)
    if steps > 1:
        for i in range(steps):
            print(f"----------------------------------------------", file=text_file)
            print(f"   Load step {(i+1):2d} compilation (adam):   {(time_dict["simulation_compiling_adam"][(2*i)+1] - time_dict["simulation_compiling_adam"][2*i]):8.3f}", file=text_file)
            print(f"   Load step {(i+1):2d} training (adam):      {(time_dict["simulation_training_adam"][(2*i)+1] - time_dict["simulation_training_adam"][2*i]):8.3f}", file=text_file)
            if lbfgs_iterations > 0:
                print(f"   Load step {(i+1):2d} compilation (L-BFGS): {(time_dict["simulation_compiling_lbfgs"][(2*i)+1] - time_dict["simulation_compiling_lbfgs"][2*i]):8.3f}", file=text_file)
                print(f"   Load step {(i+1):2d} training (L-BFGS):    {(time_dict["simulation_training_lbfgs"][(2*i)+1] - time_dict["simulation_training_lbfgs"][2*i]):8.3f}", file=text_file)
            print(f"   Load step {(i+1):2d} prediction:           {(time_dict["simulation_prediction"][(2*i)+1] - time_dict["simulation_prediction"][2*i]):8.3f}", file=text_file)
        print(f"==============================================", file=text_file)
    print(f"Total compilation (adam):         {(sum(time_dict["simulation_compiling_adam"][1::2]) - (sum(time_dict["simulation_compiling_adam"][::2]))):12.3f}", file=text_file)
    print(f"Total training (adam):            {(sum(time_dict["simulation_training_adam"][1::2]) - (sum(time_dict["simulation_training_adam"][::2]))):12.3f}", file=text_file)
    if lbfgs_iterations > 0:
        print(f"Total compilation (L-BFGS):       {(sum(time_dict["simulation_compiling_lbfgs"][1::2]) - (sum(time_dict["simulation_compiling_lbfgs"][::2]))):12.3f}", file=text_file)
        print(f"Total training (L-BFGS):          {(sum(time_dict["simulation_training_lbfgs"][1::2]) - (sum(time_dict["simulation_training_lbfgs"][::2]))):12.3f}", file=text_file)
    print(f"Total prediction:                 {(sum(time_dict["simulation_prediction"][1::2]) - (sum(time_dict["simulation_prediction"][::2]))):12.3f}", file=text_file)
    print(f"==============================================", file=text_file)
    print(f"Total:                            {(time_dict["total"][1] - time_dict["total"][0]):12.3f}", file=text_file)