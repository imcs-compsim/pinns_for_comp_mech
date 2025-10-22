import numpy as np
import matplotlib.pyplot as plt
import os
# os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from deepxde import backend as bkd
import torch
from pathlib import Path

import pyvista as pv
import time
from utils.postprocess.custom_callbacks import LossPlateauStopping, WeightsBiasPlateauStopping

dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''
@author: svoelkl

Torsion test for a 3D block, done with an incremental approach.
'''

from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Block_3D_hex
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import get_stress_tensor, get_elastic_strain_3d, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

from utils.deep_energy.deep_pde import DeepEnergyPDE
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.vpinns.quad_rule import GaussQuadratureRule

from utils.hyperelasticity import hyperelasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, first_piola_stress_tensor_3D, cauchy_stress_3D

from utils.postprocess.save_normals_tangentials_to_vtk import export_normals_tangentials_to_vtk
from deepxde.optimizers.config import LBFGS_options

time_dict = {"meshing":[],
             "element_information":[],
             "setup":[],
             "relaxation_compiling":[],
             "relaxation_training":[],
             "simulation_compiling_adam":[],
             "simulation_training_adam":[],
             "simulation_compiling_lbfgs":[],
             "simulation_training_lbfgs":[],
             "simulation_prediction":[],
             "total":[]}
time_dict["total"].append(time.time())
time_dict["meshing"].append(time.time())

length = 4
height = 1
width = 1
seed_l = 40
seed_h = 10
seed_w = 10
origin = [0, -0.5, -0.5]

# The applied pressure 
pressure = -0.1

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())

domain_dimension = 3
quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = domain_dimension - 1
quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=2) # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

def on_back(x):
    return np.isclose(x[0], length)

boundary_selection_map = [{"boundary_function" : on_back, "tag" : "on_back"}]

geom = GmshGeometryElementDeepEnergy(
                           gmsh_model,
                           dimension=domain_dimension, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           coord_quadrature_boundary=coord_quadrature_boundary,
                           boundary_dim=boundary_dimension,
                           weight_quadrature_boundary=weight_quadrature_boundary,
                           boundary_selection_map=boundary_selection_map)
time_dict["element_information"].append(time.time())
time_dict["setup"].append(time.time())
# export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="block_boundary_normals")# # change global variables in elasticity_utils
# hyperelasticity_utils.e_modul = 1.33
# hyperelasticity_utils.nu = 0.3
# nu,lame,shear,e_modul = compute_elastic_properties()

# # change global variables in elasticity_utils
# elasticity_utils.lame = lame
# elasticity_utils.shear = shear

# The applied pressure

pressure = 1
# hyperelasticity_utils.lame = 115.38461538461539
# hyperelasticity_utils.shear = 76.92307692307692
hyperelasticity_utils.e_modul = 1.33
hyperelasticity_utils.nu = 0.33

nu,lame,shear,e_modul = compute_elastic_properties()
applied_disp_y = -pressure/e_modul*(1-nu**2)*1

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
    
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)[beg_pde:beg_boundary]
    
    internal_energy = global_element_weights_t[:,0:1]*global_element_weights_t[:,1:2]*global_element_weights_t[:,2:3]*(internal_energy_density)*jacobian_t
    
    # internal_energy = torch.ones_like(internal_energy)-torch.exp(-internal_energy)
    # internal_energy = torch.log(internal_energy+1)
    # internal_energy = torch.tanh(internal_energy)
    # internal_energy = torch.arcsinh(internal_energy)
    # internal_energy = torch.pow(internal_energy, np.pi/4)
    # internal_energy = torch.pow(internal_energy,1/np.sqrt(2))
    # internal_energy = torch.sqrt(internal_energy)
    # internal_energy = torch.arccosh(internal_energy+1)
   

    return [internal_energy]

def points_at_back(x, on_boundary):
    points_bottom = np.isclose(x[0],0)
    
    return on_boundary and points_bottom

bc_u_y = dde.DirichletBC(geom, lambda _: 0, points_at_back, component=1)
bc_u_z = dde.DirichletBC(geom, lambda _: 0, points_at_back, component=2)

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

def output_transform(x, y):
    # displacement field (u, v, w)
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]

    y0, z0 = 0.0, 0.0
    # theta = 2*np.pi / 3
    # theta_deg = 150
    theta = np.radians(theta_deg)
    s = x_loc / length
    # print(theta_deg)

    # rotation displacement at x = L
    v_l = (y0 + (y_loc - y0) * np.cos(theta) - (z_loc - z0) * np.sin(theta) - y_loc)
    w_l = (z0 + (y_loc - y0) * np.sin(theta) + (z_loc - z0) * np.cos(theta) - z_loc)
    
    # Simplified version for theta_deg = 180, and the center is y0, z0 = 0.0, 0.0
    # v_l = -2*y_loc
    # w_l = -2*z_loc 

    u_out = s * (1-s) * u  # no u_x prescribed, just fix at x=0
    v_out = s * v_l + s * (1 - s) * v  # smooth blend
    w_out = s * w_l + s * (1 - s) * w

    return bkd.concat([u_out, v_out, w_out], axis=1)

# 3 inputs, 3 outputs for 3D 
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights=None

model = dde.Model(data, net)

# Model parameters 
steps = 10
torsion_angle = 150
model_path = str(Path(__file__).parent)
simulation_case = f"3d_block_torsion_nonlinear_displacement_incremental_exponential_decay"
learning_rate_adam = 1E-3
learning_rate_total_decay = 1E-3
adam_iterations = 5000
exponential_decay = learning_rate_total_decay ** (1 / adam_iterations)
lbfgs_iterations = 0
rel_err_l2_disp = []
rel_err_l2_stress = []
rel_err_l2_int_disp = []
rel_err_l2_int_stress = []
l2_iteration = []
relaxation_adam_iterations = 0 # just to not get any errors when not using it (undefined variable in naming)
relaxation = False
earlystopping = True
earlystopping_choice = "weightsbiases" # "loss" or "weightsbiases"
time_dict["setup"].append(time.time())

if relaxation:
    time_dict["relaxation_compiling"].append(time.time())
    relaxation_epsilon = 1e0
    relaxation_adam_iterations = 5000
    print(f"\nRelaxation step using a factor of {relaxation_epsilon} of the step width with {relaxation_adam_iterations} iterations.\n")
    theta_deg = relaxation_epsilon * torsion_angle / steps
    model.compile("adam", lr=learning_rate_adam)
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(iterations=relaxation_adam_iterations, display_every=100)
    time_dict["relaxation_training"].append(time.time())

if earlystopping:
    if earlystopping_choice == "loss":
        early = LossPlateauStopping(patience=500, min_delta=1e-5)
    elif earlystopping_choice == "weightsbiases":
        early = WeightsBiasPlateauStopping(patience=500, min_delta=1e-3, norm_choice="fro")
    else:
        raise ValueError("The specified stopping choice is not implemented or correct.")

# Incremental loop
for i in range(steps):
    theta_deg = torsion_angle/steps*(i+1)
    print(f"\nTraining for an angle of {theta_deg}°.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile("adam", lr=learning_rate_adam, decay=("exponential", exponential_decay))
    time_dict["simulation_compiling_adam"].append(time.time())
    time_dict["simulation_training_adam"].append(time.time())
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100, callbacks=[early for _ in [1] if earlystopping])
    time_dict["simulation_training_adam"].append(time.time())

    if lbfgs_iterations>0:
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
        model.compile("L-BFGS")
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        time_dict["simulation_training_lbfgs"].append(time.time())
        losshistory, train_state = model.train(display_every=1000)
        time_dict["simulation_training_lbfgs"].append(time.time())

    # Save results
    time_dict["simulation_prediction"].append(time.time())
    points, _, cell_types, elements = geom.get_mesh()
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    output = model.predict(points)
    displacement_pred = np.column_stack((output[:,0:1], output[:,1:2], output[:,2:3]))
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx, sigma_yz, sigma_zy = model.predict(points, operator=cauchy_stress_3D)
    cauchy_stress_pred = np.column_stack((sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz))
    grid.point_data["pred_displacement"] = displacement_pred
    grid.point_data["pred_cauchy_stress"] = cauchy_stress_pred

    ## Compare with FEM reference
    if (theta_deg % 15 == 0) & (theta_deg <= torsion_angle):
        fem_path = str(Path(__file__).parent.parent)
        fem_reference = pv.read(fem_path+f"/fem_reference/fem_reference_3d_block_torsion_angle_{int(theta_deg):03}.vtu")
        points_fem = fem_reference.points
        displacement_fem = fem_reference.point_data["displacement"]
        cauchy_stress_fem = fem_reference.point_data["nodal_cauchy_stresses_xyz"]

        # Compute values on FEM nodes
        displacement_pred_on_fem_mesh = model.predict(points_fem)
        sigma_xx_pred_on_fem_mesh, sigma_yy_pred_on_fem_mesh, sigma_zz_pred_on_fem_mesh, sigma_xy_pred_on_fem_mesh, _, sigma_xz_pred_on_fem_mesh, _, sigma_yz_pred_on_fem_mesh, _ = model.predict(points_fem, operator=cauchy_stress_3D)
        cauchy_stress_pred_on_fem_mesh = np.column_stack((sigma_xx_pred_on_fem_mesh, sigma_yy_pred_on_fem_mesh, sigma_zz_pred_on_fem_mesh, sigma_xy_pred_on_fem_mesh, sigma_yz_pred_on_fem_mesh, sigma_xz_pred_on_fem_mesh))

        # Compute L2-error
        l2_iteration.append(train_state.step)
        rel_err_l2_disp.append(np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem) / np.linalg.norm(displacement_fem))
        print(f"Relative L2 error for displacement (discrete):   {rel_err_l2_disp[-1]}")
        rel_err_l2_stress.append(np.linalg.norm(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem) / np.linalg.norm(cauchy_stress_fem))
        print(f"Relative L2 error for stress (discrete):         {rel_err_l2_stress[-1]}")

        # Compute L2-error with integrals
        volume_integral = fem_reference.copy()
        volume_integral.point_data["squared_error_disp"] = np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem) ** 2
        volume_integral.point_data["squared_disp"] = np.linalg.norm(displacement_fem) ** 2
        volume_integral.point_data["squared_error_stress"] = np.linalg.norm(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem) ** 2
        volume_integral.point_data["squared_stress"] = np.linalg.norm(cauchy_stress_fem) ** 2
        volume_integral = volume_integral.integrate_data()
        rel_err_l2_int_disp.append(np.sqrt(volume_integral.point_data["squared_error_disp"][0] / volume_integral.point_data["squared_disp"][0]))
        print(f"Relative L2 error for displacement (continuous): {rel_err_l2_int_disp[-1]}")
        rel_err_l2_int_stress.append(np.sqrt(volume_integral.point_data["squared_error_stress"][0] / volume_integral.point_data["squared_stress"][0]))
        print(f"Relative L2 error for stress (continuous):       {rel_err_l2_int_stress[-1]}")

    file_path = os.path.join(model_path, f"{simulation_case}_{int(theta_deg):03}")
    grid.save(f"{file_path}.vtu")
    time_dict["simulation_prediction"].append(time.time())

model.save(f"{model_path}/{simulation_case}")
dde.saveplot(
    losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
    loss_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_loss.dat", 
    train_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_train.dat", 
    test_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_test.dat"
)

fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(losshistory.steps, [sum(l) for l in losshistory.loss_train], color="b", lw=2, label="Internal energy", marker="x")
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("Energy", size=17)
ax1.set_yscale("log")
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig(f"{model_path}/{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_loss_plot.png", dpi=300)

if l2_iteration:
    fig2, ax2 = plt.subplots(figsize=(10,8))
    ax2.plot(l2_iteration, rel_err_l2_disp, color="b", lw=2, label="$L_2$-error for displacement", marker="x")
    ax2.plot(l2_iteration, rel_err_l2_stress, color="r", lw=2, label="$L_2$-error for cauchy stress", marker="x")
    ax2.set_xlabel("Iterations", size=17)
    ax2.set_ylabel("$L_2$ norm", size=17)
    ax2.set_yscale("log")
    ax2.tick_params(axis="both", labelsize=15)
    ax2.legend(fontsize=17)
    ax2.grid()
    plt.tight_layout()
    fig2.savefig(f"{model_path}/{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_l2_norm_over_iterations.png", dpi=300)
time_dict["total"].append(time.time())

# Print times to output file
with open(f"{model_path}/{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_times.txt", "w") as text_file:
    print(f"Compilation and training times in       [s]", file=text_file)
    print(f"==============================================", file=text_file)
    print(f"Meshing:                              {(time_dict["meshing"][1] - time_dict["meshing"][0]):8.3f}", file=text_file)
    print(f"Building element information:         {(time_dict["element_information"][1] - time_dict["element_information"][0]):8.3f}", file=text_file)
    if relaxation:
        print(f"Relaxation compilation:               {(time_dict["relaxation_compiling"][1] - time_dict["relaxation_compiling"][0]):8.3f}", file=text_file)
        print(f"Relaxation training:                  {(time_dict["relaxation_training"][1] - time_dict["relaxation_training"][0]):8.3f}", file=text_file)
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