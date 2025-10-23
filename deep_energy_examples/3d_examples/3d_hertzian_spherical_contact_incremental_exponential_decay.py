import numpy as np
import matplotlib.pyplot as plt
import os
import deepxde as dde
from deepxde import backend as bkd
from pathlib import Path
import pyvista as pv
import time
from utils.postprocess.custom_callbacks import LossPlateauStopping, WeightsBiasPlateauStopping

dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)

import torch
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
'''
@author: svoelkl

Torsion test for a 3D block, done with an incremental approach.
'''

from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.geometry.gmsh_models import SphereEighthHertzian
from utils.geometry.geometry_utils import polar_transformation_3d_spherical

from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import get_stress_tensor, get_elastic_strain_3d, problem_parameters
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

from utils.deep_energy.deep_pde import DeepEnergyPDE
from utils.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from utils.vpinns.quad_rule import GaussQuadratureRule

from utils.hyperelasticity import hyperelasticity_utils
from utils.hyperelasticity.hyperelasticity_utils import strain_energy_neo_hookean_3d, compute_elastic_properties, first_piola_stress_tensor_3D, cauchy_stress_3D, green_lagrange_strain_3D
from utils.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from utils.contact_mech import contact_utils

from utils.experiment_logger import ExperimentLogger

from deepxde.optimizers.config import LBFGS_options

# Create an experiment logger
with ExperimentLogger("./experiment_logs.h5") as explog:

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

    radius = 1
    center = [0,0,0]

    Block_3D_obj = SphereEighthHertzian(radius=radius, center=center)

    gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)
    time_dict["meshing"].append(time.time())
    explog.log_time("meshing")
    time_dict["element_information"].append(time.time())

    def on_boundary_circle_contact(x):
        return np.isclose(np.linalg.norm(x - center, axis=-1), radius)

    def on_top(x):
        return np.isclose(x[1],0)

    boundary_selection_map = [{"boundary_function" : on_boundary_circle_contact, "tag" : "on_boundary_circle_contact"},
                            {"boundary_function" : on_top, "tag" : "on_top"},]

    domain_dimension = 3
    quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=domain_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
    coord_quadrature, weight_quadrature = quad_rule.generate()

    boundary_dimension = domain_dimension - 1
    quad_rule_boundary_integral = GaussQuadratureRule(rule_name="gauss_legendre", dimension=boundary_dimension, ngp=4, element_type="simplex") # gauss_legendre gauss_labotto
    coord_quadrature_boundary, weight_quadrature_boundary = quad_rule_boundary_integral.generate()

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
    explog.log_time("element_information")
    time_dict["setup"].append(time.time())

    # change global variables in elasticity_utils
    hyperelasticity_utils.e_modul = 50
    hyperelasticity_utils.nu = 0.3
    nu,lame,shear,e_modul = compute_elastic_properties()

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
        
        internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)
        
        internal_energy = global_element_weights_t*(internal_energy_density[beg_pde:beg_boundary])*jacobian_t
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
        
        gap_n = calculate_gap_in_normal_direction_deep_energy(inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond)
        #gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
        #gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
        eta=3e4
        contact_force_density = 1/2*eta*bkd.relu(-gap_n)*bkd.relu(-gap_n)
        contact_work = global_weights_boundary_t[cond]*(contact_force_density)*jacobian_boundary_t[cond]
        
        return [internal_energy, -external_work, contact_work]

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
        u = y[:, 0:1]
        v = y[:, 1:2]
        w = y[:, 2:3]
        
        x_loc = x[:, 0:1]
        y_loc = x[:, 1:2]
        z_loc = x[:, 2:3]
        
        # define surfaces
        # top_surface = -y_loc
        x_0_surface = x_loc
        z_0_surface = z_loc
        
        return bkd.concat([u*(x_0_surface)/e_modul, #displacement in x direction is 0 at x=0
                        v/e_modul,
                        w*(z_0_surface)/e_modul, #displacement in z direction is 0 at z=0
                        ], axis=1)

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
    max_ext_traction = 5
    model_path = str(Path(__file__).parent)
    simulation_case = f"3d_hertzian_spherical_contact_incremental_exponential_decay"
    learning_rate_adam = 1E-3
    learning_rate_total_decay = 1E-3
    adam_iterations = 50000
    exponential_decay = learning_rate_total_decay ** (1 / 5000)
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
    explog.log_time("setup")
    time_dict["setup"].append(time.time())

    if relaxation:
        time_dict["relaxation_compiling"].append(time.time())
        relaxation_epsilon = 1e0
        relaxation_adam_iterations = 5000
        print(f"\nRelaxation step using a factor of {relaxation_epsilon} of the step width with {relaxation_adam_iterations} iterations.\n")
        ext_traction = relaxation_epsilon * max_ext_traction / steps
        model.compile("adam", lr=learning_rate_adam)
        time_dict["relaxation_compiling"].append(time.time())
        explog.log_time("relaxation_compiling")
        time_dict["relaxation_training"].append(time.time())
        losshistory, train_state = model.train(iterations=relaxation_adam_iterations, display_every=100)
        time_dict["relaxation_training"].append(time.time())
        explog.log_time("relaxation_training")

    if earlystopping:
        if earlystopping_choice == "loss":
            early = LossPlateauStopping(patience=500, min_delta=1e-5)
        elif earlystopping_choice == "weightsbiases":
            early = WeightsBiasPlateauStopping(patience=500, min_delta=1e-4, norm_choice="fro")
        else:
            raise ValueError("The specified stopping choice is not implemented or correct.")

    # Incremental loop
    for i in range(steps):
        ext_traction = max_ext_traction/steps*(i+1)
        print(f"\nTraining for a traction of {ext_traction}.\n")
        time_dict["simulation_compiling_adam"].append(time.time())
        model.compile("adam", lr=learning_rate_adam, decay=("exponential", exponential_decay))
        time_dict["simulation_compiling_adam"].append(time.time())
        explog.log_time("simulation_compiling_adam")
        time_dict["simulation_training_adam"].append(time.time())
        losshistory, train_state = model.train(iterations=adam_iterations, display_every=100, callbacks=[early for _ in [1] if earlystopping])
        time_dict["simulation_training_adam"].append(time.time())
        explog.log_time("simulation_training_adam")

        if lbfgs_iterations>0:
            time_dict["simulation_compiling_lbfgs"].append(time.time())
            dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
            model.compile("L-BFGS")
            time_dict["simulation_compiling_lbfgs"].append(time.time())
            explog.log_time("simulation_compiling_lbfgs")
            time_dict["simulation_training_lbfgs"].append(time.time())
            losshistory, train_state = model.train(display_every=1000)
            time_dict["simulation_training_lbfgs"].append(time.time())
            explog.log_time("simulation_training_lbfgs")

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
        if ((ext_traction * 10) % 2 == 0) & (ext_traction <= max_ext_traction):
            fem_path = str(Path(__file__).parent.parent)
            fem_reference = pv.read(fem_path+f"/fem_reference/3d_sphere_contact_fem_reference_{int((ext_traction * 10)):02}.vtu")
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
            explog.log_metric("rel_l2_error_disp_disc", rel_err_l2_disp[-1])
            print(f"Relative L2 error for displacement (discrete):   {rel_err_l2_disp[-1]}")
            rel_err_l2_stress.append(np.linalg.norm(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem) / np.linalg.norm(cauchy_stress_fem))
            explog.log_metric("rel_l2_error_stress_disc", rel_err_l2_stress[-1])
            print(f"Relative L2 error for stress (discrete):         {rel_err_l2_stress[-1]}")

            # Compute L2-error with integrals
            volume_integral = fem_reference.copy()
            volume_integral.point_data["squared_error_disp"] = np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem) ** 2
            volume_integral.point_data["squared_disp"] = np.linalg.norm(displacement_fem) ** 2
            volume_integral.point_data["squared_error_stress"] = np.linalg.norm(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem) ** 2
            volume_integral.point_data["squared_stress"] = np.linalg.norm(cauchy_stress_fem) ** 2
            volume_integral = volume_integral.integrate_data()
            rel_err_l2_int_disp.append(np.sqrt(volume_integral.point_data["squared_error_disp"][0] / volume_integral.point_data["squared_disp"][0]))
            explog.log_metric("rel_l2_error_disp_cont", rel_err_l2_int_disp[-1])
            print(f"Relative L2 error for displacement (continuous): {rel_err_l2_int_disp[-1]}")
            rel_err_l2_int_stress.append(np.sqrt(volume_integral.point_data["squared_error_stress"][0] / volume_integral.point_data["squared_stress"][0]))
            explog.log_metric("rel_l2_error_stress_cont", rel_err_l2_int_stress[-1])
            print(f"Relative L2 error for stress (continuous):       {rel_err_l2_int_stress[-1]}")

        file_path = os.path.join(model_path, f"{simulation_case}_{int(ext_traction * 10):02}")
        grid.save(f"{file_path}.vtu")
        time_dict["simulation_prediction"].append(time.time())
        explog.log_time("simulation_prediction")

    model.save(f"{model_path}/{simulation_case}")
    dde.saveplot(
        losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
        loss_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_loss.dat", 
        train_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_train.dat", 
        test_fname=f"{simulation_case}-{relaxation_adam_iterations+steps*(adam_iterations+lbfgs_iterations)}_test.dat"
    )

    fig1, ax1 = plt.subplots(1,2,figsize=(20,8))
    ax1[0].plot(losshistory.steps, [loss[0] for loss in losshistory.loss_train], label="Internal energy", marker="x")
    ax1[0].plot(losshistory.steps, [loss[1] for loss in losshistory.loss_train], label="External work", marker="x")
    ax1[0].plot(losshistory.steps, [loss[2] for loss in losshistory.loss_train], label="Contact work", marker="x")
    ax1[0].plot(losshistory.steps, [sum(losses) for losses in losshistory.loss_train], label="Total energy", marker="x")
    ax1[0].set_xlabel("Iterations", size=17)
    ax1[0].set_ylabel("Energy", size=17)
    ax1[0].tick_params(axis="both", labelsize=15)
    ax1[0].legend(fontsize=17)
    ax1[0].grid()

    ax1[1].plot(losshistory.steps, [abs(loss[0]) for loss in losshistory.loss_train], label="Internal energy", marker="x")
    ax1[1].plot(losshistory.steps, [abs(loss[1]) for loss in losshistory.loss_train], label="External work", marker="x")
    ax1[1].plot(losshistory.steps, [abs(loss[2]) for loss in losshistory.loss_train], label="Contact work", marker="x")
    ax1[1].plot(losshistory.steps, [abs(sum(losses)) for losses in losshistory.loss_train], label="Total energy", marker="x")
    ax1[1].set_xlabel("Iterations", size=17)
    ax1[1].set_ylabel("Energy", size=17)
    ax1[1].set_yscale("log")
    ax1[1].tick_params(axis="both", labelsize=15)
    ax1[1].legend(fontsize=17)
    ax1[1].grid()
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