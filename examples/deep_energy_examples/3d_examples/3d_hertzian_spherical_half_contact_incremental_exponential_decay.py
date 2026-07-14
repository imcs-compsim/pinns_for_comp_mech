"""
Hertzian Spherical Contact Under Incremental Traction
==================================================

This example solves a three-dimensional Hertzian contact problem for a half
sphere under incrementally applied traction with the Energy-based PINN (EPINN).

It is modelled using a Neo-Hookean hyperelastic material. Contact is enforced
through a penalty contribution to the energy, and the applied traction is
increased in load steps. In each load step the model is trained using the Adam
optimizer with an exponentially decaying learning rate.

Results are exported as VTU files for post-processing, while errors, timings,
and loss information are also saved.
"""

import os
import time
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from deepxde import backend as bkd

from compsim_pinns.contact_mech import contact_utils
from compsim_pinns.contact_mech.contact_utils import (
    calculate_gap_in_normal_direction_deep_energy,
)
from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import SphereHalfHertzian
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_3D,
    compute_elastic_properties,
    strain_energy_neo_hookean_3d,
)
from compsim_pinns.postprocess.custom_callbacks import (
    LossPlateauStopping,
    WeightsBiasPlateauStopping,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

# Set default float type to double precision for L-BFGS optimizer
dde.config.set_default_float("float64")

# Fix random seeds for reproducibility
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create a dictionary to store timing information
time_dict = {
    "meshing": [],
    "element_information": [],
    "setup": [],
    "relaxation_compiling": [],
    "relaxation_training": [],
    "simulation_compiling_adam": [],
    "simulation_training_adam": [],
    "simulation_compiling_lbfgs": [],
    "simulation_training_lbfgs": [],
    "simulation_prediction": [],
    "total": [],
}
time_dict["total"].append(time.time())
time_dict["meshing"].append(time.time())

# Geometry and mesh generation
radius = 1
center = [0, 0, 0]
Block_3D_obj = SphereHalfHertzian(radius=radius, center=center)
gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())

# Set up quadrature rules for domain and boundary integration
domain_dimension = 3
boundary_dimension = domain_dimension - 1
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre",
    dimension=domain_dimension,
    ngp=4,
    element_type="simplex",
)
coord_quadrature, weight_quadrature = quad_rule.generate()
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre",
    dimension=boundary_dimension,
    ngp=4,
    element_type="simplex",
)
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)


# Define BCs
def on_boundary_circle_contact(x):
    """Check whether a point satisfies the `on_boundary_circle_contact` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_boundary_circle_contact` evaluation.
    """
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius)


def on_top(x):
    """Check whether a point satisfies the `on_top` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_top` evaluation.
    """
    return np.isclose(x[1], 0)


boundary_selection_map = [
    {
        "boundary_function": on_boundary_circle_contact,
        "tag": "on_boundary_circle_contact",
    },
    {"boundary_function": on_top, "tag": "on_top"},
]

# Define EPINN geometry
geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=domain_dimension,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    coord_quadrature_boundary=coord_quadrature_boundary,
    boundary_dim=boundary_dimension,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
)
time_dict["element_information"].append(time.time())
time_dict["setup"].append(time.time())

# Set material parameters as global variables in elasticity_utils
hyperelasticity_utils.youngs_modulus = 50
hyperelasticity_utils.nu = 0.3
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# Set contact surface definition
elasticity_utils.geom = geom
projection_plane = {"y": -1}  # projection plane formula
contact_utils.projection_plane = projection_plane


# Define the potential energy function for the EPINN
def potential_energy(
    X,
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
):
    """Compute potential energy for this example setup.

    Args:
        X: Input coordinates used by this callback.
        inputs: Value for inputs.
        outputs: Value for outputs.
        beg_pde: Value for beg pde.
        beg_boundary: Value for beg boundary.
        n_e: Value for n e.
        n_gp: Value for n gp.
        n_e_boundary: Value for n e boundary.
        n_gp_boundary: Value for n gp boundary.
        jacobian_t: Value for jacobian t.
        global_element_weights_t: Value for global element weights t.
        mapped_normal_boundary_t: Value for mapped normal boundary t.
        jacobian_boundary_t: Value for jacobian boundary t.
        global_weights_boundary_t: Value for global weights boundary t.
        boundary_selection_tag: Value for boundary selection tag.

    Returns:
        Any: Computed value returned by `potential_energy`.
    """
    # Internal energy
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)
    internal_energy = (
        global_element_weights_t
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )
    # External work
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:, 1:2][beg_boundary:][cond]
    external_force_density = -ext_traction * u_y
    external_work = (
        global_weights_boundary_t[cond]
        * external_force_density
        * jacobian_boundary_t[cond]
    )
    # Contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]
    gap_n = calculate_gap_in_normal_direction_deep_energy(
        inputs[beg_boundary:],
        outputs[beg_boundary:],
        X,
        mapped_normal_boundary_t,
        cond,
    )
    eta = 3e3
    contact_force_density = 1 / 2 * eta * bkd.relu(-gap_n) * bkd.relu(-gap_n)
    contact_work = (
        global_weights_boundary_t[cond]
        * contact_force_density
        * jacobian_boundary_t[cond]
    )

    return [internal_energy, -external_work, contact_work]


# Create the data object for the EPINN
n_dummy = 1
data = DeepEnergyPDE(
    geom,
    potential_energy,
    [],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution="Sobol",
)


# Define the output transform
def output_transform(x, y):
    """Compute output transform for this example setup.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed value returned by `output_transform`.
    """
    return bkd.concat(
        [
            y[:, 1:2] / youngs_modulus,
            y[:, 0:1] / youngs_modulus,
            y[:, 2:3] / youngs_modulus,
        ],
        axis=1,
    )


# Define the neural network architecture
# 3 inputs, 3 outputs for 3D: u_x, u_y, u_z
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# Set the training parameters
steps = 1
max_ext_traction = 5
model_path = str(Path(__file__).parent)
simulation_case = f"3d_hertzian_spherical_half_contact_incremental_exponential_decay"
learning_rate_adam = 1e-3
learning_rate_total_decay = 1e-8
adam_iterations = 50000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 0
rel_err_l2_disp = []
rel_err_l2_stress = []
l2_iteration = []
relaxation_adam_iterations = 0  # avoid undefined variable in naming
relaxation = False
earlystopping = True
earlystopping_choice = "weightsbiases"  # "loss" or "weightsbiases"
time_dict["setup"].append(time.time())

# Optional relaxation step before incremental loading
if relaxation:
    time_dict["relaxation_compiling"].append(time.time())
    relaxation_epsilon = 1e0
    relaxation_adam_iterations = 5000
    print(
        f"\nRelaxation step using a factor of {relaxation_epsilon} "
        f"of the step width with {relaxation_adam_iterations} iterations.\n"
    )
    ext_traction = relaxation_epsilon * max_ext_traction / steps
    model.compile("adam", lr=learning_rate_adam)
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(
        iterations=relaxation_adam_iterations, display_every=100
    )
    time_dict["relaxation_training"].append(time.time())

# Define the early stopping callback
if earlystopping:
    if earlystopping_choice == "loss":
        early = LossPlateauStopping(patience=500, min_delta=1e-5)
    elif earlystopping_choice == "weightsbiases":
        early = WeightsBiasPlateauStopping(
            patience=500, min_delta=1e-4, norm_choice="fro"
        )
    else:
        raise ValueError("The specified stopping choice is not implemented or correct.")

# Train the network in an incremental manner and predict the results in each load step
for i in range(steps):
    ext_traction = max_ext_traction / steps * (i + 1)
    print(f"\nTraining for a traction of {ext_traction}.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile(
        "adam", lr=learning_rate_adam, decay=("exponential", exponential_decay)
    )
    time_dict["simulation_compiling_adam"].append(time.time())
    time_dict["simulation_training_adam"].append(time.time())
    losshistory, train_state = model.train(
        iterations=adam_iterations,
        display_every=100,
        callbacks=[early for _ in [1] if earlystopping],
    )
    time_dict["simulation_training_adam"].append(time.time())

    if lbfgs_iterations > 0:
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
        model.compile("L-BFGS")
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        time_dict["simulation_training_lbfgs"].append(time.time())
        losshistory, train_state = model.train(display_every=1000)
        time_dict["simulation_training_lbfgs"].append(time.time())

    # Save the EPINN results
    time_dict["simulation_prediction"].append(time.time())
    points, _, cell_types, elements = geom.get_mesh()
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    output = model.predict(points)
    displacement_pred = np.column_stack(
        (output[:, 0:1], output[:, 1:2], output[:, 2:3])
    )
    (
        sigma_xx,
        sigma_yy,
        sigma_zz,
        sigma_xy,
        _,
        sigma_xz,
        _,
        sigma_yz,
        _,
    ) = model.predict(points, operator=cauchy_stress_3D)
    cauchy_stress_pred = np.column_stack(
        (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz)
    )
    grid.point_data["pred_displacement"] = displacement_pred
    grid.point_data["pred_cauchy_stress"] = cauchy_stress_pred

    # Compare the results with the FEM reference
    if ((ext_traction * 10) % 2 == 0) & (ext_traction <= max_ext_traction):
        fem_path = str(Path(__file__).parent.parent)
        fem_reference = pv.read(
            fem_path
            + f"/fem_reference/3d_sphere_contact_fem_reference_{int((ext_traction * 10)):02}.vtu"
        )
        points_fem = fem_reference.points
        displacement_fem = fem_reference.point_data["displacement"]
        cauchy_stress_fem = fem_reference.point_data["nodal_cauchy_stresses_xyz"]

        # Compute predictions on FEM nodes
        displacement_pred_on_fem_mesh = model.predict(points_fem)
        (
            sigma_xx_pred_on_fem_mesh,
            sigma_yy_pred_on_fem_mesh,
            sigma_zz_pred_on_fem_mesh,
            sigma_xy_pred_on_fem_mesh,
            _,
            sigma_xz_pred_on_fem_mesh,
            _,
            sigma_yz_pred_on_fem_mesh,
            _,
        ) = model.predict(points_fem, operator=cauchy_stress_3D)
        cauchy_stress_pred_on_fem_mesh = np.column_stack(
            (
                sigma_xx_pred_on_fem_mesh,
                sigma_yy_pred_on_fem_mesh,
                sigma_zz_pred_on_fem_mesh,
                sigma_xy_pred_on_fem_mesh,
                sigma_yz_pred_on_fem_mesh,
                sigma_xz_pred_on_fem_mesh,
            )
        )
        tensor_cauchy_stress_pred_on_fem_mesh = np.transpose(
            np.array(
                [
                    [
                        sigma_xx_pred_on_fem_mesh.flatten(),
                        sigma_xy_pred_on_fem_mesh.flatten(),
                        sigma_xz_pred_on_fem_mesh.flatten(),
                    ],
                    [
                        sigma_xy_pred_on_fem_mesh.flatten(),
                        sigma_yy_pred_on_fem_mesh.flatten(),
                        sigma_yz_pred_on_fem_mesh.flatten(),
                    ],
                    [
                        sigma_xz_pred_on_fem_mesh.flatten(),
                        sigma_yz_pred_on_fem_mesh.flatten(),
                        sigma_zz_pred_on_fem_mesh.flatten(),
                    ],
                ]
            ),
            (2, 0, 1),
        )
        tensor_cauchy_stress_fem = np.array(
            [
                [
                    cauchy_stress_fem[:, 0],
                    cauchy_stress_fem[:, 3],
                    cauchy_stress_fem[:, 5],
                    cauchy_stress_fem[:, 3],
                    cauchy_stress_fem[:, 1],
                    cauchy_stress_fem[:, 4],
                    cauchy_stress_fem[:, 5],
                    cauchy_stress_fem[:, 4],
                    cauchy_stress_fem[:, 2],
                ]
            ]
        ).T.reshape(-1, 3, 3)

        # Compute relative L2-error
        volume_integral = fem_reference.copy()
        volume_integral.point_data["squared_error_disp"] = (
            np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem, axis=1)
            ** 2
        )
        volume_integral.point_data["squared_disp"] = (
            np.linalg.norm(displacement_fem, axis=1) ** 2
        )
        volume_integral.point_data["squared_error_stress"] = (
            np.linalg.norm(
                tensor_cauchy_stress_pred_on_fem_mesh - tensor_cauchy_stress_fem,
                axis=(1, 2),
                ord="fro",
            )
            ** 2
        )
        volume_integral.point_data["squared_stress"] = (
            np.linalg.norm(tensor_cauchy_stress_fem, axis=(1, 2), ord="fro") ** 2
        )
        volume_integral = volume_integral.integrate_data()
        l2_iteration.append(train_state.step)
        rel_err_l2_disp.append(
            np.sqrt(
                volume_integral.point_data["squared_error_disp"][0]
                / volume_integral.point_data["squared_disp"][0]
            )
        )
        print(f"Relative L2 error for displacement:   {rel_err_l2_disp[-1]}")
        rel_err_l2_stress.append(
            np.sqrt(
                volume_integral.point_data["squared_error_stress"][0]
                / volume_integral.point_data["squared_stress"][0]
            )
        )
        print(f"Relative L2 error for stress:         {rel_err_l2_stress[-1]}")

        # Compute mean absolute error
        print(
            f"Mean absolute error for displacement: {np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem) / len(displacement_fem)}"
        )
        print(
            f"Mean absolute error for stress:       {np.mean(np.linalg.norm(tensor_cauchy_stress_pred_on_fem_mesh - tensor_cauchy_stress_fem, axis=(1, 2), ord='fro'))}"
        )

        # Create output with relative pointwise errors
        fem_reference.point_data["displacement_prediction"] = (
            displacement_pred_on_fem_mesh
        )
        fem_reference.point_data["cauchy_stresses_prediction"] = (
            cauchy_stress_pred_on_fem_mesh
        )
        fem_reference.point_data["absolute_displacement_error"] = abs(
            displacement_pred_on_fem_mesh - displacement_fem
        )
        fem_reference.point_data["absolute_cauchy_stress_error"] = abs(
            cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem
        )
        fem_reference.point_data["relative_displacement_error"] = np.divide(
            np.abs(displacement_pred_on_fem_mesh - displacement_fem),
            np.abs(displacement_fem),
            out=np.zeros_like(displacement_fem, dtype=float),
            where=displacement_fem != 0,
        )
        fem_reference.point_data["relative_cauchy_stress_error"] = np.divide(
            np.abs(cauchy_stress_pred_on_fem_mesh - cauchy_stress_fem),
            np.abs(cauchy_stress_fem),
            out=np.zeros_like(cauchy_stress_fem, dtype=float),
            where=cauchy_stress_fem != 0,
        )
        file_path_fem_compare = os.path.join(
            model_path, f"{simulation_case}_fem_compare_{int(ext_traction * 10):02}"
        )
        fem_reference.save(f"{file_path_fem_compare}.vtu")

    file_path = os.path.join(
        model_path, f"{simulation_case}_{int(ext_traction * 10):02}"
    )
    grid.save(f"{file_path}.vtu")
    time_dict["simulation_prediction"].append(time.time())

# Save the trained network parameters
model.save(f"{model_path}/{simulation_case}")
dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=False,
    output_dir=model_path,
    loss_fname=f"{simulation_case}-{train_state.step}_loss.dat",
    train_fname=f"{simulation_case}-{train_state.step}_train.dat",
    test_fname=f"{simulation_case}-{train_state.step}_test.dat",
)

# Plot the energy over the iterations
fig1, ax1 = plt.subplots(1, 2, figsize=(20, 8))
ax1[0].plot(
    losshistory.steps,
    [loss[0] for loss in losshistory.loss_train],
    label="Internal energy",
    marker="x",
)
ax1[0].plot(
    losshistory.steps,
    [loss[1] for loss in losshistory.loss_train],
    label="External work",
    marker="x",
)
ax1[0].plot(
    losshistory.steps,
    [loss[2] for loss in losshistory.loss_train],
    label="Contact work",
    marker="x",
)
ax1[0].plot(
    losshistory.steps,
    [sum(losses) for losses in losshistory.loss_train],
    label="Total energy",
    marker="x",
)
ax1[0].set_xlabel("Iterations", size=17)
ax1[0].set_ylabel("Energy", size=17)
ax1[0].tick_params(axis="both", labelsize=15)
ax1[0].legend(fontsize=17)
ax1[0].grid()

ax1[1].plot(
    losshistory.steps,
    [abs(loss[0]) for loss in losshistory.loss_train],
    label="Internal energy",
    marker="x",
)
ax1[1].plot(
    losshistory.steps,
    [abs(loss[1]) for loss in losshistory.loss_train],
    label="External work",
    marker="x",
)
ax1[1].plot(
    losshistory.steps,
    [abs(loss[2]) for loss in losshistory.loss_train],
    label="Contact work",
    marker="x",
)
ax1[1].plot(
    losshistory.steps,
    [abs(sum(losses)) for losses in losshistory.loss_train],
    label="Total energy",
    marker="x",
)
ax1[1].set_xlabel("Iterations", size=17)
ax1[1].set_ylabel("Energy", size=17)
ax1[1].set_yscale("log")
ax1[1].tick_params(axis="both", labelsize=15)
ax1[1].legend(fontsize=17)
ax1[1].grid()
plt.tight_layout()
fig1.savefig(
    f"{model_path}/{simulation_case}-{train_state.step}_loss_plot.png", dpi=300
)

if l2_iteration:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(
        l2_iteration,
        rel_err_l2_disp,
        color="b",
        lw=2,
        label="$L_2$-error for displacement",
        marker="x",
    )
    ax2.plot(
        l2_iteration,
        rel_err_l2_stress,
        color="r",
        lw=2,
        label="$L_2$-error for cauchy stress",
        marker="x",
    )
    ax2.set_xlabel("Iterations", size=17)
    ax2.set_ylabel("$L_2$ norm", size=17)
    ax2.set_yscale("log")
    ax2.tick_params(axis="both", labelsize=15)
    ax2.legend(fontsize=17)
    ax2.grid()
    plt.tight_layout()
    fig2.savefig(
        f"{model_path}/{simulation_case}-{train_state.step}_l2_norm_over_iterations.png",
        dpi=300,
    )
time_dict["total"].append(time.time())

# Print times to output file
with open(
    f"{model_path}/{simulation_case}-{train_state.step}_times.txt", "w"
) as text_file:
    print(f"Compilation and training times in       [s]", file=text_file)
    print(f"==============================================", file=text_file)
    print(
        f"Meshing:                              {(time_dict['meshing'][1] - time_dict['meshing'][0]):8.3f}",
        file=text_file,
    )
    print(
        f"Building element information:         {(time_dict['element_information'][1] - time_dict['element_information'][0]):8.3f}",
        file=text_file,
    )
    if relaxation:
        print(
            f"Relaxation compilation:               {(time_dict['relaxation_compiling'][1] - time_dict['relaxation_compiling'][0]):8.3f}",
            file=text_file,
        )
        print(
            f"Relaxation training:                  {(time_dict['relaxation_training'][1] - time_dict['relaxation_training'][0]):8.3f}",
            file=text_file,
        )
    if steps > 1:
        for i in range(steps):
            print(f"----------------------------------------------", file=text_file)
            print(
                f"   Load step {(i + 1):2d} compilation (adam):   {(time_dict['simulation_compiling_adam'][(2 * i) + 1] - time_dict['simulation_compiling_adam'][2 * i]):8.3f}",
                file=text_file,
            )
            print(
                f"   Load step {(i + 1):2d} training (adam):      {(time_dict['simulation_training_adam'][(2 * i) + 1] - time_dict['simulation_training_adam'][2 * i]):8.3f}",
                file=text_file,
            )
            if lbfgs_iterations > 0:
                print(
                    f"   Load step {(i + 1):2d} compilation (L-BFGS): {(time_dict['simulation_compiling_lbfgs'][(2 * i) + 1] - time_dict['simulation_compiling_lbfgs'][2 * i]):8.3f}",
                    file=text_file,
                )
                print(
                    f"   Load step {(i + 1):2d} training (L-BFGS):    {(time_dict['simulation_training_lbfgs'][(2 * i) + 1] - time_dict['simulation_training_lbfgs'][2 * i]):8.3f}",
                    file=text_file,
                )
            print(
                f"   Load step {(i + 1):2d} prediction:           {(time_dict['simulation_prediction'][(2 * i) + 1] - time_dict['simulation_prediction'][2 * i]):8.3f}",
                file=text_file,
            )
        print(f"==============================================", file=text_file)
    print(
        f"Total compilation (adam):         {(sum(time_dict['simulation_compiling_adam'][1::2]) - (sum(time_dict['simulation_compiling_adam'][::2]))):12.3f}",
        file=text_file,
    )
    print(
        f"Total training (adam):            {(sum(time_dict['simulation_training_adam'][1::2]) - (sum(time_dict['simulation_training_adam'][::2]))):12.3f}",
        file=text_file,
    )
    if lbfgs_iterations > 0:
        print(
            f"Total compilation (L-BFGS):       {(sum(time_dict['simulation_compiling_lbfgs'][1::2]) - (sum(time_dict['simulation_compiling_lbfgs'][::2]))):12.3f}",
            file=text_file,
        )
        print(
            f"Total training (L-BFGS):          {(sum(time_dict['simulation_training_lbfgs'][1::2]) - (sum(time_dict['simulation_training_lbfgs'][::2]))):12.3f}",
            file=text_file,
        )
    print(
        f"Total prediction:                 {(sum(time_dict['simulation_prediction'][1::2]) - (sum(time_dict['simulation_prediction'][::2]))):12.3f}",
        file=text_file,
    )
    print(f"==============================================", file=text_file)
    print(
        f"Total:                            {(time_dict['total'][1] - time_dict['total'][0]):12.3f}",
        file=text_file,
    )
