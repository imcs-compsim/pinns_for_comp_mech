"""
Mixed-Formulation Nonlinear Beam Under Shear Loading
==================================================

This example solves a two-dimensional nonlinear beam under shear loading with a
mixed PINN formulation.

It is modelled using a Neo-Hookean hyperelastic material under plane strain
assumptions. The network predicts displacement and first Piola stress
components, while stress equilibrium and constitutive residuals are enforced in
the PDE loss.

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

from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.gmsh_models import Block_2D_square
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    compute_elastic_properties,
    deformation_gradient_2D,
    first_piola_stress_tensor_2D,
)
from compsim_pinns.postprocess.custom_callbacks import (
    LossPlateauStopping,
    WeightsBiasPlateauStopping,
)

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
domain_dimension = 2
coords_lower_left_corner = [0, -1]
coords_upper_right_corner = [20, 1]
mesh_size = 0.25  # default 0.25
gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 11}
block_2d = Block_2D_square(
    coord_left_corner=coords_lower_left_corner,
    coord_right_corner=coords_upper_right_corner,
    mesh_size=mesh_size,
    gmsh_options=gmsh_options,
)
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
l_beam = block_2d.coord_right_corner[0] - block_2d.coord_left_corner[0]
h_beam = block_2d.coord_right_corner[1] - block_2d.coord_left_corner[1]
revert_curve_list = []
revert_normal_dir_list = [2, 2, 1, 2]
time_dict["meshing"].append(time.time())
time_dict["setup"].append(time.time())


# Define PINN geometry
geom = GmshGeometry2D(
    gmsh_model,
    revert_curve_list=revert_curve_list,
    revert_normal_dir_list=revert_normal_dir_list,
)


# Define BCs
def boundary_right(x, on_boundary):
    """Check whether a point satisfies the `boundary_right` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Whether the point is on the boundary.

    Returns:
        bool: Result of the `boundary_right` evaluation.
    """
    return on_boundary and np.isclose(x[0], l_beam)


def boundary_free_surf(x, on_boundary):
    """Check whether a point satisfies the `boundary_free_surf` condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Whether the point is on the boundary.

    Returns:
        bool: Result of the `boundary_free_surf` evaluation.
    """
    return (
        on_boundary
        and (np.isclose(x[1], h_beam / 2) or np.isclose(x[1], -h_beam / 2))
        and not (np.isclose(x[0], l_beam) or np.isclose(x[0], 0))
    )


bc1 = dde.DirichletBC(geom, lambda _: 0.0, boundary_right, component=2)  # Pxx = 0
bc2 = dde.DirichletBC(
    geom, lambda _: -shear_load, boundary_right, component=4
)  # Pyx = -t
bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_free_surf, component=3)  # Pxy = 0
bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_free_surf, component=5)  # Pyy = 0

# Set material parameters as global variables in elasticity_utils
hyperelasticity_utils.lame = 2.78
hyperelasticity_utils.shear = 4.17
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()


def momentum_nonlinear_2d_mixed(x, y):
    """Compute mixed nonlinear momentum and constitutive residuals.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        list: Momentum balance and stress-constitutive residuals.
    """
    # Governing equation from Pxx, Pxy, Pyx, Pyy
    p_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_xy_y = dde.grad.jacobian(y, x, i=3, j=1)
    p_yx_x = dde.grad.jacobian(y, x, i=4, j=0)
    p_yy_y = dde.grad.jacobian(y, x, i=5, j=1)

    momentum_x = p_xx_x + p_xy_y  # no volume loads
    momentum_y = p_yx_x + p_yy_y  # no volume loads

    # Calculate strain terms from displacements
    p_u_xx, p_u_yy, p_u_xy, p_u_yx = first_piola_stress_tensor_2D(x, y[:, 0:2])
    residuum_Pxx = p_u_xx - y[:, 2:3]
    residuum_Pxy = p_u_xy - y[:, 3:4]
    residuum_Pyx = p_u_yx - y[:, 4:5]
    residuum_Pyy = p_u_yy - y[:, 5:6]

    return [
        momentum_x,
        momentum_y,
        residuum_Pxx,
        residuum_Pxy,
        residuum_Pyx,
        residuum_Pyy,
    ]


# Create the data object for the PINN
n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_nonlinear_2d_mixed,
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
    u_x = y[:, 0:1]
    u_y = y[:, 1:2]
    P_xx = y[:, 2:3]
    P_xy = y[:, 3:4]
    P_yx = y[:, 4:5]
    P_yy = y[:, 5:6]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    rel_height_pos = (h_beam - 2 * y_loc) / h_beam
    rel_height_neg = (h_beam + 2 * y_loc) / h_beam
    rel_length_left = x_loc / l_beam
    rel_length_right = (l_beam - x_loc) / l_beam

    return bkd.concat(
        [
            rel_length_left * h_beam * u_x,
            rel_length_left * h_beam * u_y,
            rel_length_right * youngs_modulus * P_xx,
            rel_height_pos * rel_height_neg * youngs_modulus * P_xy,
            -shear_load + rel_length_right * youngs_modulus * P_yx,
            rel_height_pos * rel_height_neg * youngs_modulus * P_yy,
        ],
        axis=1,
    )


# Define the neural network architecture
# 2 inputs, 6 outputs for 2D: ux, uy, Pxx, Pxy, Pyx, Pyy
layer_size = [2] + [50] * 5 + [6]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# Set the training parameters
steps = 1  # try also 20, 40
max_shear_load = 1e-2
model_path = str(Path(__file__).parent)
simulation_case = "Beam_under_shear_load_nonlinear_incremental_mixed"
learning_rate_adam = 1e-3
learning_rate_total_decay = 1e-3
adam_iterations = 5000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 2000
earlystopping = False
earlystopping_choice = "weightsbiases"  # "loss" or "weightsbiases"
time_dict["setup"].append(time.time())

# Create outline of beam to plot it during the load steps
shape_points_x = int(
    (coords_upper_right_corner[0] - coords_lower_left_corner[0]) / mesh_size
)
shape_points_y = int(
    (coords_upper_right_corner[1] - coords_lower_left_corner[1]) / mesh_size
)
edge_space_x = np.linspace(
    coords_lower_left_corner[0], coords_upper_right_corner[0], shape_points_x + 1
)
edge_space_y = np.linspace(
    coords_lower_left_corner[1], coords_upper_right_corner[1], shape_points_y + 1
)
coords_edge_points = np.unique(
    np.vstack(
        [
            np.stack(
                np.meshgrid(edge_space_x, [coords_lower_left_corner[1]]), -1
            ).reshape(-1, 2),
            np.stack(
                np.meshgrid(edge_space_x, [coords_upper_right_corner[1]]), -1
            ).reshape(-1, 2),
            np.stack(
                np.meshgrid([coords_lower_left_corner[0]], edge_space_y), -1
            ).reshape(-1, 2),
            np.stack(
                np.meshgrid([coords_upper_right_corner[0]], edge_space_y), -1
            ).reshape(-1, 2),
        ]
    ),
    axis=0,
)
coords_corners = np.array(
    (
        coords_upper_right_corner,
        [coords_lower_left_corner[0], coords_upper_right_corner[1]],
        coords_lower_left_corner,
        [coords_upper_right_corner[0], coords_lower_left_corner[1]],
    )
)
trajectory_edge_points = np.empty(
    (steps + 1, coords_edge_points.shape[0], domain_dimension)
)
trajectory_edge_points[0, :, :] = coords_edge_points
trajectory_corners = np.empty((steps + 1, coords_corners.shape[0], domain_dimension))
trajectory_corners[0, :, :] = coords_corners

# Create error lists to store relative L2 errors
rel_err_l2_disp = []
rel_err_l2_stress = []
l2_iteration = []

# Define the early stopping callback
train_callbacks = []
if earlystopping:
    if earlystopping_choice == "loss":
        early = LossPlateauStopping(patience=500, min_delta=1e-5)
    elif earlystopping_choice == "weightsbiases":
        early = WeightsBiasPlateauStopping(
            patience=500, min_delta=1e-4, norm_choice="fro"
        )
    else:
        raise ValueError("The specified stopping choice is not implemented or correct.")
    train_callbacks.append(early)

# Weights
w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5, w_pde_6 = (
    1e0,
    1e0,
    1e0,
    1e0,
    1e0,
    1e0,
)
loss_weights = [w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5, w_pde_6]


def compute_displacement_sigma_nonlinear_mixed(model, points):
    """Compute displacement and Cauchy stress from the mixed model output.

    Args:
        model: Trained DeepXDE model used for prediction.
        points: Coordinates where displacement and stress are evaluated.

    Returns:
        tuple: Predicted displacement and Cauchy stress arrays.
    """
    output = model.predict(points)
    displacement = np.column_stack((output[:, 0:1], output[:, 1:2]))
    f_xx, f_yy, f_xy, f_yx = model.predict(points, operator=deformation_gradient_2D)
    det_f = f_xx * f_yy - f_xy * f_yx
    sigma_xx = 1 / det_f * (output[:, 2:3] * f_xx + output[:, 3:4] * f_xy)
    sigma_xy = 1 / det_f * (output[:, 2:3] * f_yx + output[:, 3:4] * f_yy)
    sigma_yy = 1 / det_f * (output[:, 4:5] * f_yx + output[:, 5:6] * f_yy)
    sigma = np.column_stack((sigma_xx, sigma_yy, sigma_xy))
    return displacement, sigma


# Train the network in an incremental manner and predict the results in each load step
for i in range(steps):
    shear_load = max_shear_load / steps * (i + 1)
    print(f"\nTraining for a shear load of {shear_load}.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile("adam", lr=learning_rate_adam, loss_weights=loss_weights)
    time_dict["simulation_compiling_adam"].append(time.time())
    time_dict["simulation_training_adam"].append(time.time())
    losshistory, train_state = model.train(
        iterations=adam_iterations,
        display_every=100,
        callbacks=train_callbacks,
    )
    time_dict["simulation_training_adam"].append(time.time())

    if lbfgs_iterations > 0:
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
        model.compile("L-BFGS", loss_weights=loss_weights)
        time_dict["simulation_compiling_lbfgs"].append(time.time())
        time_dict["simulation_training_lbfgs"].append(time.time())
        losshistory, train_state = model.train(display_every=1000)
        time_dict["simulation_training_lbfgs"].append(time.time())

    # Save the PINN results
    time_dict["simulation_prediction"].append(time.time())
    node_tags, node_coords, _ = geom.gmsh_model.mesh.getNodes(
        2, -1, includeBoundary=True
    )
    points, _, _ = geom.order_coordinates(node_coords, node_tags)
    element_types, _, node_tags_per_element = geom.gmsh_model.mesh.getElements(2, -1)
    element_type = element_types[0]
    _, _, _, n_nodes_per_cell, _, _ = geom.gmsh_model.mesh.getElementProperties(
        element_type
    )
    elements = node_tags_per_element[0].reshape(-1, n_nodes_per_cell) - 1
    vtk_cell_type_map = {2: 5, 3: 9}
    cell_types = np.full(
        elements.shape[0], vtk_cell_type_map[element_type], dtype=np.uint8
    )
    n_nodes_per_cell = elements.shape[1]
    n_cells = elements.shape[0]
    n_points = points.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(
        cells, cell_types, np.c_[points, np.zeros((n_points, 1))]
    )
    displacement_pred, cauchy_stress_pred = compute_displacement_sigma_nonlinear_mixed(
        model, points
    )
    grid.point_data["pred_displacement"] = np.c_[
        displacement_pred, np.zeros((n_points, 1))
    ]
    grid.point_data["pred_cauchy_stress"] = np.column_stack(
        (
            cauchy_stress_pred[:, 0],
            cauchy_stress_pred[:, 1],
            np.zeros((n_points, 1)),
            cauchy_stress_pred[:, 2],
            np.zeros((n_points, 1)),
            np.zeros((n_points, 1)),
        )
    )

    # Compare the results with the FEM reference
    if any(abs(shear_load * 1000 - i) <= 1e-12 for i in range(1, 11)):
        reference_path = (
            str(Path(__file__).parent.parent.parent.parent.parent)
            + "/fem_references/paper-epinn-data-reference/2d_bending_beam"
        )
        fem_reference = pv.read(
            reference_path
            + f"/fem_reference_2d_bending_beam_{int(round(shear_load * 1e3)):02}.vtu"
        )
        points_fem = fem_reference.points
        displacement_fem = fem_reference.point_data["displacement"]
        cauchy_stress_fem = fem_reference.point_data["nodal_cauchy_stresses_xyz"]

        # Compute predictions on FEM nodes
        displacement_pred_on_fem_mesh, sigma_pred_on_fem_mesh = (
            compute_displacement_sigma_nonlinear_mixed(model, points_fem[:, 0:2])
        )
        cauchy_stress_pred_on_fem_mesh = np.column_stack(
            (
                sigma_pred_on_fem_mesh[:, 0],
                sigma_pred_on_fem_mesh[:, 1],
                np.zeros_like(sigma_pred_on_fem_mesh[:, 0]),
                sigma_pred_on_fem_mesh[:, 2],
                np.zeros_like(sigma_pred_on_fem_mesh[:, 0]),
                np.zeros_like(sigma_pred_on_fem_mesh[:, 0]),
            )
        )
        tensor_cauchy_stress_pred_on_fem_mesh = np.transpose(
            np.array(
                [
                    [sigma_pred_on_fem_mesh[:, 0], sigma_pred_on_fem_mesh[:, 2]],
                    [sigma_pred_on_fem_mesh[:, 2], sigma_pred_on_fem_mesh[:, 1]],
                ]
            ),
            (2, 0, 1),
        )
        tensor_cauchy_stress_fem = np.array(
            [
                [
                    cauchy_stress_fem[:, 0],
                    cauchy_stress_fem[:, 3],
                    cauchy_stress_fem[:, 3],
                    cauchy_stress_fem[:, 1],
                ]
            ]
        ).T.reshape(-1, 2, 2)

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
            "Mean absolute error for displacement: "
            f"{np.linalg.norm(displacement_pred_on_fem_mesh - displacement_fem) / len(displacement_fem)}"
        )
        print(
            "Mean absolute error for stress:       "
            f"{np.mean(np.linalg.norm(tensor_cauchy_stress_pred_on_fem_mesh - tensor_cauchy_stress_fem, axis=(1, 2), ord='fro'))}"
        )

        # Create output with relative pointwise errors
        fem_reference.point_data["displacement_prediction"] = np.hstack(
            (
                displacement_pred_on_fem_mesh,
                np.zeros_like(displacement_pred_on_fem_mesh[:, 0:1]),
            )
        )
        fem_reference.point_data["cauchy_stresses_prediction"] = (
            cauchy_stress_pred_on_fem_mesh
        )
        fem_reference.point_data["absolute_displacement_error"] = np.hstack(
            (
                abs(displacement_pred_on_fem_mesh - displacement_fem),
                np.zeros_like(displacement_pred_on_fem_mesh[:, 0:1]),
            )
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
            model_path,
            f"{simulation_case}_fem_compare_{int(round(shear_load * 1e3)):02}",
        )
        fem_reference.save(f"{file_path_fem_compare}.vtu")

    # Predict shape of the beam in each time step
    trajectory_edge_points[i + 1, :, :] = (
        coords_edge_points + model.predict(coords_edge_points)[:, 0:2]
    )
    trajectory_corners[i + 1, :, :] = (
        coords_corners + model.predict(coords_corners)[:, 0:2]
    )

    file_path = os.path.join(
        model_path, f"{simulation_case}_{int(shear_load * 1e3):02}"
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

# Plot the beam outline and trajectory over load steps
fig2, ax2 = plt.subplots(1, 2, figsize=(20, 8))
cmap = plt.colormaps[("cool")]
colors = cmap(np.linspace(0, 1, steps + 1))
center = coords_edge_points.mean(axis=0)
angles = np.arctan2(
    coords_edge_points[:, 1] - center[1], coords_edge_points[:, 0] - center[0]
)
sort_idx = np.argsort(angles)
trajectory_edge_points_sorted = trajectory_edge_points[:, sort_idx, :]
for t, P in enumerate(trajectory_edge_points_sorted):
    P = np.vstack([P, P[0]])
    ax2[0].plot(
        P[:, 0],
        P[:, 1],
        color=colors[t],
        lw=2,
        label=f"Shear load of {t / steps * max_shear_load:1.3f}",
        marker=".",
    )
ax2[0].set_xlabel("$x$", size=17)
ax2[0].set_ylabel("$y$", size=17)
ax2[0].tick_params(axis="both", labelsize=15)
ax2[0].grid()
ax2[0].legend()
ax2[1].plot(
    trajectory_corners[:, 0, 0],
    trajectory_corners[:, 0, 1],
    color="b",
    lw=2,
    label="Trajectory of upper right corner",
    marker="x",
)
ax2[1].plot(
    trajectory_corners[:, 3, 0],
    trajectory_corners[:, 3, 1],
    color="r",
    lw=2,
    label="Trajectory of lower right corner",
    marker="x",
)
ax2[1].set_xlabel("$x$", size=17)
ax2[1].set_ylabel("$y$", size=17)
ax2[1].grid()
ax2[1].legend()
plt.tight_layout()
fig2.savefig(
    f"{model_path}/{simulation_case}-{train_state.step}_edge_trajectory.png", dpi=300
)

# Save the trajectory of the corner points and the relative L2 errors for displacement and stress
np.savez(
    f"{model_path}/{simulation_case}_edge_trajectory_meshsize_{2 / mesh_size:03.0f}.npz",
    x=trajectory_edge_points_sorted,
    y=trajectory_corners,
)
np.savez(
    f"{model_path}/{simulation_case}_l2_errors.npz",
    x=rel_err_l2_disp,
    y=rel_err_l2_stress,
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
