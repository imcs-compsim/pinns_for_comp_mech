"""
Leguan Bridge Under Incremental Displacement Loading
==================================================

This example solves a three-dimensional quarter model of the 26 m Leguan
tank bridge under incrementally applied displacement loading with the
Energy-based PINN (EPINN).

It is modelled using a Neo-Hookean hyperelastic material. Symmetry and support
conditions are enforced through the output transform, while the prescribed top
slope displacement is increased in load steps. The model is trained with Adam
followed by L-BFGS.

Results are exported as VTU files for post-processing, while timings and loss
information are also saved.
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

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import leguan_bridge_quarter
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_3D,
    compute_elastic_properties,
    strain_energy_neo_hookean_3d,
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
quarter_bridge_leguan = leguan_bridge_quarter()
gmsh_model, slope_corner_points, seating_corner_points = (
    quarter_bridge_leguan.generateGmshModel(visualize_mesh=False)
)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())

# Set up quadrature rules for domain and boundary integration
domain_dimension = 3
boundary_dimension = domain_dimension - 1
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=domain_dimension, ngp=2
)
coord_quadrature, weight_quadrature = quad_rule.generate()
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=boundary_dimension, ngp=2
)
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

# Define top surface normals
normal_top_slope = np.cross(
    slope_corner_points[0, :] - slope_corner_points[4, :],
    slope_corner_points[1, :] - slope_corner_points[5, :],
)
normal_top_slope = normal_top_slope / np.linalg.norm(normal_top_slope)

normal_bottom_slope = np.cross(
    slope_corner_points[1, :] - slope_corner_points[3, :],
    slope_corner_points[2, :] - slope_corner_points[4, :],
)
normal_bottom_slope = normal_bottom_slope / np.linalg.norm(normal_bottom_slope)


# Define BCs
def on_top_slope(x):
    """Check whether a point satisfies the `on_top_slope` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_top_slope` evaluation.
    """
    return np.isclose(np.dot(normal_top_slope, x - slope_corner_points[0, :]), 0.0)


def on_bottom_slope(x):
    """Check whether a point satisfies the `on_bottom_slope` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_bottom_slope` evaluation.
    """
    return np.isclose(np.dot(normal_bottom_slope, x - slope_corner_points[2, :]), 0.0)


boundary_selection_map = [
    {"boundary_function": on_top_slope, "tag": "on_top_slope"},
    {"boundary_function": on_bottom_slope, "tag": "on_bottom_slope"},
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
hyperelasticity_utils.youngs_modulus = 1.33
hyperelasticity_utils.nu = 0.33
nu, lame, shear, youngs_modulus = compute_elastic_properties()


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
    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)[
        beg_pde:beg_boundary
    ]
    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * global_element_weights_t[:, 2:3]
        * (internal_energy_density)
        * jacobian_t
    )

    return [internal_energy]


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
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]

    seating_coord_z = seating_corner_points[0, 2]

    # Top slope plane:
    # n_x * (x - x0) + n_y * (y - y0) + n_z * (z - z0) = 0
    top_point = slope_corner_points[0, :]
    z_top = (
        top_point[2]
        - (
            normal_top_slope[0] * (x_loc - top_point[0])
            + normal_top_slope[1] * (y_loc - top_point[1])
        )
        / normal_top_slope[2]
    )

    # Hard BCs
    u_out = x_loc * u / youngs_modulus  # u = 0 on x = 0
    v_out = y_loc * v / youngs_modulus  # v = 0 on y = 0
    w_out = (z_loc - seating_coord_z) * (
        z_loc - z_top
    ) * w / youngs_modulus + displacement_top * (z_loc - seating_coord_z) / (
        z_top - seating_coord_z
    )  # w = 0 on z = seating_coord_z, w = displacement_top on the top slope

    return bkd.concat([u_out, v_out, w_out], axis=1)


# Define the neural network architecture
# 3 inputs, 3 outputs for 3D: u_x, u_y, u_z
layer_size = [3] + [50] * 5 + [3]  # also try [3] + [50] * 10 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights = None
model = dde.Model(data, net)

# Set the training parameters
steps = 1
displacement_max = -0.05
model_path = str(Path(__file__).parent)
simulation_case = f"3d_leguan_bridge_incremental_displacement"
learning_rate_adam = 1e-3
learning_rate_total_decay = 1e-3
adam_iterations = 2000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 2000
relaxation_adam_iterations = (
    0  # just to not get any errors when not using it (undefined variable in naming)
)
relaxation = False
time_dict["setup"].append(time.time())

# Optional relaxation step before incremental loading
if relaxation:
    time_dict["relaxation_compiling"].append(time.time())
    displacement_top = displacement_max / steps
    relaxation_adam_iterations = 1000
    print(f"\nRelaxation step for initial displacement of {displacement_top}.\n")
    model.compile("adam", lr=learning_rate_adam)
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(
        iterations=relaxation_adam_iterations, display_every=100
    )
    time_dict["relaxation_training"].append(time.time())
    time_dict["relaxation_compiling"].append(time.time())
    relaxation_lbfgs_iterations = 1000
    dde.optimizers.config.set_LBFGS_options(maxiter=relaxation_lbfgs_iterations)
    model.compile("L-BFGS")
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(display_every=1000)
    time_dict["relaxation_training"].append(time.time())

# Train the network in an incremental manner and predict the results in each load step
for i in range(steps):
    displacement_top = displacement_max / steps * (i + 1)
    print(f"\nTraining for a displacement of {displacement_top}.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile(
        "adam", lr=learning_rate_adam
    )  # , decay=("exponential", exponential_decay))
    time_dict["simulation_compiling_adam"].append(time.time())
    time_dict["simulation_training_adam"].append(time.time())
    losshistory, train_state = model.train(
        iterations=adam_iterations, display_every=100
    )  # , callbacks=[early for _ in [1] if earlystopping])
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

    file_path = os.path.join(
        model_path, f"{simulation_case}_{int(abs(displacement_top * 1e3)):03}"
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
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(
    losshistory.steps,
    [sum(l) for l in losshistory.loss_train],
    color="b",
    lw=2,
    label="Internal energy",
    marker="x",
)
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("Energy", size=17)
ax1.set_yscale("log")
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig(
    f"{model_path}/{simulation_case}-{train_state.step}_loss_plot.png", dpi=300
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
            f"Relaxation compilation (adam):        {(time_dict['relaxation_compiling'][1] - time_dict['relaxation_compiling'][0]):8.3f}",
            file=text_file,
        )
        print(
            f"Relaxation training (adam):           {(time_dict['relaxation_training'][1] - time_dict['relaxation_training'][0]):8.3f}",
            file=text_file,
        )
        print(
            f"Relaxation compilation (L-BFGS):      {(time_dict['relaxation_compiling'][3] - time_dict['relaxation_compiling'][2]):8.3f}",
            file=text_file,
        )
        print(
            f"Relaxation training (L-BFGS):         {(time_dict['relaxation_training'][3] - time_dict['relaxation_training'][2]):8.3f}",
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
