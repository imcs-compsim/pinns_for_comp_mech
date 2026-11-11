"""
Nonlinear 2D Cook's Cantilever Under Shear Loading
==================================================

This example solves a two-dimensional Cook's cantilever problem under shear
loading with the Energy-based PINN (EPINN).

It is modelled using a modified Neo-Hookean hyperelastic material. The clamped
boundary conditions are enforced through the output transform, while the shear
load is applied to the cantilever end through the potential energy.

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
from compsim_pinns.geometry.gmsh_models import CooksCantilever2D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    bkd_log,
    compute_elastic_properties,
    deformation_gradient_2D,
    matrix_determinant_2D,
)
from compsim_pinns.postprocess.custom_callbacks import (
    LossPlateauStopping,
    PredictionTracker,
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
length = 0.048
web_height = 0.044
load_height = 0.016
thickness = 0.001
n_elements_length = 48
n_elements_height = 16
cantilever = CooksCantilever2D(
    length=length,
    web_height=web_height,
    load_height=load_height,
    divisions=[n_elements_length, n_elements_height],
)
gmsh_model = cantilever.generateGmshModel(visualize_mesh=False)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())

# Set up quadrature rules for domain and boundary integration
domain_dimension = 2
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


# Define BCs
def cantilever_end(x):
    """Check whether a point satisfies the `cantilever_end` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `cantilever_end` evaluation.
    """
    return np.isclose(x[0], length)


boundary_selection_map = [
    {"boundary_function": cantilever_end, "tag": "cantilever_end"}
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
hyperelasticity_utils.lame = 400889.8e6
hyperelasticity_utils.shear = 80.194e6
nu, lame, shear, youngs_modulus = compute_elastic_properties()


def strain_energy_neo_hookean_2D_modified(x, y):
    """Compute modified Neo-Hookean strain energy density.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        Any: Computed strain energy density.
    """
    # Deformation gradient (2x2)
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)

    # Construct C = F^T F (right Cauchy-Green tensor)
    C_xx = f_xx * f_xx + f_yx * f_yx
    C_yy = f_xy * f_xy + f_yy * f_yy

    I_1 = C_xx + C_yy
    det_f = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)
    W = (
        0.5 * shear * (I_1 - 2)
        + lame / 4 * (det_f**2 - 1)
        - (lame / 2 + shear) * bkd_log(det_f)
    )

    return W


def cauchy_stress_2D_modified(x, y):
    """Compute modified Neo-Hookean Cauchy stress components.

    Args:
        x: Input coordinates used to evaluate the function.
        y: Field values or model outputs associated with `x`.

    Returns:
        tuple: Cauchy stress components in 2D.
    """
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)
    det_f = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)

    factor_1 = (lame * (det_f**2 - 1) - 2 * shear) / (2 * det_f)
    factor_2 = shear / det_f

    # Left Cauchy-Green tensor b = F * F^T
    b_xx = f_xx**2 + f_xy**2
    b_yy = f_yx**2 + f_yy**2
    b_xy = f_xx * f_yx + f_xy * f_yy

    T_xx = factor_1 + factor_2 * b_xx
    T_yy = factor_1 + factor_2 * b_yy
    T_xy = factor_2 * b_xy
    T_yx = T_xy

    return T_xx, T_yy, T_xy, T_yx


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
    internal_energy_density = strain_energy_neo_hookean_2D_modified(inputs, outputs)[
        beg_pde:beg_boundary
    ]

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * internal_energy_density
        * jacobian_t
        * thickness
    )
    # External work
    cond = boundary_selection_tag["cantilever_end"]
    u_y = outputs[:, 1:2][beg_boundary:][cond]
    external_force_density = shear_force * u_y
    external_work = (
        global_weights_boundary_t[:, 0:1][cond]
        * external_force_density
        * jacobian_boundary_t[cond]
        * thickness
    )

    return [internal_energy, -external_work]


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

    x_loc = x[:, 0:1]

    u_out = x_loc * u
    v_out = x_loc * v

    return bkd.concat([u_out, v_out], axis=1)


# Define the neural network architecture
# 2 inputs, 2 outputs for 2D: u_x, u_y
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# Set the training parameters
steps = 1
max_shear_force = 8e6
model_path = str(Path(__file__).parent)
simulation_case = "cooks_cantilever_2d_nonlinear"
learning_rate_adam = 1e-3
learning_rate_total_decay = 1e-3
adam_iterations = 5000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 2000
rel_err_l2_disp = []
rel_err_l2_stress = []
l2_iteration = []
relaxation_adam_iterations = 0  # avoid undefined variable in naming
relaxation_lbfgs_iterations = 0  # avoid undefined variable in naming
relaxation = False
earlystopping = False
earlystopping_choice = "weightsbiases"  # "loss" or "weightsbiases"
point_A = np.array([[length, web_height + load_height]])
output_A_prediction = PredictionTracker(
    period=1000,
    points=point_A,
    filename=os.path.join(model_path, f"prediction_tracker_{learning_rate_adam}.csv"),
)
train_callbacks = [output_A_prediction]
time_dict["setup"].append(time.time())

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
    train_callbacks.append(early)

# Optional relaxation step before incremental loading
if relaxation:
    time_dict["relaxation_compiling"].append(time.time())
    shear_force = max_shear_force / steps
    relaxation_adam_iterations = 2000
    print(f"\nRelaxation step for initial shear force of {shear_force}.\n")
    model.compile("adam", lr=learning_rate_adam)
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(
        iterations=relaxation_adam_iterations, display_every=100
    )
    time_dict["relaxation_training"].append(time.time())

    time_dict["relaxation_compiling"].append(time.time())
    relaxation_lbfgs_iterations = 2000
    dde.optimizers.config.set_LBFGS_options(maxiter=relaxation_lbfgs_iterations)
    model.compile("L-BFGS")
    time_dict["relaxation_compiling"].append(time.time())
    time_dict["relaxation_training"].append(time.time())
    losshistory, train_state = model.train(display_every=1000)
    time_dict["relaxation_training"].append(time.time())

# Train the network in an incremental manner and predict the results in each load step
for i in range(steps):
    shear_force = max_shear_force / steps * (i + 1)
    print(f"\nTraining for a shear force of {shear_force}.\n")
    time_dict["simulation_compiling_adam"].append(time.time())
    model.compile("adam", lr=learning_rate_adam)
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
    n_points = points.shape[0]
    cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(
        cells, cell_types, np.c_[points, np.zeros((n_points, 1))]
    )
    output = model.predict(points)
    displacement_pred = np.column_stack((output[:, 0:1], output[:, 1:2]))
    sigma_xx, sigma_yy, sigma_xy, _ = model.predict(
        points, operator=cauchy_stress_2D_modified
    )
    cauchy_stress_pred = np.column_stack((sigma_xx, sigma_yy, sigma_xy))
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

    # Print displacement of point A
    displacement_A = model.predict(point_A)
    print(f"The predicted y-displacement in point A is {displacement_A[:, 1][0]:.4e}.")

    file_path = os.path.join(model_path, f"{simulation_case}_{int(shear_force):03}")
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
    [abs(loss[0]) for loss in losshistory.loss_train],
    label="Internal energy",
    marker="x",
)
ax1.plot(
    losshistory.steps,
    [abs(loss[1]) for loss in losshistory.loss_train],
    label="External work",
    marker="x",
)
ax1.plot(
    losshistory.steps,
    [abs(sum(losses)) for losses in losshistory.loss_train],
    label="Total energy",
    marker="x",
)
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("abs(Energy)", size=17)
ax1.set_yscale("log")
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
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
