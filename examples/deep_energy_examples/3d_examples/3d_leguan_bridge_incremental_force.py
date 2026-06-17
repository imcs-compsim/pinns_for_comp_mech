import os
import time
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from deepxde import backend as bkd

dde.config.set_default_float("float32")  # use double precision (needed for L-BFGS)
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
"""
@author: svoelkl

Load test for the quarter 26m brige of the Leguan bridge vehicle.
"""

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

# Geometry definition and meshing
quarter_bridge_leguan = leguan_bridge_quarter()

gmsh_model, slope_corner_points, seating_corner_points = (
    quarter_bridge_leguan.generateGmshModel(visualize_mesh=False)
)
time_dict["meshing"].append(time.time())
time_dict["element_information"].append(time.time())

domain_dimension = 3
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=domain_dimension, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = domain_dimension - 1
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=boundary_dimension, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

# define top surface normals
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


def on_top_slope(x):
    return np.isclose(np.dot(normal_top_slope, x - slope_corner_points[0, :]), 0.0)


def on_bottom_slope(x):
    return np.isclose(np.dot(normal_bottom_slope, x - slope_corner_points[2, :]), 0.0)


boundary_selection_map = [
    {"boundary_function": on_top_slope, "tag": "on_top_slope"},
    {"boundary_function": on_bottom_slope, "tag": "on_bottom_slope"},
]

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

pressure = 1
hyperelasticity_utils.e_modul = 1.33
hyperelasticity_utils.nu = 0.33

nu, lame, shear, e_modul = compute_elastic_properties()


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


def output_transform(x, y):
    # displacement field (u, v, w)
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

    # Hard BCs:
    u_out = x_loc * u / e_modul  # u = 0 on x = 0
    v_out = y_loc * v / e_modul  # v = 0 on y = 0
    w_out = (z_loc - seating_coord_z) * (
        z_loc - z_top
    ) * w / e_modul + displacement_top * (z_loc - seating_coord_z) / (
        z_top - seating_coord_z
    )  # w = 0 on z = seating_coord_z, w = displacement_top on the top slope

    return bkd.concat([u_out, v_out, w_out], axis=1)


# 3 inputs, 3 outputs for 3D
layer_size = [3] + [50] * 5 + [3]  # also try [3] + [50] * 10 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights = None

model = dde.Model(data, net)

# Model parameters
steps = 1
displacement_max = -0.05
model_path = str(Path(__file__).parent)
simulation_case = f"3d_leguan_bridge_incremental_force"
learning_rate_adam = 1e-3
learning_rate_total_decay = 1e-3
adam_iterations = 2000
exponential_decay = learning_rate_total_decay ** (1 / 5000)
lbfgs_iterations = 2000
rel_err_l2_disp = []
rel_err_l2_stress = []
l2_iteration = []
relaxation_adam_iterations = (
    0  # just to not get any errors when not using it (undefined variable in naming)
)
relaxation = False
time_dict["setup"].append(time.time())

if relaxation:
    time_dict["relaxation_compiling"].append(time.time())
    displacement_top = displacement_max / steps
    relaxation_adam_iterations = 1000
    print(f"\nRelaxation step for initial force of {displacement_top}.\n")
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

# Incremental loop
for i in range(steps):
    displacement_top = displacement_max / steps * (i + 1)
    print(f"\nTraining for a force of {displacement_top}.\n")
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
    displacement_pred = np.column_stack(
        (output[:, 0:1], output[:, 1:2], output[:, 2:3])
    )
    (
        sigma_xx,
        sigma_yy,
        sigma_zz,
        sigma_xy,
        sigma_yx,
        sigma_xz,
        sigma_zx,
        sigma_yz,
        sigma_zy,
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
