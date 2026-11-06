"""
Physics-Informed Neural Networks (PINNs) for nonlinear beam under shear load using Deep Energy method.

This example is to test the capabilities of the API between 4C and pinns_for_comp_mech

@author: svoelkl
"""

import os
import pickle
from pathlib import Path

import deepxde as dde
import numpy as np
import pyvista as pv
from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import APIGeometry
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_2D,
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

# Load mesh from 4C output
path_pickle_files = os.path.join(os.getcwd(), "context-and-states")
with open(f"{path_pickle_files}/context.pkl", "rb") as f:
    context = pickle.load(f)

# Generate mesh for EPINN
problem_dim = context["problem_dim"]
FourCgeometry = APIGeometry(problem_dim, context["mesh"])
gmsh_model = FourCgeometry.generateGmshModel(visualize_mesh=False)

# Generate quadratures for EPINN
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=problem_dim, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=problem_dim - 1, ngp=4
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

# get corner node coordinates
_, arr, _ = gmsh_model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True)
node_xy = arr.reshape(-1, 3)[:, :2]
mn, mx = node_xy.min(0), node_xy.max(0)

bottom_left_corner_node_coords = node_xy[
    np.linalg.norm(node_xy - np.array([mn[0], mx[1]]), axis=1).argmin()
]
bottom_right_corner_node_coords = node_xy[
    np.linalg.norm(node_xy - np.array([mx[0], mx[1]]), axis=1).argmin()
]
top_right_corner_node_coords = node_xy[
    np.linalg.norm(node_xy - np.array([mx[0], mn[1]]), axis=1).argmin()
]
top_left_corner_node_coords = node_xy[
    np.linalg.norm(node_xy - np.array([mn[0], mn[1]]), axis=1).argmin()
]


# define BCs for problem
def boundary_right(x):
    """Check whether a point satisfies the `boundary_right` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `boundary_right` evaluation.
    """
    return np.isclose(x[0], top_right_corner_node_coords[0])


boundary_selection_map = [
    {"boundary_function": boundary_right, "tag": "boundary_right"}
]

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=problem_dim,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    revert_curve_list=[],
    revert_normal_dir_list=[],
    coord_quadrature_boundary=coord_quadrature_boundary,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
)

# change global variables in elasticity_utils
hyperelasticity_utils.lame = 2.78
hyperelasticity_utils.shear = 4.17
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# Neumann boundary condition
max_shear_load = 1e-2


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
    internal_energy_density = strain_energy_neo_hookean_2d(inputs, outputs)

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["boundary_right"]
    nx = mapped_normal_boundary_t[:, 0:1][cond]
    ny = mapped_normal_boundary_t[:, 1:2][cond]

    u_x = outputs[:, 0:1][beg_boundary:][cond]
    u_y = outputs[:, 1:2][beg_boundary:][cond]

    external_force_density = -shear_load * u_y
    external_work = (
        global_weights_boundary_t[cond]
        * (external_force_density)
        * jacobian_boundary_t[cond]
    )

    return [internal_energy, -external_work]


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

    return bkd.concat(
        [
            u * x_loc / top_right_corner_node_coords[0],
            v * x_loc / top_right_corner_node_coords[0],
        ],
        axis=1,
    )


# two inputs x and y, output is ux and uy
layer_size = [2] + [64] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# get number of steps
steps = 10
# model parameters
model_path = str(Path(__file__).parent)
simulation_case = f"Beam_API_load"
learning_rate_adam = 1e-3
adam_iterations = 2000
lbfgs_iterations = 3000

# incremental learning with the network
for i in range(steps):
    with open(f"{path_pickle_files}/state-{i + 1}.pkl", "rb") as f:
        state = pickle.load(f)
    shear_load = max_shear_load / steps * (i + 1)
    print(f"\nTraining for a load of {shear_load}.\n")
    model.compile("adam", lr=learning_rate_adam)
    losshistory, train_state = model.train(
        iterations=adam_iterations, display_every=100
    )

    if lbfgs_iterations > 0:
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
        model.compile("L-BFGS")
        losshistory, train_state = model.train(display_every=1000)

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
    output = model.predict(points)
    displacement_pred = np.column_stack((output[:, 0:1], output[:, 1:2]))
    sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(
        points, operator=cauchy_stress_2D
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
    file_path = os.path.join(
        model_path, f"{simulation_case}_{int(shear_load * 1e3):02}"
    )
    grid.save(f"{file_path}.vtu")

    output_to_4C = output.flatten()
    model.save(f"{model_path}/{simulation_case}_load_step_{i + 1}")
    dde.saveplot(
        losshistory,
        train_state,
        issave=True,
        isplot=False,
        output_dir=model_path,
        loss_fname=f"{simulation_case}_load_step_{i + 1}-{train_state.step}_loss.dat",
        train_fname=f"{simulation_case}_load_step_{i + 1}-{train_state.step}_train.dat",
        test_fname=f"{simulation_case}_load_step_{i + 1}-{train_state.step}_test.dat",
    )
