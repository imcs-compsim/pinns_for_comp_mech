"""
This example demonstrates the application of deep energy methods to solve a Hertzian contact problem.
The setup consists of a quarter disc in contact with a rigid surface, where the disc is subjected
to an external pressure on its top surface. The code defines the geometry of the problem using Gmsh,
sets up the potential energy functional that includes internal energy, external work, and contact
penalty, and trains a neural network to minimize this energy.
The results are then compared to a finite element solution, and the predictions, FEM results, and
errors are saved in VTK format for visualization.
"""

import os

import deepxde as dde
import numpy as np
from deepxde import backend as bkd

from compsim_pinns.contact_mech import contact_utils
from compsim_pinns.contact_mech.contact_utils import (
    calculate_gap_in_normal_direction_deep_energy,
)
from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import QuarterDisc
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_2D,
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.postprocess.custom_callbacks import EpochTracker, SaveModelVTU
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 11}
radius = 1
center = [0, 0]

Quarter_Disc = QuarterDisc(
    radius=radius,
    center=center,
    mesh_size=0.04,
    angle=225,
    refine_times=10,
    gmsh_options=gmsh_options,
)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(
    visualize_mesh=False
)

revert_curve_list = []
revert_normal_dir_list = [1, 2, 2, 1]


def on_boundary_circle_contact(x):
    """Check whether a point satisfies the `on_boundary_circle_contact` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_boundary_circle_contact` evaluation.
    """
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (
        x[0] >= x_loc_partition
    )


def on_top(x):
    """Compute on top for this example setup.

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

quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=2, ngp=2
)  # gauss_legendre gauss_lobatto
coord_quadrature, weight_quadrature = quad_rule.generate()

quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=1, ngp=6
)  # gauss_legendre gauss_lobatto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=2,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    revert_curve_list=revert_curve_list,
    revert_normal_dir_list=revert_normal_dir_list,
    coord_quadrature_boundary=coord_quadrature_boundary,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
    lagrange_method=True,
)

# change global variables in elasticity_utils
hyperelasticity_utils.youngs_modulus = 50
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# zero neumann BC functions need the geom variable to be
elasticity_utils.geom = geom

projection_plane = {"y": -1}  # projection plane formula
contact_utils.projection_plane = projection_plane

# The applied pressure
ext_traction = 5
epochs = 5000
steps = 10

# stabilization model epoch
stabilization_model_epoch = None
increment_tracker = steps * [True]


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
    lagrange_parameter_boundary,
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
        lagrange_parameter_boundary: Value for lagrange parameter boundary.

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
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    if model.data.current_epoch is not None:
        if stabilization_model_epoch is not None:
            current_epoch = model.data.current_epoch - stabilization_model_epoch
        else:
            current_epoch = model.data.current_epoch
        step_size = epochs / steps  # e.g., 10
        current_step = int(current_epoch // step_size)
        step_load = (current_step + 1) * ext_traction / steps
    else:
        step_load = 0

    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:, 1:2][beg_boundary:][cond]

    external_force_density = -step_load * u_y
    external_work = (
        global_weights_boundary_t[cond]
        * (external_force_density)
        * jacobian_boundary_t[cond]
    )
    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]

    gap_n = calculate_gap_in_normal_direction_deep_energy(
        inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond
    )

    eta = 3e4
    contact_force_density = (
        1
        / (2 * eta)
        * bkd.relu(-lagrange_parameter_boundary[cond] - eta * gap_n)
        * bkd.relu(-lagrange_parameter_boundary[cond] - eta * gap_n)
    )
    contact_work = (
        global_weights_boundary_t[cond]
        * (contact_force_density)
        * jacobian_boundary_t[cond]
    )

    # update lambda
    if model.data.current_epoch is not None:
        if not (current_epoch == 0) and not (current_epoch == 1):
            if ((current_epoch + 1) % step_size) == 0:
                # The first increment is for the test data, however we dont want to to that
                if increment_tracker[current_step]:
                    lagrange_parameter_boundary[cond] = -eta * bkd.relu(-gap_n)
                if increment_tracker[current_step] == True:
                    increment_tracker[current_step] = False

    return [internal_energy, -external_work, contact_work], lagrange_parameter_boundary


n_dummy = 1
data = DeepEnergyPDE(
    geom, potential_energy, [], num_domain=n_dummy, num_boundary=n_dummy, num_test=None
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
    y_loc = x[:, 1:2]

    return bkd.concat([u * (-x_loc) / youngs_modulus, v / youngs_modulus], axis=1)


# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

file_path = os.path.join(
    os.getcwd(), "Hertzian_normal_contact_nonlinear_incremental_lagrange"
)
epoch_tracker = EpochTracker()
model_saver_incremental = SaveModelVTU(
    op=cauchy_stress_2D,
    period=500,
    stabilization_epoch=stabilization_model_epoch,
    filename=file_path,
)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(
    epochs=epochs, callbacks=[epoch_tracker, model_saver_incremental], display_every=100
)
