"""
2D Contact Ring Instability Simulation on a Quarter Torus Geometry.

This example demonstrates the use of energy-based Physics-Informed Neural Networks (PINNs)
to simulate contact mechanics in a hyperelastic semicircle arc under displacement loading. The
simulation models a ring instability scenario where contact occurs between the inner
surface and a rigid plane.

Key components include:
- Geometry definition using Gmsh for a 2D semicircle arc with specified major and minor radii.
- Hyperelastic material model based on Neo-Hookean strain energy.
- Contact enforcement via a penalty method in the potential energy formulation.
- Neural network training with DeepXDE, including Adam and L-BFGS optimization.
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
from compsim_pinns.geometry.gmsh_models import RingHalf
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.postprocess.custom_callbacks import EpochTracker, SaveModelVTU
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

radius = 1
center = [0, 0]

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = RingHalf(
    center=[0, 0, 0],
    inner_radius=0.9,
    outer_radius=1,
    mesh_size=0.025,
    gmsh_options=gmsh_options,
)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_3"]
revert_normal_dir_list = [1, 1, 2, 1]


def on_boundary_circle_contact(x):
    """Check whether a point satisfies the `on_boundary_circle_contact` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_boundary_circle_contact` evaluation.
    """
    return np.isclose(np.linalg.norm(x - center, axis=-1), radius)


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
)
# change global variables in elasticity_utils
hyperelasticity_utils.youngs_modulus = 10
hyperelasticity_utils.nu = 0.3
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()

applied_disp_y = -0.8

# zero neumann BC functions need the geom variable to be
elasticity_utils.geom = geom

projection_plane = {"y": -1}  # projection plane formula
contact_utils.projection_plane = projection_plane


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

    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]

    gap_n = calculate_gap_in_normal_direction_deep_energy(
        inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond
    )
    eta = 3e4
    contact_force_density = 1 / 2 * eta * bkd.relu(-gap_n) * bkd.relu(-gap_n)
    contact_work = (
        global_weights_boundary_t[cond]
        * (contact_force_density)
        * jacobian_boundary_t[cond]
    )

    return [internal_energy, contact_work]


n_dummy = 1
data = DeepEnergyPDE(
    geom, potential_energy, [], num_domain=n_dummy, num_boundary=n_dummy, num_test=None
)

# stabilization model epoch
stabilization_model_epoch = None
epochs = 50000
steps = 10


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

    if model.data.current_epoch is not None:
        if stabilization_model_epoch is not None:
            current_epoch = model.data.current_epoch - stabilization_model_epoch
        else:
            current_epoch = model.data.current_epoch

        step_size = epochs / steps  # e.g., 10
        current_step = int(current_epoch // step_size)
        displacement_chunk = (current_step + 1) * applied_disp_y / steps
    else:
        displacement_chunk = 0

    # tf.math.abs(x_loc)
    return bkd.concat(
        [
            u * (-y_loc) / youngs_modulus,
            v * (-y_loc) / youngs_modulus + displacement_chunk,
        ],
        axis=1,
    )


# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
file_path = os.path.join(os.getcwd(), "contact_ring_instability_half_incremental")
epoch_tracker = EpochTracker()
model_saver_incremental = SaveModelVTU(period=int(epochs / steps), filename=file_path)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(
    epochs=epochs, callbacks=[epoch_tracker, model_saver_incremental], display_every=100
)
