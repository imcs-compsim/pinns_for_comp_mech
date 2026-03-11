"""
Example for a 3‑D deep‑energy PINN solving a quarter of a ring with contact.

This module sets up and trains a DeepEnergyPDE model using DeepXDE and
`compsim_pinns` utilities to compute the deformation of a hyper‑elastic
ring sector that is driven into contact with a rigid plane.  The geometry is
created with Gmsh (`RingQuarter3D`), quadrature rules are generated for both
volume and surface integrals, and boundary regions are tagged via simple
indicator functions.

The potential energy functional combines the neo‑Hookean strain energy
density with a penalty‑style contact work term evaluated on the outer
cylindrical boundary.  Elastic properties (Young’s modulus, Poisson’s ratio,
etc.) are defined globally through the `hyperelasticity_utils` helpers, and a
custom output transform enforces the prescribed vertical displacement on the
top face.

A feed‑forward neural network with five hidden layers is trained first with
Adam and then with L‑BFGS; a pretrained model can optionally be restored.
After training the script exports the unstructured mesh together with the
predicted displacements and Cauchy stress components to a VTU file for
post‑processing (via PyVista).
"""

from pathlib import Path

import deepxde as dde
import numpy as np
import pyvista as pv
from deepxde import backend as bkd
from deepxde.optimizers.config import LBFGS_options

from compsim_pinns.contact_mech import contact_utils
from compsim_pinns.contact_mech.contact_utils import (
    calculate_gap_in_normal_direction_deep_energy,
)
from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import RingQuarter3D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_3D,
    compute_elastic_properties,
    strain_energy_neo_hookean_3d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

inner_radius = 0.9
outer_radius = 1.0
center = [0, 0, 0]

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = RingQuarter3D(
    center=center,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
    mesh_size=0.025,
    num_elements=5,
)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)


def on_boundary_circle_contact(x):
    """Check whether a point satisfies the `on_boundary_circle_contact` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `on_boundary_circle_contact` evaluation.
    """
    return np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), outer_radius)


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

domain_dimension = 3
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=domain_dimension, ngp=2
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=boundary_dimension, ngp=3
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=domain_dimension,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    boundary_dim=boundary_dimension,
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
    """Calculate the potential energy of the system based on strain energy density.

    This function computes the total internal energy by integrating the Neo-Hookean
    strain energy density over the domain using numerical quadrature.

    Args:
        X (Tensor): The coordinate points in the domain.
        inputs (Tensor): Input features for the neural network (e.g., coordinates).
        outputs (Tensor): Output predictions from the neural network (e.g., displacements).
        beg_pde (int): Beginning index for PDE domain elements.
        beg_boundary (int): Beginning index for boundary elements.
        n_e (int): Number of elements in the domain.
        n_gp (int): Number of Gauss points per element.
        n_e_boundary (int): Number of boundary elements.
        n_gp_boundary (int): Number of Gauss points per boundary element.
        jacobian_t (Tensor): Jacobian determinants at domain integration points.
        global_element_weights_t (Tensor): Gaussian quadrature weights for domain integration.
        mapped_normal_boundary_t (Tensor): Mapped normal vectors at boundary Gauss points.
        jacobian_boundary_t (Tensor): Jacobian determinants at boundary integration points.
        global_weights_boundary_t (Tensor): Gaussian quadrature weights for boundary integration.
        boundary_selection_tag (Tensor): Tags identifying boundary regions.

    Returns:
        list: A list containing a single Tensor with the integrated internal energy
            over all domain elements.
    """

    internal_energy_density = strain_energy_neo_hookean_3d(inputs, outputs)

    internal_energy = (
        global_element_weights_t[:, 0:1]
        * global_element_weights_t[:, 1:2]
        * global_element_weights_t[:, 2:3]
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )

    ####################################################################################################################
    # contact work
    cond = boundary_selection_tag["on_boundary_circle_contact"]

    # gap_y = inputs[:,1:2][beg_boundary:][cond] + outputs[:,1:2][beg_boundary:][cond] + radius
    # gap_n = tf.math.divide_no_nan(gap_y, tf.math.abs(mapped_normal_boundary_t[:,1:2][cond]))
    gap_n = calculate_gap_in_normal_direction_deep_energy(
        inputs[beg_boundary:], outputs[beg_boundary:], X, mapped_normal_boundary_t, cond
    )
    eta = 3e4
    contact_force_density = 1 / 2 * eta * bkd.relu(-gap_n) * bkd.relu(-gap_n)
    contact_work = (
        global_weights_boundary_t[:, 0:1][cond]
        * global_weights_boundary_t[:, 1:2][cond]
        * (contact_force_density)
        * jacobian_boundary_t[cond]
    )

    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # contact_work_reshaped = bkd.sum(bkd.reshape(contact_work, (n_e_boundary_contact, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) - bkd.reduce_sum(external_work_reshaped) + bkd.reduce_sum(contact_work_reshaped)

    return [internal_energy, contact_work]


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
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    # tf.math.abs(x_loc)
    return bkd.concat(
        [
            u * (y_loc * x_loc) / youngs_modulus,
            v * (-y_loc) / youngs_modulus + applied_disp_y,
            w * (-z_loc) / youngs_modulus,
        ],
        axis=1,
    )


# 3 inputs, 3 outputs for 3D
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
restore_model = False
model_path = (
    str(Path(__file__).parent.parent.parent)
    + f"/pretrained_models/deep_energy_examples/3d_contact_ring_instability/3d_contact_ring_instability_quarter_nonlinear"
)

if not restore_model:
    # model.compile("adam", lr=0.001)
    # losshistory, train_state = model.train(epochs=stabilization_model_epoch, display_every=100)

    model.compile("adam", lr=0.001)
    # losshistory, train_state = model.train(epochs=5000, display_every=100)
    # if you want to save the model, run the following
    losshistory, train_state = model.train(
        epochs=5000, display_every=100, model_save_path=model_path
    )

    # For pytorch
    # LBFGS_options["iter_per_step"] = 1
    # LBFGS_options["maxiter"] = 500

    LBFGS_options["maxiter"] = 1000
    model.compile("L-BFGS")
    # losshistory, train_state = model.train(display_every=100)
    losshistory, train_state = model.train(
        display_every=100, model_save_path=model_path
    )
else:
    n_epochs = 5471
    model_restore_path = model_path + "-" + str(n_epochs) + ".ckpt"

    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

# Get mesh data from your class
points, _, cell_types, elements = geom.get_mesh()

# 1. Flatten elements and prepend number of nodes per cell
n_nodes_per_cell = elements.shape[1]
n_cells = elements.shape[0]

cells = np.hstack([np.insert(elem, 0, n_nodes_per_cell) for elem in elements])

# 2. Make sure data types are correct
cells = np.array(cells, dtype=np.int64)
cell_types = np.array(cell_types, dtype=np.uint8)

# 3. Create the UnstructuredGrid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Predict the displacements
output = model.predict(points)

# Predict stress components
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

cauchy_stress = np.column_stack(
    (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz)
)
displacement = np.column_stack((output[:, 0:1], output[:, 1:2], output[:, 2:3]))


grid.point_data["pred_displacement"] = displacement
grid.point_data["pred_cauchy_stress"] = cauchy_stress

grid.save("deep_energy_3d_contact_ring_instability_quarter.vtu")
