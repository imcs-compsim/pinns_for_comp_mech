"""
3D Hertzian Spherical Contact Analysis Using Deep Energy Method

This module implements a Physics-Informed Neural Network (PINN) for solving 3D Hertzian
spherical contact problems using the Deep Energy method. The problem involves a sphere
in contact with a flat surface under applied pressure, with contact constraints handled
through a penalty method.

Key Features:
    - 3D geometry modeling using Gmsh with eighth-symmetry sphere
    - Neo-Hookean hyperelastic material model
    - Variational formulation with volumetric and boundary integrals
    - Contact mechanics enforcement via gap functions and penalty method
    - Cauchy stress computation and von Mises stress evaluation
    - VTK output for visualization of displacements and stresses

The neural network approximates the displacement field while satisfying:
    - Boundary conditions on fixed surfaces (x=0, z=0)
    - Contact non-penetration on the spherical surface
    - Applied traction on the top surface
    - Stress equilibrium through energy minimization

Outputs are saved in VTU format containing displacement, Cauchy stress components,
and von Mises stress distributions.

@author: tsahin
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
from compsim_pinns.geometry.gmsh_models import SphereEighthHertzian
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_3D,
    compute_elastic_properties,
    strain_energy_neo_hookean_3d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

radius = 1
center = [0, 0, 0]

Block_3D_obj = SphereEighthHertzian(radius=radius, center=center)

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)


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

domain_dimension = 3
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre",
    dimension=domain_dimension,
    ngp=4,
    element_type="simplex",
)  # gauss_legendre gauss_lobatto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 2
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre",
    dimension=boundary_dimension,
    ngp=4,
    element_type="simplex",
)  # gauss_legendre gauss_lobatto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

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

# change global variables in elasticity_utils
hyperelasticity_utils.youngs_modulus = 50
hyperelasticity_utils.nu = 0.3
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# The applied pressure
ext_traction = 5

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
        global_element_weights_t
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )
    ####################################################################################################################
    # get the external work
    # select the points where external force is applied
    cond = boundary_selection_tag["on_top"]
    u_y = outputs[:, 1:2][beg_boundary:][cond]

    external_force_density = -ext_traction * u_y
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
    contact_force_density = 1 / 2 * eta * bkd.relu(-gap_n) * bkd.relu(-gap_n)
    contact_work = (
        global_weights_boundary_t[cond]
        * (contact_force_density)
        * jacobian_boundary_t[cond]
    )

    return [internal_energy, -external_work, contact_work]


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
    w = y[:, 2:3]

    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]

    # define surfaces
    # top_surface = -y_loc
    x_0_surface = x_loc
    z_0_surface = z_loc

    return bkd.concat(
        [
            u
            * (x_0_surface)
            / youngs_modulus,  # displacement in x direction is 0 at x=0
            v / youngs_modulus,
            w
            * (z_0_surface)
            / youngs_modulus,  # displacement in z direction is 0 at z=0
        ],
        axis=1,
    )


# 3 inputs, 3 outputs for 3D
layer_size = [3] + [50] * 5 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights = None

model = dde.Model(data, net)

restore_model = False
model_path = (
    str(Path(__file__).parent.parent.parent)
    + f"/pretrained_models/deep_energy_examples/3d_herzian_spherical_contact/3d_herzian_spherical_contact_nonlinear"
)

if not restore_model:
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=5000, display_every=100)

    LBFGS_options["maxiter"] = 1000
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=100)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
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

von_mises = np.sqrt(
    0.5
    * (
        (sigma_xx - sigma_yy) ** 2
        + (sigma_yy - sigma_zz) ** 2
        + (sigma_zz - sigma_xx) ** 2
        + 6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
    )
)

grid.point_data["pred_displacement"] = displacement
grid.point_data["pred_cauchy_stress"] = cauchy_stress
grid.point_data["von_mises"] = von_mises

grid.save("3d_hertzian_spherical_contact.vtu")
