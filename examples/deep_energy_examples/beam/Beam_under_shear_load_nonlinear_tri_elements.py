"""
Physics-Informed Neural Network for nonlinear beam under shear load using deep energy method.

This module implements a PINN solver for a 2D beam subjected to shear loading using the deep energy
approach with triangular finite elements. It uses the DeepXDE framework combined with hyperelasticity
theory (neo-Hookean material) to solve the nonlinear structural mechanics problem.

The solution is obtained by minimizing the total potential energy (strain energy minus external work)
using automatic differentiation and neural networks. Results are visualized by exporting displacement
and stress fields to VTK format.

Key features:
    - Nonlinear hyperelastic material model (neo-Hookean)
    - Gmsh-based mesh generation with custom geometry handling
    - Gauss quadrature integration for accurate energy computation
    - Output transformation with material-dependent scaling
    - Visualization of displacement and stress (Cartesian and polar coordinates)

@author: tsahin
"""

import os

import deepxde as dde
import numpy as np
from deepxde import backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import (
    GmshGeometryElementDeepEnergy,
)
from compsim_pinns.geometry.geometry_utils import (
    polar_transformation_2d,
)
from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    cauchy_stress_2D,
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

gmsh_options = {"General.Terminal": 1, "Mesh.Algorithm": 6}
block_2d = Block_2D(
    coord_left_corner=[0, -1],
    coord_right_corner=[20, 1],
    mesh_size=0.2,
    gmsh_options=gmsh_options,
)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

domain_dimension = 2
quad_rule = GaussQuadratureRule(
    rule_name="gauss_legendre",
    element_type="simplex",
    dimension=domain_dimension,
    ngp=4,
)  # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

boundary_dimension = 1
quad_rule_boundary_integral = GaussQuadratureRule(
    rule_name="gauss_legendre", dimension=boundary_dimension, ngp=4
)  # gauss_legendre gauss_labotto
coord_quadrature_boundary, weight_quadrature_boundary = (
    quad_rule_boundary_integral.generate()
)

l_beam = block_2d.coord_right_corner[0] - block_2d.coord_left_corner[0]
h_beam = block_2d.coord_right_corner[1] - block_2d.coord_left_corner[1]


def boundary_right(x):
    """Check whether a point satisfies the `boundary_right` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.

    Returns:
        bool: Result of the `boundary_right` evaluation.
    """
    return np.isclose(x[0], l_beam)


boundary_selection_map = [
    {"boundary_function": boundary_right, "tag": "boundary_right"}
]

revert_curve_list = []
revert_normal_dir_list = [1, 2, 1, 1]

geom = GmshGeometryElementDeepEnergy(
    gmsh_model,
    dimension=domain_dimension,
    coord_quadrature=coord_quadrature,
    weight_quadrature=weight_quadrature,
    revert_curve_list=revert_curve_list,
    revert_normal_dir_list=revert_normal_dir_list,
    boundary_dim=boundary_dimension,
    coord_quadrature_boundary=coord_quadrature_boundary,
    weight_quadrature_boundary=weight_quadrature_boundary,
    boundary_selection_map=boundary_selection_map,
)

# change global variables in elasticity_utils
hyperelasticity_utils.lame = 2.78
hyperelasticity_utils.shear = 4.17
hyperelasticity_utils.stress_state = "plane_strain"
nu, lame, shear, youngs_modulus = compute_elastic_properties()

# The applied pressure
shear_load = 1e-2


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
        global_element_weights_t
        * (internal_energy_density[beg_pde:beg_boundary])
        * jacobian_t
    )
    ####################################################################################################################
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

    ####################################################################################################################
    # Reshape energy-work terms and sum over the gauss points
    # internal_energy_reshaped = bkd.sum(bkd.reshape(internal_energy, (n_e, n_gp)), dim=1)
    # external_work_reshaped = bkd.sum(bkd.reshape(external_work, (n_e_boundary_external, n_gp_boundary)), dim=1)
    # sum over the elements and get the overall loss
    # total_energy = bkd.reduce_sum(internal_energy_reshaped) #- bkd.reduce_sum(external_work_reshaped)

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

    return bkd.concat([u * x_loc / youngs_modulus, v * x_loc / youngs_modulus], axis=1)


# two inputs x and y, output is ux and uy
layer_size = [2] + [64] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# if we want to save the model, we use "model_save_path=model_path" during training, if we want to load trained model, we use "model_restore_path=return_restore_path(model_path, num_epochs)"
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=10000, display_every=1000)

model.compile("L-BFGS")
# model.train_step.optimizer_kwargs["options"]['maxiter']=2000
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy, sigma_yx = model.predict(X, operator=cauchy_stress_2D)
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(
    sigma_xx, sigma_yy, sigma_xy, X
)

combined_disp = tuple(
    np.vstack(
        (
            np.array(displacement[:, 0].tolist()),
            np.array(displacement[:, 1].tolist()),
            np.zeros(displacement[:, 0].shape[0]),
        )
    )
)
combined_stress = tuple(
    np.vstack(
        (
            np.array(sigma_xx.flatten().tolist()),
            np.array(sigma_yy.flatten().tolist()),
            np.array(sigma_xy.flatten().tolist()),
        )
    )
)
combined_stress_polar = tuple(
    np.vstack(
        (
            np.array(sigma_rr.tolist()),
            np.array(sigma_theta.tolist()),
            np.array(sigma_rtheta.tolist()),
        )
    )
)

file_path = os.path.join(os.getcwd(), "Beam_under_shear_load_nonlinear")

x = X[:, 0].flatten()
y = X[:, 1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(
    file_path,
    x,
    y,
    z,
    dol_triangles.flatten(),
    offset,
    cell_types,
    pointData={
        "displacement": combined_disp,
        "stress": combined_stress,
        "stress_polar": combined_stress_polar,
    },
)
