import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

# 4C may launch this predictor in an MPI environment. DeepXDE interprets
# OMPI_COMM_WORLD_SIZE as a request for Horovod parallel training, which is not
# supported with its PyTorch backend.
os.environ.pop("OMPI_COMM_WORLD_SIZE", None)

import deepxde as dde
import numpy as np
import torch
from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import APIGeometry
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    compute_elastic_properties,
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.contact_mech import contact_utils
from compsim_pinns.contact_mech.contact_utils import calculate_gap_in_normal_direction_deep_energy
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

###########
# GLOBALS #
###########

# global variables that will be persistent during the simulation and in between calls to python
geom = None
num_load_steps = None
points = None
reduced_node_mask = None

#############
# FUNCTIONS #
#############

@contextmanager
def nvtx_range(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

# executed once, optional
def setup(context):
    global geom, num_load_steps, points, reduced_node_mask

    # store the number of load steps to be performed
    num_load_steps = context.step_max

    problem_dim = context.problem_dim
    coordinates = np.asarray(context.mesh.coordinates)
    node_ids = np.asarray(context.mesh.node_ids, dtype=np.int64)
    element_ids = np.asarray(context.mesh.element_ids, dtype=np.int64)
    connectivity = np.asarray(context.mesh.connectivity, dtype=np.int64)
    disp_dof_ids = np.asarray(context.mesh.disp_dof_ids, dtype=np.int64)

    # reduce the system to the elastic part
    rigid_cond = (coordinates[:, 1] <= -1) & (~np.isclose(coordinates[:, 0], 0.0))
    rigid_node_ids = node_ids[rigid_cond]
    element_keep = ~np.isin(connectivity, rigid_node_ids).any(axis=1)
    reduced_connectivity = connectivity[element_keep]
    reduced_element_ids = element_ids[element_keep]

    used_node_ids = np.unique(reduced_connectivity)
    node_keep = np.isin(node_ids, used_node_ids)
    reduced_mesh = SimpleNamespace(
        coordinates=coordinates[node_keep],
        node_ids=node_ids[node_keep],
        disp_dof_ids=disp_dof_ids[node_keep],
        element_ids=reduced_element_ids,
        connectivity=reduced_connectivity,
    )
    points = coordinates
    reduced_node_mask = node_keep

    # Generate mesh for EPINN
    FourCgeometry = APIGeometry(problem_dim, reduced_mesh)
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
    # compute elastic properties
    hyperelasticity_utils.youngs_modulus = 50
    hyperelasticity_utils.nu = 0.3
    hyperelasticity_utils.stress_state = "plane_strain"
    compute_elastic_properties()

    # definitions for BCs
    radius = 1
    center = [0, 0]
    x_contact_probable = 0.7
    def on_boundary_circle_contact(x):
        """Check whether a point satisfies the `on_boundary_circle_contact` boundary condition.

        Args:
            x: Input coordinates used to evaluate the function.

        Returns:
            bool: Result of the `on_boundary_circle_contact` evaluation.
        """
        return np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (
            abs(x[0]) < x_contact_probable * radius
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

    elasticity_utils.geom = geom
    projection_plane = {"y": -1}  # projection plane formula
    contact_utils.projection_plane = projection_plane


# executed in every step, mandatory
def compute(state):
    global geom, num_load_steps, points, reduced_node_mask

    with nvtx_range("PROFILE_4C_ML_STEP"):
        with nvtx_range(f"ML predictor step {state.step}"):
            with nvtx_range("build DeepXDE model"):
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
                    cond = boundary_selection_tag["on_top"]
                    nx = mapped_normal_boundary_t[:, 0:1][cond]
                    ny = mapped_normal_boundary_t[:, 1:2][cond]

                    u_x = outputs[:, 0:1][beg_boundary:][cond]
                    u_y = outputs[:, 1:2][beg_boundary:][cond]

                    external_force_density = shear_load * u_y
                    external_work = (
                        global_weights_boundary_t[cond]
                        * (external_force_density)
                        * jacobian_boundary_t[cond]
                    )
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
                    x_loc = x[:, 0:1]
                    y_loc = x[:, 1:2]

                    return bkd.concat([u / hyperelasticity_utils.youngs_modulus, v / hyperelasticity_utils.youngs_modulus], axis=1)

                # two inputs x and y, output is ux and uy
                layer_size = [2] + [64] * 5 + [2]
                activation = "tanh"
                initializer = "Glorot uniform"
                net = dde.maps.FNN(layer_size, activation, initializer)
                net.apply_output_transform(output_transform)
                model = dde.Model(data, net)

            with nvtx_range("train NN - Adam"):
                # Training parameters
                learning_rate_adam = 1e-3
                adam_iterations = 2000
                lbfgs_iterations = 3000

                max_shear_load = -5
                shear_load = state.time * max_shear_load
                print(f"\nTraining for a load of {shear_load}.\n")
                model.compile("adam", lr=learning_rate_adam)
                losshistory, train_state = model.train(
                    iterations=adam_iterations, display_every=100
                )

            if lbfgs_iterations > 0:
                with nvtx_range("train NN - L-BFGS"):
                    dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iterations)
                    model.compile("L-BFGS")
                    losshistory, train_state = model.train(display_every=1000)

            with nvtx_range("predict displacement"):
                output = model.predict(points[reduced_node_mask, :2])
                output_full = np.zeros((points.shape[0], output.shape[1]), dtype=output.dtype)
                output_full[reduced_node_mask] = output
                output_to_4C = output_full.flatten()

            if output_to_4C.size != state.dis_np.size:
                raise ValueError(
                    f"Predictor output has {output_to_4C.size} entries, "
                    f"but 4C expects {state.dis_np.size} displacement DOFs."
                )

            with nvtx_range("copy prediction to 4C state"):
                # for now, just pass the state back to 4C
                state.dis_np[:] = output_to_4C[:]
                state.vel_np.fill(0.0)
                state.acc_np.fill(0.0)
