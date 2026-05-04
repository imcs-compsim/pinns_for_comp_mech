import deepxde as dde
import numpy as np
from deepxde import backend as bkd

from compsim_pinns.deep_energy.deep_pde import DeepEnergyPDE
from compsim_pinns.geometry.custom_geometry import GmshGeometryElementDeepEnergy
from compsim_pinns.geometry.gmsh_models import APIGeometry
from compsim_pinns.hyperelasticity import hyperelasticity_utils
from compsim_pinns.hyperelasticity.hyperelasticity_utils import (
    strain_energy_neo_hookean_2d,
)
from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule

from contextlib import contextmanager
import torch

###########
# GLOBALS #
###########

# global variables that will be persistent during the simulation and in between calls to python
geom = None
num_load_steps = None
points = None

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
    global geom, num_load_steps, points
    # Load mesh from 4C output
    # with open(context, "rb") as f:
    #     my_context = pickle.load(f)

    # store the number of load steps to be performed
    num_load_steps = context.step_max

    # store the global points for evaluation in the compute function
    points = context.mesh.coordinates

    # Generate mesh for EPINN
    problem_dim = context.problem_dim
    FourCgeometry = APIGeometry(problem_dim, context.mesh)
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
    hyperelasticity_utils.lame = 2.78
    hyperelasticity_utils.shear = 4.17
    hyperelasticity_utils.stress_state = "plane_strain"

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


# executed in every step, mandatory
def compute(state):
    global geom, num_load_steps, points

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

                # get corner node coordinates
                _, arr, _ = geom.gmsh_model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True)
                node_xy = arr.reshape(-1, 3)[:, :2]
                mn, mx = node_xy.min(0), node_xy.max(0)
                top_right_corner_node_coords = node_xy[
                    np.linalg.norm(node_xy - np.array([mx[0], mn[1]]), axis=1).argmin()
                ]

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

            with nvtx_range("train NN - Adam"):
                # Training parameters
                learning_rate_adam = 1e-3
                adam_iterations = 2000
                lbfgs_iterations = 3000

                max_shear_load = 1e-2
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
                output = model.predict(points[:, :2])
                output_to_4C = output.flatten()

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