"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import deepxde as dde
import numpy as np

# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import torch

from compsim_pinns.elasticity.elasticity_utils import momentum_2d, stress_plane_strain
from compsim_pinns.postprocess.elasticity_postprocessing import (
    meshGeometry,
    postProcess,
)

geom_rectangle = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 1])
geom_disk = dde.geometry.Disk([1, 1], 1)
geom = dde.geometry.csg.CSGIntersection(geom1=geom_rectangle, geom2=geom_disk)


def boundary_upper(x, on_boundary):
    """Check whether a point satisfies the `boundary_upper` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `boundary_upper` evaluation.
    """
    return on_boundary and np.isclose(x[1], 1)


def boundary_circle(x, on_boundary):
    """Check whether a point satisfies the `boundary_circle` boundary condition.

    Args:
        x: Input coordinates used to evaluate the function.
        on_boundary: Boundary indicator provided by the geometry callback.

    Returns:
        bool: Result of the `boundary_circle` evaluation.
    """
    return on_boundary and np.isclose(np.linalg.norm(x - [1, 1], axis=-1), 1)


def disp_on_middle_points(x):
    """
    Applies zero displacement for the nodes that are created at the middle of the half circle.
    """
    return 0


def calculate_gap(x, y, X):
    """
    Controls the gap between each node and y coordinates of contact constraint (y_0) using sign function (gap>=0, first condition of KarushKuhnTucker-KKT).
    If y_0 is set to 0, the gap will be y coordinate + displacement in y direction (since displacement is negative).

    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    (1-torch.sign(gap))*gap: tensor
        gap between each node and contact level
    """
    y_0 = 0
    y_coordinate = x[:, 1:2]
    y_displacement = y[:, 1:2]

    gap = y_coordinate + y_displacement + y_0

    return (1 - torch.sign(gap)) * gap


def calculate_pressure(x, y, X):
    """
    Controls the pressure on the surface using sign function (pressure<=0, second condition of KarushKuhnTucker-KKT).

    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    (1+torch.sign(sigma_yy))*sigma_yy: tensor
        pressure on the surface
    """

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x, y)

    return (1 + torch.sign(sigma_yy)) * sigma_yy


def product_gap_pressure(x, y, X):
    """
    Controls the third (complimentary) condition of KarushKuhnTucker-KKT) which is the multiplication of gap by pressure (gap*pressure=0)

    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    gap*sigma_yy: tensor
        complimentary part of KarushKuhnTucker-KKT conditions
    """
    gap = x[:, 1:2] + y[:, 1:2]
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x, y)

    return gap * sigma_yy


n_mid_points = 50
middle_points_u = np.vstack(
    (np.full(n_mid_points, 1), np.linspace(0, 1, num=n_mid_points))
).T

observe_u = dde.PointSetBC(
    middle_points_u, disp_on_middle_points(middle_points_u), component=0
)

bc1 = dde.DirichletBC(
    geom, lambda _: 0.0, boundary_upper, component=0
)  # fixed in x direction
bc2 = dde.DirichletBC(
    geom, lambda _: -0.1, boundary_upper, component=1
)  # apply disp in y direction
bc_gap = dde.OperatorBC(geom, calculate_gap, boundary_circle)
bc_pressure = dde.OperatorBC(geom, calculate_pressure, boundary_circle)
bc_multip = dde.OperatorBC(geom, product_gap_pressure, boundary_circle)

data = dde.data.PDE(
    geom,
    momentum_2d,
    [bc1, bc2, observe_u, bc_gap, bc_pressure, bc_multip],
    num_domain=1000,
    num_boundary=200,
    anchors=middle_points_u,
    num_test=200,
)

# two inputs x and y, output is ux and uy
layer_size = [2] + [60] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1, 1, 1, 1, 1, 1, 1, 1])
losshistory, train_state = model.train(iterations=3000, display_every=500)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, triangles = meshGeometry(
    geom, n_boundary=130, max_mesh_area=0.01, boundary_distribution="Sobol"
)

postProcess(model, X, triangles, output_name="displacement")
