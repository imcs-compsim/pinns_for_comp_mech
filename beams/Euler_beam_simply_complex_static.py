"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

"""
This script is used to create the PINN model of clamped beam having non-uniform cross-section under arbitrary load
see the manuscript for the example, Section 4, a complex consideration, Fig. 4.5, Deep Learning in Computational Mechanics
"""
from deepxde.backend import get_preferred_backend
backend_name = get_preferred_backend()
if (backend_name == "tensorflow.compat.v1") or ((backend_name == "tensorflow")):
    import tensorflow as bkd
elif (backend_name == "pytorch"):
    import torch as bkd
else:
    raise NameError(f'The backend {backend_name} is not available. Please use ') 

def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


L = 1


def p(x):
    return -(
        8
        * np.pi**2
        * (
            (2 * np.pi**2 * x**2 - 1) * bkd.sin(2 * np.pi * x)
            - 4 * np.pi * x * bkd.cos(2 * np.pi * x)
        )
    )


EI_material = lambda x: x**2


def pde(x, y):
    dy_xx = ddy(x, y)
    EI_dy_xx = EI_material(x) * dy_xx
    dy_xxxx = dde.grad.hessian(EI_dy_xx, x)
    return dy_xxxx + p(x)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def func(x):
    return np.sin(2 * np.pi * x)


geom = dde.geometry.Interval(0, L)

bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_l)
bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)


data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=20,
    num_boundary=2,
    solution=func,
    num_test=100,
)
layer_size = [1] + [30] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=45000, display_every=1000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
