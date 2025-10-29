"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

'''
This script is used to create the PINN model of cantiliver beam
see the manuscript for the example, Section 4, Figure 4.2 and 4.3, Deep Learning in Computational Mechanics
'''

def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# p = lambda x: x
L = 1.0
q = 1.0
EI = 1.0

def p(x):
    return x

EI_material = lambda x: 1


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + p(x)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

def func(x):
   return -q*x**2/(120*EI)*(20*L**3 - 10*L**2*x + x**3)


geom = dde.geometry.Interval(0, L)

bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
bc4 = dde.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

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
losshistory, train_state = model.train(epochs=30000, display_every=1000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
