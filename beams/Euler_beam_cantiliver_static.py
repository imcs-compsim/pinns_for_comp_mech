"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

"""
This script is used to create the PINN model of clamped beam
see the manuscript for the example, Section 4, Figure 4.2 and 4.3, Deep Learning in Computational Mechanics
"""


# Helper functions for pde, 2. derivative of w w.r.t x
def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


# Pressure as a function, but constant
p = lambda x: 1
# E --> Young's modulus , I --> Moment of inertia
EI_material = lambda x: x


# We define our pde using the beam theory. d’’’’ + p = 0
def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + p(x)


# We define boundary function for the left side, which is x=0, x[0] --> means for this example x-axis
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


# We define boundary function for the right side, which is x=1, x[0] --> means for this example x-axis
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


# The analytical solution
def func(x):
    return -p(x) * x**2 / (24 * 1) * ((x - 1) ** 2)


# We generate 1D beam, start point 0, end point 1.
geom = dde.geometry.Interval(0, 1)

# We define boundary conditions.
# On the left side, w=0 and dw/dx/slope of the beam=0
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
# On the right side (end of the interval) x=1, w=0 and dw/dx/slope of the beam=0
bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.NeumannBC(geom, lambda x: 0, boundary_r)

# We generate 20 points in domain, 2 on the boundary, we provide func which is the solution, number of test points
data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=20,
    num_boundary=2,
    solution=func,
    num_test=100,
)

# We set input dimension --> 1D --> [1], number of layers --> 3, number of neurons each has 30, output is 1D
# alternatively, you can define [1,30,30,30,1]
layer_size = [1] + [30] * 3 + [1]
# We choose activation function, such as tanh
activation = "tanh"
# We define how to initialize networks weights
initializer = "Glorot uniform"
# We generate our neural network
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

# We define the optimizer, in this case "adam", lr --> learning rate
model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
# We train our model using 40000 iterations, we display results every 1000 iterations
losshistory, train_state = model.train(epochs=40000, display_every=1000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)

"""
fname="case_1"
loss_fname = fname + "_" + "loss.dat"
train_fname = fname + "_" + "train.dat"
test_fname = fname + "_" + "test.dat"
dde.saveplot(losshistory, train_state, issave=True, isplot=True, plot_name=fname,
loss_fname=loss_fname, train_fname=train_fname, test_fname=test_fname, 
output_dir="/home/a11btasa/git_repos/deepxde_2/beam_results")
"""
