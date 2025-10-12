"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import numpy as np

"""
This script is used to create the PINN model of simply supported beam under point load (heaviside function)
four point bending test
"""
from deepxde.backend import get_preferred_backend

# Check what backend is imported and check whether it works for this code
backend_name = get_preferred_backend()
if (backend_name == "tensorflow.compat.v1") or ((backend_name == "tensorflow")):
    import tensorflow as bkd
else:
    raise NameError(f"The backend {backend_name} is not available. Please use other.")


# Helper functions for pde, 2. derivative of w w.r.t x
def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


# p = lambda x: 1
L = 1
"""
def p(x):
    if x==0.5:
        return 1.
    else:
        return 0.
"""


# Pressure as a function, but not constant
def p(x):
    return (
        bkd.experimental.numpy.heaviside(x - 0.3, 1)
        - bkd.experimental.numpy.heaviside(x - 0.3 - 0.005, 1)
    ) + 20 * (
        bkd.experimental.numpy.heaviside(x - 0.6, 1)
        - bkd.experimental.numpy.heaviside(x - 0.7, 1)
    )


# def p(x):
#    return 4*(x-1)**2

# Not being used
EI_material = lambda x: 1


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
    return -(x**6) / 90 + x**5 / 15 + -(x**4) / 6 + x**3 / 6 - x / 18


# We generate 1D beam, start point 0, end point 1.
geom = dde.geometry.Interval(0, L)

# We define boundary conditions.
# On the left side, w=0 and ddw/ddx/bending moment=0
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_l)
# On the right side (end of the interval) x=1, w=0 and dw/dx/bending moment=0
bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)

# We generate 20 points in domain, 2 on the boundary, we provide number of test points
data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=40,
    num_boundary=2,
    # solution=None, # analytical solution is not known
    # num_test=100,
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
model.compile("adam", lr=0.001)
# We train our model using 50000 iterations, we display results every 1000 iterations
losshistory, train_state = model.train(epochs=50000, display_every=1000)

# X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
