"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
# Set the backend as tensorflow.compat.v1 before importing DeepXDE
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import tensorflow as tf

'''
This script is used to create the PINN model of simply supported beam under point load (heaviside function)
four point bending test
'''

def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# p = lambda x: 1
L = 1
'''
def p(x):
    if x==0.5:
        return 1.
    else:
        return 0.
'''
def p(x):
    return (tf.experimental.numpy.heaviside(x-0.3,1) - tf.experimental.numpy.heaviside(x-0.3 - 0.005,1)) + 20*(tf.experimental.numpy.heaviside(x-0.6,1) - tf.experimental.numpy.heaviside(x-0.7,1))


#def p(x):
#    return 4*(x-1)**2
    

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
    return -x**6/90 + x**5/15 + -x**4/6 + x**3/6 -x/18



geom = dde.geometry.Interval(0, L)

bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_l)
bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
bc4 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)

data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=40,
    num_boundary=2,
    #solution=None, # analytical solution is not known
    #num_test=100,
)

layer_size = [1] + [30] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=50000, display_every=1000)

# X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
