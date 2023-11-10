"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import numpy as np
import pandas as pd


"""
This script is used to create the PINN model of clamped Euler-Lagrange beam under arbitrary load (space-time)
see the manuscript for the example, Section 4, a complex consideration, Fig. 4.5, Deep Learning in Computational Mechanics
"""
from deepxde.backend import get_preferred_backend

backend_name = get_preferred_backend()
if (backend_name == "tensorflow.compat.v1") or ((backend_name == "tensorflow")):
    import tensorflow as bkd
else:
    raise NameError(f"The backend {backend_name} is not available. Please use ")


def d_xx(x, y):
    return dde.grad.hessian(y, x)


def d_xxx(x, y):
    return dde.grad.jacobian(d_xx(x, y), x)


def pde(x, y):
    dy_xx = d_xx(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    d_tt = dde.grad.hessian(y, x, i=1, j=1)

    return dy_xxxx + d_tt + p(x)


def p(x):
    pi = np.pi
    x_t = x[:, 1:2]
    x_s = x[:, 0:1]
    return -(bkd.sin(np.pi * x_s) * bkd.exp(-x_t) * (np.pi**4 * (x_t + 1) + x_t - 1))


l_spatial = 0
r_spatial = 1
l_time = 0
r_time = 1.0

L = 1


def boundary_l_space(x, on_boundary):
    return on_boundary and np.isclose(x[0], l_spatial)


def boundary_r_space(x, on_boundary):
    return on_boundary and np.isclose(x[0], r_spatial)


def sol(x):
    x, t = np.split(x, 2, axis=1)
    # return np.sin(np.pi*x)*np.cos(np.pi**2*t)
    return np.sin(np.pi * x) * (t + 1) * np.exp(-t)


geom = dde.geometry.Interval(l_spatial, r_spatial)
timedomain = dde.geometry.TimeDomain(l_time, r_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc1 = dde.DirichletBC(geomtime, lambda x: 0, boundary_l_space)
bc2 = dde.OperatorBC(geomtime, lambda x, y, _: d_xx(x, y), boundary_l_space)
bc3 = dde.DirichletBC(geomtime, lambda x: 0, boundary_r_space)
bc4 = dde.OperatorBC(geomtime, lambda x, y, _: d_xx(x, y), boundary_r_space)

ic1 = dde.IC(
    geomtime, lambda x: bkd.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)
ic2 = dde.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda _, on_initial: on_initial,
)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc1, bc2, bc3, bc4, ic1, ic2],
    num_domain=2500,
    num_boundary=50,
    num_initial=50,
    solution=sol,
    num_test=1000,
)

layer_size = [2] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
# model.train(epochs=1000, display_every=200)
# model.compile("L-BFGS")
losshistory, train_state = model.train(epochs=20000, display_every=1000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)

"""
output_dir="/home/a11btasa/git_repos/deepxde_2/beam_results"
fname="case_2_time_final"
csv_file = output_dir + "/" + fname + ".csv"

X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
df_X_test = pd.DataFrame(X_test, columns=["x", "t"])
df_y_best = pd.DataFrame(best_y, columns=["w_predicted"])
df_y_test = pd.DataFrame(y_test, columns=["w_truth"])
df_combined = pd.concat([df_X_test, df_y_best, df_y_test], axis=1)
df_combined.to_csv(csv_file, index=False)



loss_fname = fname + "_" + "loss.dat"
train_fname = fname + "_" + "train.dat"
test_fname = fname + "_" + "test.dat"
dde.saveplot(losshistory, train_state, issave=True, isplot=True, plot_name=fname,
loss_fname=loss_fname, train_fname=train_fname, test_fname=test_fname, 
output_dir=output_dir)
"""
