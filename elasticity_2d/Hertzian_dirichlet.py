"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK 

from .. utils.elasticity_utils import stress_plane_strain, momentum_2d 
from .. utils.geometry_utils import polar_transformation_2d

geom_rectangle = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 1])
geom_disk = dde.geometry.Disk([1, 1], 1)
geom = dde.geometry.csg.CSGIntersection(geom1=geom_rectangle, geom2=geom_disk)


def boundary_upper(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def fun_middle_point(x):
    return 0

n_mid_points = 20
middle_points_u = np.vstack((np.full(n_mid_points, 1),np.linspace(0, 1, num=n_mid_points))).T
middle_points_u = middle_points_u[[0,-1]]
middle_points_u_2 = middle_points_u[[0]]

observe_u  = dde.PointSetBC(middle_points_u, fun_middle_point(middle_points_u), component=0)
observe_v  = dde.PointSetBC(middle_points_u_2, fun_middle_point(middle_points_u_2), component=1)

bc1 = dde.DirichletBC(geom, lambda _: 0.0, boundary_upper, component=0) # fixed in x direction
bc2 = dde.DirichletBC(geom, lambda _: -1.0, boundary_upper, component=1) # apply disp in y direction

data = dde.data.PDE(
    geom,
    momentum_2d,
    [bc1, bc2, observe_u, observe_v],
    num_domain=400,
    num_boundary=80,
    anchors=middle_points_u,
    num_test=100,
)

# two inputs x and y, output is ux and uy 
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=3000, display_every=1000)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################
X = geom.random_points(600, random="Sobol")
boun = geom.uniform_boundary_points(100)
X = np.vstack((X,boun))

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
sigma_rr, sigma_theta, sigma_rtheta, theta_radian = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triang = tri.Triangulation(x, y)
dol_triangles = triang.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
cell_types = np.ones(dol_triangles.shape[0])*5

file_path = "/home/a11btasa/git_repos/phd_materials/pinns/Lame_problem/Hertzian_DIRICHLET_test"

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress_polar" : combined_stress_polar, "stress": combined_stress})