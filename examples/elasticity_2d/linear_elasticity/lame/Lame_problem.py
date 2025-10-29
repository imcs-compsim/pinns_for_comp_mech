"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

from compsim_pinns.elasticity.elasticity_utils import stress_plane_strain, momentum_2d, stress_to_traction_2d
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.postprocess.elasticity_postprocessing import meshGeometry, postProcess

radius_inner = 1
center_inner = [0,0]
radius_outer = 2
center_outer = [0,0]

geom_disk_1 = dde.geometry.Disk(center_inner, radius_inner)
geom_disk_2 = dde.geometry.Disk(center_outer, radius_outer)
geom = dde.geometry.csg.CSGDifference(geom1=geom_disk_2, geom2=geom_disk_1)

pressure_inlet = 1
pressure_outlet = 2

def pressure_inner_x(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx + pressure_inlet*normals[:,0:1]

def pressure_outer_x(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx + pressure_outlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty + pressure_inlet*normals[:,1:2]

def pressure_outer_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty + pressure_outlet*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.OperatorBC(geom, pressure_outer_x, boundary_outer)
bc4 = dde.OperatorBC(geom, pressure_outer_y, boundary_outer)

data = dde.data.PDE(
    geom,
    momentum_2d,
    [bc1, bc2, bc3, bc4],
    num_domain=800,
    num_boundary=160,
    num_test=160,
    train_distribution = "Sobol"
)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=2000, display_every=1000)

model.compile("L-BFGS")
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

def compareModelPredictionAndAnalyticalSolution(model):
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])

    rr2 = radius_inner**2 * radius_outer**2
    dr2 = radius_outer**2 - radius_inner**2
    dpdr2 = (pressure_outlet - pressure_inlet) / dr2
    dpr2dr2 = (pressure_inlet * radius_inner**2 - pressure_outlet * radius_outer**2) / dr2

    sigma_rr_analytical = rr2 * dpdr2/r**2 + dpr2dr2
    sigma_theta_analytical = - rr2 * dpdr2/r**2 + dpr2dr2

    r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
    sigma_xx, sigma_yy, sigma_xy = model.predict(r_x, operator=stress_plane_strain)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, r_x)

    plt.plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{r}$")
    plt.plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{r}$")
    plt.plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta}$")
    plt.plot(r/radius_inner, sigma_theta/radius_inner, label = r"Predicted $\sigma_{\theta}$")
    plt.legend()
    plt.xlabel("r/a")
    plt.ylabel("Normalized stress")
    plt.grid()

    plt.show()

X, triangles = meshGeometry(geom, n_boundary=100, holes=[[0, 0]], max_mesh_area=0.01)

postProcess(model, X, triangles, output_name="displacement", operator=stress_plane_strain, operator_name="stress", polar_transf = True, file_path=None)

compareModelPredictionAndAnalyticalSolution(model)
