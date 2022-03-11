"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_strain, momentum_2d
from geometry_utils import calculate_boundary_normals, polar_transformation_2d

radius_inner = 1
center_inner = [0,0]
radius_outer = 2
center_outer = [0,0]

geom_disk_1 = dde.geometry.Disk(center_inner, radius_inner)
geom_disk_2 = dde.geometry.Disk(center_outer, radius_outer)
geom = dde.geometry.csg.CSGDifference(geom1=geom_disk_2, geom2=geom_disk_1)

pressure_inlet = 1
pressure_outlet = 2

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

def meshGeometryWithHole(geom):

    import triangle as tr
    pts1 = geom.geom1.uniform_boundary_points(100)
    seg1 = tr.convex_hull(pts1)

    pts2 = geom.geom2.uniform_boundary_points(80)
    seg2 = tr.convex_hull(pts2)

    pts = np.vstack([pts1, pts2])
    seg = np.vstack([seg1, seg2 + seg1.shape[0]])

    planarStraightLineGraph = dict(vertices=pts, segments=seg, holes=[[0, 0]])
    mesh = tr.triangulate(planarStraightLineGraph, 'qpa0.01')

    return mesh['vertices'], mesh['triangles']

def postProcess(model):
    '''
    Performs test case specific post-processing of a trained model.

    Parameters
    ----------
    model : trained deepxde model

    '''
    from exportVtk import solutionFieldOnMeshToVtk

    geom = model.data.geom

    X, triangles = meshGeometryWithHole(geom)

    displacement = model.predict(X)
    sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

    combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
    # combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
    combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

    file_path = os.path.join(os.getcwd(),"Lame_problem")

    pointData = {"displacement" : combined_disp, "stress" : combined_stress_polar}

    solutionFieldOnMeshToVtk(X, triangles, pointData, file_path)


def pressure_inner_x(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure_inlet*normals[:,0:1]

def pressure_outer_x(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y + pressure_outlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure_inlet*normals[:,1:2]

def pressure_outer_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y + pressure_outlet*normals[:,1:2]

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

losshistory, train_state = model.train(epochs=6000, display_every=1000)

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

postProcess(model)

compareModelPredictionAndAnalyticalSolution(model)
