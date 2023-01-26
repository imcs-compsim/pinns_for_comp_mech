"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf

import sys
from pathlib import Path
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_strain, momentum_2d 
from geometry_utils import calculate_boundary_normals
from elasticity_postprocessing import meshGeometry, postProcess


geom_rectangle = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2, 1])
geom_disk = dde.geometry.Disk([1, 1], 1)
geom = dde.geometry.csg.CSGIntersection(geom1=geom_rectangle, geom2=geom_disk)


def boundary_upper(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def boundary_circle(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - [1,1], axis=-1), 1)

def disp_on_middle_points(x):
    '''
    Applies zero displacement for the nodes that are created at the middle of the half circle. 
    '''
    return 0

def calculate_gap(x, y, X):
    '''
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
    (1-tf.math.sign(gap))*gap: tensor
        gap between each node and contact level  
    '''
    y_0 = 0
    y_coordinate = x[:,1:2]
    y_displacement = y[:,1:2]
    
    gap = y_coordinate + y_displacement + y_0

    return (1-tf.math.sign(gap))*gap

def calculate_pressure(x,y,X):
    '''
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
    (1+tf.math.sign(sigma_yy))*sigma_yy: tensor
        pressure on the surface 
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    return (1+tf.math.sign(sigma_yy))*sigma_yy

def product_gap_pressure(x,y,X):
    '''
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
    '''
    gap = x[:,1:2] + y[:,1:2]
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    return gap*sigma_yy

n_mid_points = 50
middle_points_u = np.vstack((np.full(n_mid_points, 1),np.linspace(0, 1, num=n_mid_points))).T

observe_u  = dde.PointSetBC(middle_points_u, disp_on_middle_points(middle_points_u), component=0)

bc1 = dde.DirichletBC(geom, lambda _: 0.0, boundary_upper, component=0) # fixed in x direction
bc2 = dde.DirichletBC(geom, lambda _: -0.1, boundary_upper, component=1) # apply disp in y direction
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
model.compile("adam", lr=0.001, loss_weights=[1,1,1,1,1,1,1,1]) # loss_weights=[1,1,1,1,100,100,100,1]
losshistory, train_state = model.train(epochs=3000, display_every=500)


###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, triangles = meshGeometry(geom, n_boundary=130, max_mesh_area=0.01, boundary_distribution="Sobol")

postProcess(model, X, triangles, output_name="displacement")

# The rest of the command is used to put a block to visualize the the results more realistic

# X = geom.random_points(1000, random="Sobol")
# boun = geom.uniform_boundary_points(200)
# X = np.vstack((X,boun))

# displacement = model.predict(X)
# sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_strain)
# sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

# # combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
# # combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
# # combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

# x = X[:,0].flatten()
# y = X[:,1].flatten()
# z = np.zeros(y.shape)
# triang = tri.Triangulation(x, y)
# #dol_triangles = triang.triangles
# #offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
# #cell_types = np.ones(dol_triangles.shape[0])*5

# x = np.hstack((x,[0,0,2,2]))
# y = np.hstack((y,[0-distance,-1-distance,-1-distance,0-distance]))
# z = np.hstack((z,4*[0]))
# block_triangle = np.array([[1200,1201,1202],[1200,1202,1203]])
# dol_triangles = triang.triangles
# dol_triangles = np.vstack((dol_triangles,block_triangle))
# displacement = np.vstack((displacement, np.zeros((4,2))))
# combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
# combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
# combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

# offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
# cell_types = np.ones(dol_triangles.shape[0])*5

# file_path = os.path.join(os.getcwd(),"default_result_name")

# unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
#                       cell_types, pointData = { "displacement" : combined_disp})
# # unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
# #                       cell_types, pointData = { "displacement" : combined_disp,"stress_polar" : combined_stress_polar, "stress": combined_stress})