import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
# add utils folder to the system path
path_utils = str(Path(__file__).parent.parent.absolute()) + "/utils"
sys.path.append(path_utils)

from elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, problem_parameters, zero_neumman_plane_stress_x, zero_neumman_plane_stress_y, elastic_strain_2d
from geometry_utils import calculate_boundary_normals
from custom_geometry import GmshGeometry2D
from gmsh_models import Rectangle_4PointBending, Block_2D
import elasticity_utils


'''
In this routine, 4 point bending test example is illustrated.

            P1                P2
***********----**************----*****************
*                                                *
*                                                *
*                                                *
***----*********************************----******
    D1                                   D2

where a pressure vector is applied on P1 and P2, while it is constrained in x and y direction at regions D1 and D2 

author: @tsahin 
'''

# gmsh options
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
# the start points and the size of 4 points are given
region_size_dict={"r1":{"start":0.25, "increment":0.1}, "r2":{"start":2.65, "increment":0.1}, "r3":{"start":1.1, "increment":0.1}, "r4":{"start":1.8, "increment":0.1}}

# generate the block, gmsh model and geometry for further calculations
block_2d = Rectangle_4PointBending(l_beam=3, h_beam=0.3, region_size_dict=region_size_dict, mesh_size=0.08, refine_factor=None, gmsh_options=gmsh_options)
#block_2d = Block_2D(coord_left_corner=[0,0], coord_right_corner=[3,0.3], mesh_size=0.3, gmsh_options=gmsh_options)
gmsh_model = block_2d.generateGmshModel(visualize_mesh=True)
geom = GmshGeometry2D(gmsh_model)

# get the length and height of the beam
l = 3
#l = block_2d.l_beam
h = 0.3
#h = block_2d.h_beam

# set the force
pressure = 0.01 # 1

# change global variables in elasticity_utils
e_1 = 2000
nu_1 = 0.3
elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
elasticity_utils.shear = e_1/(2*(1+nu_1))
# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

def neumann_y(x, y, X):
    '''
    Represents the y component of the applied pressure
    '''

    normals, cond = calculate_boundary_normals(X,geom)
    #x_loc = x[:,0:1][cond]
    return zero_neumman_plane_stress_y(x, y, X) + pressure#*(x_loc-1.1)*(x_loc-1.2)/(0.05**2)

def dirichlet_region(x, on_boundary):
    '''Find the points on where the BCs are applied'''
    region_1 = np.logical_and(x[0] >= region_size_dict["r1"]["start"], x[0] <= (region_size_dict["r1"]["start"] + region_size_dict["r1"]["increment"]))
    region_2 = np.logical_and(x[0] >= region_size_dict["r2"]["start"], x[0] <= (region_size_dict["r2"]["start"] + region_size_dict["r2"]["increment"]))
    bottom_mask = np.isclose(x[1],0)
    
    return on_boundary and np.logical_or(region_1, region_2) and bottom_mask

def force_region(x, on_boundary):
    '''Find the points on where the load vectors are applied'''
    region_3 = np.logical_and(x[0] >= region_size_dict["r3"]["start"], x[0] <= (region_size_dict["r3"]["start"] + region_size_dict["r3"]["increment"]))
    region_4 = np.logical_and(x[0] >= region_size_dict["r4"]["start"], x[0] <= (region_size_dict["r4"]["start"] + region_size_dict["r4"]["increment"]))
    top_mask = np.isclose(x[1],h)
    
    return on_boundary and np.logical_or(region_3, region_4) and top_mask

def free_regions(x, on_boundary):
    '''Find the points on where no force or boundary conditions are applied (zero Neumann)'''
    region_1 = np.logical_and(x[0] >= region_size_dict["r1"]["start"], x[0] <= (region_size_dict["r1"]["start"] + region_size_dict["r1"]["increment"]))
    region_2 = np.logical_and(x[0] >= region_size_dict["r2"]["start"], x[0] <= (region_size_dict["r2"]["start"] + region_size_dict["r2"]["increment"]))
    region_3 = np.logical_and(x[0] >= region_size_dict["r3"]["start"], x[0] <= (region_size_dict["r3"]["start"] + region_size_dict["r3"]["increment"]))
    region_4 = np.logical_and(x[0] >= region_size_dict["r4"]["start"], x[0] <= (region_size_dict["r4"]["start"] + region_size_dict["r4"]["increment"]))

    bottom_points = np.logical_or(region_1, region_2) and np.isclose(x[1],0)
    top_points = np.logical_or(region_3, region_4) and np.isclose(x[1],h)

    #corner_points = np.logical_and(np.isclose(x[0],0),np.isclose(x[1],0)) or np.logical_and(np.isclose(x[0],l),np.isclose(x[1],0)) or np.logical_and(np.isclose(x[0],0),np.isclose(x[1],h)) or np.logical_and(np.isclose(x[0],l),np.isclose(x[1],h))

    return on_boundary and ~bottom_points and ~top_points #and ~corner_points

bc1 = dde.DirichletBC(geom, lambda _: 0.0, dirichlet_region, component=0)
bc2 = dde.DirichletBC(geom, lambda _: 0.0, dirichlet_region, component=1)
bc3 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, free_regions)
bc4 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, free_regions)
bc5 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, force_region)
bc6 = dde.OperatorBC(geom, neumann_y, force_region)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc3, bc4, bc5, bc6],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    return tf.concat([ u*1e-3, v*1e-3], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
loss_weights=[1,1,1e6,1e6,1,1,1,1]
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=loss_weights)

losshistory, train_state = model.train(epochs=10000, display_every=200)
model.compile("L-BFGS",loss_weights=loss_weights)
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

X, offset, cell_types, dol_triangles = geom.get_mesh()

displacement = model.predict(X)
sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=stress_plane_stress)
eps_xx, eps_yy, eps_xy = model.predict(X, operator=elastic_strain_2d)



combined_disp = tuple(np.vstack((np.array(displacement[:,0].tolist()),np.array(displacement[:,1].tolist()),np.zeros(displacement[:,0].shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))
combined_strain = tuple(np.vstack((np.array(eps_xx.flatten().tolist()),np.array(eps_yy.flatten().tolist()),np.array(eps_xy.flatten().tolist()))))


file_path = os.path.join(os.getcwd(), "Beam2D")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                    cell_types, pointData = { "displacement" : combined_disp,
                    "stress" : combined_stress, "strain": combined_strain})


