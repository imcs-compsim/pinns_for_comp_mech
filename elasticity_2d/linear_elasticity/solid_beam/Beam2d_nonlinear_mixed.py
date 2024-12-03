import deepxde as dde
import numpy as np
import os
from deepxde.backend import tf
from pyevtk.hl import unstructuredGridToVTK
from deepxde import backend as bkd
import pandas as pd
from pathlib import Path
from matplotlib import tri
import pyvista as pv

from utils.elasticity.elasticity_utils import problem_parameters, first_piola_stress_tensor, momentum_mixed, problem_parameters, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation, cauchy_stress
from utils.geometry.geometry_utils import calculate_boundary_normals
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import Block_2D
from utils.elasticity import elasticity_utils


'''
The correct order for the normals --> 1 2 1 1

Reference solution:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.504.4507&rep=rep1&type=pdf

@author: tsahin
'''

height = 1
width = 5
applied_displacement = -0.1
elasticity_utils.model_complexity = "nonlinear"     #with "linear" --> linear strain definition, everyhing else i.e. "hueicii" nonlinear
elasticity_utils.model_type = "plane_strain"        #with "plane_strain" --> plane strain, everyhing else i.e. "hueicii" plane stress
model_type = elasticity_utils.model_type 
model_complexity = elasticity_utils.model_complexity

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
block_2d = Block_2D(coord_left_corner=[-width/2,-height/2], coord_right_corner=[width/2,height/2], mesh_size=0.095, gmsh_options=gmsh_options) #0.095

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

l = block_2d.coord_right_corner[0] -block_2d.coord_left_corner[0] #5
h = block_2d.coord_right_corner[1] -block_2d.coord_left_corner[1]

# change global variables in elasticity_utils
e_1 = 10
nu_1 = 0.3
# elasticity_utils.lame = e_1*nu_1/((1+nu_1)*(1-2*nu_1))
# elasticity_utils.shear = e_1/(2*(1+nu_1))
elasticity_utils.lame = 75/13
elasticity_utils.shear = 50/13
# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

nu, lame, shear, e_modul = problem_parameters()

def top_bottom(x, on_boundary):
    not_included_points = np.logical_or(np.isclose(x[0],width/2), np.isclose(x[0],-width/2))
    points_top = np.logical_and(np.isclose(x[1],height/2),~not_included_points)
    points_bottom = np.logical_and(np.isclose(x[1],-height/2),~not_included_points)
    
    return on_boundary and np.logical_or(points_top, points_bottom)

def top_bottom_right(x, on_boundary):
    not_included_points = np.logical_or(np.isclose(x[0],width/2), np.isclose(x[0],-width/2))
    points_top = np.logical_and(np.isclose(x[1],height/2),~not_included_points)
    points_bottom = np.logical_and(np.isclose(x[1],-height/2),~not_included_points)
    points_right = np.isclose(x[0],width/2)
    
    return on_boundary and np.logical_or(np.logical_or(points_top, points_bottom), points_right)

def left(x, on_boundary):
    return on_boundary and np.isclose(x[0],-width/2)

def right(x, on_boundary):
    return on_boundary and np.isclose(x[0],width/2)


bc1 = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, top_bottom_right)
bc2 = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, top_bottom)
bc3 = dde.DirichletBC(geom, lambda _: applied_displacement, right, component=1)
bc4 = dde.DirichletBC(geom, lambda _: 0, left, component=0)
bc5 = dde.DirichletBC(geom, lambda _: 0, left, component=1)


n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_mixed,
    [bc1, bc2, bc3, bc4, bc5],
    num_domain = n_dummy,
    num_boundary = n_dummy,
    num_test = None,
    train_distribution = "Sobol",
)

def output_transform(x, y):
    u = y[:, 0:1]       #x-displacement
    v = y[:, 1:2]       #y-displacement
    x_loc = x[:, 0:1] 
    print(x_loc)                  
    y_loc = x[:, 1:2]
    left_side = (width/2+x_loc)
    right_side = (width/2-x_loc)
    return bkd.concat([(u)*width, (v)/e_modul], axis=1)
    #return bkd.concat([(u*left_side/(e_modul**2*width)), (v*left_side/(e_modul))], axis=1)
    # return bkd.concat([(u*left_side)/e_modul, (v*right_side*left_side+left_side*-1.5/width)/e_modul], axis=1)                                  #Hard enforcement of DBC on the right

# in case hard Dirichlet is desired (no scaling!! so it must be tested)
# def output_transform(x, y):
#     x_loc = x[:,0:1]
#     y_loc = x[:,1:2]
#     u_x_analy = y[:,0:1]*shear_y*y_loc/(6*e_modul*Inertia)*((6*l-3*x_loc)*x_loc + (2+nu)*(y_loc**2-h**2/4))
#     u_y_analy = -y[:,1:2]*shear_y/(6*e_modul*Inertia)*(3*nu*y_loc**2*(l-x_loc) + (4+5*nu)*h**2*x_loc/4 + (3*l-x_loc)*x_loc**2)
#     return tf.concat([ u_x_analy, u_y_analy], axis=1)

# two inputs x and y, output is ux, uy, Txx, Tyy, Txy, Tyx
layer_size = [2] + [50] * 3 + [6]
activation = "swish"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
# net.apply_output_transform(output_transform)
loss_weights=[1,1,1,1,1]

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=None)
losshistory, train_state = model.train(epochs=6000, display_every=200)

model.compile("L-BFGS",loss_weights=None)
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

file_path =  f"/home_student/kappen/Comparison_FE_to_PINN_in_paraview/ba-kappen-reference-results-main/bending_beam/{model_complexity}/bending_beam_{model_complexity}_E=10.0_disp={applied_displacement:.1f}/bending_beam_{model_complexity}_E=10.0_disp={applied_displacement:.1f}-structure.pvd"

# Convert the Path object to a string
reader = pv.get_reader(file_path)

reader.set_active_time_point(-1)
data = reader.read()[0]

X = data.points

# Predict the outputs using the model
predictions = model.predict(X[:, 0:2])  # Input the spatial points (x, y)

# Extract the displacements
u_x = predictions[:, 0]  # First column corresponds to u_x
u_y = predictions[:, 1]  # Second column corresponds to u_y

# Extract the Cauchy stress components
T_xx = predictions[:, 2]  # Third column corresponds to T_xx
T_yy = predictions[:, 3]  # Fourth column corresponds to T_yy
T_xy = predictions[:, 4]  # Fifth column corresponds to T_xy
T_yx = T_xy  # If you assume the stress tensor is symmetric

# T_xx = model.predict(X[:,2])
# T_yy = model.predict(X[:,3])
# T_xy = model.predict(X[:,4])
# T_yx = model.predict(X[:,4])

displacement = model.predict(X[:,0:2])
#T_xx, T_yy, T_xy, T_yx = model.predict(X[:,0:2], operator=cauchy_stress)
P_xx, P_yy, P_xy, P_yx = model.predict(X[:,0:2], operator=first_piola_stress_tensor)

first_piola = np.column_stack((P_xx, P_yy, P_xy))
cauchy = np.column_stack((T_xx, T_yy, T_xy))

displacement_extended = np.hstack((displacement, np.zeros_like(displacement[:,0:1])))

data.point_data['pred_first_piola'] = first_piola
data.point_data['pred_displacement'] = displacement_extended
data.point_data['pred_stress'] = cauchy

disp_fem = data.point_data['displacement']
stress_fem = data.point_data['nodal_cauchy_stresses_xyz']

error_disp = abs((disp_fem - displacement[:, 0:2]))
data.point_data['pointwise_displacement_error'] = error_disp
# select xx, yy, and xy component (1st, 2nd and 4th column)
columns = [0,1,3]
error_stress = abs((stress_fem[:, columns] - cauchy))
data.point_data['pointwise_cauchystress_error'] = error_stress
#data.point_data['pointwise_cauchystress_error'].column_names

data.save(f"Beam2D_mixed_{model_complexity}_u_{applied_displacement:.1f}_{activation}_{model_type}.vtu")

print("NOTE THAT 'Warp By Vector' DOES NOT WORK HERE AS THE Z-DIMENSION VALUES ARE ILL-DEFINED.")
print("USE CALCULATION WITH 'displacement_X*iHat + displacement_Y*jHat + 0*kHat' AND THEN APPLY 'Warp By Vector'.")