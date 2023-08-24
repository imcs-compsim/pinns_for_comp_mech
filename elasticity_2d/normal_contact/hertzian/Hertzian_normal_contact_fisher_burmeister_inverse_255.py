import deepxde as dde
import numpy as np
import pandas as pd
import os
from pathlib import Path
from deepxde.backend import tf
import matplotlib.pyplot as plt
import seaborn as sns


from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.geometry.geometry_utils import calculate_boundary_normals

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, calculate_traction_mixed_formulation
from utils.elasticity.elasticity_utils import zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.contact_mech.contact_utils import zero_tangential_traction
from utils.elasticity import elasticity_utils
from utils.contact_mech import contact_utils

'''
Solves an inverse problem to identify external pressure in Hertzian contact example.

@author: tsahin
'''
###########################################
#dde.config.real.set_float64()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
radius = 1
center = [0,0]

Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=255, refine_times=100, gmsh_options=gmsh_options)

gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

# Material properties
e_actual = 199.99999999999997
nu_actual = 0.3
e_predicted = e_actual
nu_predicted = nu_actual

# change global variables in elasticity_utils
elasticity_utils.geom = geom
elasticity_utils.lame = e_predicted*nu_predicted/((1+nu_predicted)*(1-2*nu_predicted))
elasticity_utils.shear = e_predicted/(2*(1+nu_predicted))

# The applied pressure 
ext_traction_actual = -0.5 
ext_traction_predicted= dde.Variable(-0.1)

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom
contact_utils.geom = geom

# how far above the block from ground
distance = 0

# assign local parameters from the current file in contact_utils and elasticity_utils
contact_utils.distance = distance

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction
    '''
    # calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + radius + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def zero_fischer_burmeister(x,y,X):
    '''
    Enforces KKT conditions using Fisher-Burmeister equation
    '''
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction_mixed_formulation(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_circle_not_contact)

# Contact BC
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_fischer_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)

bcs = [bc_zero_traction_x,bc_zero_traction_y,bc_zero_tangential_traction,bc_zero_fischer_burmeister]

add_external_data = True

if add_external_data:
    # load external data
    fem_path = str(Path(__file__).parent.parent.parent)+"/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
    df = pd.read_csv(fem_path)
    fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
    fem_results = fem_results.to_numpy()

    # shuffle fem_results so that we do not slice a specific part of mesh
    np.random.seed(12) # We will always use the same points #reproducibility
    np.random.shuffle(fem_results)

    # coordinates, diplacements and stresses in fem 
    node_coords_xy = fem_results[:,0:2]
    displacement_fem = fem_results[:,2:4]
    stress_fem = fem_results[:,4:7]

    # define condition to find boundary points 
    on_radius = np.isclose(np.linalg.norm(node_coords_xy - center, axis=-1), radius)
    on_right = np.isclose(node_coords_xy[:,0], center[0])
    on_top = np.isclose(node_coords_xy[:,1], center[1])
    on_boundary_ = np.logical_or(np.logical_or(on_radius,on_right),on_top)
    
    # we will take only 100 points from boundary and 100 points from domain
    n_boundary = 200
    n_domain = 200

    # 
    ex_data_xy = np.vstack((node_coords_xy[on_boundary_][:n_boundary],node_coords_xy[~on_boundary_][:n_domain]))
    ex_data_disp = np.vstack((displacement_fem[on_boundary_][:n_boundary],displacement_fem[~on_boundary_][:n_domain]))
    ex_data_stress = np.vstack((stress_fem[on_boundary_][:n_boundary],stress_fem[~on_boundary_][:n_domain]))

    # define boundary conditions for experimental data
    observe_u = dde.PointSetBC(ex_data_xy, ex_data_disp[:,0:1], component=0)
    observe_v = dde.PointSetBC(ex_data_xy, ex_data_disp[:,1:2], component=1)
    observe_sigma_xx = dde.PointSetBC(ex_data_xy, ex_data_stress[:,0:1], component=2)
    observe_sigma_yy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,1:2], component=3)
    observe_sigma_xy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,2:3], component=4)
    
    bcs_data = [observe_u, observe_v, observe_sigma_xx, observe_sigma_yy, observe_sigma_xy]
    
    bcs.extend(bcs_data)


n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
    anchors=(ex_data_xy if add_external_data else None)
)

def output_transform(x, y):
    '''
    Hard BCs:
        Dirichlet terms
            u(x=0)=0
        
        Neumann terms:
            sigma_yy(y=0) = ext_traction
            sigma_xy(x=0) = 0
            sigma_xy(y=0) = 0
    
    General formulation to enforce BC hardly:
        N'(x) = g(x) + l(x)*N(x)
    
        where N'(x) is network output before transformation, N(x) is network output after transformation, g(x) Non-homogenous part of the BC and 
            if x is on the boundary
                l(x) = 0 
            else
                l(x) < 0
    
    For instance sigma_yy(y=0) = -ext_traction
        N'(x) = N(x) = sigma_yy
        g(x) = ext_traction
        l(x) = -y
    so
        u' = g(x) + l(x)*N(x)
        sigma_yy = ext_traction + -y*sigma_yy
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    #return tf.concat([u*(-x_loc), ext_dips + v*(-y_loc), sigma_xx, sigma_yy, sigma_xy*(x_loc)*(y_loc)], axis=1)
    return tf.concat([u*(-x_loc)/e_predicted, v/e_predicted, sigma_xx, ext_traction_predicted + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

# 2 inputs: x and y, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

# weights due to PDE
w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5 = 1e0,1e0,1e0,1e0,1e0
# weights due to Neumann BC
w_zero_traction_x, w_zero_traction_y = 1e0,1e0
# weights due to Contact BC
w_zero_tangential_traction = 1e0
w_zero_fischer_burmeister = 1e3
# weights due to external data
w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy = 1e4,1e4,1e-1,1e-1,1e-1

loss_weights = [w_pde_1,w_pde_2,w_pde_3,w_pde_4,w_pde_5,w_zero_traction_x,w_zero_traction_y,w_zero_tangential_traction,w_zero_fischer_burmeister]

if add_external_data:
    loss_weights_data = [w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy]
    loss_weights.extend(loss_weights_data)

model = dde.Model(data, net)

external_var_list = []

if not isinstance(e_predicted, float):
    external_var_list.append(e_predicted)
if not isinstance(nu_predicted, float):
    external_var_list.append(nu_predicted)
if not isinstance(ext_traction_predicted, float):
    external_var_list.append(ext_traction_predicted)

parameter_file_name = str(Path(__file__).parent)+"/identified_pressure.dat"

variable = dde.callbacks.VariableValue(external_var_list, period=10, filename=parameter_file_name)

n_iter_adam = 2000
model.compile("adam", lr=0.001, external_trainable_variables=external_var_list)
losshistory, train_state = model.train(epochs=n_iter_adam, callbacks=[variable], display_every=100)

model.compile("L-BFGS-B", external_trainable_variables=external_var_list)
losshistory, train_state = model.train(callbacks=[variable], display_every=100)