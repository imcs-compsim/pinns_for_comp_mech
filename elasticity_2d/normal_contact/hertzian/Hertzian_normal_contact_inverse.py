### Quarter disc hertzian contact problem finding external pressure with external data
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float('float64')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, stress_to_traction_2d
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.elasticity import elasticity_utils

## Create geometry
# Dimensions of disk
radius = 1
center = [0,0]
# Create the quarter disk using gmsh
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=265, refine_times=1, gmsh_options=gmsh_options)
gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)
# Modifications to define a proper outer normal
revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

## Adjust material parameters
# We want to have e_modul=200 and nu=0.3
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters()

## Define the parameters we want to predict
ext_traction_actual = -0.5 
ext_traction_predicted= dde.Variable(-0.1, dtype=tf.float64)

## Preliminary calculations for contact conditions
elasticity_utils.geom = geom
distance = 0

def calculate_gap_in_normal_direction(x,y,X):
    '''
    Calculates the gap in normal direction
    '''
    # Calculate the gap in y direction    
    gap_y = x[:,1:2] + y[:,1:2] + radius + distance

    # calculate the boundary normals
    normals, cond = calculate_boundary_normals(X,geom)

    # Here is the idea to calculate gap_n:
    # gap_n/|n| = gap_y/|ny| --> since n is unit vector |n|=1
    gap_n = tf.math.divide_no_nan(gap_y[cond],tf.math.abs(normals[:,1:2]))
    
    return gap_n

def calculate_traction(x, y, X):
    '''
    Calculates x component of any traction vector using by Cauchy stress tensor
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

## Enforce Karush-Kuhn-Tucker conditions for frictionless contact
#      gn >= 0
#      Pn <= 0
# gn * Pn  = 0
# using nonlinear complementarity problem function Fisher-Burmeister
# f(a,b) = a + b - sqrt(a^2 + b^2) (zero_fisher_burmeister)
# where a = gn, b = -Pn
#       Tt = 0 (zero_tangential_traction)
def zero_fisher_burmeister(x,y,X):
    '''
    Enforces KKT conditions using Fisher-Burmeister equation
    '''
    # ref https://www.math.uwaterloo.ca/~ltuncel/publications/corr2007-17.pdf
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)
    gn = calculate_gap_in_normal_direction(x, y, X)
    
    a = gn
    b = -Pn
    
    return a + b - tf.sqrt(tf.maximum(a**2+b**2, 1e-9))

def zero_tangential_traction(x,y,X):
    '''
    Enforces tangential part of contact traction (Tt) to be zero.
    Tt = 0, Frictionless contact.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Tt

## Define BCs
# Applied pressure 
ext_traction = -0.5

### maybe replace with zero_neumann_*_mixed_formulation ###remove
def zero_neumann_x(x,y,X):
    '''
    Enforces x component of zero Neumann BC to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Tx

def zero_neumann_y(x,y,X):
    '''
    Enforces y component of zero Neumann BC to be zero.
    '''
    Tx, Ty, Pn, Tt = calculate_traction(x, y, X)

    return Ty

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y, boundary_circle_not_contact)

# Contact BC
bc_zero_fisher_burmeister = dde.OperatorBC(geom, zero_fisher_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)
bcs = [bc_zero_traction_x,bc_zero_traction_y,bc_zero_fisher_burmeister,bc_zero_tangential_traction]

## Add external data for the prediction
# Load external data
fem_path = str(Path(__file__).parent.parent.parent)+"/Hertzian_fem/Hertzian_fem_fine_mesh.csv"
df = pd.read_csv(fem_path)
fem_results = df[["Points_0","Points_1","displacement_0","displacement_1","nodal_cauchy_stresses_xyz_0","nodal_cauchy_stresses_xyz_1","nodal_cauchy_stresses_xyz_3"]]
fem_results = fem_results.to_numpy()

# Shuffle fem_results so that we do not slice a specific part of mesh
np.random.seed(12) # We will always use the same points #reproducibility
np.random.shuffle(fem_results)

# Coordinates, diplacements and stresses in fem 
node_coords_xy = fem_results[:,0:2]
displacement_fem = fem_results[:,2:4]
stress_fem = fem_results[:,4:7]

# Define condition to find boundary points 
on_radius = np.isclose(np.linalg.norm(node_coords_xy - center, axis=-1), radius)
on_right = np.isclose(node_coords_xy[:,0], center[0])
on_top = np.isclose(node_coords_xy[:,1], center[1])
on_boundary = np.logical_or(np.logical_or(on_radius,on_right),on_top)

# Only use 100 points from boundary and 100 points from domain
n_boundary = 100
n_domain = 100

# Create arrays with external data
ex_data_xy = np.vstack((node_coords_xy[on_boundary][:n_boundary],node_coords_xy[~on_boundary][:n_domain]))
ex_data_disp = np.vstack((displacement_fem[on_boundary][:n_boundary],displacement_fem[~on_boundary][:n_domain]))
ex_data_stress = np.vstack((stress_fem[on_boundary][:n_boundary],stress_fem[~on_boundary][:n_domain]))

# Define boundary conditions for experimental data
observe_u = dde.PointSetBC(ex_data_xy, ex_data_disp[:,0:1], component=0)
observe_v = dde.PointSetBC(ex_data_xy, ex_data_disp[:,1:2], component=1)
observe_sigma_xx = dde.PointSetBC(ex_data_xy, ex_data_stress[:,0:1], component=2)
observe_sigma_yy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,1:2], component=3)
observe_sigma_xy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,2:3], component=4)

# Append to the list of boundary conditions
bcs_data = [observe_u, observe_v, observe_sigma_xx, observe_sigma_yy, observe_sigma_xy]
bcs.extend(bcs_data)

# Setup the data object
n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution="Sobol",
    anchors=ex_data_xy
)

## Use output transformation
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
    
    return tf.concat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction_predicted + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

## Define the neural network
layer_size = [2] + [50] * 5 + [5] # 2 inputs: x and y, 5 hidden layers with 50 neurons each, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

## Adjust weights in the loss function
# Weights due to PDE
w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5 = 1e0, 1e0, 1e0, 1e0, 1e0
# Weights due to Neumann BC
w_zero_traction_x, w_zero_traction_y = 1e0, 1e0
# Weights due to Contact BC
w_zero_fisher_burmeister = 1e4
w_zero_tangential_traction = 1e0
# Weights due to external data
w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy = 1e4, 1e4, 1e-1, 1e-1, 1e-1

loss_weights = [w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5,
                w_zero_traction_x, w_zero_traction_y,
                w_zero_fisher_burmeister,
                w_zero_tangential_traction,
                w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy]


## Train the model or use a pre-trained model
model = dde.Model(data, net)
restore_model = False
model_path = str(Path(__file__).parent.parent.parent)+f"/trained_models/hertzian_normal_contact/inverse/"
parameter_file_name = model_path+"identified_pressure.dat"
external_var_list = [ext_traction_predicted]
prediction_output_step_size = 1
variable = dde.callbacks.VariableValue(external_var_list, period=prediction_output_step_size, filename=parameter_file_name, precision=8)
simulation_case = f"inverse_hertzian"
adam_iterations = 2000

if not restore_model:
    
    model.compile("adam", lr=0.001, loss_weights=loss_weights, external_trainable_variables=external_var_list)
    losshistory, train_state = model.train(iterations=adam_iterations, callbacks=[variable], display_every=100)

    model.compile("L-BFGS-B", loss_weights=loss_weights, external_trainable_variables=external_var_list)
    losshistory, train_state = model.train(callbacks=[variable], display_every=200)
   
else:
    n_iterations = 10827
    model_restore_path = model_path + simulation_case + "-"+ str(n_iterations) + ".ckpt"
    model_loss_path = model_path + simulation_case + "-"+ str(n_iterations) + "_loss.dat"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

## Visualize the predicted pressure
steps = []
pressure_predicted = []
with open(parameter_file_name) as f:
    for line in f:
        a, b = line.split()
        steps.append(float(a))
        pressure_predicted.append(float(b.strip("[]")))
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(steps, pressure_predicted, color='b', lw=2, label="Predicted pressure")
ax1.axhline(y=ext_traction_actual, color='r', lw=2,label="Actual pressure")
ax1.annotate(r"ADAM $\ \Leftarrow$ ",    xy=[adam_iterations/2,max(pressure_predicted)],   ha='center', va='bottom', size=15)
ax1.annotate(r"$\Rightarrow \ $ L-BGFS", xy=[adam_iterations*3/2,max(pressure_predicted)], ha='center', va='bottom', size=15)
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("Pressure", size=17)
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig("inverse_hertzian_pressure_prediction_plot.png", dpi=300)
plt.show()