### Quarter disc hertzian contact problem predicting external pressure with external data
### @author: tsahin, svoelkl, dwolff, apopp
### based on the initial work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "stix"
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.geometry.geometry_utils import polar_transformation_2d
from utils.elasticity import elasticity_utils
import utils.contact_mech.contact_utils as contact_utils
from utils.contact_mech.contact_utils import zero_complimentarity_function_based_fischer_burmeister, zero_tangential_traction

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = True

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

## Adjust global definitions
# Material parameters
# We want to have e_modul=200 and nu=0.3
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters()
# Communicate parameters to dependencies
elasticity_utils.geom = geom
contact_utils.geom = geom
contact_utils.distance = radius

## Define the parameters we want to predict
ext_traction_actual = -0.5 
ext_traction_predicted= dde.Variable(-0.1, dtype=tf.float64) # start value

## Define BCs
# Applied pressure 
ext_traction = -0.5

def boundary_circle_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_circle_not_contact)

# Contact BC
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_complimentarity_function_based_fischer_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)
bcs = [bc_zero_traction_x, bc_zero_traction_y, bc_zero_fischer_burmeister, bc_zero_tangential_traction]

## Add external data for the prediction
# Load external data
fem_path = f"{str(Path(__file__).parent.parent)}/fem_reference/Hertzian_fem_fine_mesh.csv"
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
w_zero_fischer_burmeister = 1e4
w_zero_tangential_traction = 1e0
# Weights due to external data
w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy = 1e4, 1e4, 1e-1, 1e-1, 1e-1

loss_weights = [w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5,
                w_zero_traction_x, w_zero_traction_y,
                w_zero_fischer_burmeister,
                w_zero_tangential_traction,
                w_ext_u, w_ext_v, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_xy]

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent)
simulation_case = f"inverse"
adam_iterations = 2000
external_var_list = [ext_traction_predicted]
prediction_output_step_size = 1

if not restore_pretrained_model:
    start_time_train = time.time()

    parameter_file_name = f"{model_path}/identified_pressure.dat"
    variable = dde.callbacks.VariableValue(external_var_list, period=prediction_output_step_size, filename=parameter_file_name, precision=8)
    
    model.compile("adam", lr=0.001, loss_weights=loss_weights, external_trainable_variables=external_var_list)
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, callbacks=[variable], display_every=100)
    end_time_adam_train = time.time()

    model.compile("L-BFGS-B", loss_weights=loss_weights, external_trainable_variables=external_var_list)
    end_time_LBFGS_compile = time.time()
    losshistory, train_state = model.train(callbacks=[variable], display_every=200, model_save_path=f"{model_path}/{simulation_case}")

    end_time_train = time.time()
    time_train = f"Total compilation and training time: {(end_time_train - start_time_train):.3f} seconds"
    print(time_train)

    # Retrieve the total number of iterations at the end of training
    n_iterations = train_state.step

    dde.saveplot(
        losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
        loss_fname=f"{simulation_case}-{n_iterations}_loss.dat", 
        train_fname=f"{simulation_case}-{n_iterations}_train.dat", 
        test_fname=f"{simulation_case}-{n_iterations}_test.dat"
    )
else:
    n_iterations = 18416
    model_restore_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}.ckpt"
    model_loss_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}_loss.dat"
    parameter_file_name = f"{model_path}/pretrained/{simulation_case}-{str(n_iterations)}_identified_pressure.dat"
    
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
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(steps, pressure_predicted, color="b", lw=2, label=r"Predicted $\tilde{p}$")
ax1.vlines(x=adam_iterations,ymin=min(pressure_predicted), ymax=max(pressure_predicted), linestyles="--", colors="k")
ax1.axhline(y=ext_traction_actual, color="r", lw=2,label=r"Actual $p$")
ax1.annotate(r"ADAM $\Leftarrow\,\Rightarrow$ L-BGFS",    xy=[adam_iterations*1.075,max(pressure_predicted)*0.8],   ha="center", va="bottom", size=12)
ax1.set_xlabel("Iterations", size=14)
ax1.set_ylabel("Pressure", size=14)
ax1.tick_params(axis="both", labelsize=10)
ax1.legend(fontsize=14)
ax1.grid()
plt.tight_layout()
fig1.savefig(f"{model_path}/{simulation_case}-{n_iterations}_pressure_prediction_plot.png", dpi=300)
plt.show()