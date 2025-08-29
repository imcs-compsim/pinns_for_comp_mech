### Quarter cylinder hertzian contact problem enhanced by the analytical solution
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float("float64") # use double precision (needed for L-BFGS)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from deepxde import backend as bkd
from pathlib import Path
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Geom_step_to_gmsh
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_3d, problem_parameters
from utils.elasticity.elasticity_utils import apply_zero_neumann_x_mixed_formulation, apply_zero_neumann_y_mixed_formulation, apply_zero_neumann_z_mixed_formulation
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D
from utils.contact_mech import contact_utils
from utils.contact_mech.contact_utils import zero_tangential_traction_component1_3d, zero_tangential_traction_component2_3d, zero_complementarity_function_based_fischer_burmeister_3d

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = True

## Create geometry
# Get geometry from step file
path_to_step_file = str(Path(__file__).parent.parent)+f"/step_files/hertzian_quarter_cylinder.stp"
# Modifications to define a proper outer normal
curve_info = {"7":15, "9":15, 
              "14":8, "18":8, 
              "8":40, "6":25,
              "2":15, "16":15}
# Create geom object
geom_obj = Geom_step_to_gmsh(path=path_to_step_file, curve_info=curve_info)
gmsh_model = geom_obj.generateGmshModel(visualize_mesh=False)
geom = GmshGeometry3D(gmsh_model)
# Define geometric parameters
projection_plane = {"y" : -1} # projection plane formula
center = [0, 0, 0]
radius = 1
b_limit = -0.25

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
contact_utils.projection_plane = projection_plane

## Define BCs
# Applied pressure 
pressure = -0.5

def boundary_not_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius) and (x[0]<b_limit)

def boundary_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius) and (x[0]>=b_limit)

def bottom_point(x, on_boundary):
    points_at_x_0 = np.isclose(x[0],0)
    points_on_the_radius = np.isclose(np.linalg.norm(x[:2] - center[:2], axis=-1), radius)

    return on_boundary and points_on_the_radius and points_at_x_0

# Neumann BCs
bc_zero_traction_x = dde.OperatorBC(geom, apply_zero_neumann_x_mixed_formulation, boundary_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, apply_zero_neumann_y_mixed_formulation, boundary_not_contact)
bc_zero_traction_z = dde.OperatorBC(geom, apply_zero_neumann_z_mixed_formulation, boundary_not_contact)

# Contact BCs
bc_zero_tangential_traction_eta = dde.OperatorBC(geom, zero_tangential_traction_component1_3d, boundary_contact)
bc_zero_tangential_traction_xi  = dde.OperatorBC(geom, zero_tangential_traction_component2_3d, boundary_contact)
# KKT using fischer_burmeister
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_complementarity_function_based_fischer_burmeister_3d, boundary_contact)
bcs = [bc_zero_traction_x, bc_zero_traction_y, bc_zero_traction_z,
       bc_zero_tangential_traction_eta, bc_zero_tangential_traction_xi,
       bc_zero_fischer_burmeister]

## Add analytical solution to enhance the training
# Define constants
p_max = 8.36
b = 0.07611333607551958
n_test = 50
z = np.linspace(0,3*b,n_test).reshape(-1,1)
# Compute stresses
s_x = -p_max*((1+2*(z**2/b**2))/(np.sqrt(1+z**2/b**2)) - 2*np.abs(z/b))
s_y = -p_max/(np.sqrt(1+z**2/b**2))
s_z = -2*nu*p_max*(np.sqrt(1+z**2/b**2) - np.abs(z/b))
# Create arrays with external data
y_coord = np.linspace(-1,-0.7642,n_test).reshape(-1,1)
ex_data_xyz_1 = np.hstack((np.zeros_like(y_coord), y_coord, -1*np.ones_like(y_coord)))
ex_data_xyz_2 = np.hstack((np.zeros_like(y_coord), y_coord, -0.5*np.ones_like(y_coord)))
ex_data_xyz_3 = np.hstack((np.zeros_like(y_coord), y_coord, np.zeros_like(y_coord)))
ex_data_xyz = np.vstack((ex_data_xyz_1,ex_data_xyz_2,ex_data_xyz_3))
s_x = np.vstack((s_x,s_x,s_x))
s_y = np.vstack((s_y,s_y,s_y))
s_z = np.vstack((s_z,s_z,s_z))

# Define boundary conditions for experimental data
observe_sigma_xx = dde.PointSetBC(ex_data_xyz, s_x, component=3)
observe_sigma_yy = dde.PointSetBC(ex_data_xyz, s_y, component=4)
observe_sigma_zz = dde.PointSetBC(ex_data_xyz, s_z, component=5)

# Append to the list of boundary conditions
bcs_data = [observe_sigma_xx, observe_sigma_yy, observe_sigma_zz]
bcs.extend(bcs_data)

# Setup the data object
n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_3d,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution="Sobol",
    anchors=ex_data_xyz
)

def output_transform(x, y):
    '''
    Enforce the following conditions in a hard way
        Dirichlet terms
            u(x=0)=0
            w(z=0)=0
        
        Neumann terms:
            sigma_yy(y=0) = ext_traction
            sigma_xy(x=0) = sigma_xy(y=0) = 0
            sigma_yz(y=0) = sigma_yz(z=0) = sigma_yz(z=radius) = 0
            sigma_xz(x=0) = sigma_xz(z=0) = sigma_xz(z=radius) = 0
    '''
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    z_loc = x[:, 2:3]
    
    # Define surfaces
    top_surface = -y_loc
    front_surface = -z_loc
    back_surface = radius + z_loc
    right_surface = -x_loc
    
    # Define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = top_surface*right_surface
    sigma_yz_surfaces = top_surface*front_surface*back_surface
    sigma_xz_surfaces = front_surface*back_surface*right_surface
    
    return bkd.concat([u*(right_surface)/e_modul,
                      v/e_modul,
                      w*(back_surface)*(front_surface)/e_modul,
                      sigma_xx, 
                      pressure + sigma_yy*(top_surface),
                      sigma_zz,
                      sigma_xy*sigma_xy_surfaces,
                      sigma_yz*sigma_yz_surfaces,
                      sigma_xz*sigma_xz_surfaces
                      ], axis=1)

## Define the neural network
layer_size = [3] + [50] * 5 + [9] # 3 inputs: x, y and z, 5 hidden layers with 50 neurons each, 9 outputs: ux, uy, uz, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz and sigma_xz
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

## Adjust weights in the loss function
# Weights due to PDE
w_momentum_xx, w_momentum_yy, w_momentum_zz = 1e0, 1e0, 1e0
# Weights due to stress-stress coupling
w_s_xx, w_s_yy, w_s_zz, w_s_xy, w_s_yz, w_s_xz = 1e0, 1e0, 1e0, 1e0, 1e0, 1e0
# Weights due to Neumann BCs
w_zero_traction_x, w_zero_traction_y, w_zero_traction_z = 1e0, 1e0, 1e0
# Weights due to Contact BCs
w_zero_tangential_traction_component1 = 1e0
w_zero_tangential_traction_component2 = 1e0
w_zero_fischer_burmeister = 5e2
# Weights due to external data
w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_zz = 1e0, 1e0, 1e0

loss_weights = [w_momentum_xx, w_momentum_yy, w_momentum_zz, 
                w_s_xx, w_s_yy, w_s_zz, w_s_xy, w_s_yz, w_s_xz,  
                w_zero_traction_x, w_zero_traction_y, w_zero_traction_z,
                w_zero_tangential_traction_component1, w_zero_tangential_traction_component2, w_zero_fischer_burmeister,
                w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_zz]

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent)
simulation_case = f"quarter_cylinder_enhanced"
adam_iterations = 2000

if not restore_pretrained_model:
    start_time_train = time.time()

    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)
    end_time_adam_train = time.time()

    model.compile("L-BFGS", loss_weights=loss_weights)
    end_time_LBFGS_compile = time.time()
    losshistory, train_state = model.train(display_every=200, model_save_path=f"{model_path}/{simulation_case}")

    end_time_train = time.time()
    time_train = f"Total compilation and training time: {(end_time_train - start_time_train):.3f} seconds"
    print(time_train)

    # Retrieve the total number of iterations at the end of training
    n_iterations = train_state.step
    
    # Print times to output file
    with open(f"{model_path}/{simulation_case}-{n_iterations}_times.txt", "w") as text_file:
        print(f"Compilation and training times in [s]", file=text_file)
        print(f"Adam compilation:    {(end_time_adam_compile - start_time_train):6.3f}", file=text_file)
        print(f"Adam training:       {(end_time_adam_train - end_time_adam_compile):6.3f}", file=text_file)
        print(f"L-BFGS compilation:  {(end_time_LBFGS_compile - end_time_adam_train):6.3f}", file=text_file)
        print(f"L-BFGS training:     {(end_time_train - end_time_LBFGS_compile):6.3f}", file=text_file)
        print(f"Total:               {(end_time_train - start_time_train):6.3f}", file=text_file)

    # Save results
    dde.saveplot(
        losshistory, train_state, issave=True, isplot=False, output_dir=model_path, 
        loss_fname=f"{simulation_case}-{n_iterations}_loss.dat", 
        train_fname=f"{simulation_case}-{n_iterations}_train.dat", 
        test_fname=f"{simulation_case}-{n_iterations}_test.dat"
    )

else:
    n_iterations = 17000
    model_restore_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}.pt"
    model_loss_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}_loss.dat"
    
    model.compile("L-BFGS")
    model.restore(save_path=model_restore_path)
    
# Output results to VTK
solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=model_path, 
                           file_name="Hertzian_contact_3d_quarter_cylinder_enhanced", 
                           polar_transformation="cylindrical")

## Add analytical solution to get the error of the training
# Define constants
p_max = 8.36
b = 0.07611333607551958
n_compare = 100
plot_width = 5*b
# Analytical Solution
y_range_analytical = np.linspace(0,plot_width,n_compare).reshape(-1,1)
s_x = -p_max*((1+2*(y_range_analytical**2/b**2))/(np.sqrt(1+y_range_analytical**2/b**2)) - 2*np.abs(y_range_analytical/b))
s_y = -p_max/(np.sqrt(1+y_range_analytical**2/b**2))
s_z = -2*nu*p_max*(np.sqrt(1+y_range_analytical**2/b**2) - np.abs(y_range_analytical/b))

# Prediction on theses points
y_range_prediction = np.linspace(-1,-1+plot_width,n_compare).reshape(-1,1)
prediction_points = np.hstack((np.zeros_like(y_range_prediction), y_range_prediction, -0.75*np.ones_like(y_range_prediction)))

start_time_predict = time.time()
prediction = model.predict(prediction_points)
end_time_predict = time.time()

# Plot predicted and analytical solution
fig, ax = plt.subplots(figsize=(8,4.5))
ax.plot(y_range_analytical/b, s_x/-p_max, linewidth = 3, color="blue", label="$\sigma_{xx}^{analytical}$")
ax.plot(y_range_analytical/b, prediction[:,3]/-p_max, linewidth = 3, color="blue", marker="*", markersize=8, markeredgecolor="black", markevery=4, alpha=0.5, label="$\sigma_{xx}^{predicted}$")
ax.plot(y_range_analytical/b, s_y/-p_max, linewidth = 3, color="orange", label="$\sigma_{yy}^{analytical}$")
ax.plot(y_range_analytical/b, prediction[:,4]/-p_max, linewidth = 3, color="orange", marker="*", markersize=8, markeredgecolor="black", markevery=4, alpha=0.5, label="$\sigma_{yy}^{predicted}$")
ax.plot(y_range_analytical/b, s_z/-p_max, linewidth = 3, color="green", label="$\sigma_{zz}^{analytical}$")
ax.plot(y_range_analytical/b, prediction[:,5]/-p_max, linewidth = 3, color="green", marker="*", markersize=8, markeredgecolor="black", markevery=4, alpha=0.5, label="$\sigma_{zz}^{predicted}$")
ax.set_xlabel(r"Distance to contact surface", fontsize=16)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: "0" if np.isclose(v, 0) else f"{v}b"))
# ax.xaxis.set_major_locator(MultipleLocator(0.5 * b))
ax.set_ylabel(r"Ratio of stress to to $p_{max}$", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(fontsize=14)
ax.grid()
plt.tight_layout()
fig.savefig(f"{model_path}/{simulation_case}-{n_iterations}_pressure.png", dpi=300)
plt.show()