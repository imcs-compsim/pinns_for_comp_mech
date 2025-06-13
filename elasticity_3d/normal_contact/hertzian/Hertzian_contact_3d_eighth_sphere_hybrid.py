### Eighth sphere hertzian contact problem enhanced with external data
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float("float64")
import numpy as np
import pyvista as pv
from deepxde import backend as bkd
from pathlib import Path
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometry3D
from utils.geometry.gmsh_models import Eighth_sphere_hertzian
from utils.elasticity import elasticity_utils
from utils.elasticity.elasticity_utils import pde_mixed_3d, problem_parameters
from utils.elasticity.elasticity_utils import apply_zero_neumann_x_mixed_formulation, apply_zero_neumann_y_mixed_formulation, apply_zero_neumann_z_mixed_formulation
from utils.contact_mech import contact_utils
from utils.contact_mech.contact_utils import zero_tangential_traction_component1_3d, zero_tangential_traction_component2_3d, zero_complementarity_function_based_fisher_burmeister_3d
from utils.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtk3D

## Set custom Flag to either restore the model from pretrained
## or simulate yourself
restore_pretrained_model = False

## Create geometry
# Dimensions of sphere
center = [0, 0, 0]
radius = 1
# Create the eighth spere using gmsh
angle_deg = 15 # Angle of refinement area
refine_times = 5 # Refinement multiplicator in refinement area
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
start_time_meshing = time.time()
Eighth_sphere = Eighth_sphere_hertzian(radius=radius, center=center, mesh_size=0.05, angle=angle_deg, refine_times=refine_times, gmsh_options=gmsh_options)
gmsh_model = Eighth_sphere.generateGmshModel(visualize_mesh=False)
end_time_meshing = time.time()
geom = GmshGeometry3D(gmsh_model)

## Adjust global definitions
# Material parameters
# We want to have e_modul=200 and nu=0.3
lame = 115.38461538461539
shear = 76.92307692307692
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu,lame,shear,e_modul = problem_parameters()
# Projection plane for contact definition
projection_plane = {"y" : -1}
# Communicate parameters to dependencies
elasticity_utils.geom = geom
contact_utils.geom = geom
contact_utils.projection_plane = projection_plane

## Define BCs
# Applied pressure 
pressure = -0.2
# Compute radius for refinement area
b_limit = Eighth_sphere.computeblimit()
# Spherical boundary which is not in contact
def boundary_not_contact(x, on_boundary):
    return on_boundary & np.isclose(np.linalg.norm(x - center, axis=-1), radius) & (np.linalg.norm(x-[center[0],center[1]-radius,center[2]], axis=-1)>b_limit)
# Spherical boundary which is in contact
def boundary_contact(x, on_boundary):
    return on_boundary & np.isclose(np.linalg.norm(x - center, axis=-1), radius) & (np.linalg.norm(x-[center[0],center[1]-radius,center[2]], axis=-1)<=b_limit)
# # Cut surface with normal along x-axis
# def boundary_cut_x(x, on_boundary):
#     return on_boundary & np.isclose(x[0],center[0])
# # Cut surface with normal along z-axis
# def boundary_cut_z(x, on_boundary):
#     return on_boundary & np.isclose(x[2],center[2])

## Apply BCs
# Neumann BCs on non-contact zones of the radial surface of the sphere
bc_zero_traction_x = dde.OperatorBC(geom, apply_zero_neumann_x_mixed_formulation, boundary_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, apply_zero_neumann_y_mixed_formulation, boundary_not_contact)
bc_zero_traction_z = dde.OperatorBC(geom, apply_zero_neumann_z_mixed_formulation, boundary_not_contact)
# Neumann BCs (sliding) on cut sections of the sphere
# bc_sliding_x = dde.OperatorBC(geom, apply_zero_neumann_x_mixed_formulation, boundary_cut_x)
# bc_sliding_z = dde.OperatorBC(geom, apply_zero_neumann_z_mixed_formulation, boundary_cut_z)
# Contact BCs
# Zero tangential tractions in contact area
bc_zero_tangential_traction_eta = dde.OperatorBC(geom, zero_tangential_traction_component1_3d, boundary_contact)
bc_zero_tangential_traction_xi  = dde.OperatorBC(geom, zero_tangential_traction_component2_3d, boundary_contact)
# KKT using fisher_burmeister
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_complementarity_function_based_fisher_burmeister_3d, boundary_contact)
bcs = [bc_zero_traction_x, bc_zero_traction_y, bc_zero_traction_z,
    #    bc_sliding_x, bc_sliding_z,
       bc_zero_tangential_traction_eta, bc_zero_tangential_traction_xi,
       bc_zero_fischer_burmeister]

## Add external data to enhance the training
# Load external data
fem_results = pv.read(str(Path(__file__).parent.parent.parent)+f"/trained_models/hertzian/hertzian_sphere_3d/fem_results_eighth_sphere_linear.vtu")

# Coordinates, diplacements and stresses in fem
prediction_points = fem_results.points
fem_displacement = fem_results.point_data["displacement"]
fem_stresses = fem_results.point_data["nodal_cauchy_stresses_xyz"]

# Shuffle fem_results so that we do not slice a specific part of mesh
np.random.seed(17) # We will always use the same points #reproducibility
random_points = np.random.permutation(prediction_points)
random_fem_displacement = np.random.permutation(fem_displacement)
random_fem_stresses = np.random.permutation(fem_stresses)

# Define condition to find boundary points
on_radius = np.isclose(np.linalg.norm(random_points - center, axis=-1), radius)
on_cut_x = np.isclose(random_points[:,0],center[0])
on_cut_z = np.isclose(random_points[:,2],center[2])
on_boundary = np.logical_or(np.logical_or(on_radius,on_cut_x),on_cut_z)

# Only use 100 points from boundary and 100 points from domain
n_boundary = 100
n_domain = 100

# Create arrays with external data
ex_data_xy = np.vstack((prediction_points[on_boundary][:n_boundary],prediction_points[~on_boundary][:n_domain]))
ex_data_disp = np.vstack((fem_displacement[on_boundary][:n_boundary],fem_displacement[~on_boundary][:n_domain]))
ex_data_stress = np.vstack((fem_stresses[on_boundary][:n_boundary],fem_stresses[~on_boundary][:n_domain]))

# Define boundary conditions for experimental data
observe_u = dde.PointSetBC(ex_data_xy, ex_data_disp[:,0:1], component=0)
observe_v = dde.PointSetBC(ex_data_xy, ex_data_disp[:,1:2], component=1)
observe_w = dde.PointSetBC(ex_data_xy, ex_data_disp[:,2:3], component=2)
observe_sigma_xx = dde.PointSetBC(ex_data_xy, ex_data_stress[:,0:1], component=3)
observe_sigma_yy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,1:2], component=4)
observe_sigma_zz = dde.PointSetBC(ex_data_xy, ex_data_stress[:,2:3], component=5)
observe_sigma_xy = dde.PointSetBC(ex_data_xy, ex_data_stress[:,3:4], component=6)
observe_sigma_yz = dde.PointSetBC(ex_data_xy, ex_data_stress[:,4:5], component=7)
observe_sigma_xz = dde.PointSetBC(ex_data_xy, ex_data_stress[:,5:6], component=8)

# Append to the list of boundary conditions
bcs_data = [observe_u, observe_v, observe_w,
            observe_sigma_xx, observe_sigma_yy, observe_sigma_zz,
            observe_sigma_xy, observe_sigma_yz, observe_sigma_xz]
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
    train_distribution = "Sobol",
    anchors=ex_data_xy
)

def output_transform(x, y):
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
    
    # define surfaces
    cut_x_surface = -x_loc
    top_surface   = -y_loc
    cut_z_surface =  z_loc
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = cut_x_surface * top_surface
    sigma_yz_surfaces = top_surface   * cut_z_surface
    sigma_xz_surfaces = cut_z_surface * cut_x_surface
    
    return bkd.concat([u/e_modul*cut_x_surface, #displacement in x direction is 0 at x=0
                       v/e_modul,
                       w/e_modul*cut_z_surface, #displacement in z direction is 0 at z=0
                       sigma_xx, 
                       pressure + sigma_yy*top_surface,
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
# w_sliding_x, w_sliding_z = 1e0, 1e0
# Weights due to Contact BCs
w_zero_tangential_traction_component1 = 1e0
w_zero_tangential_traction_component2 = 1e0
w_zero_fisher_burmeister = 5e2
# Weights due to external data
w_ext_u, w_ext_v, w_ext_w, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_zz, w_ext_sigma_xy, w_ext_sigma_yz, w_ext_sigma_xz = 1e4, 1e4, 1e4, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1

loss_weights = [w_momentum_xx, w_momentum_yy, w_momentum_zz, 
                w_s_xx, w_s_yy, w_s_zz, w_s_xy, w_s_yz, w_s_xz,# w_sliding_x, w_sliding_z, 
                w_zero_traction_x, w_zero_traction_y, w_zero_traction_z,
                w_zero_tangential_traction_component1, w_zero_tangential_traction_component2, w_zero_fisher_burmeister,
                w_ext_u, w_ext_v, w_ext_w, w_ext_sigma_xx, w_ext_sigma_yy, w_ext_sigma_zz, w_ext_sigma_xy, w_ext_sigma_yz, w_ext_sigma_xz]

## Train the model or use a pre-trained model
model = dde.Model(data, net)
model_path = str(Path(__file__).parent.parent.parent)+f"/trained_models/hertzian/hertzian_sphere_3d"
simulation_case = f"eighth_sphere_linear_hybrid"
adam_iterations = 2000

if not restore_pretrained_model:
    start_time_train = time.time()

    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)
    end_time_adam_train = time.time()

    # dde.optimizers.config.set_LBFGS_options(maxiter=1000)
    model.compile("L-BFGS", loss_weights=loss_weights)
    end_time_LBFGS_compile = time.time()
    losshistory, train_state = model.train(display_every=200, model_save_path=f"{model_path}/{simulation_case}")

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
    n_iterations = 9249
    model_restore_path = f"{model_path}/{simulation_case}-{n_iterations}.ckpt"
    model_loss_path = f"{model_path}/{simulation_case}-{n_iterations}_loss.dat"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

# Compare with results from 4C
# Predict solution at FEM nodes
start_time_predict = time.time()
prediction_results = model.predict(prediction_points)
end_time_predict = time.time()

# Compute differences
prediction_displacement = prediction_results[:,0:3]
error_displacement = prediction_displacement - fem_results.point_data["displacement"]
prediction_stress = prediction_results[:,3:9]
error_stress = prediction_stress - fem_results.point_data["nodal_cauchy_stresses_xyz"]

# Save and return them in vtu file
fem_results.point_data["displacement_prediction"] = prediction_displacement
fem_results.point_data["prediction_stress"] = prediction_stress
fem_results.point_data["error_displacement"] = error_displacement
fem_results.point_data["error_stress"] = error_stress
fem_results.save(str(Path(__file__).parent.parent.parent.parent)+f"/3D_hertzian_contact_eighth_sphere_hybrid_predictions.vtu", binary=True)

#########################################################################################################################################
#### POST-PROCESSING #####
#########################################################################################################################################

# Print times to output file
if not restore_pretrained_model:
    with open(f"{model_path}/{simulation_case}-{n_iterations}_times.txt", "w") as text_file:
        print(f"Compilation and training times in [s]", file=text_file)
        print(f"Meshing took:        {(end_time_meshing - start_time_meshing):6.3f}", file=text_file)
        print(f"Adam compilation:    {(end_time_adam_compile - start_time_train):6.3f}", file=text_file)
        print(f"Adam training:       {(end_time_adam_train - end_time_adam_compile):6.3f}", file=text_file)
        print(f"L-BFGS compilation:  {(end_time_LBFGS_compile - end_time_adam_train):6.3f}", file=text_file)
        print(f"L-BFGS training:     {(end_time_train - end_time_LBFGS_compile):6.3f}", file=text_file)
        print(f"Total:               {(end_time_train - start_time_train):6.3f}", file=text_file)
        print(f"Prediction:          {(end_time_predict - start_time_predict):6.3f}", file=text_file)

solutionFieldOnMeshToVtk3D(geom, 
                           model, 
                           save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                           file_name="3D_hertzian_contact_eighth_sphere_hybrid", 
                           polar_transformation="spherical")