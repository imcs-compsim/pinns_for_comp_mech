### Quarter disc hertzian contact problem enhanced with external data
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
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

## Add external data to enhance the training
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
    
    return tf.concat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

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
simulation_case = f"hybrid"
adam_iterations = 2000

if not restore_pretrained_model:
    start_time_train = time.time()

    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    end_time_adam_compile = time.time()
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)
    end_time_adam_train = time.time()

    model.compile("L-BFGS-B", loss_weights=loss_weights)
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

    def calculate_loss():
        losses = np.hstack(
                (
                    np.array(losshistory.steps)[:, None],
                    np.array(losshistory.loss_train),
                )
            )
        steps = losses[:,0]
        pde_loss = losses[:,1:5].sum(axis=1)
        neumann_loss = losses[:,6:9].sum(axis=1)
        
        return steps, pde_loss, neumann_loss
else:
    n_iterations = 18108
    model_restore_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}.ckpt"
    model_loss_path = f"{model_path}/pretrained/{simulation_case}-{n_iterations}_loss.dat"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)
    def calculate_loss():
        losses = np.loadtxt(model_loss_path),
        steps = losses[0][:,0]
        pde_loss = losses[0][:,1:5].sum(axis=1)
        neumann_loss = losses[0][:,6:9].sum(axis=1)
        
        return steps, pde_loss, neumann_loss

## Visualize the loss
steps, pde_loss, neumann_loss = calculate_loss()
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(steps, pde_loss/5, color="b", lw=2, label="PDE")
ax1.plot(steps, neumann_loss/4, color="r", lw=2,label="NBC")
ax1.vlines(x=adam_iterations,ymin=0, ymax=1, linestyles="--", colors="k")
ax1.annotate(r"ADAM $\ \Leftarrow$ ",    xy=[adam_iterations/2,0.5],   ha="center", va="top", size=15)
ax1.annotate(r"$\Rightarrow \ $ L-BGFS", xy=[adam_iterations*3/2,0.5], ha="center", va="top", size=15)
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("MSE", size=17)
ax1.set_yscale("log")
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig(f"{model_path}/{simulation_case}-{n_iterations}_loss_plot.png", dpi=300)

## Create a comparison with FEM results
# Load the FEM results
node_coords_xy = fem_results[:,0:2]
displacement_fem = fem_results[:,2:4]
stress_fem = fem_results[:,4:7]

X = node_coords_xy
x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)
triangles = tri.Triangulation(x, y)

# Predictions at FEM nodes
start_time_predict = time.time()
output = model.predict(X)
end_time_predict = time.time()
time_predict = f"Prediction time: {(end_time_predict - start_time_predict):.3f} seconds"
print(time_predict)

# Print times to output file
if not restore_pretrained_model:
    with open(f"{model_path}/{simulation_case}-{n_iterations}_times.txt", "w") as text_file:
        print(f"Compilation and training times in [s]", file=text_file)
        print(f"Adam compilation:    {(end_time_adam_compile - start_time_train):6.3f}", file=text_file)
        print(f"Adam training:       {(end_time_adam_train - end_time_adam_compile):6.3f}", file=text_file)
        print(f"L-BFGS compilation:  {(end_time_LBFGS_compile - end_time_adam_train):6.3f}", file=text_file)
        print(f"L-BFGS training:     {(end_time_train - end_time_LBFGS_compile):6.3f}", file=text_file)
        print(f"Total:               {(end_time_train - start_time_train):6.3f}", file=text_file)
        print(f"Prediction:          {(end_time_predict - start_time_predict):6.3f}", file=text_file)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))
combined_stress_polar_pred = tuple(np.vstack((np.array(sigma_rr_pred.tolist()),np.array(sigma_theta_pred.tolist()),np.array(sigma_rtheta_pred.tolist()))))

u_fem, v_fem = displacement_fem[:,0], displacement_fem[:,1]
sigma_xx_fem, sigma_yy_fem, sigma_xy_fem = stress_fem[:,0:1], stress_fem[:,1:2], stress_fem[:,2:3]
sigma_rr_fem, sigma_theta_fem, sigma_rtheta_fem = polar_transformation_2d(sigma_xx_fem, sigma_yy_fem, sigma_xy_fem, X)

combined_disp_fem = tuple(np.vstack((np.array(u_fem.tolist()),np.array(v_fem.tolist()),np.zeros(u_fem.shape[0]))))
combined_stress_fem = tuple(np.vstack((np.array(sigma_xx_fem.flatten().tolist()),np.array(sigma_yy_fem.flatten().tolist()),np.array(sigma_xy_fem.flatten().tolist()))))
combined_stress_polar_fem = tuple(np.vstack((np.array(sigma_rr_fem.tolist()),np.array(sigma_theta_fem.tolist()),np.array(sigma_rtheta_fem.tolist()))))

# Compute difference error of PINNs
error_disp_x = abs(np.array(u_pred.tolist()) - u_fem.flatten())
error_disp_y =  abs(np.array(v_pred.tolist()) - v_fem.flatten())
combined_error_disp = tuple(np.vstack((error_disp_x, error_disp_y,np.zeros(error_disp_x.shape[0]))))

error_stress_x = abs(np.array(sigma_xx_pred.flatten().tolist()) - sigma_xx_fem.flatten())
error_stress_y =  abs(np.array(sigma_yy_pred.flatten().tolist()) - sigma_yy_fem.flatten())
error_stress_xy =  abs(np.array(sigma_xy_pred.flatten().tolist()) - sigma_xy_fem.flatten())
combined_error_stress = tuple(np.vstack((error_stress_x, error_stress_y, error_stress_xy)))

error_polar_stress_x = abs(np.array(sigma_rr_pred.flatten().tolist()) - sigma_rr_fem.flatten())
error_polar_stress_y =  abs(np.array(sigma_theta_pred.flatten().tolist()) - sigma_theta_fem.flatten())
error_polar_stress_xy =  abs(np.array(sigma_rtheta_pred.flatten().tolist()) - sigma_rtheta_fem.flatten())
combined_error_polar_stress = tuple(np.vstack((error_polar_stress_x, error_polar_stress_y, error_polar_stress_xy)))

# Output results to VTK
file_path = os.path.join(os.getcwd(), f"{model_path}/{simulation_case}-{n_iterations}_results")

dol_triangles = triangles.triangles
offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1]).astype(dol_triangles.dtype)
cell_types = np.ones(dol_triangles.shape[0])*5

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = {"displacement_pred" : combined_disp_pred,
                                               "displacement_fem" : combined_disp_fem,
                                               "stress_pred" : combined_stress_pred, 
                                               "stress_fem" : combined_stress_fem, 
                                               "polar_stress_pred" : combined_stress_polar_pred, 
                                               "polar_stress_fem" : combined_stress_polar_fem, 
                                               "error_disp" : combined_error_disp, 
                                               "error_stress" : combined_error_stress, 
                                               "error_polar_stress" : combined_error_polar_stress
                                            })

## Calculate the l2-error between FEM and PINN results
u_combined_pred = np.asarray(combined_disp_pred).T
s_combined_pred = np.asarray(combined_stress_pred).T
u_combined_fem = np.asarray(combined_disp_fem).T
s_combined_fem = np.asarray(combined_stress_fem).T

# Output l2-error into console and file
rel_err_l2_disp = np.linalg.norm(u_combined_pred - u_combined_fem) / np.linalg.norm(u_combined_fem)
print("Relative L2 error for displacement: ", rel_err_l2_disp)
rel_err_l2_stress = np.linalg.norm(s_combined_pred - s_combined_fem) / np.linalg.norm(s_combined_fem)
print("Relative L2 error for stress:       ", rel_err_l2_stress)
with open(f"{model_path}/{simulation_case}-{n_iterations}_L2_error_norm.txt", "w") as text_file:
    print(f"Relative L2 error for displacement: {rel_err_l2_disp:.8e}",   file=text_file)
    print(f"Relative L2 error for stress:       {rel_err_l2_stress:.8e}", file=text_file)

## Plot the normal traction on contact domain, analytical vs predicted
nu,lame,shear,e_modul = problem_parameters()
X, _, _, _ = geom.get_mesh()

output = model.predict(X)
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

x_lim = -0.25
x_contact_lim = 2*np.sqrt(2*radius**2*abs(ext_traction)*(1-nu**2)/(e_modul*np.pi))
x_contact_cond = np.logical_and(np.isclose(np.linalg.norm(X - center, axis=-1), radius), X[:,0]>x_lim)
node_coords_x_contact = -X[x_contact_cond][:,0]
idx = np.argsort(node_coords_x_contact)

pc_analytical = -np.nan_to_num(4*radius*ext_traction/(np.pi*x_contact_lim**2)*np.sqrt(x_contact_lim**2-node_coords_x_contact**2))
pc_predicted = -sigma_rr_pred[x_contact_cond]

fig2, ax2 = plt.subplots(figsize=(5,4))
ax2.plot(node_coords_x_contact[idx], pc_analytical[idx], label="Analytical", lw = 3)
ax2.plot(node_coords_x_contact[idx], pc_predicted[idx], label="Prediction", lw = 3, color = "tab:orange", linestyle = '--')
ax2.set_xlabel(r"$-x$", fontsize=16)
ax2.set_ylabel(r"$p_\mathrm{C}$", fontsize=16)
ax2.tick_params(axis="both", which="major", labelsize=12)
ax2.legend(fontsize=14)
ax2.grid()
plt.tight_layout()
fig2.savefig(f"{model_path}/{simulation_case}-{n_iterations}_pressure_distribution.png", dpi=300)
plt.show()