### Quarter disc hertzian contact problem as a surrogate model
### @author: svoelkl, dwolff, apopp
### based on the work of tsahin
# Import required libraries
import deepxde as dde
dde.config.set_default_float('float64') # use double precision (needed for L-BFGS)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from deepxde.backend import tf
import matplotlib.tri as tri
from pyevtk.hl import unstructuredGridToVTK
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
import time

# Import custom modules
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterDisc
from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.geometry.geometry_utils import polar_transformation_2d
from utils.elasticity import elasticity_utils
import utils.contact_mech.contact_utils as contact_utils
from utils.contact_mech.contact_utils import zero_complimentarity_function_based_fischer_burmeister, zero_tangential_traction

## Create geometry
# Dimensions of disk
radius = 1
center = [0,0]
# Create the quarter disk using gmsh
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
Quarter_Disc = QuarterDisc(radius=radius, center=center, mesh_size=0.005, angle=265, refine_times=1, gmsh_options=gmsh_options)
gmsh_model, x_loc_partition, y_loc_partition = Quarter_Disc.generateGmshModel(visualize_mesh=False)
batch_size = gmsh_model.mesh.getNodes(2, -1, includeBoundary=True)[2].shape[0]//2
# Modifications to define a proper outer normal
revert_curve_list = []
revert_normal_dir_list = [1,2,2,1]
external_dim_size = 5
borders = [-0.2,-1.0]
geom = GmshGeometry2D(gmsh_model,external_dim_size=external_dim_size, borders=borders, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)
# extenal_dim_size defines a "third" dimension which is applied layerwise between borders in extenal_dim_size steps

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
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center, axis=-1), radius) and (x[0]<x_loc_partition)

def boundary_circle_contact(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x[:2] - center, axis=-1), radius) and (x[0]>=x_loc_partition)

# Neumann BC
bc_zero_traction_x = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_circle_not_contact)
bc_zero_traction_y = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_circle_not_contact)

# Contact BC
bc_zero_fischer_burmeister = dde.OperatorBC(geom, zero_complimentarity_function_based_fischer_burmeister, boundary_circle_contact)
bc_zero_tangential_traction = dde.OperatorBC(geom, zero_tangential_traction, boundary_circle_contact)
bcs = [bc_zero_traction_x,bc_zero_traction_y,bc_zero_fischer_burmeister,bc_zero_tangential_traction]

# Setup the data object
n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution="Sobol"
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
    ext_traction = x[:, 2:3]
    
    return tf.concat([u*(-x_loc)/e_modul, v/e_modul, sigma_xx, ext_traction + sigma_yy*(-y_loc),sigma_xy*(x_loc)*(y_loc)], axis=1)

## Define the neural network
# This time we also define the pressure as an input (in the previously described layers between borders)
layer_size = [3] + [50] * 5 + [5] # 3 inputs: x, y and p, 5 hidden layers with 50 neurons each, 5 outputs: ux, uy, sigma_xx, sigma_yy and sigma_xy
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

loss_weights = [w_pde_1, w_pde_2, w_pde_3, w_pde_4, w_pde_5,
                w_zero_traction_x, w_zero_traction_y,
                w_zero_fischer_burmeister,
                w_zero_tangential_traction]

## Train the model or use a pre-trained model
model = dde.Model(data, net)
restore_model = True
model_path = str(Path(__file__).parent)
simulation_case = f"surrogate"
adam_iterations = 2000

if not restore_model:
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    losshistory, train_state = model.train(iterations=adam_iterations, display_every=100)

    model.compile("L-BFGS-B", loss_weights=loss_weights)
    losshistory, train_state = model.train(display_every=200)
   
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
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
    n_iterations = 18702
    model_restore_path = model_path + "/" + simulation_case + "-"+ str(n_iterations) + ".ckpt"
    model_loss_path = model_path + "/" + simulation_case + "-"+ str(n_iterations) + "_loss.dat"
    
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
ax1.plot(steps, pde_loss/5, color='b', lw=2, label="PDE")
ax1.plot(steps, neumann_loss/4, color='r', lw=2,label="NBC")
ax1.vlines(x=adam_iterations,ymin=0, ymax=1, linestyles='--', colors="k")
ax1.annotate(r"ADAM $\ \Leftarrow$ ",    xy=[adam_iterations/2,0.5],   ha='center', va='top', size=15)
ax1.annotate(r"$\Rightarrow \ $ L-BGFS", xy=[adam_iterations*3/2,0.5], ha='center', va='top', size=15)
ax1.set_xlabel("Iterations", size=17)
ax1.set_ylabel("MSE", size=17)
ax1.set_yscale('log')
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(fontsize=17)
ax1.grid()
plt.tight_layout()
fig1.savefig(simulation_case+"_loss_plot.png", dpi=300)

## Visualize the pressure for different values
# Define variable space
theta = np.linspace(0, 2 * np.pi, num=15000, endpoint=False)
X = np.vstack((np.cos(theta), np.sin(theta))).T
node_coords_xy = radius*X + center

borders=[-0.45,-1.5]
external_dim_size = 3
p_applied = np.linspace(borders[0],borders[1],external_dim_size).reshape(-1,1).astype(np.dtype('f8'))
R = radius
p = abs(p_applied)

b = 2*np.sqrt(2*R**2*p*(1-nu**2)/(e_modul*np.pi))
n = 50
pc = []
x = []

# Calculate analytical solution pc
for i, p_ in enumerate(p):
    x.append(np.linspace(0,b[i],50))
    pc.append(4*R*p[i]/(np.pi*b[i]**2)*np.sqrt(b[i]**2-x[i]**2))

x_loc = []
pc_pred_list = []

# Calculate pc pinn
def f_pc_pinn(model, points):
    output = model.predict(points)
    u_pred, v_pred = output[:,0], output[:,1]
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
    sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, points)
    
    return -sigma_rr_pred

for i in range(external_dim_size):
    xy = node_coords_xy
    x_lim = -0.25
    condition = (xy[:,1]<=0) & (xy[:,0]>=x_lim) & (xy[:,0]<=0)

    contact_pts = xy[condition]
    
    network_input = np.hstack((contact_pts, p_applied[i]*np.ones((contact_pts.shape[0],1))))
    
    x_loc.append(contact_pts[:,0])
    pc_pred_list.append(f_pc_pinn(model, network_input).reshape(-1,1))

## Visualize the predicted pressure
fig, ax = plt.subplots(1,external_dim_size,figsize=(external_dim_size*4,4))
lw = 2
s = 20
lines =[]
titles=[fr"$p={p_i[0]:1.2}$" for p_i in p]

for i in range(external_dim_size):
    ax[i].grid()
    l1, = ax[i].plot(x[i], pc[i], label="Analytical", lw=lw, zorder=2)
    ax[i].hlines(y=0, xmin=b[i], xmax=abs(x_lim), lw=lw, zorder=3)
    l3 = ax[i].plot(-x_loc[i], pc_pred_list[i], color = "tab:orange", lw=lw, zorder=4)
    ax[i].set_xlabel(r"$|x|$", fontsize=22)
    ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[i].tick_params(axis='both', which='major', labelsize=13)
    ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))
    ax[i].set_title(titles[i], size=22)

ax[0].set_ylabel(r"$p_c$", fontsize=22)
fig.subplots_adjust(bottom=0.3, wspace=0.175)
plt.tight_layout()
fig.savefig(simulation_case+"_pressure_distribution.png", dpi=300)

## Compute the error of the solutions
def abs_error(actual, pred):
    mape = np.mean(np.abs((actual - pred)/pred))
    return mape

def mse(actual, pred):
    return np.mean((actual-pred)**2)

def l2_norm(actual, pred):
    l2_error_stress = np.linalg.norm(actual - pred) / np.linalg.norm(actual)
    return l2_error_stress

## Print and save the error
l2_error = []
for i in range(external_dim_size):
    x_e2 = -x_loc[i]
    pc_pred = pc_pred_list[i]
    pc_actual = 4*R*p[i]/(np.pi*b[i]**2)*np.sqrt(b[i]**2-x_e2**2)
    pc_actual[np.isnan(pc_actual)]=0
    
    l2_error.append(l2_norm(pc_actual.flatten(), pc_pred.flatten()))
    print(f"Relative L2 error for pressure: {float(l2_error[i]):.2%}")

with open("L2_error_norm.txt", "w") as text_file:
    for i in range(external_dim_size):
        print(f"At an external pressure of {titles[i][1:-1]} the relative L2 error for contact pressure is {float(l2_error[i]):.8e}",   file=text_file)
plt.show()