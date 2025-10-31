import deepxde as dde
import numpy as np
from pathlib import Path
import gmsh
from deepxde import config
import numpy as np
from deepxde import backend as bkd

# import helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
#import time
from pathlib import Path
from matplotlib import ticker

# set plot options
mpl.rcParams['mathtext.fontset'] = 'stix'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

'''
@author: tsahin

Simple block under compression test for a 2D block for elastodynamics under sin load in time direction => p = -pressure*sin(2\pi*t) for t [0,1]) 
This problem includes also body force.
In this example, z direction is considered as time direction. We generate the geometry using 3D mesh, but we consider the 3. dimension as time. 
Therefore, z direction must be the time direction. The followings must be taken into account:
- The z-direction must be the time direction. 
- The boundary normals must be consistent through the z-direction. The reason is that in Neumann BCs, we only take nx and ny terms assuming that nz is zero.   
'''

from compsim_pinns.geometry.custom_geometry import GmshGeometry3D
from compsim_pinns.geometry.gmsh_models import Block_3D_hex
from compsim_pinns.elasticity import elasticity_utils
from compsim_pinns.elasticity.elasticity_utils import pde_mixed_plane_strain_time_dependent, get_tractions_mixed_2d_time, problem_parameters
from compsim_pinns.postprocess.elasticity_postprocessing import solutionFieldOnMeshToVtkSpaceTime
from compsim_pinns.postprocess.save_normals_tangentials_to_vtk import export_normals_tangentials_to_vtk
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals_3D

# This is for visualization
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.gmsh_models import Block_2D

length = 1
height = 1
width = 1
seed_l = 10
seed_h = 10
seed_w = 10
origin = [0, 0, 0]

# The applied pressure 
pressure = 0.1
nu,lame,shear,e_modul = problem_parameters()

Block_3D_obj = Block_3D_hex(origin=origin, 
                            length=length,
                            height=height,
                            width=width,
                            divisions=[seed_l, seed_h, seed_w])

gmsh_model = Block_3D_obj.generateGmshModel(visualize_mesh=False)
geom = GmshGeometry3D(gmsh_model, target_surface_ids=[4])

# This allows for visualization of boundary normals in Paraview
export_normals_tangentials_to_vtk(geom, save_folder_path=str(Path(__file__).parent.parent.parent.parent), file_name="block_boundary_normals")

def analytical_solution(x, t):
    x_loc = x[:,0:1]
    y_loc = x[:,1:2]
    constant_x = lame/(4*shear*(lame+shear))
    constant_y = -(lame+2*shear)/(4*shear*(lame+shear))
    u_x = constant_x*np.sin(2*np.pi*t)*x_loc
    u_y = constant_y*np.sin(2*np.pi*t)*y_loc
    sigma_xx = np.zeros_like(x_loc)
    sigma_yy = -np.ones_like(x_loc)*np.sin(2*np.pi*t)
    sigma_xy = np.zeros_like(x_loc)
    
    return u_x, u_y, sigma_xx, sigma_yy, sigma_xy

def body_force(x):
    x_loc = x[:,0:1]
    y_loc = x[:,1:2]
    t_loc = x[:,2:3]
    
    f_x = - (lame*x_loc)/(4*shear*(lame + shear))*4*np.pi**2*bkd.sin(2*np.pi*t_loc)
    f_y = ((lame+2*shear)*y_loc)/(4*shear*(lame + shear))*4*np.pi**2*bkd.sin(2*np.pi*t_loc)
    
    return f_x, f_y

elasticity_utils.geom = geom
elasticity_utils.body_force_function = body_force 
# Top surface
def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1],height)

# Front surface
def boundary_initial(x, on_boundary):
    time_dimension = x[2]
    return on_boundary and np.isclose(time_dimension,0)

# Neumann BC on top
def apply_pressure_y_top(x,y,X):
    Tx, Ty, Tn, Tt = get_tractions_mixed_2d_time(x, y, X)
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    
    t_loc = x[:, 2:3][cond] # time only for boundary points
    
    return Ty + (bkd.sin(2*np.pi*t_loc))

# Initial BC for velocity component x
def apply_velocity_in_x(x,y,X):
    du_x_t = dde.grad.jacobian(y, x, i=0, j=2)# i=0 represents u_x, j=2 is time
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    x_loc = x[:,0:1][cond]
    t_loc = x[:,2:3][cond]

    return du_x_t[cond] - ((lame*x_loc)/(4*shear*(lame + shear))*2*np.pi*bkd.cos(2*np.pi*t_loc))

# Initial BC for velocity component y
def apply_velocity_in_y(x,y,X):
    du_y_t = dde.grad.jacobian(y, x, i=1, j=2) # i=1 represents u_y, j=2 is time
    _, _, _, cond = calculate_boundary_normals_3D(X,geom)
    y_loc = x[:,1:2][cond]
    t_loc = x[:,2:3][cond]

    return du_y_t[cond] - (-((lame+2*shear)*y_loc)/(4*shear*(lame + shear))*2*np.pi*bkd.cos(2*np.pi*t_loc))

# Neumann BC
bc_pressure_y_top = dde.OperatorBC(geom, apply_pressure_y_top, boundary_top)
# Initial BCs for velocities
ic_velocity_in_x = dde.OperatorBC(geom, apply_velocity_in_x, boundary_initial)
ic_velocity_in_y = dde.OperatorBC(geom, apply_velocity_in_y, boundary_initial)    
# for displacements
# bc_u_x = dde.DirichletBC(geom, lambda _: 0, boundary_initial, component=0)
# bc_u_y = dde.DirichletBC(geom, lambda _: 0, boundary_initial, component=1)

bcs = [bc_pressure_y_top, ic_velocity_in_x, ic_velocity_in_y] 

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain_time_dependent,
    bcs,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx =  y[:, 2:3]
    sigma_yy =  y[:, 3:4]
    sigma_xy =  y[:, 4:5]
    
    x_loc = x[:, 0:1] # coord x
    y_loc = x[:, 1:2] # coord y
    t_loc = x[:, 2:3] # time
    
    # define surfaces
    y_at_h = (height-y_loc)
    y_at_0 = (y_loc)
    x_at_l = (length-x_loc)
    x_at_0 = (x_loc)
    t_at_0 = (t_loc)
    t_at_w = (width-t_loc)
    
    # define the surfaces where shear forces will be applied.
    sigma_xy_surfaces = (y_at_h)*(y_at_0)*(x_at_l)*(x_at_0)
    
    return bkd.concat([u*(x_at_0)*t_at_0, # u_x is 0 at x=0 (Dirichlet BC) + u_x = 0 at t=0 (Initial BC) 
                      v*(y_at_0)*t_at_0, # u_y is 0 at y=0 (Dirichlet BC) + u_y = 0 at t=0 (Initial BC) 
                      sigma_xx*(x_at_l), 
                      sigma_yy, 
                      sigma_xy*sigma_xy_surfaces
                      ], axis=1)

# 3 inputs, 5 outputs for 3D 
layer_size = [3] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=1000, display_every=200)

model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=200)

#########################################################################################################################################
#### POST-PROCESSING #####
#########################################################################################################################################
############## 2D plots ######################
time_steps = 2
time_interval = [0, 1]
n_points = 100
x_coord = np.linspace(0,1,n_points).reshape(-1,1)
X = np.hstack((x_coord, np.ones_like(x_coord)*0.5))

time_element = [1/8, 1/4, 1/2, 3/4, 7/8]

fig, axs = plt.subplots(1,3,figsize=(15,5))

# Define a colormap for time steps
colors = plt.cm.viridis(np.linspace(0, 1, len(time_element)))

for i, current_time in enumerate(time_element):    
    # Combine spatial coordinates (X) with the current time into spacetime input
    X_spacetime = np.hstack((X, np.full([len(X), 1], current_time, dtype=config.real(np))))
    output = model.predict(X_spacetime)
    
    # Extract predicted displacements and stresses
    u_pred, v_pred = output[:,0:1], output[:,1:2]
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
    
    # Get analytical solution for the current time
    u_x, v_y, sigma_xx, sigma_yy, sigma_xy = analytical_solution(X, current_time)
    
    displacement_a = np.vstack((u_x.reshape(-1,1),v_y.reshape(-1,1)))
    displacement_pred = np.vstack((u_pred.reshape(-1,1),v_pred.reshape(-1,1)))
    rel_err_l2_disp = np.linalg.norm(displacement_a.flatten() - displacement_pred.flatten()) / np.linalg.norm(displacement_a)
    print(f"Relative L2 error for displacement at t={current_time}: ", rel_err_l2_disp)
    
    sigma = np.vstack((sigma_xx.reshape(-1,1),sigma_yy.reshape(-1,1), sigma_xy.reshape(-1,1)))
    sigma_pred = np.vstack((sigma_xx_pred.reshape(-1,1),sigma_yy_pred.reshape(-1,1), sigma_xy_pred.reshape(-1,1)))
    rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
    print(f"Relative L2 error for stress at t={current_time}: ", rel_err_l2_stress)
    
    # Separate x and y coordinates
    x = X[:,0]
    y = X[:,1]
    
    # Set the color for the current iteration
    color = colors[i]

    # First plot: displacement in x-direction (u_x)
    axs[0].plot(x.flatten(), u_x.flatten(), label=f"t = {current_time}", color=color)
    axs[0].plot(x.flatten(), u_pred.flatten(), marker='o', markersize=5, markevery=5, color=color)
    axs[0].set_xlabel(r"$x$", fontsize=17)
    axs[0].set_ylabel(r"$u_x$", fontsize=17)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[0].legend(fontsize=13)
    axs[0].grid()

    # Second plot: displacement in y-direction (v_y)
    axs[1].plot(x.flatten(), v_y.flatten(), label=f"t = {current_time}", color=color)
    axs[1].plot(x.flatten(), v_pred.flatten(), marker='o', markersize=5, markevery=5, color=color)
    axs[1].set_xlabel(r"$x$", fontsize=17)
    axs[1].set_ylabel(r"$u_y$", fontsize=17)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[1].legend(fontsize=13)
    axs[1].grid()

    # Third plot: stress in yy direction (sigma_yy)
    axs[2].plot(x.flatten(), sigma_yy.flatten(), label=f"t = {current_time}", color=color)
    axs[2].plot(x.flatten(), sigma_yy_pred.flatten(), marker='o', markersize=5, markevery=5, color=color)
    axs[2].set_xlabel(r"$x$", fontsize=17)
    axs[2].set_ylabel(r"$\sigma_{yy}$", fontsize=17)
    axs[2].tick_params(axis='both', labelsize=12)
    axs[2].legend(fontsize=13)
    axs[2].grid()

# Custom legend: one for analytical solutions (lines) and one for predictions (marked lines)
analytical_legend = mlines.Line2D([], [], color='black', label='Analytical Solution')
predicted_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label='Prediction')

# Add the custom legend below all plots
fig.legend(handles=[analytical_legend, predicted_legend], loc='lower center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, -0.05))


fig.tight_layout()
plt.savefig("block_time_result.png", dpi=300, bbox_inches="tight")
# plt.show()

##############################################################################################################
############## 3D visualization ######################
# ##############################################################################################################
gmsh.clear()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[0,0]
coord_right_corner=[1,1]
l_beam = coord_right_corner[0] - coord_left_corner[0]
h_beam = coord_right_corner[1] - coord_left_corner[1]

block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

time_steps = 10
time_interval = [0, 1]

solutionFieldOnMeshToVtkSpaceTime(geom, 
                            model,
                            time_interval=time_interval,
                            time_steps=time_steps, 
                            save_folder_path=str(Path(__file__).parent.parent.parent.parent), 
                            file_name="2D_block_time_sin_load",
                            analytical_solution=analytical_solution)








