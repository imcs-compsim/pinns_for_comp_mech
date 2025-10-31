import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import deepxde.backend as bkd
from pyevtk.hl import unstructuredGridToVTK
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
import time
from pathlib import Path

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain
from compsim_pinns.elasticity.elasticity_utils import calculate_traction_mixed_formulation, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement
from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.elasticity import elasticity_utils

'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

Reference for PINNs formulation:
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

@author: tsahin
'''

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.1, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometryElement(gmsh_model, dimension=2, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

radius_inner = quarter_circle_with_hole.inner_radius
center_inner = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]
radius_outer = quarter_circle_with_hole.outer_radius
center_outer = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]

# change global variables in elasticity_utils
elasticity_utils.geom = geom
# change global variables in elasticity_utils
elasticity_utils.lame = 1153.846
elasticity_utils.shear = 769.23
nu,lame,shear,e_modul = problem_parameters()

# The applied pressure 
pressure_inlet = 1

def pressure_inner_x(x, y, X):
    
    normals, _ = calculate_boundary_normals(X,geom)
    Tx, _, _, _  = calculate_traction_mixed_formulation(x, y, X)

    return Tx + pressure_inlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    normals, _ = calculate_boundary_normals(X,geom)
    _, Ty, _, _  = calculate_traction_mixed_formulation(x, y, X)

    return Ty + pressure_inlet*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner) #and ~np.logical_and(np.isclose(x[0],1),np.isclose(x[1],0)) and ~np.logical_and(np.isclose(x[0],0),np.isclose(x[1],1))

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_outer)
bc4 = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_outer)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_mixed_plane_strain,
    [bc1,bc2,bc3,bc4],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)


def output_transform(x, y):
    '''
    Hard BCs:
        Dirichlet terms
            u(x=0)=0
        
        Neumann terms:
            sigma_xx(x=l) = 0
            sigma_yy(y=h) = ext_traction
            sigma_xy(x=l) = 0, sigma_xy(x=0) = 0 and sigma_xy(y=h) = 0

        where h:=h_beam and l:=l_beam.
    
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
    
    return bkd.concat([u*(x_loc)/e_modul,v*(y_loc)/e_modul, sigma_xx, sigma_yy, sigma_xy*x_loc*y_loc], axis=1)
# two inputs x and y, output is ux and uy
layer_size = [2] + [30] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

restore_model = True
model_path = str(Path(__file__).parent.parent.parent.parent)+f"/pretrained_models/elasticity/lame_structured/lame"

if not restore_model:
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=2000, display_every=100)

    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=200)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
else:
    n_epochs = 5484 
    model_restore_path = model_path + "-"+ str(n_epochs) + ".ckpt"
    
    model.compile("adam", lr=0.001)
    model.restore(save_path=model_restore_path)

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

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''
    nu,lame,shear,e_modul = problem_parameters()
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])

    dr2 = (radius_outer**2 - radius_inner**2)
    sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
    sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2

    inv_dr2 = (1/radius_inner**2 - 1/radius_outer**2)
    a = -pressure_inlet/inv_dr2
    c = -a/(2*radius_outer**2)
    
    u_rad = (1+nu)/e_modul*(-a/r+2*(1-2*nu)*c*r)

    r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
    output = model.predict(r_x)
    u_pred, v_pred = output[:,0:1], output[:,1:2]
    u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
    sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, r_x)
    
    rel_err_l2_disp = np.linalg.norm(u_rad.flatten() - u_rad_pred.flatten()) / np.linalg.norm(u_rad)
    print("Relative L2 error for displacement: ", rel_err_l2_disp)
    
    sigma = np.vstack((sigma_rr_analytical.reshape(-1,1),sigma_theta_analytical.reshape(-1,1)))
    sigma_pred = np.vstack((sigma_rr.reshape(-1,1),sigma_theta.reshape(-1,1)))
    rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
    print("Relative L2 error for stress: ", rel_err_l2_stress)


    steps, pde_loss, neumann_loss = calculate_loss()
    
    fig, axs = plt.subplots(1,3,figsize=(15,5))

    axs[0].plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{rr}/R_i$",color="tab:blue")
    axs[0].plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{rr}/R_i$", color="tab:blue", marker='o', markersize=5, markevery=5)
    # axs[0].scatter(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{rr}/R_i$", s=10, c="tab:blue", marker='o', edgecolors="tab:orange")
    axs[0].plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta\theta}/R_i$",color="tab:orange")
    axs[0].plot(r/radius_inner, sigma_theta/radius_inner,label = r"Predicted $\sigma_{\theta\theta}/R_i$",color="tab:orange", marker='o', markersize=5, markevery=5)
    axs[0].set_xlabel(r"$r \ /R_i$", fontsize=17)
    axs[0].set_ylabel(r"$\sigma \ /R_i$", fontsize=17)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[0].legend(fontsize=13)
    axs[0].grid()
    
    axs[1].plot(r/radius_inner, u_rad/radius_inner, label = r"Analytical $u_r/R_i$",color="tab:orange")
    axs[1].plot(r/radius_inner, u_rad_pred/radius_inner, label = r"Predicted $u_r/R_i$",color="tab:orange", marker='o', markersize=5, markevery=5)
    axs[1].set_xlabel(r"$r \ /R_i$", fontsize=17)
    axs[1].set_ylabel(r"$u \ /R_i$", fontsize=17)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[1].legend(fontsize=13)
    axs[1].yaxis.set_major_formatter(formatter)
    axs[1].grid()
    
    axs[2].plot(steps, pde_loss/5, color='b', lw=2, label="PDE")
    axs[2].plot(steps, neumann_loss/4, color='r', lw=2,label="NBC")
    axs[2].vlines(x=2000,ymin=0, ymax=1, linestyles='--', colors="k")
    axs[2].annotate(r"ADAM $\ \Leftarrow$ ", xy=[610,0.5], size=13)
    axs[2].annotate(r"$\Rightarrow \ $ L-BGFS", xy=[2150,0.5], size=13)
    axs[2].tick_params(axis="both", labelsize=12)
    axs[2].set_xlabel("Epochs", size=17)
    axs[2].set_ylabel("MSE", size=17)
    axs[2].set_yscale('log')
    axs[2].legend(fontsize=13)
    axs[2].grid()
    
    
    fig.tight_layout()

    plt.savefig("Lame_quarter_gmsh_mixed_scaled_structured.png", dpi=300)
    plt.show()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.02, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometryElement(gmsh_model, dimension=2, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

X, offset, cell_types, dol_triangles = geom.get_mesh()

def computeError(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''
    nu,lame,shear,e_modul = problem_parameters()
    
    pts_xy = X
    r = np.sqrt(X[:,0]**2+X[:,1]**2) 


    dr2 = (radius_outer**2 - radius_inner**2)
    sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
    sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2

    inv_dr2 = (1/radius_inner**2 - 1/radius_outer**2)
    a = -pressure_inlet/inv_dr2
    c = -a/(2*radius_outer**2)
    
    u_rad = (1+nu)/e_modul*(-a/r+2*(1-2*nu)*c*r)

    output = model.predict(pts_xy)
    u_pred, v_pred = output[:,0:1], output[:,1:2]
    u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
    sigma_xx, sigma_yy, sigma_xy = output[:,2:3], output[:,3:4], output[:,4:5]
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, pts_xy)
    
    rel_err_l2_disp = np.linalg.norm(u_rad.flatten() - u_rad_pred.flatten()) / np.linalg.norm(u_rad)
    print("Relative L2 error for displacement (all): ", rel_err_l2_disp)
    
    sigma = np.vstack((sigma_rr_analytical.reshape(-1,1),sigma_theta_analytical.reshape(-1,1)))
    sigma_pred = np.vstack((sigma_rr.reshape(-1,1),sigma_theta.reshape(-1,1)))
    rel_err_l2_stress = np.linalg.norm(sigma.flatten() - sigma_pred.flatten()) / np.linalg.norm(sigma)
    print("Relative L2 error for stress (all): ", rel_err_l2_stress)

computeError(model)

start_time_calc = time.time()
output = model.predict(X)
end_time_calc = time.time()
final_time = f'Prediction time: {(end_time_calc - start_time_calc):.3f} seconds'
print(final_time)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]
sigma_rr_pred, sigma_theta_pred, sigma_rtheta_pred = polar_transformation_2d(sigma_xx_pred, sigma_yy_pred, sigma_xy_pred, X)

combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))
combined_stress_polar_pred = tuple(np.vstack((np.array(sigma_rr_pred.tolist()),np.array(sigma_theta_pred.tolist()),np.array(sigma_rtheta_pred.tolist()))))

file_path = os.path.join(os.getcwd(), "Lame_quarter_gmsh_mixed_scaled_structured")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp_pred,"stress" : combined_stress_pred, "stress_polar": combined_stress_polar_pred})

if not restore_model:
    compareModelPredictionAndAnalyticalSolution(model)