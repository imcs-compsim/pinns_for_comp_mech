import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from pyevtk.hl import unstructuredGridToVTK
from pathlib import Path
import time

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

from utils.geometry.gmsh_models import QuarterCirclewithHole
from utils.elasticity import elasticity_utils

from utils.geometry.gmsh_models import QuarterCirclewithHole
from utils.geometry.custom_geometry import GmshGeometryElement

from utils.vpinns.quad_rule import GaussQuadratureRule
from utils.vpinns.quad_rule import get_test_function_properties

from utils.vpinns.v_pde import VariationalPDE

from utils.elasticity.elasticity_utils import problem_parameters, pde_mixed_plane_strain, lin_iso_elasticity_plane_strain
from utils.elasticity.elasticity_utils import calculate_traction_mixed_formulation, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.geometry.custom_geometry import GmshGeometry2D
from utils.geometry.gmsh_models import QuarterCirclewithHole
from utils.elasticity import elasticity_utils

from deepxde import backend as bkd

#dde.config.set_default_float('float64')

"""
Solves the quarter Lame problem using VPINNs.

  * * * * * *
  *            *
  *              *
  *                *     
     *              *   
       *             * 
        *             *    
  y      *            *
  |__x   * * * * * * **
  -------| --> R_i
  ---------------------| -->R_o     
 
Dirichlet BCs:

u_x(x=0,y) = 0
u_y(x,y=0) = 0

where u_x represents the displacement in x direction, while u_y represents the displacement in y direction. 

Neumann boundary conditions (in polar coordinates)
P(r=R_i,\theta) = 1 

In this problem set the material properties as follows:
    - lame : 1153.846
    - shear: 769.23

which will lead Young's modulus: 2000 and Poisson's coeff: 0.3. In this example, the Dirichlet boundary conditions are enforced hardly by choosing a surrogate model as follows:

u_s = u_x*x
v_s = u_y*y

where u_x and u_y are the network predictions.   

VPINNs weak formulation based on R(1) and R(2): https://www.worldscientific.com/doi/10.1142/S1758825123500655

The problem definition and analytical solution:
https://par.nsf.gov/servlets/purl/10100420

@author: tsahin
"""

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.3, gmsh_options=gmsh_options)

quad_rule = GaussQuadratureRule(rule_name="gauss_labotto", dimension=2, ngp=3) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 5
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")


# generate gmsh model
gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]

geom = GmshGeometryElement(gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           test_function=test_function, 
                           test_function_derivative=test_function_derivative, 
                           n_test_func=n_test_func,
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list)



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

# def boundary_left(x, on_boundary):
#     return on_boundary and np.isclose(x[0],0)

# def boundary_bottom(x, on_boundary):
#     return on_boundary and np.isclose(x[1],0)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_outer)
bc4 = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_outer)


def constitutive_law(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation in case of plane strain

    Parameters
    ----------
    x : tensor
        the input arguments
    y: tensor
        the network output

    Returns
    -------
    momentum_x, momentum_y, term_x, term_y, term_xy: tensor
        momentum_x, momentum_y: momentum terms based on derivatives of predicted stresses
        term_x, term_y, term_xy: difference between predicted stresses and calculated stresses in X, Y and XY direction
    '''
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_strain(x,y)

    return [term_x, term_y, term_xy]

residual_form = "1"

def weak_form_x(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    
    if residual_form == "1":
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        sigma_xx_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
        sigma_xy_y = dde.grad.jacobian(outputs, inputs, i=4, j=1)
        
        residual_x = vx*vy*(sigma_xx_x[beg:] + sigma_xy_y[beg:])
        
    elif residual_form == "2":
        sigma_xx = outputs[:, 2:3]
        sigma_xy = outputs[:, 4:5]
        
        vx_x = g_test_function_derivative[:,0:1]
        vy_y = g_test_function_derivative[:,1:2]
        
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        residual_x = -(sigma_xx[beg:]*vx_x*vy + sigma_xy[beg:]*vx*vy_y)
    
    weighted_residual_x = g_weights[:,0:1]*g_weights[:,1:2]*(residual_x)*g_jacobian
    
    return bkd.reshape(weighted_residual_x, (n_e, n_gp))

def weak_form_y(inputs, outputs, beg, n_e, n_gp, g_jacobian, g_weights, g_test_function, g_test_function_derivative):
    
    if residual_form == "1":
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        sigma_yy_y = dde.grad.jacobian(outputs, inputs, i=3, j=1)
        sigma_xy_x = dde.grad.jacobian(outputs, inputs, i=4, j=0)
        
        residual_y = vx*vy*(sigma_yy_y[beg:] + sigma_xy_x[beg:])
    
    elif residual_form == "2":
        sigma_yy = outputs[:, 3:4]
        sigma_xy = outputs[:, 4:5]
        
        vx_x = g_test_function_derivative[:,0:1]
        vy_y = g_test_function_derivative[:,1:2]
        
        vx = g_test_function[:,0:1]
        vy = g_test_function[:,1:2]
        
        residual_y = -(sigma_xy[beg:]*vx_x*vy + sigma_yy[beg:]*vx*vy_y)
    
    weighted_residual_y = g_weights[:,0:1]*g_weights[:,1:2]*(residual_y)*g_jacobian
    
    return bkd.reshape(weighted_residual_y, (n_e, n_gp))

n_dummy = 1
data = VariationalPDE(
    geom,
    [weak_form_x,weak_form_y],
    [bc1,bc2,bc3,bc4],
    constitutive_law,
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

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

model.compile("adam", lr=0.001, loss=mean_squared_error)
losshistory, train_state = model.train(epochs=2000, display_every=100)

model.compile("L-BFGS", loss=mean_squared_error)
model.train_step.optimizer_kwargs["options"]["maxiter"]=25000
losshistory, train_state = model.train(display_every=200)


def calculate_loss():
    losses = np.hstack(
            (
                np.array(losshistory.steps)[:, None],
                np.array(losshistory.loss_train),
            )
        )
    steps = losses[:,0]
    pde_loss = losses[:,1:5].sum(axis=1)
    neumann_loss = losses[:,5:9].sum(axis=1)
    
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

    plt.savefig(f"Lame_mixed_variational_r_{residual_form}.png", dpi=300)
    #plt.show()

gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.02, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel(visualize_mesh=False)

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

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

file_path = os.path.join(os.getcwd(), f"Lame_mixed_variational_r_{residual_form}")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros_like(y)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp_pred,"stress" : combined_stress_pred, "stress_polar": combined_stress_polar_pred})

compareModelPredictionAndAnalyticalSolution(model)