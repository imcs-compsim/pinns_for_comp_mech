import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from deepxde import backend as bkd
from pyevtk.hl import unstructuredGridToVTK

from utils.elasticity.elasticity_utils import stress_plane_stress, problem_parameters, stress_to_traction_2d
from utils.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from utils.geometry.custom_geometry import GmshGeometryElement
from utils.geometry.gmsh_models import QuarterCirclewithHole
from utils.elasticity import elasticity_utils

'''
Solves a hollow quarter cylinder under internal pressure (Lame problem)

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

Reference for PINNs formulation:
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel()

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometryElement(gmsh_model, dimension=2, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

radius_inner = quarter_circle_with_hole.inner_radius
center_inner = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]
radius_outer = quarter_circle_with_hole.outer_radius
center_outer = [quarter_circle_with_hole.center[0],quarter_circle_with_hole.center[1]]


# change global variables in elasticity_utils, they are used for getting the material properties for analytical model
lame = 1153.846
shear = 769.23
elasticity_utils.lame = lame
elasticity_utils.shear = shear
nu_analy,lame_analy,shear_analy,e_modul_analy = problem_parameters() # with dimensions, will be used for analytical solution

# The applied pressure 
pressure_inlet = 1

# characteristic quantities which are used for non-dimensionalization
characteristic_nu = shear # characteristic shear modulus
characteristic_disp = 1/e_modul_analy*pressure_inlet # characteristic displacement
characteristic_length = 1 # characteristic length
characteristic_stress = characteristic_nu*characteristic_disp/characteristic_length 

elasticity_utils.lame = lame/characteristic_nu    # non-dimensionalized, used for PINNs
elasticity_utils.shear = shear/characteristic_nu  # non-dimensionalized, used for PINNs

# Non-dimensionalized pressure
pressure_inlet_norm = pressure_inlet/characteristic_stress # non-dimensionalized, used for PINNs

# zero neumann BC functions need the geom variable to be 
elasticity_utils.geom = geom

def pde_stress(x,y):
    # governing equation
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]

def lin_iso_elasticity_plane_stress(x,y):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    term_x = sigma_xx - y[:, 2:3]
    term_y = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_x, term_y, term_xy

def zero_neumann_x(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx

def zero_neumann_y(x,y,X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 

    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty

def pressure_inner_x(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx + pressure_inlet_norm*normals[:,0:1]

def pressure_inner_y(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty + pressure_inlet_norm*normals[:,1:2]

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
bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, zero_neumann_x, boundary_outer)
bc6 = dde.OperatorBC(geom, zero_neumann_y, boundary_outer)

n_dummy = 1
data = dde.data.PDE(
    geom,
    pde_stress,
    [bc1, bc2, bc3, bc4, bc5],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol"
)

# non-dimensionalize the input using characteristic length 
def input_transform(x):
    return bkd.concat([x[:,0:1]/characteristic_length, x[:,1:2]/characteristic_length], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [30] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.PFNN(layer_size, activation, initializer)
net.apply_feature_transform(input_transform)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=4000, display_every=200)

model.compile("L-BFGS")
model.train()

###################################################################################
############################## VISUALIZATION PARTS ################################
###################################################################################

def compareModelPredictionAndAnalyticalSolution(model):
    '''
    This function plots analytical solutions and the predictions. 
    '''
    
    r = np.linspace(radius_inner, radius_outer,100)
    y = np.zeros(r.shape[0])

    dr2 = (radius_outer**2 - radius_inner**2)

    sigma_rr_analytical = radius_inner**2*pressure_inlet/dr2*(r**2-radius_outer**2)/r**2
    sigma_theta_analytical = radius_inner**2*pressure_inlet/dr2*(r**2+radius_outer**2)/r**2
    u_rad = radius_inner**2*pressure_inlet*r/(e_modul_analy*(radius_outer**2-radius_inner**2))*(1-nu_analy+(radius_outer/r)**2*(1+nu_analy))

    r_x = np.hstack((r.reshape(-1,1),y.reshape(-1,1)))
    output = model.predict(r_x)
    u_pred, v_pred = output[:,0:1]*characteristic_disp/characteristic_length, output[:,1:2]*characteristic_disp/characteristic_length
    u_rad_pred = np.sqrt(u_pred**2+v_pred**2)
    sigma_xx, sigma_yy, sigma_xy = output[:,2:3]*characteristic_stress, output[:,3:4]*characteristic_stress, output[:,4:5]*characteristic_stress
    sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, r_x)

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    axs[0].plot(r/radius_inner, sigma_rr_analytical/radius_inner, label = r"Analytical $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_rr/radius_inner, label = r"Predicted $\sigma_{r}$")
    axs[0].plot(r/radius_inner, sigma_theta_analytical/radius_inner, label = r"Analytical $\sigma_{\theta}$")
    axs[0].plot(r/radius_inner, sigma_theta/radius_inner, label = r"Predicted $\sigma_{\theta}$")
    axs[0].set(ylabel="Normalized stress", xlabel = "r/a")
    axs[1].plot(r/radius_inner, u_rad/radius_inner, label = r"Analytical $u_r$")
    axs[1].plot(r/radius_inner, u_rad_pred/radius_inner, label = r"Predicted $u_r$")
    axs[1].set(ylabel="Normalized radial displacement", xlabel = "r/a")
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()
    fig.tight_layout()

    plt.savefig("Lame_quarter_gmsh_nondimensionalized_newformulation")
    plt.show()

X, offset, cell_types, dol_triangles = geom.get_mesh()

output = model.predict(X)
u_pred, v_pred = output[:,0]*characteristic_disp/characteristic_length, output[:,1]*characteristic_disp/characteristic_length
sigma_xx, sigma_yy, sigma_xy = output[:,2]*characteristic_stress,output[:,3]*characteristic_stress,output[:,4]*characteristic_stress
sigma_rr, sigma_theta, sigma_rtheta = polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X)

combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(u_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress = tuple(np.vstack((np.array(sigma_xx.tolist()),np.array(sigma_yy.tolist()),np.array(sigma_xy.tolist()))))
combined_stress_polar = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))

file_path = os.path.join(os.getcwd(), "Lame_quarter_gmsh_nondimensionalized_newformulation")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros(y.shape)

unstructuredGridToVTK(file_path, x, y, z, dol_triangles.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp,"stress" : combined_stress, "stress_polar": combined_stress_polar})

compareModelPredictionAndAnalyticalSolution(model)