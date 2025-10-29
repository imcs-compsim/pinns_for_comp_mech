"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.interpolate
import os
import deepxde.backend as bkd

from compsim_pinns.elasticity.elasticity_utils import stress_plane_strain, momentum_2d
from compsim_pinns.elasticity import elasticity_utils

from pyevtk.hl import unstructuredGridToVTK
import gmsh

from compsim_pinns.geometry.gmsh_models import Block_2D
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement

from compsim_pinns.elasticity.elasticity_utils import problem_parameters, lin_iso_elasticity_plane_strain
from compsim_pinns.elasticity.elasticity_utils import calculate_traction_mixed_formulation
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.vpinns.quad_rule import get_test_function_properties

from compsim_pinns.vpinns.v_pde import VariationalPDE


'''
This script is used to create the PINN model of 2D Elasticity example. The example is taken from
A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics with the following link
https://www.semanticscholar.org/paper/A-physics-informed-deep-learning-framework-for-and-Haghighat-Raissi/e420b8cd519909b4298b16d1a46fbd015c86fc4e
'''

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
coord_left_corner=[0,0]
coord_right_corner=[1,1]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.1, gmsh_options=gmsh_options) #0.1

quad_rule = GaussQuadratureRule(rule_name="gauss_legendre", dimension=2, ngp=5) # gauss_legendre gauss_labotto
coord_quadrature, weight_quadrature = quad_rule.generate()

n_test_func = 10
test_function, test_function_derivative = get_test_function_properties(n_test_func, coord_quadrature, approach="2")

_,lame,shear,_ = problem_parameters()
Q_param = 4

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)

revert_curve_list = []
revert_normal_dir_list = [1,1,1,1]


geom = GmshGeometryElement(gmsh_model,
                           dimension=2, 
                           coord_quadrature=coord_quadrature, 
                           weight_quadrature= weight_quadrature, 
                           test_function=test_function, 
                           test_function_derivative=test_function_derivative, 
                           n_test_func=n_test_func,
                           revert_curve_list=revert_curve_list, 
                           revert_normal_dir_list=revert_normal_dir_list
                           )

# change global variables in elasticity_utils
elasticity_utils.geom = geom

def fun_sigma_yy(x, y, X):
    
    _, Ty, _, _  = calculate_traction_mixed_formulation(x, y, X)
    _, cond = calculate_boundary_normals(X,geom)

    return Ty - (lame+2*shear)*Q_param*bkd.sin(np.pi*x[:,0:1][cond])

def boundary_t(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)

def func(x):
    x_coord = x[:,0:1]
    y_coord = x[:,1:2]
    u_x = np.cos(2*np.pi*x_coord)*np.sin(np.pi*y_coord)
    u_y = np.sin(np.pi*x_coord)*(Q_param*y_coord**4)/4
    return np.hstack((u_x,u_y))


#geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

bc = dde.OperatorBC(geom, fun_sigma_yy, boundary_t)

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
        
    # Extract spatial coordinates x_s and y_s from the network inputs x
    x_s = inputs[:,0:1]
    y_s = inputs[:,1:2]
    
    # body forces
    f_x = lame*(4*np.pi**2*bkd.cos(2*np.pi*x_s)*bkd.sin(np.pi*y_s)-np.pi*bkd.cos(np.pi*x_s)*Q_param*y_s**3) + shear*(9*np.pi**2*bkd.cos(2*np.pi*x_s)*bkd.sin(np.pi*y_s)-np.pi*bkd.cos(np.pi*x_s)*Q_param*y_s**3)
    body_force_residual_x = f_x[beg:]*vx

    weighted_residual_x = g_weights[:,0:1]*g_weights[:,1:2]*(residual_x-body_force_residual_x)*g_jacobian
    
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
        
    # Extract spatial coordinates x_s and y_s from the network inputs x
    x_s = inputs[:,0:1]
    y_s = inputs[:,1:2]
    
    # body forces
    f_y = lame*(-3*bkd.sin(np.pi*x_s)*Q_param*y_s**2+2*np.pi**2*bkd.sin(2*np.pi*x_s)*bkd.cos(np.pi*y_s)) + shear*(-6*bkd.sin(np.pi*x_s)*Q_param*y_s**2+2*np.pi**2*bkd.sin(2*np.pi*x_s)*bkd.cos(np.pi*y_s)+np.pi**2*bkd.sin(np.pi*x_s)*Q_param*y_s**4/4)
    
    body_force_residual_y = f_y[beg:]*vy
    
    weighted_residual_y = g_weights[:,0:1]*g_weights[:,1:2]*(residual_y-body_force_residual_y)*g_jacobian
    
    return bkd.reshape(weighted_residual_y, (n_e, n_gp))
    
n_dummy = 1
data = VariationalPDE(
    geom,
    [weak_form_x,weak_form_y],
    [bc],
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
    
    return bkd.concat([u*y_loc*(1-y_loc),
                       v*x_loc*(1-x_loc)*y_loc, 
                       sigma_xx*x_loc*(1-x_loc), 
                       sigma_yy, #(lame+2*nu)*4*bkd.sin(np.pi*x_loc)+sigma_yy*(1-y_loc)
                       sigma_xy], axis=1)


# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
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
losshistory, train_state = model.train(display_every=200)

################ Post-processing ################
gmsh.clear()
gmsh.finalize()

# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
coord_left_corner=[0,0]
coord_right_corner=[1,1]

# create a block
block_2d = Block_2D(coord_left_corner=coord_left_corner, coord_right_corner=coord_right_corner, mesh_size=0.05, gmsh_options=gmsh_options)

# generate gmsh model
gmsh_model = block_2d.generateGmshModel(visualize_mesh=False)
geom = GmshGeometryElement(gmsh_model, dimension=2, only_get_mesh=True)

X, offset, cell_types, elements = geom.get_mesh()

output = model.predict(X)

u_pred, v_pred = output[:,0], output[:,1]
sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = output[:,2:3], output[:,3:4], output[:,4:5]

combined_disp_pred = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),np.zeros(u_pred.shape[0]))))
combined_stress_pred = tuple(np.vstack((np.array(sigma_xx_pred.flatten().tolist()),np.array(sigma_yy_pred.flatten().tolist()),np.array(sigma_xy_pred.flatten().tolist()))))

file_path = os.path.join(os.getcwd(), "Rectangle_2d_weak")

x = X[:,0].flatten()
y = X[:,1].flatten()
z = np.zeros_like(y)

unstructuredGridToVTK(file_path, x, y, z, elements.flatten(), offset, 
                      cell_types, pointData = { "displacement" : combined_disp_pred,"stress" : combined_stress_pred})
