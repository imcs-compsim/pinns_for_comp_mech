'''
@author: tsahin
'''
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path

from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.elasticity import elasticity_utils

from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.geometry.custom_geometry import GmshGeometryElement

from compsim_pinns.vpinns.quad_rule import GaussQuadratureRule
from compsim_pinns.vpinns.quad_rule import get_test_function_properties

from compsim_pinns.vpinns.v_pde import VariationalPDE

from compsim_pinns.elasticity.elasticity_utils import lin_iso_elasticity_plane_stress
from compsim_pinns.elasticity.elasticity_utils import calculate_traction_mixed_formulation, zero_neumann_x_mixed_formulation, zero_neumann_y_mixed_formulation
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, polar_transformation_2d
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.elasticity import elasticity_utils

from deepxde import backend as bkd


# Define GMSH and geometry parameters
gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 11}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.5, gmsh_options=gmsh_options)

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

# Material properties
e_actual = 1.33
nu_actual = 0.33
e_predicted = dde.Variable(5.0) #initial guess
nu_predicted = dde.Variable(0.49) #initial guess

# Applied pressure 
pressure_inlet_actual = 1.0
pressure_inlet_predicted = dde.Variable(5.0)

# change global variables in elasticity_utils
elasticity_utils.geom = geom
elasticity_utils.lame = e_predicted*nu_predicted/((1+nu_predicted)*(1-2*nu_predicted))
elasticity_utils.shear = e_predicted/(2*(1+nu_predicted))

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
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x,y)

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

def pressure_inner_x(x, y, X):
    
    normals, _ = calculate_boundary_normals(X,geom)
    Tx, _, _, _  = calculate_traction_mixed_formulation(x, y, X)

    return Tx + pressure_inlet_predicted*normals[:,0:1]

def pressure_inner_y(x, y, X):

    normals, _ = calculate_boundary_normals(X,geom)
    _, Ty, _, _  = calculate_traction_mixed_formulation(x, y, X)

    return Ty + pressure_inlet_predicted*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner) #and ~np.logical_and(np.isclose(x[0],1),np.isclose(x[1],0)) and ~np.logical_and(np.isclose(x[0],0),np.isclose(x[1],1))

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.OperatorBC(geom, zero_neumann_x_mixed_formulation, boundary_outer)
bc4 = dde.OperatorBC(geom, zero_neumann_y_mixed_formulation, boundary_outer)

inv_data = str(Path(__file__).parent.parent.parent)+"/elasticity_2d/data/lame/Inverse_mixed_variable"
data = np.loadtxt(inv_data)
np.random.seed(42)
np.random.shuffle(data)
data = data[:100]
data_xy, data_u, _, _, data_sigma_yy = data[:,0:2], data[:,2:3], data[:,3:4], data[:,4:5] , data[:,5:6]

observe_u = dde.PointSetBC(data_xy, data_u, component=0)
observe_sigma_yy = dde.PointSetBC(data_xy, data_sigma_yy, component=3)

n_dummy = 1
data = VariationalPDE(
    geom,
    [weak_form_x,weak_form_y],
    [bc1, bc2, bc3, bc4, observe_u, observe_sigma_yy],
    constitutive_law,
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=None,
    train_distribution = "Sobol",
    anchors=data_xy
)

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    sigma_xx = y[:, 2:3]
    sigma_yy = y[:, 3:4]
    sigma_xy = y[:, 4:5]
    x_loc = x[:, 0:1]
    y_loc = x[:, 1:2]
    
    return bkd.concat([u*(x_loc),v*(y_loc), sigma_xx, sigma_yy, sigma_xy*x_loc*y_loc], axis=1)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

external_var_list = [e_predicted, nu_predicted, pressure_inlet_predicted]
parameter_file_name = "lame_inverse_mixed_variable.dat"
variable = dde.callbacks.VariableValue(external_var_list, period=10, filename=parameter_file_name, precision=4)

def mean_squared_error(y_true, y_pred):
    return bkd.mean(bkd.square(y_true - y_pred), dim=0)

n_iter_adam = 2000
model.compile("adam", lr=0.001, loss=mean_squared_error, external_trainable_variables=external_var_list)
losshistory, train_state = model.train(epochs=n_iter_adam, callbacks=[variable], display_every=100)

model.compile("L-BFGS-B", loss=mean_squared_error, external_trainable_variables=external_var_list)
losshistory, train_state = model.train(callbacks=[variable], display_every=100)

# ######## PLOT estimated parameters vs actuals ##########

# set dark thema
sns.set_theme(style="darkgrid")

parameter_file = os.path.join(os.getcwd(),parameter_file_name)

df = pd.read_csv(parameter_file, sep=" ", header=None)
df[1] = pd.to_numeric(df[1].str[1:-1])
df[2] = pd.to_numeric(df[2].str[0:-1])
df[3] = pd.to_numeric(df[3].str[0:-1])

df_sliced = df[:350] #pickupt only epochs until 3500. Since we have period of 10, the last value will be 350

fig, ax = plt.subplots(figsize=(15,10))

line_1, = ax.plot(df_sliced[0], df_sliced[1], color='b', lw=2)
line_2, = ax.plot(df_sliced[0], df_sliced[2], color='r', lw=2)
line_3, = ax.plot(df_sliced[0], df_sliced[3], color='g', lw=2)

ax.vlines(x=n_iter_adam,ymin=df_sliced[2].min()*1.5, ymax=df_sliced[1].max()*1.1, linestyles='--', colors="k")
ax.hlines(y=e_actual,xmin=0, xmax=df_sliced[0].max(), linestyles='-.', colors="b", label=f"Actual $E={e_actual}$")
ax.hlines(y=nu_actual,xmin=0, xmax=df_sliced[0].max(), linestyles='-.', colors="r", label=r"Actual $\nu=$"+f"{nu_actual}")
ax.hlines(y=pressure_inlet_actual,xmin=0, xmax=df_sliced[0].max(), linestyles='-.', colors="g", label=r"Actual $P=$"+f"{pressure_inlet_actual}")

ax.annotate(r"ADAM $\Leftarrow$ ", xy=[n_iter_adam-600,df_sliced[1].max()*1.05], size=25)
ax.annotate(r"$\Rightarrow$ L-BGFS", xy=[n_iter_adam+50,df_sliced[1].max()*1.05], size=25)

line_1.set_label(r'Predicted $E$')
line_2.set_label(r'Predicted $\nu$')
line_3.set_label(r'Predicted $P$')

ax.legend(prop={'size': 25}, loc=6)

ax.set_xlabel("Epochs", size=25)
ax.set_ylabel(r"Parameters", size=25)
ax.tick_params(labelsize=20)

fig.savefig("lame_inverse_mixed_variable.png", dpi=300)