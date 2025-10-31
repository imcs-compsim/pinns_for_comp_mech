import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path

from compsim_pinns.elasticity.elasticity_utils import stress_plane_stress, momentum_2d_plane_stress, stress_to_traction_2d, zero_neumman_plane_stress_x, zero_neumman_plane_stress_y
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals
from compsim_pinns.geometry.custom_geometry import GmshGeometry2D
from compsim_pinns.geometry.gmsh_models import QuarterCirclewithHole
from compsim_pinns.elasticity import elasticity_utils

'''
Solves inverse problem for a hollow quarter cylinder under internal pressure (Lame problem) using displacement based formulation.
Unknown quantities:
    - Young's modulus
    - Poisson's ratio
    
Experimental data is generated using PINNs. Data file name:
    - Inverse_displacement
    
We add displacements to training from the experimental data

Reference solution:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6132

@author: tsahin
'''


gmsh_options = {"General.Terminal":1, "Mesh.Algorithm": 6}
quarter_circle_with_hole = QuarterCirclewithHole(center=[0,0,0], inner_radius=1, outer_radius=2, mesh_size=0.05, gmsh_options=gmsh_options)

gmsh_model = quarter_circle_with_hole.generateGmshModel()

revert_curve_list = ["curve_2"]
revert_normal_dir_list = [2,2,1,2]
geom = GmshGeometry2D(gmsh_model, revert_curve_list=revert_curve_list, revert_normal_dir_list=revert_normal_dir_list)

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
pressure_inlet = 1.0

# change global variables in elasticity_utils
elasticity_utils.geom = geom
elasticity_utils.lame = e_predicted*nu_predicted/((1+nu_predicted)*(1-2*nu_predicted))
elasticity_utils.shear = e_predicted/(2*(1+nu_predicted))

def pressure_inner_x(x, y, X):
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx + pressure_inlet*normals[:,0:1]

def pressure_inner_y(x, y, X):

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty + pressure_inlet*normals[:,1:2]

def boundary_outer(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_outer, axis=-1), radius_outer)

def boundary_inner(x, on_boundary):
    return on_boundary and np.isclose(np.linalg.norm(x - center_inner, axis=-1), radius_inner)

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0],0)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1],0)

bc1 = dde.OperatorBC(geom, pressure_inner_x, boundary_inner)
bc2 = dde.OperatorBC(geom, pressure_inner_y, boundary_inner)
bc3 = dde.DirichletBC(geom, lambda _: 0.0, boundary_left, component=0)
bc4 = dde.DirichletBC(geom, lambda _: 0.0, boundary_bottom, component=1)
bc5 = dde.OperatorBC(geom, zero_neumman_plane_stress_x, boundary_outer)
bc6 = dde.OperatorBC(geom, zero_neumman_plane_stress_y, boundary_outer)

inv_data = str(Path(__file__).parent.parent.parent)+"/data/lame/Inverse_displacement"
data = np.loadtxt(inv_data)
np.random.seed(42)
np.random.shuffle(data)
data = data[:100]
data_xy, data_u, data_v = data[:,0:2], data[:,2:3], data[:,3:4]

observe_u = dde.PointSetBC(data_xy, data_u, component=0)
observe_v = dde.PointSetBC(data_xy, data_v, component=1)

n_dummy = 1
data = dde.data.PDE(
    geom,
    momentum_2d_plane_stress,
    [bc1, bc2, bc3, bc4, bc5, bc6, observe_u, observe_v],
    num_domain=n_dummy,
    num_boundary=n_dummy,
    num_test=n_dummy,
    train_distribution = "Sobol",
    anchors=data_xy
)

# two inputs x and y, output is ux and uy
layer_size = [2] + [50] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)


model = dde.Model(data, net)

external_var_list = [e_predicted, nu_predicted]

parameter_file_name = "lame_inverse_displacement.dat"

variable = dde.callbacks.VariableValue(external_var_list, period=10, filename=parameter_file_name)

model.compile("adam", lr=0.001, external_trainable_variables=external_var_list)
n_iter_adam = 2000
losshistory, train_state = model.train(epochs=n_iter_adam, callbacks=[variable], display_every=100)

model.compile("L-BFGS-B", external_trainable_variables=external_var_list)
losshistory, train_state = model.train(callbacks=[variable], display_every=100)

######## PLOT estimated parameters vs actuals ##########

# set dark thema
sns.set_theme(style="darkgrid")

parameter_file = os.path.join(os.getcwd(),parameter_file_name)

df = pd.read_csv(parameter_file, sep=" ", header=None)
df[1] = pd.to_numeric(df[1].str[1:-1])
df[2] = pd.to_numeric(df[2].str[0:-1])

df_sliced = df[:350] #pickupt only epochs until 3500. Since we have period of 10, the last value will be 350

fig, ax = plt.subplots(figsize=(15,10))

line_1, = ax.plot(df_sliced[0], df_sliced[1], color='b', lw=2)
line_2, = ax.plot(df_sliced[0], df_sliced[2], color='r', lw=2)

ax.vlines(x=n_iter_adam,ymin=df_sliced[2].min()*1.5, ymax=df_sliced[1].max()*1.1, linestyles='--', colors="k")
ax.hlines(y=e_actual,xmin=0, xmax=df_sliced[0].max(), linestyles='-.', colors="b", label=f"Actual $E={e_actual}$")
ax.hlines(y=nu_actual,xmin=0, xmax=df_sliced[0].max(), linestyles='-.', colors="r", label=r"Actual $\nu=$"+f"{nu_actual}")

ax.annotate(r"ADAM $\Leftarrow$ ", xy=[n_iter_adam-600,df_sliced[1].max()*1.05], size=25)
ax.annotate(r"$\Rightarrow$ L-BGFS", xy=[n_iter_adam+50,df_sliced[1].max()*1.05], size=25)

line_1.set_label(r'Predicted $E$')
line_2.set_label(r'Predicted $\nu$')

ax.legend(prop={'size': 25}, loc=6)

ax.set_xlabel("Epochs", size=25)
ax.set_ylabel(r"Estimated parameter values", size=25)
ax.tick_params(labelsize=20)

fig.savefig("lame_inverse_displacement.png", dpi=300)