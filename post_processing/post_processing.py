from pyevtk.hl import unstructuredGridToVTK
import deepxde as dde
import numpy as np
import warnings
import matplotlib.tri as tri
import os

def elasticity_post_processing(X, model, calculate_stress=False, polar_transformation=False, triangulation=None, condition=None, file_path=None, material_parameters={"lame":1, "shear":0.5}):
    global lame
    global shear
    
    try:
        lame = material_parameters["lame"]
        shear = material_parameters["shear"]
    except:
        raise ValueError("Please define the material parameters as lame and shear! Check https://en.wikipedia.org/wiki/Shear_modulus for conversion") 

    if X.shape[1]!= 2:
        warnings.warn("3D implementation is not done yet! So z component will be skipped!")
    
    x = X[:,0].flatten()
    y = X[:,1].flatten()
    z = np.zeros(y.shape)
    
    if condition is not None:
        dol_triangles = triangulation.triangles[condition]
    else:
        triang = tri.Triangulation(x, y)
        dol_triangles = triang.triangles
    
    displacement = model.predict(X)

    u_pred = displacement[:,0]
    v_pred = displacement[:,1]
    w_pred = np.zeros(v_pred.shape[0])
    combined_disp = tuple(np.vstack((np.array(u_pred.tolist()),np.array(v_pred.tolist()),w_pred)))
    
    if calculate_stress:
        sigma_xx, sigma_yy, sigma_xy = model.predict(X, operator=get_stress)

        if polar_transformation:
            theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
            theta_radian = theta*np.pi/180

            sigma_rr = ((sigma_xx + sigma_yy)/2 + (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 + sigma_xy*np.sin(2*theta_radian)).flatten()
            sigma_theta = ((sigma_xx + sigma_yy)/2 - (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 - sigma_xy*np.sin(2*theta_radian)).flatten()
            sigma_rtheta = np.zeros(sigma_theta.shape[0])
            combined_stress = tuple(np.vstack((np.array(sigma_rr.tolist()),np.array(sigma_theta.tolist()),np.array(sigma_rtheta.tolist()))))
        else:
            combined_stress = tuple(np.vstack((np.array(sigma_xx.flatten().tolist()),np.array(sigma_yy.flatten().tolist()),np.array(sigma_xy.flatten().tolist()))))

    if file_path:
        file_name = file_path
    else:
        file_name = os.path.join(os.getcwd(), "default_result")
    
    offset = np.arange(3,dol_triangles.shape[0]*dol_triangles.shape[1]+1,dol_triangles.shape[1])
    cell_types = np.ones(dol_triangles.shape[0])*5

    if calculate_stress:
        unstructuredGridToVTK(file_name, x, y, z, dol_triangles.flatten(), offset, 
                                cell_types, pointData = { "displacement" : combined_disp, "stress" : combined_stress}) #pointData = {"disp" : u_pred}
    else:
        unstructuredGridToVTK(file_name, x, y, z, dol_triangles.flatten(), offset, 
                                cell_types, pointData = { "displacement" : combined_disp})

def get_stress(x, y):    
    # calculate strain terms (kinematics, small strain theory)
    eps_xx, eps_yy, eps_xy = get_strain(x,y)

    constant,nu,lame,shear = problem_parameters()
    
    # calculate stress terms (constitutive law - plane strain)
    sigma_xx = constant*((1-nu)*eps_xx+nu*eps_yy)
    sigma_yy = constant*(nu*eps_xx+(1-nu)*eps_yy)
    sigma_xy = constant*((1-2*nu)*eps_xy)

    return [sigma_xx, sigma_yy, sigma_xy]

def get_strain(x,y):
    '''
    From displacement strain is obtained using automatic differentiation
    '''
    eps_xx = dde.grad.jacobian(y, x, i=0, j=0)
    eps_yy = dde.grad.jacobian(y, x, i=1, j=1)
    eps_xy = 1/2*(dde.grad.jacobian(y, x, i=1, j=0)+dde.grad.jacobian(y, x, i=0, j=1))
    return eps_xx, eps_yy, eps_xy

def problem_parameters():
    e_modul = shear*(3*lame+2*shear)/(lame+shear)
    nu = lame/(2*(lame+shear))
    constant = e_modul/((1+nu)*(1-2*nu))
    return constant,nu,lame,shear