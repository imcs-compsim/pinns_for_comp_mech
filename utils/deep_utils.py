import deepxde as dde
import numpy as np
from deepxde import utils

def elastic_strain_2d(x,y):
    '''
    From displacement, strain is obtained using automatic differentiation
    '''
    eps_xx = dde.grad.jacobian(y, x, i=0, j=0)
    eps_yy = dde.grad.jacobian(y, x, i=1, j=1)
    eps_xy = 1/2*(dde.grad.jacobian(y, x, i=1, j=0)+dde.grad.jacobian(y, x, i=0, j=1))
    
    return eps_xx, eps_yy, eps_xy


def problem_parameters():
    
    lame = 1
    shear = 0.5

    e_modul = shear*(3*lame+2*shear)/(lame+shear)
    nu = lame/(2*(lame+shear))
    constant = e_modul/((1+nu)*(1-2*nu))
    
    return constant, nu, lame, shear, e_modul


def stress_plane_strain(x,y):
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(x,y)

    constant,nu,lame,shear,e_modul = problem_parameters()
    
    # calculate stress terms (constitutive law - plane strain)
    sigma_xx = constant*((1-nu)*eps_xx+nu*eps_yy)
    sigma_yy = constant*(nu*eps_xx+(1-nu)*eps_yy)
    sigma_xy = constant*((1-2*nu)*eps_xy)

    return sigma_xx, sigma_yy, sigma_xy


def stress_plane_stress(x,y):
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(x,y)

    constant,nu,lame,shear,e_modul = problem_parameters()

    sigma_xx = e_modul/(1-nu**2)*(eps_xx+nu*eps_yy)
    sigma_yy = e_modul/(1-nu**2)*(nu*eps_xx+eps_yy)
    sigma_xy = e_modul/(1-nu**2)*((1-nu)*eps_xy)

    return sigma_xx, sigma_yy, sigma_xy


def calculate_boundary_normals(X,geom):
    # boundary points
    cond = geom.on_boundary(X)
    boundary_points = X[cond]

    # convert them the normal function to tensor
    boundary_normal_tensor = utils.return_tensor(geom.boundary_normal)

    # normals
    normals = boundary_normal_tensor(boundary_points)

    return normals, cond


def polar_transformation_2d(sigma_xx, sigma_yy, sigma_xy, X):
    
    theta = np.degrees(np.arctan2(X[:,1],X[:,0])).reshape(-1,1) # in degree
    theta_radian = theta*np.pi/180

    sigma_rr = ((sigma_xx + sigma_yy)/2 + (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 + sigma_xy*np.sin(2*theta_radian)).flatten()
    sigma_theta = ((sigma_xx + sigma_yy)/2 - (sigma_xx - sigma_yy)*np.cos(2*theta_radian)/2 - sigma_xy*np.sin(2*theta_radian)).flatten()
    sigma_rtheta = np.zeros(sigma_theta.shape[0])
    
    return sigma_rr, sigma_theta, sigma_rtheta