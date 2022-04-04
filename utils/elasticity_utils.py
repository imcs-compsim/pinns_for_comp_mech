import deepxde as dde
import numpy as np
from deepxde import utils

# global variables
lame = 1
shear = 0.5

def momentum_2d(x, y):    
    # calculate strain terms (kinematics, small strain theory)

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    # governing equation
    sigma_xx_x = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
    sigma_yy_y = dde.grad.jacobian(sigma_yy, x, i=0, j=1)
    sigma_xy_x = dde.grad.jacobian(sigma_xy, x, i=0, j=0)
    sigma_xy_y = dde.grad.jacobian(sigma_xy, x, i=0, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x

    return [momentum_x, momentum_y]

def momentum_2d_plane_stress(x, y):    
    # calculate strain terms (kinematics, small strain theory)

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    # governing equation
    sigma_xx_x = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
    sigma_yy_y = dde.grad.jacobian(sigma_yy, x, i=0, j=1)
    sigma_xy_x = dde.grad.jacobian(sigma_xy, x, i=0, j=0)
    sigma_xy_y = dde.grad.jacobian(sigma_xy, x, i=0, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x

    return [momentum_x, momentum_y]

def elastic_strain_2d(x,y):
    '''
    Calculates the strain tensor components for plane stress condition in 2D.

    Parameters
    ----------
    x : Placeholder (tf)
        contains the placeholder for coordinates of input points
    y : Placeholder (tf)
        contains the placeholder for network output

    Returns 
    -------
    sigma_xx, sigma_yy, sigma_xy: Placeholder (tf)
        contains the components of stress tensor 
    '''
    eps_xx = dde.grad.jacobian(y, x, i=0, j=0)
    eps_yy = dde.grad.jacobian(y, x, i=1, j=1)
    eps_xy = 1/2*(dde.grad.jacobian(y, x, i=1, j=0)+dde.grad.jacobian(y, x, i=0, j=1))
    
    return eps_xx, eps_yy, eps_xy


def problem_parameters():
    '''
    Calculates the elastic properties for given Lame and shear modulus.

    Returns
    -------
        nu : Float
            Poisson's ratio
        lame: Float
            Lame parameter
        shear: Float
            Shear modulus
        e_modul: Float
            Young's modulus
    '''
    e_modul = shear*(3*lame+2*shear)/(lame+shear)
    nu = lame/(2*(lame+shear))
    
    return nu, lame, shear, e_modul


def stress_plane_strain(x,y):
    '''
    Calculates the stress tensor components for plane stress condition.

    Parameters
    ----------
    x : Placeholder (tf)
        contains the placeholder for coordinates of input points
    y : Placeholder (tf)
        contains the placeholder for network output

    Returns 
    -------
    sigma_xx, sigma_yy, sigma_xy: Placeholder (tf)
        contains the components of stress tensor 
    '''
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(x,y)

    nu,lame,shear,e_modul = problem_parameters()
    
    # calculate stress terms (constitutive law - plane strain)
    sigma_xx = e_modul/((1+nu)*(1-2*nu))*((1-nu)*eps_xx+nu*eps_yy)
    sigma_yy = e_modul/((1+nu)*(1-2*nu))*(nu*eps_xx+(1-nu)*eps_yy)
    sigma_xy = e_modul/((1+nu)*(1-2*nu))*((1-2*nu)*eps_xy)

    return sigma_xx, sigma_yy, sigma_xy


def stress_plane_stress(x,y):
    '''
    Calculates the stress tensor components for plane stress condition.

    Parameters
    ----------
    x : Placeholder (tf)
        contains the placeholder for coordinates of input points
    y : Placeholder (tf)
        contains the placeholder for network output

    Returns 
    -------
    sigma_xx, sigma_yy, sigma_xy: Placeholder (tf)
        contains the components of stress tensor 
    '''
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(x,y)

    nu,lame,shear,e_modul = problem_parameters()

    sigma_xx = e_modul/(1-nu**2)*(eps_xx+nu*eps_yy)
    sigma_yy = e_modul/(1-nu**2)*(nu*eps_xx+eps_yy)
    sigma_xy = e_modul/(1-nu**2)*((1-nu)*eps_xy)

    return sigma_xx, sigma_yy, sigma_xy