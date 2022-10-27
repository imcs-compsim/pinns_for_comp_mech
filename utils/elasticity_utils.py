import deepxde as dde
import numpy as np
from deepxde import utils
from geometry_utils import calculate_boundary_normals

# global variables
lame = 1
shear = 0.5
geom = None

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

def zero_neumman_plane_stress_x(x, y, X):
    '''
    Calculates x component of the homogeneous Neumann BC
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    sigma_xx_n_x + sigma_xy_n_y: tensor
        x component of the homogeneous Neumann BC
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y

def zero_neumman_plane_stress_y(x, y, X):
    '''
    Calculates y component of the homogeneous Neumann BC

    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    sigma_yx_n_x + sigma_yy_n_y: tensor
        y component of the homogeneous Neumann BC
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y

def zero_neumman_plane_strain_x(x, y, X):
    '''
    Calculates x component of the homogeneous Neumann BC
    
    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    sigma_xx_n_x + sigma_xy_n_y: tensor
        x component of the homogeneous Neumann BC
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X, geom)

    sigma_xx_n_x = sigma_xx[cond]*normals[:,0:1]
    sigma_xy_n_y = sigma_xy[cond]*normals[:,1:2]

    return sigma_xx_n_x + sigma_xy_n_y

def zero_neumman_plane_strain_y(x, y, X):
    '''
    Calculates y component of the homogeneous Neumann BC

    Parameters
    ----------
    x : tensor
        the input arguments (coordinates x and y)
    y: tensor
        the network output (predicted displacement in x and y direction)
    X: np.array
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    sigma_yx_n_x + sigma_yy_n_y: tensor
        y component of the homogeneous Neumann BC
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)

    normals, cond = calculate_boundary_normals(X,geom)

    sigma_yx_n_x = sigma_xy[cond]*normals[:,0:1]
    sigma_yy_n_y = sigma_yy[cond]*normals[:,1:2]

    return sigma_yx_n_x + sigma_yy_n_y

def lin_iso_elasticity_plane_stress(x,y):
    '''
    Calculates the difference between predicted stresses and calculated stresses based on linear isotropic material law and predicted displacements in plane stress condition.

    Parameters
    ----------
    x : tensor
        the input arguments
    y: tensor
        the network output

    Returns
    -------
    term_x, term_y, term_xy: tensor
        difference between predicted stresses and calculated stresses in X, Y and XY direction 
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    term_x = sigma_xx - y[:, 2:3]
    term_y = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_x, term_y, term_xy

def pde_mixed_plane_stress(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation in case of plane stress

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
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]

def lin_iso_elasticity_plane_strain(x,y):
    '''
    Calculates the difference between predicted stresses and calculated stresses based on linear isotropic material law and predicted displacements in plane strain condition.

    Parameters
    ----------
    x : tensor
        the input arguments
    y: tensor
        the network output

    Returns
    -------
    term_x, term_y, term_xy: tensor
        difference between predicted stresses and calculated stresses in X, Y and XY direction 
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    term_x = sigma_xx - y[:, 2:3]
    term_y = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_x, term_y, term_xy

def pde_mixed_plane_strain(x,y):
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
    # governing equation
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_x, term_y, term_xy = lin_iso_elasticity_plane_strain(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]

