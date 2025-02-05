import deepxde as dde
from utils.geometry.geometry_utils import calculate_boundary_normals, calculate_boundary_normals_3D
import deepxde.backend as bkd
import numpy as np
import tensorflow as tf

# global variables
lame = 1        # Lame ist lambda
shear = 0.5     # shear ist mü
geom = None
model_type = "plane_strain"
model_complexity = "nonlinear"          #with "linear" --> linear strain defintion, everyhing else i.e. "hueicii" nonlinear
green_lagrange_tensor_type = "second"    #type "first" is the formulation in question whether its right or not - "hviuvi" calls the second formulation

def deformation_gradient(x, y):

    f_xx = dde.grad.jacobian(y, x, i=0, j=0) + 1
    f_yy = dde.grad.jacobian(y, x, i=1, j=1) + 1
    f_xy = dde.grad.jacobian(y, x, i=0, j=1)
    f_yx = dde.grad.jacobian(y, x, i=1, j=0)

    return f_xx, f_yy, f_xy, f_yx

def green_lagrange_strain_tensor_1(x, y):

    f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)
    
    e_xx = 0.5*(f_xx * f_xx + f_yx * f_yx - 1)              #**2 durch reinen multiplikation ersetzen --> sollten dann gleich sein (ist nicht so)
    e_yy = 0.5*(f_xy * f_xy + f_yy * f_yy - 1)
    e_xy = 0.5*(f_xx * f_xy + f_yx * f_yy)
    e_yx = 0.5*(f_xy * f_xx + f_yy * f_yx)
    
    return e_xx, e_yy, e_xy, e_yx

def green_lagrange_strain_tensor_2(x,y):
   
    e_xx = 1/2 * (2 * dde.grad.jacobian(y, x, i=0, j=0) + (dde.grad.jacobian(y, x, i=0, j=0) * dde.grad.jacobian(y, x, i=0, j=0) + dde.grad.jacobian(y, x, i=1, j=0) * dde.grad.jacobian(y, x, i=1, j=0)))
    e_yy = 1/2 * (2 * dde.grad.jacobian(y, x, i=1, j=1) + (dde.grad.jacobian(y, x, i=0, j=1) * dde.grad.jacobian(y, x, i=0, j=1) + dde.grad.jacobian(y, x, i=1, j=1) * dde.grad.jacobian(y, x, i=1, j=1)))
    e_xy = 1/2 * (dde.grad.jacobian(y, x, i=1, j=0) + dde.grad.jacobian(y, x, i=0, j=1) + (dde.grad.jacobian(y, x, i=0, j=0) * dde.grad.jacobian(y, x, i=0, j=1) + dde.grad.jacobian(y, x, i=1, j=1) * dde.grad.jacobian(y, x, i=1, j=0)))
    e_yx = 1/2 * (dde.grad.jacobian(y, x, i=1, j=0) + dde.grad.jacobian(y, x, i=0, j=1) + (dde.grad.jacobian(y, x, i=0, j=0) * dde.grad.jacobian(y, x, i=0, j=1) + dde.grad.jacobian(y, x, i=1, j=1) * dde.grad.jacobian(y, x, i=1, j=0)))
    
    return e_xx, e_yy, e_xy, e_yx

"Keine Aufteilung in plane stress or strain mehr"

def second_piola_stress_tensor(x, y):
    
    if model_complexity == "linear":
        e_xx, e_yy, e_xy = elastic_strain_2d(x, y)     # When checking for linear results
    else:
        if green_lagrange_tensor_type == "first":
            e_xx, e_yy, e_xy, e_yx = green_lagrange_strain_tensor_1(x, y)
        else: 
            e_xx, e_yy, e_xy, e_yx = green_lagrange_strain_tensor_2(x, y)
    
    nu, lame, shear, e_modul = problem_parameters()
    
    # Check the value of model_typedef second_piola_stress_tensor(x, y):
    
    if model_type == "plane_strain":                            # based on linear neo hookean material behavior (small-moderate deformations)               
        e_yx = e_xy
        
        # calculate stress terms (constitutive law - plane strain)
        s_xx = e_modul/((1+nu)*(1-2*nu))*((1-nu)*e_xx+nu*e_yy)
        s_yy = e_modul/((1+nu)*(1-2*nu))*(nu*e_xx+(1-nu)*e_yy)
        s_xy = e_modul/((1+nu)*(1-2*nu))*((1-2*nu)*e_xy)
        s_yx = s_xy
        
    elif model_type == "plane_stress":
        
        s_xx = e_modul / (1 - nu**2) * (e_xx + nu * e_yy)
        s_yy = e_modul / (1 - nu**2) * (nu * e_xx + e_yy)
        s_xy = e_modul / (1 - nu**2) * (1 - nu) * e_xy
        s_yx = s_xy
    
    else:   # Hyperelastic second piola based on neo hookean model but modified
        
        nu, lame, shear, e_modul = problem_parameters()
        
        f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)
        determinant_deform_grad = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)
        
        C_xx, C_yy, C_xy, C_yx = right_cauchy_green_2D(f_xx, f_yy, f_xy, f_yx)
        C_xx_inv, C_yy_inv, C_xy_inv, C_yx_inv = matrix_inverse_2D(C_xx, C_yy, C_xy, C_yx)
        
        s_xx = lame / 2 * (determinant_deform_grad**2 - 1) * C_xx_inv + shear * (1 - C_xx_inv)
        
        s_yy = lame / 2 * (determinant_deform_grad**2 - 1) * C_yy_inv + shear * (1 - C_yy_inv)
        
        s_xy = lame / 2 * (determinant_deform_grad**2 - 1) * C_xy_inv + shear * (1 - C_xy_inv)
        
        s_yx = lame / 2 * (determinant_deform_grad**2 - 1) * C_yx_inv + shear * (1 - C_yx_inv)
        
    return s_xx, s_yy, s_xy, s_yx

def first_piola_stress_tensor(x,y):
    
    s_xx, s_yy, s_xy, s_yx = second_piola_stress_tensor(x, y)
    f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)

    p_xx = f_xx * s_xx + f_xy * s_yx
    p_yy = f_yx * s_xy + f_yy * s_yy
    p_xy = f_xx * s_xy + f_xy * s_yy
    p_yx = f_yx * s_xx + f_yy * s_yx

    return p_xx, p_yy, p_xy, p_yx

def cauchy_stress(x, y):
    
    f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)
    
    #det_F = 1/(f_xx * f_yy - f_xy * f_yx)
    
    det_F = 1/matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)
    
    T_xx = det_F * (p_xx * f_xx + p_xy * f_xy)      #Alternative formulation
    T_xy = det_F * (p_xx * f_yx + p_xy * f_yy)
    T_yx = det_F * (p_yx * f_xx + p_yy * f_xy)
    T_yy = det_F * (p_yx * f_yx + p_yy * f_yy)
    
    return T_xx, T_yy, T_xy, T_yx

def cauchy_stress_mixed_P(x, y):

    f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)
    p_xx, p_yy, p_xy, p_yx = y[:, 2:3] ,y[:, 3:4] ,y[:, 4:5], y[:, 5:6]
    
    # det_F = 1/(f_xx * f_yy - f_xy * f_yx)
    
    det_F = 1/matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)
    
    T_xx = det_F * (p_xx * f_xx + p_xy * f_xy)      #Alternative formulation
    T_xy = det_F * (p_xx * f_yx + p_xy * f_yy)
    T_yx = det_F * (p_yx * f_xx + p_yy * f_xy)
    T_yy = det_F * (p_yx * f_yx + p_yy * f_yy)
    
    return T_xx, T_yy, T_xy, T_yx

def momentum_2d_firstpiola(x, y):                           # independet of plane strain and stress as this is applied via S -> P

    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)

    # governing equation
    p_xx_x = dde.grad.jacobian(p_xx, x, i=0, j=0)
    p_yy_y = dde.grad.jacobian(p_yy, x, i=0, j=1)
    p_xy_y = dde.grad.jacobian(p_xy, x, i=0, j=1)
    p_yx_x = dde.grad.jacobian(p_yx, x, i=0, j=0)

    momentum_x = p_xx_x + p_xy_y
    momentum_y = p_yy_y + p_yx_x

    return [momentum_x, momentum_y]

def momentum_2d_plane_strain(x, y):    
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
    nu = lame/(2*(lame+shear))
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
    
    nu = lame/(2*(lame+shear))
    e_modul = shear*(3*lame+2*shear)/(lame+shear) #Original
    
    # Alternativ:
    #e_modul = (lame*(1+nu)*(1-2*nu))/(nu)
    
    "lambda_ = (nu * e_modul)/(1 - nu - 2*nu^(2))" "Lame ist lambda und shear ist mü"
    
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
        contains the components of stress tensor -04, 1.43e-04, 9
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

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx

def zero_neumman_first_piola_x(x, y, X):
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
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond)
    
    return Tx

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
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty

def zero_neumman_first_piola_y(x, y, X):
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
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)

    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond)

    return Ty

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
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx

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
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty

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

def pde_mixed_plane_stress(x,y):                #Tariks ursprüngliche mixed_implementierung funktioniert nicht mehr
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
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
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

def stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond):
    '''
    Calculates the traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential)).

    Parameters
    -----------
        sigma_xx (any): Stress component in x direction
        sigma_yy (any): Stress component in y direction
        sigma_xy (any): Stress component in xy direction (shear)
        normals (vector): Normal vectors
        cond (boolean): Dimensions of stresses and and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''
    
    nx = normals[:,0:1]
    ny = normals[:,1:2]

    sigma_xx_n_x = sigma_xx[cond]*nx
    sigma_xy_n_y = sigma_xy[cond]*ny

    sigma_yx_n_x = sigma_xy[cond]*nx
    sigma_yy_n_y = sigma_yy[cond]*ny
    
    Tx = sigma_xx_n_x + sigma_xy_n_y
    Ty = sigma_yx_n_x + sigma_yy_n_y
    Tn = Tx*nx + Ty*ny
    Tt = -Tx*ny + Ty*nx # Direction is clockwise --> if you go from normal tangetial

    return Tx, Ty, Tn, Tt

def piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond):
    '''
    Calculates the traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential)).

    Parameters
    -----------
        sigma_xx (any): Stress component in x direction
        sigma_yy (any): Stress component in y direction
        sigma_xy (any): Stress component in xy direction (shear)
        normals (vector): Normal vectors
        cond (boolean): Dimensions of stresses and and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''
    
    nx = normals[:,0:1]
    ny = normals[:,1:2]

    p_xx_n_x = p_xx[cond]*nx
    p_xy_n_y = p_xy[cond]*ny

    p_yx_n_x = p_yx[cond]*nx
    p_yy_n_y = p_yy[cond]*ny
    
    Tx = p_xx_n_x + p_xy_n_y
    Ty = p_yx_n_x + p_yy_n_y
    Tn = Tx*nx + Ty*ny
    Tt = -Tx*ny + Ty*nx # Direction is clockwise --> if you go from normal tangetial

    return Tx, Ty, Tn, Tt

def cauchy_stress_to_traction_2d(x, y, T_xx, T_yy, T_xy, normals, cond):
    '''
    Calculates the traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential)).

    Parameters
    -----------
        sigma_xx (any): Stress component in x direction
        sigma_yy (any): Stress component in y direction
        sigma_xy (any): Stress component in xy direction (shear)
        normals (vector): Normal vectors
        cond (boolean): Dimensions of stresses and and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''
    
    s_xx, s_yy, s_xy, s_yx = second_piola_stress_tensor(x, y)
    
    f_xx, f_yy, f_xy, f_yx = deformation_gradient(x, y)
    
    nx = normals[:,0:1]
    ny = normals[:,1:2]

    s_xx_n_x = s_xx[cond]*nx
    s_xy_n_y = s_xy[cond]*ny

    s_yx_n_x = s_xy[cond]*nx
    s_yy_n_y = s_yy[cond]*ny
    
    T_xx_ = s_xx_n_x + s_xy_n_y
    T_xy_ = s_yx_n_x + s_yy_n_y
    
    print("f_yx shape:", tf.shape(f_yx))
    print("f_yy shape:", tf.shape(f_yx))
    print("T_xx_ shape:", tf.shape(T_xx_))
    print("T_xy_ shape:", tf.shape(T_xy_))
    
    t_0_x = f_xx * T_xx_ + f_xy * T_xy_
    t_0_y = f_yx * T_xx_ + f_yy * T_xy_
    
    return t_0_x, t_0_y

def calculate_traction_mixed_from_piola_formulation(x, y, X):
    '''
    Calculates traction components in the mixed formulation. 
    
    Parameters
    -----------    #     term_x = sigma_xx - y[:, 2:3]
    # term_y = sigma_yy - y[:, 3:4]
    # term_xy = sigma_xy - y[:, 4:5]
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''

    p_xx, p_yy, p_xy, p_yx = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tn, Tt = piola_stress_to_traction_2d(p_xx, p_yy, p_xy, p_yx, normals, cond)

    return Tx, Ty, Tn, Tt

def calculate_traction_mixed_from_cauchy_stress_formulation(x, y, X):
    '''
    Calculates traction components in the mixed formulation. 
    
    Parameters
    -----------    #     term_x = sigma_xx - y[:, 2:3]
    # term_y = sigma_yy - y[:, 3:4]
    # term_xy = sigma_xy - y[:, 4:5]
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''

    T_xx, T_yy, T_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5]
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty = cauchy_stress_to_traction_2d(x, y, T_xx, T_yy, T_xy, normals, cond)

    return Tx, Ty

def zero_neumann_x_mixed_P_formulation(x, y, X):
    '''
    Calculates/Enforces x component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in x direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx: any
            x component of traction vector
    '''
    
    
    Tx, _, _, _ = calculate_traction_mixed_from_piola_formulation(x, y, X)

    return Tx

def zero_neumann_y_mixed_P_formulation(x, y, X):
    '''
    Calculates/Enforces y component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in y direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Ty: any
            y component of traction vector
    '''
    
    
    _, Ty, _, _ = calculate_traction_mixed_from_piola_formulation(x, y, X)

    return Ty

def zero_neumann_x_mixed_T_formulation(x, y, X):
    '''
    Calculates/Enforces x component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in x direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx: any
            x component of traction vector
    '''
    
    
    Tx, _ = calculate_traction_mixed_from_cauchy_stress_formulation(x, y, X)

    return Tx

def zero_neumann_y_mixed_T_formulation(x, y, X):
    '''
    Calculates/Enforces y component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in y direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Ty: any
            y component of traction vector
    '''
    
    
    _, Ty = calculate_traction_mixed_from_cauchy_stress_formulation(x, y, X)

    return Ty


# Consistency function: ideas

def momentum_mixed_T(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation

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
    T_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    T_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    T_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    T_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = T_xx_x + T_xy_y
    momentum_y = T_yy_y + T_xy_x
    
    # material law
    term_x, term_y, term_xy = iso_elasticity_T(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy]

def iso_elasticity_T(x,y):
    '''
    Calculates the difference between predicted T and calculated T based on isotropic material law and predicted displacements

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
    
    T_xx, T_yy, T_xy, T_yx = cauchy_stress(x,y)
    
    term_x = T_xx - y[:, 2:3]
    term_y = T_yy - y[:, 3:4]
    term_xy = T_xy - y[:, 4:5]
    
    return term_x, term_y, term_xy

def momentum_mixed_P(x,y):
    '''
    Calculates the momentum equation using predicted 1st piola k stresses and generates the terms for pde of the mixed-variable formulation

    Parameters
    ----------
    x : tensor
        the input arguments
    y: tensor
        the network output

    Returns
    -------
    momentum_x, momentum_y, term_x, term_y, term_xy, term_yx: tensor
        momentum_x, momentum_y: momentum terms based on derivatives of predicted stresses
        term_x, term_y, term_xy, term_yx: difference between predicted stresses and calculated stresses in X, Y, XY and YX direction
    '''
    # governing equation
    P_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    P_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    P_xy_y = dde.grad.jacobian(y, x, i=4, j=1)
    P_yx_x = dde.grad.jacobian(y, x, i=5, j=0)

    momentum_x = P_xx_x + P_xy_y
    momentum_y = P_yy_y + P_yx_x
    
    # material law
    term_x, term_y, term_xy, term_yx = iso_elasticity_P(x,y)

    return [momentum_x, momentum_y, term_x, term_y, term_xy, term_yx]

def iso_elasticity_P(x,y):
    '''
    Calculates the difference between predicted T and calculated T based on isotropic material law and predicted displacements

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
    
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor(x,y)
    
    term_x = p_xx - y[:, 2:3]
    term_y = p_yy - y[:, 3:4]
    term_xy = p_xy - y[:, 4:5]
    term_yx = p_yx - y[:, 5:6]
    
    return term_x, term_y, term_xy, term_yx

def compute_relative_l2_error(fem_data, pinn_data, column=0):
    # Extract the specified column as a 1D array (flatten)
    fem_col = fem_data[:, column] if fem_data.ndim > 1 else fem_data
    pinn_col = pinn_data[:, column] if pinn_data.ndim > 1 else pinn_data

    # Compute the relative L2 error
    numerator = np.sum((fem_col - pinn_col) ** 2)
    denominator = np.sum(fem_col ** 2)
    return np.sqrt(numerator / denominator)

def matrix_determinant_2D(a_11, a_22, a_12, a_21):
    # Calculate the determinant of the 2x2 matrix
    determinant = a_11 * a_22 - a_12 * a_21
    return determinant

def matrix_inverse_2D(a_11, a_22, a_12, a_21):
    # Calculate the determinant
    determinant = matrix_determinant_2D(a_11, a_22, a_12, a_21)
    
    # Check if the determinant is zero
    if determinant == 0:
        raise ValueError("The matrix is singular and does not have an inverse.")
    
    a_xx_new = a_22 / determinant
    a_yy_new = a_11 / determinant
    a_xy_new = -a_12 / determinant
    a_yx_new = -a_21 / determinant
    
    return a_xx_new, a_yy_new, a_xy_new, a_yx_new

def right_cauchy_green_2D(f_xx, f_yy, f_xy, f_yx):

    C_xx = f_xx**2 + f_yx**2
    C_yy = f_yy**2 + f_xy**2
    C_xy = f_xx * f_xy + f_yx * f_yy
    C_yx = f_xy * f_xx + f_yy * f_yx
    
    return C_xx, C_yy, C_xy, C_yx

#################################################################################################################################################################################
# Equations for 3D elasticity
#################################################################################################################################################################################
def get_elastic_strain_3d(x,y):
    '''
    Calculates the strain tensor components in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points
    y : Placeholder (tensor)
        contains the placeholder for network output

    Returns 
    -------
    eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz: tensor
        contains the components of strain tensor in 3D
    '''
    # Normal strains
    eps_xx = dde.grad.jacobian(y, x, i=0, j=0)
    eps_yy = dde.grad.jacobian(y, x, i=1, j=1)
    eps_zz = dde.grad.jacobian(y, x, i=2, j=2)
    
    # Shear strains
    eps_xy = 1/2 * (dde.grad.jacobian(y, x, i=1, j=0) + dde.grad.jacobian(y, x, i=0, j=1))
    eps_yz = 1/2 * (dde.grad.jacobian(y, x, i=2, j=1) + dde.grad.jacobian(y, x, i=1, j=2))
    eps_xz = 1/2 * (dde.grad.jacobian(y, x, i=2, j=0) + dde.grad.jacobian(y, x, i=0, j=2))

    return eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz 

def get_stress_tensor(x,y):
    '''
    Calculates the stress tensor components in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points
    y : Placeholder (tensor)
        contains the placeholder for network output

    Returns 
    -------
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz: tensor
        contains the components of stress tensor in 3D
    '''
    eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz = get_elastic_strain_3d(x,y)

    nu,lame,shear,e_modul = problem_parameters()
    
    # calculate stress terms (constitutive law)
    factor = e_modul / ((1 + nu) * (1 - 2*nu))
    
    sigma_xx = factor * ((1 - nu)*eps_xx + nu*eps_yy + nu*eps_zz)
    sigma_yy = factor * ((1 - nu)*eps_yy + nu*eps_xx + nu*eps_zz)
    sigma_zz = factor * ((1 - nu)*eps_zz + nu*eps_xx + nu*eps_yy)
    
    sigma_xy = factor * (1 - 2*nu) * eps_xy
    sigma_yz = factor * (1 - 2*nu) * eps_yz
    sigma_xz = factor * (1 - 2*nu) * eps_xz

    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz

def get_stress_coupling(x,y):
    '''
    Calculates the difference between predicted stresses and calculated stresses based on linear isotropic material law and predicted displacements in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, disp_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz

    Returns
    -------
    term_xx, term_yy, term_zz, term_xy, term_yz, term_xz: tensor
        difference between predicted stresses and calculated stresses in 3D
    '''
    
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz = get_stress_tensor(x,y)
    
    term_xx = sigma_xx - y[:, 3:4]
    term_yy = sigma_yy - y[:, 4:5]
    term_zz = sigma_zz - y[:, 5:6]
    term_xy = sigma_xy - y[:, 6:7]
    term_yz = sigma_yz - y[:, 7:8]
    term_xz = sigma_xz - y[:, 8:9]
    
    return term_xx, term_yy, term_zz, term_xy, term_yz, term_xz

def pde_mixed_3d(x, y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for PDE of the mixed-variable formulation in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, disp_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz

    Returns
    -------
    momentum_x, momentum_y, momentum_z, term_xx, term_yy, term_zz, term_xy, term_yz, term_xz: tensor
        momentum_x, momentum_y, momentum_z: momentum terms based on derivatives of predicted stresses
        term_xx, term_yy, term_zz, term_xy, term_yz, term_xz: difference between predicted stresses and calculated stresses in X, Y, Z, XY, YZ, and XZ directions
    '''
    # Stress derivatives
    sigma_xx_x = dde.grad.jacobian(y, x, i=3, j=0)
    #sigma_xx_y = dde.grad.jacobian(y, x, i=3, j=1)
    #sigma_xx_z = dde.grad.jacobian(y, x, i=3, j=2)
    
    #sigma_yy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=4, j=1)
    #sigma_yy_z = dde.grad.jacobian(y, x, i=4, j=2)
    
    #sigma_zz_x = dde.grad.jacobian(y, x, i=5, j=0)
    #sigma_zz_y = dde.grad.jacobian(y, x, i=5, j=1)
    sigma_zz_z = dde.grad.jacobian(y, x, i=5, j=2)
    
    sigma_xy_x = dde.grad.jacobian(y, x, i=6, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=6, j=1)
    #sigma_xy_z = dde.grad.jacobian(y, x, i=6, j=2)
    
    #sigma_yz_x = dde.grad.jacobian(y, x, i=7, j=0)
    sigma_yz_y = dde.grad.jacobian(y, x, i=7, j=1)
    sigma_yz_z = dde.grad.jacobian(y, x, i=7, j=2)
    
    sigma_xz_x = dde.grad.jacobian(y, x, i=8, j=0)
    #sigma_xz_y = dde.grad.jacobian(y, x, i=8, j=1)
    sigma_xz_z = dde.grad.jacobian(y, x, i=8, j=2)
    
    # Momentum equations
    momentum_x = sigma_xx_x + sigma_xy_y + sigma_xz_z
    momentum_y = sigma_yy_y + sigma_xy_x + sigma_yz_z
    momentum_z = sigma_zz_z + sigma_xz_x + sigma_yz_y

    # Material law
    term_xx, term_yy, term_zz, term_xy, term_yz, term_xz = get_stress_coupling(x, y)

    return [momentum_x, momentum_y, momentum_z, term_xx, term_yy, term_zz, term_xy, term_yz, term_xz]

def stress_to_traction_3d(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, normals, tangentials_1, tangentials_2, cond):
    '''
    Calculates the traction components in Cartesian (x, y, z) and polar coordinates (n and t) in 3D.

    Parameters
    ----------
    sigma_xx : any
        Stress component in the x direction
    sigma_yy : any
        Stress component in the y direction
    sigma_zz : any
        Stress component in the z direction
    sigma_xy : any
        Shear stress component in the xy plane
    sigma_xz : any
        Shear stress component in the xz plane
    sigma_yz : any
        Shear stress component in the yz plane
    normals : numpy.ndarray
        Normal vectors
    tangentials_1 : numpy.ndarray
        Tangential vectors in the first coordinate direction
    tangentials_2 : numpy.ndarray
        Tangential vectors in the second coordinate direction
    cond : boolean
        Dimensions of stresses and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
    Tx, Ty, Tz, Tn, Tt_1, Tt_2: any
        Traction components in Cartesian (x, y, z) coordinates and in normal (n) and tangential directions (t_1, t_2)
    '''
    # normals 
    nx = normals[:,0:1]
    ny = normals[:,1:2]
    nz = normals[:,2:3]
    
    # tangentials in epsilon direction
    t1x = tangentials_1[:,0:1]
    t1y = tangentials_1[:,1:2]
    t1z = tangentials_1[:,2:3]
    
    # tangentials in eta direction
    t2x = tangentials_2[:,0:1]
    t2y = tangentials_2[:,1:2]
    t2z = tangentials_2[:,2:3]

    # Calculate the traction components in Cartesian coordinates
    Tx = sigma_xx[cond]*nx + sigma_xy[cond]*ny + sigma_xz[cond]*nz
    Ty = sigma_xy[cond]*nx + sigma_yy[cond]*ny + sigma_yz[cond]*nz
    Tz = sigma_xz[cond]*nx + sigma_yz[cond]*ny + sigma_zz[cond]*nz
    
    # Calculate the traction components in polar coordinates (normal and tangential)
    # Calculate normal traction
    Tn = Tx*nx + Ty*ny + Tz*nz
    # Calculate tangential tractions (popp thesis page 23). 
    Tt_1 = Tx*t1x + Ty*t1y + Tz*t1z
    Tt_2 = Tx*t2x + Ty*t2y + Tz*t2z
    
    return Tx, Ty, Tz, Tn, Tt_1, Tt_2

def get_tractions_mixed_3d(x, y, X):
    '''
    Calculates traction components in the mixed formulation. 
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx, Ty, Tz, Tn, Tt_1, Tt_2: tensor
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''    
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    normals, tangentials_1, tangentials_2, cond = calculate_boundary_normals_3D(X,geom)

    Tx, Ty, Tz, Tn, Tt_1, Tt_2 = stress_to_traction_3d(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, normals, tangentials_1, tangentials_2, cond)

    return Tx, Ty, Tz, Tn, Tt_1, Tt_2

def apply_zero_neumann_x_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces x component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in x direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tx: tensor
            x component of traction vector
    '''
    
    Tx, Ty, Tz, Tn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)

    return Tx

def apply_zero_neumann_y_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces y component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in y direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Ty: tensor
            y component of traction vector
    '''
    
    Tx, Ty, Tz, Tn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)

    return Ty

def apply_zero_neumann_z_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces z component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in z direction.  
    
    Parameters
    -----------
        x : tensor
            Network input
        y: tensor
            Network output
        X : np array
            Network input as numpy array

    Returns
    -------
        Tz: tensor
            z component of traction vector
    '''
    
    Tx, Ty, Tz, Tn, Tt_1, Tt_2 = get_tractions_mixed_3d(x, y, X)

    return Tz