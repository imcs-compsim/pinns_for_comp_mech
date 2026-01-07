import deepxde as dde
from compsim_pinns.geometry.geometry_utils import calculate_boundary_normals, calculate_boundary_normals_3D
import deepxde.backend as bkd
from deepxde import utils

# global variables
lame = 1
shear = 0.5
rho = 1
body_force_function = None
geom = None
spacetime_domain = None

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
    sigma_xx : Placeholder (tf)
        xx component of stress tensor
    sigma_yy : Placeholder (tf)
        yy component of stress tensor
    sigma_xy : Placeholder (tf)
        xy component of stress tensor
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
    nu : float
        Poisson's ratio
    lame : float
        Lame parameter
    shear : float
        Shear modulus
    e_modul : float
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
    sigma_xx : Placeholder (tf)
        xx component of stress tensor
    sigma_yy : Placeholder (tf)
        yy component of stress tensor
    sigma_xy : Placeholder (tf)
        xy component of stress tensor
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
    sigma_xx : Placeholder (tf)
        xx component of stress tensor
    sigma_yy : Placeholder (tf)
        yy component of stress tensor
    sigma_xy : Placeholder (tf)
        xy component of stress tensor
    '''
    eps_xx, eps_yy, eps_xy = elastic_strain_2d(x,y)

    nu,lame,shear,e_modul = problem_parameters()

    sigma_xx = e_modul/(1-nu**2)*(eps_xx+nu*eps_yy)
    sigma_yy = e_modul/(1-nu**2)*(nu*eps_xx+eps_yy)
    sigma_xy = e_modul/(1-nu**2)*((1-nu)*eps_xy)

    return sigma_xx, sigma_yy, sigma_xy

def zero_neumman_plane_stress_x(x, y, X):
    '''
    Calculates x component of the homogeneous Neumann BC.
    
    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments (coordinates x and y)
    y : torch.Tensor or tf.Tensor
        the network output (predicted displacement in x and y direction)
    X : numpy.ndarray
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    Tx : torch.Tensor or tf.Tensor
        x component of the homogeneous Neumann BC
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx

def zero_neumman_plane_stress_y(x, y, X):
    '''
    Calculates y component of the homogeneous Neumann BC.

    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments (coordinates x and y)
    y : torch.Tensor or tf.Tensor
        the network output (predicted displacement in x and y direction)
    X : numpy.ndarray
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    Ty : torch.Tensor or tf.Tensor
        y component of the homogeneous Neumann BC
    '''

    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)

    normals, cond = calculate_boundary_normals(X,geom)
    _, Ty, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Ty

def zero_neumman_plane_strain_x(x, y, X):
    '''
    Calculates x component of the homogeneous Neumann BC.
    
    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments (coordinates x and y)
    y : torch.Tensor or tf.Tensor
        the network output (predicted displacement in x and y direction)
    X : numpy.ndarray
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    Tx : torch.Tensor or tf.Tensor
        x component of the homogeneous Neumann BC
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx

def zero_neumman_plane_strain_y(x, y, X):
    '''
    Calculates y component of the homogeneous Neumann BC.

    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments (coordinates x and y)
    y : torch.Tensor or tf.Tensor
        the network output (predicted displacement in x and y direction)
    X : numpy.ndarray
        the input arguments as an array (coordinates x and y)

    Returns
    -------
    Ty : torch.Tensor or tf.Tensor
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
    x : torch.Tensor or tf.Tensor
        the input arguments
    y : torch.Tensor or tf.Tensor
        the network output

    Returns
    -------
    term_xx : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_stress(x,y)
    
    term_xx = sigma_xx - y[:, 2:3]
    term_yy = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_xx, term_yy, term_xy

def pde_mixed_plane_stress(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation in case of plane stress.

    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments
    y : torch.Tensor or tf.Tensor
        the network output

    Returns
    -------
    momentum_x : torch.Tensor or tf.Tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : torch.Tensor or tf.Tensor
        momentum terms in y based on derivatives of predicted stresses
    term_xx : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_xx, term_yy, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [momentum_x, momentum_y, term_xx, term_yy, term_xy]

def lin_iso_elasticity_plane_strain(x,y):
    '''
    Calculates the difference between predicted stresses and calculated stresses based on linear isotropic material law and predicted displacements in plane strain condition.

    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments
    y : torch.Tensor or tf.Tensor
        the network output

    Returns
    -------
    term_xx :torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    
    sigma_xx, sigma_yy, sigma_xy = stress_plane_strain(x,y)
    
    term_xx = sigma_xx - y[:, 2:3]
    term_yy = sigma_yy - y[:, 3:4]
    term_xy = sigma_xy - y[:, 4:5]
    
    return term_xx, term_yy, term_xy

def pde_mixed_plane_strain(x,y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for pde of the mixed-variable formulation in case of plane strain.

    Parameters
    ----------
    x : torch.Tensor or tf.Tensor
        the input arguments
    y : torch.Tensor or tf.Tensor
        the network output

    Returns
    -------
    momentum_x : torch.Tensor or tf.Tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : torch.Tensor or tf.Tensor
        momentum terms in y based on derivatives of predicted stresses
    term_xx : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : torch.Tensor or tf.Tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    # governing equation
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    momentum_x = sigma_xx_x + sigma_xy_y
    momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_xx, term_yy, term_xy = lin_iso_elasticity_plane_strain(x,y)

    return [momentum_x, momentum_y, term_xx, term_yy, term_xy]

def stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond):
    '''
    Calculates the traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential)).

    Parameters
    -----------
    sigma_xx : any
        Stress component in x direction
    sigma_yy : any
        Stress component in y direction
    sigma_xy : any
        Stress component in xy direction (shear)
    normals : vector
        Normal vectors
    cond : boolean
        Dimensions of stresses and and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
    Tx : any
        Traction components in cartesian x direction
    Ty : any
        Traction components in cartesian y direction
    Tn : any
        Traction components in polar n (normal) direction
    Tt : any
        Traction components in polar t (tangential) direction
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

def calculate_traction_mixed_formulation(x, y, X):
    '''
    Calculates traction components in the mixed formulation. 
    
    Parameters
    -----------
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Tx : any
        Traction components in cartesian x direction
    Ty : any
        Traction components in cartesian y direction
    Tn : any
        Traction components in polar n (normal) direction
    Tt : any
        Traction components in polar t (tangential) direction
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

def zero_neumann_x_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces x component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in x direction.  
    
    Parameters
    -----------
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Tx: any
        x component of traction vector
    '''
    
    
    Tx, _, _, _ = calculate_traction_mixed_formulation(x, y, X)

    return Tx

def zero_neumann_y_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces y component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in y direction.  
    
    Parameters
    -----------
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Ty: any
        y component of traction vector
    '''
    
    
    _, Ty, _, _ = calculate_traction_mixed_formulation(x, y, X)

    return Ty
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
    eps_xx : tensor
        contains the xx components of strain tensor in 3D
    eps_yy : tensor
        contains the yy components of strain tensor in 3D
    eps_zz : tensor
        contains the zz components of strain tensor in 3D
    eps_xy : tensor
        contains the xy components of strain tensor in 3D
    eps_yz : tensor
        contains the yz components of strain tensor in 3D
    eps_xz : tensor
        contains the xz components of strain tensor in 3D
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
    sigma_xx : tensor
        contains the xx components of stress tensor in 3D
    sigma_yy : tensor
        contains the yy components of stress tensor in 3D
    sigma_zz : tensor
        contains the zz components of stress tensor in 3D
    sigma_xy : tensor
        contains the xy components of stress tensor in 3D
    sigma_yz : tensor
        contains the yz components of stress tensor in 3D
    sigma_xz : tensor
        contains the xz components of stress tensor in 3D
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
    term_xx : tensor
        difference between predicted stresses and calculated stresses in xx component in 3D
    term_yy : tensor
        difference between predicted stresses and calculated stresses in yy component in 3D
    term_zz : tensor
        difference between predicted stresses and calculated stresses in zz component in 3D
    term_xy : tensor
        difference between predicted stresses and calculated stresses in xy component in 3D
    term_yz : tensor
        difference between predicted stresses and calculated stresses in yz component in 3D
    term_xz : tensor
        difference between predicted stresses and calculated stresses in xz component in 3D
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
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    momentum_z : tensor
        momentum terms in z based on derivatives of predicted stresses
    term_xx : tensor
        difference between predicted stresses and calculated stresses in xx component in 3D
    term_yy : tensor
        difference between predicted stresses and calculated stresses in yy component in 3D
    term_zz : tensor
        difference between predicted stresses and calculated stresses in zz component in 3D
    term_xy : tensor
        difference between predicted stresses and calculated stresses in xy component in 3D
    term_yz : tensor
        difference between predicted stresses and calculated stresses in yz component in 3D
    term_xz : tensor
        difference between predicted stresses and calculated stresses in xz component in 3D
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
    Tx : any
        Traction components in cartesian x direction
    Ty : any
        Traction components in cartesian y direction
    Tz : any
        Traction components in cartesian z direction
    Tn : any
        Traction components in polar n (normal) direction
    Tt_1 : any
        Traction components in polar t_1 (tangential) direction
    Tt_2 : any
        Traction components in polar t_2 (tangential) direction
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
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Tx : torch.Tensor or tf.Tensor
        Traction components in cartesian x direction
    Ty : torch.Tensor or tf.Tensor
        Traction components in cartesian y direction
    Tz : torch.Tensor or tf.Tensor
        Traction components in cartesian z direction
    Tn : torch.Tensor or tf.Tensor
        Traction components in polar n (normal) direction
    Tt_1 : torch.Tensor or tf.Tensor
        Traction components in polar t_1 (tangential) direction
    Tt_2 : torch.Tensor or tf.Tensor
        Traction components in polar t_2 (tangential) direction
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
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Tx: torch.Tensor or tf.Tensor
        x component of traction vector
    '''
    
    Tx, _, _, _, _, _ = get_tractions_mixed_3d(x, y, X)

    return Tx

def apply_zero_neumann_y_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces y component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in y direction.  
    
    Parameters
    -----------
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Ty: torch.Tensor or tf.Tensor
        y component of traction vector
    '''
    
    _, Ty, _, _, _, _ = get_tractions_mixed_3d(x, y, X)

    return Ty

def apply_zero_neumann_z_mixed_formulation(x, y, X):
    '''
    Calculates/Enforces z component of traction vector in the mixed formulation. This is also known as zero Neumann boundary conditions in z direction.  
    
    Parameters
    -----------
    x : torch.Tensor or tf.Tensor
        Network input
    y : torch.Tensor or tf.Tensor
        Network output
    X : numpy.ndarray
        Network input as numpy array

    Returns
    -------
    Tz: torch.Tensor or tf.Tensor
        z component of traction vector
    '''
    
    _, _, Tz, _, _, _ = get_tractions_mixed_3d(x, y, X)

    return Tz
#################################################################################################################################################################################
# Equations for 2D elastodynamics
#################################################################################################################################################################################
def pde_mixed_plane_stress_time_dependent(x,y):
    '''
    Calculates the momentum equation using predicted stresses and displacements. 
    It generates the terms for pde of the displacement-stress mixed-variable formulation in case of plane stress.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points and time: x, y, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, sigma_xx, sigma_yy, sigma_xy

    Returns
    -------
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    term_xx : tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    # get ddu_x/dt2
    u_x_tt = dde.grad.hessian(y, x, component=0, i=2, j=2) # component is u_x, tt is the third input which is time tt=>ij
    # get ddu_y/dt2
    u_y_tt = dde.grad.hessian(y, x, component=1, i=2, j=2) # component is u_y, tt is the third input which is time tt=>ij
    # get stress derivatives
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)

    if body_force_function:
        body_force_x, body_force_y = body_force_function(x)
        momentum_x = sigma_xx_x + sigma_xy_y + body_force_x - rho*u_x_tt
        momentum_y = sigma_yy_y + sigma_xy_x + body_force_y - rho*u_y_tt
    else:
        momentum_x = sigma_xx_x + sigma_xy_y
        momentum_y = sigma_yy_y + sigma_xy_x
    
    # material law
    term_xx, term_yy, term_xy = lin_iso_elasticity_plane_stress(x,y)

    return [momentum_x, momentum_y, term_xx, term_yy, term_xy]

def pde_mixed_plane_strain_time_dependent(x,y):
    '''
    Calculates the momentum equation using predicted stresses and displacements. 
    It generates the terms for pde of the displacement-stress mixed-variable formulation in case of plane strain for elastodynamics.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points and time: x, y, t
    y : Placeholder (tensor)
        contains the placeholder for network output: c

    Returns
    -------
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    term_xx : tensor
        difference between predicted stresses and calculated stresses for xx component
    term_yy : tensor
        difference between predicted stresses and calculated stresses for yy component
    term_xy : tensor
        difference between predicted stresses and calculated stresses for xy component
    '''
    # get ddu_x/dt2
    u_x_tt = dde.grad.hessian(y, x, component=0, i=2, j=2) # component is u_x, tt is the third input which is time tt=>ij
    # get ddu_y/dt2
    u_y_tt = dde.grad.hessian(y, x, component=1, i=2, j=2) # component is u_y, tt is the third input which is time tt=>ij
    # get stress derivatives
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)
    
    # momentum terms
    if body_force_function:
        body_force_x, body_force_y = body_force_function(x)
        momentum_x = sigma_xx_x + sigma_xy_y + body_force_x - rho*u_x_tt
        momentum_y = sigma_yy_y + sigma_xy_x + body_force_y - rho*u_y_tt
    else:
        momentum_x = sigma_xx_x + sigma_xy_y - rho*u_x_tt
        momentum_y = sigma_yy_y + sigma_xy_x - rho*u_y_tt
    
    # material law
    term_xx, term_yy, term_xy = lin_iso_elasticity_plane_strain(x,y)

    return [momentum_x, momentum_y, term_xx, term_yy, term_xy]

def get_tractions_mixed_2d_time(x, y, X):
    '''
    Calculates traction components in 2D mixed formulation with time. Note that this method works only if the generated GMSH geometry is 3D. 
    
    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points and time: x, y, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, sigma_xx, sigma_yy, sigma_xy
    X : numpy.ndarray
        Network input as numpy array (x,y,t)

    Returns
    -------
    Tx : any
        Traction components in cartesian x direction
    Ty : any
        Traction components in cartesian y direction
    Tn : any
        Traction components in polar n (normal) direction
    Tt : any
        Traction components in polar t (tangential) direction
    '''    
    sigma_xx =  y[:, 2:3]
    sigma_yy =  y[:, 3:4]
    sigma_xy =  y[:, 4:5]
    
    normals, _, _, cond = calculate_boundary_normals_3D(X,geom)

    # get the traction vector, here we assume that time is the z-direction therefore nz = 0 on the target edges (surfaces).
    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

def get_tractions_mixed_2d_spacetime(x, y, X):
    '''
    Calculates traction components in the mixed formulation. Note that this method works only if space-time domain (dde.geometry.GeometryXTime) is generated.
    
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points and time: x, y, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, sigma_xx, sigma_yy, sigma_xy
    X : numpy.ndarray
        Network input as numpy array (x,y,t)

    Returns
    -------
    Tx : any
        Traction components in cartesian x direction
    Ty : any
        Traction components in cartesian y direction
    Tn : any
        Traction components in polar n (normal) direction
    Tt : any
        Traction components in polar t (tangential) direction
    '''

    sigma_xx, sigma_yy, sigma_xy = y[:, 2:3], y[:, 3:4], y[:, 4:5] 
    
    normals, cond = calculate_boundary_normals(X,spacetime_domain)

    Tx, Ty, Tn, Tt = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

    return Tx, Ty, Tn, Tt

def strain_rate_2d(x,y):
    '''
    Calculates the strain rate tensor components. 

    Parameters
    ----------
    x : Placeholder (tf)
        contains the placeholder for coordinates of input points and time 
    y : Placeholder (tf)
        contains the placeholder for network output: disp_x, disp_y, sigma_xx, sigma_yy, sigma_xy, velocity_x, velocity_y

    Returns 
    -------
    sigma_xx : Placeholder (tf)
        xx component of stress tensor
    sigma_yy : Placeholder (tf)
        yy component of stress tensor
    sigma_xy : Placeholder (tf)
        xy component of stress tensor
    '''
    eps_xx_rate = dde.grad.jacobian(y, x, i=5, j=0)
    eps_yy_rate = dde.grad.jacobian(y, x, i=6, j=1)
    eps_xy_rate = 1/2*(dde.grad.jacobian(y, x, i=6, j=0)+dde.grad.jacobian(y, x, i=5, j=1))
    
    return eps_xx_rate, eps_yy_rate, eps_xy_rate

def stress_rate_plane_strain(x,y):
    '''
    Calculates the derivative of stress tensor components for plane strain condition.

    Parameters
    ----------
    x : Placeholder (tf)
        contains the placeholder for coordinates of input points
    y : Placeholder (tf)
        contains the placeholder for network output

    Returns 
    -------
    sigma_xx : Placeholder (tf)
        xx component of stress tensor
    sigma_yy : Placeholder (tf)
        yy component of stress tensor
    sigma_xy : Placeholder (tf)
        xy component of stress tensor
    '''
    # get the strain rates for plane strain
    eps_xx_rate, eps_yy_rate, eps_xy_rate = strain_rate_2d(x,y)

    nu,_,_,e_modul = problem_parameters()
    
    # calculate stress rates (hooke's law - plane strain)
    sigma_xx_t = e_modul/((1+nu)*(1-2*nu))*((1-nu)*eps_xx_rate+nu*eps_yy_rate)
    sigma_yy_t = e_modul/((1+nu)*(1-2*nu))*(nu*eps_xx_rate+(1-nu)*eps_yy_rate)
    sigma_xy_t = e_modul/((1+nu)*(1-2*nu))*((1-2*nu)*eps_xy_rate)

    return sigma_xx_t, sigma_yy_t, sigma_xy_t

def get_plane_strain_stress_rate_coupling(x,y):
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
    term_x_velocity : tensor
        difference between predicted velocity and calulated velocity in x direction
    term_y_velocity : tensor
        difference between predicted velocity and calulated velocity in y direction
    '''
    # get stress rates from velocity fields
    sigma_xx_t_v, sigma_yy_t_v, sigma_xy_t_v = stress_rate_plane_strain(x,y)
    # get stress rates from stress fields
    sigma_xx_t_s = dde.grad.jacobian(y, x, i=2, j=2)
    sigma_yy_t_s = dde.grad.jacobian(y, x, i=3, j=2)
    sigma_xy_t_s = dde.grad.jacobian(y, x, i=4, j=2)
    # couple the stress rates
    term_xx_rate = sigma_xx_t_v - sigma_xx_t_s
    term_yy_rate = sigma_yy_t_v - sigma_yy_t_s
    term_xy_rate = sigma_xy_t_v - sigma_xy_t_s
    
    return term_xx_rate, term_yy_rate, term_xy_rate

def get_velocity_coupling_2d(x,y):
    # coupling of displacement and velocities
    u_x_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_y_t = dde.grad.jacobian(y, x, i=1, j=2)
    v_x = y[:, 5:6]
    v_y = y[:, 6:7]
    term_x_velocity = u_x_t - v_x
    term_y_velocity = u_y_t - v_y
    
    return term_x_velocity, term_y_velocity

def pde_mixed_velocity_plane_strain_time_dependent(x,y):
    '''
    Calculates the momentum equation using predicted stresses and velocity fields. 
    It generates the terms for pde of the displacement-velocity mixed-variable formulation in case of plane strain for elastodynamics.
    Additionally, it couples the velocity fields with displacements.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points and time: x, y, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, sigma_xx, sigma_yy, sigma_xy, velocity_x, velocity_y

    Returns
    -------
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    term_xx_rate : tensor
        difference between predicted stress rates and calculated stress rates in xx component
    term_yy_rate : tensor
        difference between predicted stress rates and calculated stress rates in yy component
    term_xy_rate : tensor
        difference between predicted stress rates and calculated stress rates in xy component
    term_x_velocity : tensor
        difference between predicted velocity and calulated velocity in x direction
    term_y_velocity : tensor
        difference between predicted velocity and calulated velocity in y direction
    term_xx : tensor
        difference between predicted stresses and calculated stresses in xx component
    term_yy : tensor
        difference between predicted stresses and calculated stresses in yy component
    term_xy : tensor
        difference between predicted stresses and calculated stresses in xy component
    '''
    # get stress derivatives
    sigma_xx_x = dde.grad.jacobian(y, x, i=2, j=0)
    sigma_yy_y = dde.grad.jacobian(y, x, i=3, j=1)
    sigma_xy_x = dde.grad.jacobian(y, x, i=4, j=0)
    sigma_xy_y = dde.grad.jacobian(y, x, i=4, j=1)
    # get velocity derivatives
    v_x_t = dde.grad.jacobian(y, x, i=5, j=2)
    # get ddu_y/dt2
    v_y_t = dde.grad.jacobian(y, x, i=6, j=2)
    
    # momentum terms
    if body_force_function:
        body_force_x, body_force_y = body_force_function(x)
        momentum_x = sigma_xx_x + sigma_xy_y + body_force_x - rho*v_x_t
        momentum_y = sigma_yy_y + sigma_xy_x + body_force_y - rho*v_y_t
    else:
        momentum_x = sigma_xx_x + sigma_xy_y - rho*v_x_t
        momentum_y = sigma_yy_y + sigma_xy_x - rho*v_y_t
    
   # Coupling of stress rates
    term_xx_rate, term_yy_rate, term_xy_rate = get_plane_strain_stress_rate_coupling(x, y)
    
    # Coupling of velocities
    term_x_velocity, term_y_velocity = get_velocity_coupling_2d(x,y)
    
    # material law
    term_xx, term_yy, term_xy = lin_iso_elasticity_plane_strain(x,y)

    return [momentum_x, momentum_y, term_xx_rate, term_yy_rate, term_xy_rate, term_x_velocity, term_y_velocity, term_xx, term_yy, term_xy]

#################################################################################################################################################################################
# Equations for 3D elastodynamics
#################################################################################################################################################################################
def pde_mixed_3d_time(x, y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for PDE of the mixed-variable formulation in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points an time: x, y, z, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, disp_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz

    Returns
    -------
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    momentum_z : tensor
        momentum terms in z based on derivatives of predicted stresses
    term_xx : tensor
        difference between predicted stresses and calculated stresses in xx component in 3D
    term_yy : tensor
        difference between predicted stresses and calculated stresses in yy component in 3D
    term_zz : tensor
        difference between predicted stresses and calculated stresses in zz component in 3D
    term_xy : tensor
        difference between predicted stresses and calculated stresses in xy component in 3D
    term_yz : tensor
        difference between predicted stresses and calculated stresses in yz component in 3D
    term_xz : tensor
        difference between predicted stresses and calculated stresses in xz component in 3D
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
    
    
    # get ddu_x/dt2
    u_x_tt = dde.grad.hessian(y, x, component=0, i=3, j=3) # component is u_x, tt is the fourth input which is time tt=>ij
    # get ddu_y/dt2
    u_y_tt = dde.grad.hessian(y, x, component=1, i=3, j=3) # component is u_y, tt is the fourth input which is time tt=>ij
    # get ddu_z/dt2
    u_z_tt = dde.grad.hessian(y, x, component=2, i=3, j=3) # component is u_z, tt is the fourth input which is time tt=>ij
    
    # momentum terms
    if body_force_function:
        body_force_x, body_force_y, body_force_z = body_force_function(x)
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_xz_z + body_force_x - rho*u_x_tt
        momentum_y = sigma_yy_y + sigma_xy_x + sigma_yz_z + body_force_y - rho*u_y_tt
        momentum_z = sigma_zz_z + sigma_xz_x + sigma_yz_y + body_force_z - rho*u_z_tt
    else:
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_xz_z - rho*u_x_tt
        momentum_y = sigma_yy_y + sigma_xy_x + sigma_yz_z - rho*u_y_tt
        momentum_z = sigma_zz_z + sigma_xz_x + sigma_yz_y - rho*u_z_tt   

    # Material law
    term_xx, term_yy, term_zz, term_xy, term_yz, term_xz = get_stress_coupling(x, y)

    return [momentum_x, momentum_y, momentum_z, term_xx, term_yy, term_zz, term_xy, term_yz, term_xz]

def get_tractions_mixed_3d_spacetime(x, y, X):
    '''
    Calculates traction components in the mixed formulation. 
    
    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points an time: x, y, z, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, disp_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz
    X : numpy.ndarray
        Network input as numpy array (x,y,z,t)

    Returns
    -------
    Tx : tensor
        Traction components in cartesian x direction
    Ty : tensor
        Traction components in cartesian y direction
    Tz : tensor
        Traction components in cartesian z direction
    Tn : tensor
        Traction components in polar n (normal) direction
    Tt_1 : tensor
        Traction components in polar t_1 (tangential) direction
    Tt_2 : tensor
        Traction components in polar t_2 (tangential) direction
    '''    
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    normals, tangentials_1, tangentials_2, cond = calculate_boundary_normals_3D(X,spacetime_domain)

    Tx, Ty, Tz, Tn, Tt_1, Tt_2 = stress_to_traction_3d(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, normals, tangentials_1, tangentials_2, cond)

    return Tx, Ty, Tz, Tn, Tt_1, Tt_2

#################################################################################################################################################################################
# Equations for 3D elastodynamics in case of stress-velocity mixed formulation approach
#################################################################################################################################################################################
def elastic_strain_rate_3d(x,y):
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
    eps_xx_rate : tensor
        contains the xx components of strain rate tensor in 3D
    eps_yy_rate : tensor
        contains the yy components of strain rate tensor in 3D
    eps_zz_rate : tensor
        contains the zz components of strain rate tensor in 3D
    eps_xy_rate : tensor
        contains the xy components of strain rate tensor in 3D
    eps_yz_rate : tensor
        contains the yz components of strain rate tensor in 3D
    eps_xz_rate : tensor
        contains the xz components of strain rate tensor in 3D
    '''
    # Normal strain rates
    eps_xx_rate = dde.grad.jacobian(y, x, i=9, j=0)
    eps_yy_rate = dde.grad.jacobian(y, x, i=10, j=1)
    eps_zz_rate = dde.grad.jacobian(y, x, i=11, j=2)
    
    # Shear strain rates
    eps_xy_rate = 1/2 * (dde.grad.jacobian(y, x, i=10, j=0) + dde.grad.jacobian(y, x, i=9, j=1))
    eps_yz_rate = 1/2 * (dde.grad.jacobian(y, x, i=11, j=1) + dde.grad.jacobian(y, x, i=10, j=2))
    eps_xz_rate = 1/2 * (dde.grad.jacobian(y, x, i=11, j=0) + dde.grad.jacobian(y, x, i=9, j=2))

    return eps_xx_rate, eps_yy_rate, eps_zz_rate, eps_xy_rate, eps_yz_rate, eps_xz_rate 

def stress_rate_tensor_3d(x,y):
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
    sigma_xx_rate : tensor
        contains the xx components of stress rate tensor in 3D
    sigma_yy_rate : tensor
        contains the yy components of stress rate tensor in 3D
    sigma_zz_rate : tensor
        contains the zz components of stress rate tensor in 3D
    sigma_xy_rate : tensor
        contains the xy components of stress rate tensor in 3D
    sigma_yz_rate : tensor
        contains the yz components of stress rate tensor in 3D
    sigma_xz_rate : tensor
        contains the xz components of stress rate tensor in 3D
    '''
    eps_xx_rate, eps_yy_rate, eps_zz_rate, eps_xy_rate, eps_yz_rate, eps_xz_rate = elastic_strain_rate_3d(x,y)

    nu,lame,shear,e_modul = problem_parameters()
    
    # calculate stress terms (constitutive law)
    factor = e_modul / ((1 + nu) * (1 - 2*nu))
    
    sigma_xx_rate = factor * ((1 - nu)*eps_xx_rate + nu*eps_yy_rate + nu*eps_zz_rate)
    sigma_yy_rate = factor * ((1 - nu)*eps_yy_rate + nu*eps_xx_rate + nu*eps_zz_rate)
    sigma_zz_rate = factor * ((1 - nu)*eps_zz_rate + nu*eps_xx_rate + nu*eps_yy_rate)
    
    sigma_xy_rate = factor * (1 - 2*nu) * eps_xy_rate
    sigma_yz_rate = factor * (1 - 2*nu) * eps_yz_rate
    sigma_xz_rate = factor * (1 - 2*nu) * eps_xz_rate

    return sigma_xx_rate, sigma_yy_rate, sigma_zz_rate, sigma_xy_rate, sigma_yz_rate, sigma_xz_rate

def get_stress_rate_coupling_3d(x,y):
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
    term_xx_rate : tensor
        difference between predicted stress rates and calculated stress rates in xx component
    term_yy_rate : tensor
        difference between predicted stress rates and calculated stress rates in yy component
    term_zz_rate : tensor
        difference between predicted stress rates and calculated stress rates in zz component
    term_xy_rate : tensor
        difference between predicted stress rates and calculated stress rates in xy component
    term_yz_rate : tensor
        difference between predicted stress rates and calculated stress rates in yz component
    term_xz_rate : tensor
        difference between predicted stress rates and calculated stress rates in xz component
    '''
    # get the stress rate from velocity fields
    sigma_xx_rate_v, sigma_yy_rate_v, sigma_zz_rate_v, sigma_xy_rate_v, sigma_yz_rate_v, sigma_xz_rate_v = stress_rate_tensor_3d(x,y)
    
    # get the stress rate from stress fields
    sigma_xx_rate_s = dde.grad.jacobian(y, x, i=3, j=3)
    sigma_yy_rate_s = dde.grad.jacobian(y, x, i=4, j=3)
    sigma_zz_rate_s = dde.grad.jacobian(y, x, i=5, j=3)
    sigma_xy_rate_s = dde.grad.jacobian(y, x, i=6, j=3)
    sigma_yz_rate_s = dde.grad.jacobian(y, x, i=7, j=3)
    sigma_xz_rate_s = dde.grad.jacobian(y, x, i=8, j=3)
    
    term_xx_rate = sigma_xx_rate_v - sigma_xx_rate_s
    term_yy_rate = sigma_yy_rate_v - sigma_yy_rate_s
    term_zz_rate = sigma_zz_rate_v - sigma_zz_rate_s
    term_xy_rate = sigma_xy_rate_v - sigma_xy_rate_s
    term_yz_rate = sigma_yz_rate_v - sigma_yz_rate_s
    term_xz_rate = sigma_xz_rate_v - sigma_xz_rate_s
    
    return term_xx_rate, term_yy_rate, term_zz_rate, term_xy_rate, term_yz_rate, term_xz_rate

def get_velocity_coupling_3d(x,y):
    # coupling of displacement and velocities
    u_x_t = dde.grad.jacobian(y, x, i=0, j=3)
    u_y_t = dde.grad.jacobian(y, x, i=1, j=3)
    u_z_t = dde.grad.jacobian(y, x, i=2, j=3)
    
    v_x = y[:, 9:10]
    v_y = y[:, 10:11]
    v_z = y[:, 11:12]
    
    term_x_velocity = u_x_t - v_x
    term_y_velocity = u_y_t - v_y
    term_z_velocity = u_z_t - v_z
    
    return term_x_velocity, term_y_velocity, term_z_velocity

def pde_mixed_velocity_3d_time(x, y):
    '''
    Calculates the momentum equation using predicted stresses and generates the terms for PDE of the mixed-variable formulation in 3D.

    Parameters
    ----------
    x : Placeholder (tensor)
        contains the placeholder for coordinates of input points an time: x, y, z, t
    y : Placeholder (tensor)
        contains the placeholder for network output: disp_x, disp_y, disp_z, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz

    Returns
    -------
    momentum_x : tensor
        momentum terms in x based on derivatives of predicted stresses
    momentum_y : tensor
        momentum terms in y based on derivatives of predicted stresses
    momentum_z : tensor
        momentum terms in z based on derivatives of predicted stresses
    term_xx_rate : tensor
        difference between predicted stress rates and calculated stress rates in xx component
    term_yy_rate : tensor
        difference between predicted stress rates and calculated stress rates in yy component
    term_zz_rate : tensor
        difference between predicted stress rates and calculated stress rates in zz component
    term_xy_rate : tensor
        difference between predicted stress rates and calculated stress rates in xy component
    term_yz_rate : tensor
        difference between predicted stress rates and calculated stress rates in yz component
    term_xz_rate : tensor
        difference between predicted stress rates and calculated stress rates in xz component
    term_x_velocity : tensor
        difference between predicted velocity and calulated velocity in x direction
    term_y_velocity : tensor
        difference between predicted velocity and calulated velocity in y direction
    term_z_velocity : tensor
        difference between predicted velocity and calulated velocity in z direction
    term_xx : tensor
        difference between predicted stresses and calculated stresses in xx component
    term_yy : tensor
        difference between predicted stresses and calculated stresses in yy component
    term_zz : tensor
        difference between predicted stresses and calculated stresses in zz component
    term_xy : tensor
        difference between predicted stresses and calculated stresses in xy component
    term_yz : tensor
        difference between predicted stresses and calculated stresses in yz component
    term_xz : tensor
        difference between predicted stresses and calculated stresses in xz component
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
    
    # get velocity derivatives
    v_x_t = dde.grad.jacobian(y, x, i=9, j=3)
    # get ddu_y/dt2
    v_y_t = dde.grad.jacobian(y, x, i=10, j=3)
    # get ddu_y/dt2
    v_z_t = dde.grad.jacobian(y, x, i=11, j=3)
    
    # momentum terms
    if body_force_function:
        body_force_x, body_force_y, body_force_z = body_force_function(x)
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_xz_z + body_force_x - rho*v_x_t
        momentum_y = sigma_yy_y + sigma_xy_x + sigma_yz_z + body_force_y - rho*v_y_t
        momentum_z = sigma_zz_z + sigma_xz_x + sigma_yz_y + body_force_z - rho*v_z_t
    else:
        momentum_x = sigma_xx_x + sigma_xy_y + sigma_xz_z - rho*v_x_t
        momentum_y = sigma_yy_y + sigma_xy_x + sigma_yz_z - rho*v_y_t
        momentum_z = sigma_zz_z + sigma_xz_x + sigma_yz_y - rho*v_z_t   

    # Coupling of stress rates
    term_xx_rate, term_yy_rate, term_zz_rate, term_xy_rate, term_yz_rate, term_xz_rate = get_stress_rate_coupling_3d(x, y)
    
    # Coupling of velocities
    term_x_velocity, term_y_velocity, term_z_velocity = get_velocity_coupling_3d(x,y)
    
    # Material law
    term_xx, term_yy, term_zz, term_xy, term_yz, term_xz = get_stress_coupling(x, y)

    return [momentum_x, momentum_y, momentum_z, 
            term_xx_rate, term_yy_rate, term_zz_rate, term_xy_rate, term_yz_rate, term_xz_rate, 
            term_x_velocity, term_y_velocity, term_z_velocity,
            term_xx, term_yy, term_zz, term_xy, term_yz, term_xz]