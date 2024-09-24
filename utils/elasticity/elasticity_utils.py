import deepxde as dde
from utils.geometry.geometry_utils import calculate_boundary_normals

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

    normals, cond = calculate_boundary_normals(X,geom)
    Tx, _, _, _ = stress_to_traction_2d(sigma_xx, sigma_yy, sigma_xy, normals, cond)

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

def calculate_traction_mixed_formulation(x, y, X):
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
        Tx, Ty, Tn, Tt: any
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
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
    
    
    Tx, _, _, _ = calculate_traction_mixed_formulation(x, y, X)

    return Tx

def zero_neumann_y_mixed_formulation(x, y, X):
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
    eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz: tensor
        contains the components of strain tensor in 3D
    '''
    # Normal strains
    eps_xx = dde.grad.jacobian(y, x, i=0, j=0)
    eps_yy = dde.grad.jacobian(y, x, i=1, j=1)
    eps_zz = dde.grad.jacobian(y, x, i=2, j=2)
    
    # Shear strains
    eps_xy = 1/2 * (dde.grad.jacobian(y, x, i=1, j=0) + dde.grad.jacobian(y, x, i=0, j=1))
    eps_xz = 1/2 * (dde.grad.jacobian(y, x, i=2, j=0) + dde.grad.jacobian(y, x, i=0, j=2))
    eps_yz = 1/2 * (dde.grad.jacobian(y, x, i=2, j=1) + dde.grad.jacobian(y, x, i=1, j=2))
    
    return eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz

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
    eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz = get_elastic_strain_3d(x,y)

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
        contains the placeholder for network output

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
        contains the placeholder for network output

    Returns
    -------
    momentum_x, momentum_y, momentum_z, term_x, term_y, term_z, term_xy, term_xz, term_yz: tensor
        momentum_x, momentum_y, momentum_z: momentum terms based on derivatives of predicted stresses
        term_x, term_y, term_z, term_xy, term_xz, term_yz: difference between predicted stresses and calculated stresses in X, Y, Z, XY, XZ, and YZ directions
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

def stress_to_traction_3d(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, normals, cond):
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
    cond : boolean
        Dimensions of stresses and normals have to match. Normals are calculated on the boundary, while stresses are calculated everywhere.

    Returns
    -------
    Tx, Ty, Tz, Tn, Tt, Tn: any
        Traction components in Cartesian (x, y, z) and polar coordinates (n and t)
    '''
    nx = normals[:,0:1]
    ny = normals[:,1:2]
    nz = normals[:,2:3]

    # Calculate the traction components in Cartesian coordinates
    Tx = sigma_xx[cond]*nx + sigma_xy[cond]*ny + sigma_xz[cond]*nz
    Ty = sigma_xy[cond]*nx + sigma_yy[cond]*ny + sigma_yz[cond]*nz
    Tz = sigma_xz[cond]*nx + sigma_yz[cond]*ny + sigma_zz[cond]*nz
    
    # Calculate the traction components in polar coordinates (normal and tangential)
    Tn = Tx*nx + Ty*ny + Tz*nz
    # For tangential components, I have some doubts, so this has to be reconsidered again.
    Tt_x = -Ty*ny - Tz*nz
    Tt_y = Tx*nx - Tz*nz
    Tt_z = Tx*nx + Ty*ny
    
    return Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z

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
        Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z: tensor
            Traction components in cartesian (x,y) and polar coordinates (n (normal) and t (tangential))
    '''    
    sigma_xx =  y[:, 3:4]
    sigma_yy =  y[:, 4:5]
    sigma_zz =  y[:, 5:6]
    sigma_xy =  y[:, 6:7]
    sigma_yz =  y[:, 7:8]
    sigma_xz =  y[:, 8:9]
    
    normals, cond = calculate_boundary_normals(X,geom)

    Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z = stress_to_traction_3d(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz, normals, cond)

    return Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z

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
    
    Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z = get_tractions_mixed_3d(x, y, X)

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
    
    Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z = get_tractions_mixed_3d(x, y, X)

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
    
    Tx, Ty, Tz, Tn, Tt_x, Tt_y, Tt_z = get_tractions_mixed_3d(x, y, X)

    return Tz