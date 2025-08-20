import deepxde as dde
from deepxde.backend import get_preferred_backend
from deepxde.backend import tf, torch

# global variables
lame = None
shear = None
nu = None
e_modul = None
stress_state = "plane_strain"

def compute_elastic_properties():
    '''
    Computes all elastic parameters given any two.

    Parameters
    ----------
    e_modul : float, optional
        Young's modulus
    nu : float, optional
        Poisson's ratio
    shear : float, optional
        Shear modulus (mu)
    lame : float, optional
        Lame's first parameter (lambda)

    Returns
    -------
    nu : float
        Poisson's ratio
    lame : float
        Lame's parameter (lambda)
    shear : float
        Shear modulus (mu)
    e_modul : float
        Young's modulus
    '''
    global e_modul, nu, shear, lame
    
    known = {
        'e_modul': e_modul,
        'nu': nu,
        'shear': shear,
        'lame': lame
    }

    num_known = sum(v is not None for v in known.values())
    if num_known < 2:
        raise ValueError("Please provide at least two parameters among e_modul, nu, shear, and lame.")

    # Case 1: e_modul and nu are known
    if e_modul is not None and nu is not None:
        shear = e_modul / (2 * (1 + nu))
        lame = e_modul * nu / ((1 + nu) * (1 - 2 * nu))

    # Case 2: shear and nu are known
    elif shear is not None and nu is not None:
        e_modul = 2 * shear * (1 + nu)
        lame = 2 * shear * nu / (1 - 2 * nu)

    # Case 3: e_modul and shear are known
    elif e_modul is not None and shear is not None:
        nu = e_modul / (2 * shear) - 1
        lame = shear * (e_modul - 2 * shear) / (3 * shear - e_modul)

    # Case 4: lame and shear are known
    elif lame is not None and shear is not None:
        e_modul = shear * (3 * lame + 2 * shear) / (lame + shear)
        nu = lame / (2 * (lame + shear))

    # Case 5: e_modul and lame are known
    elif e_modul is not None and lame is not None:
        nu = lame / (e_modul - lame)
        shear = e_modul / (2 * (1 + nu))

    # Case 6: lame and nu are known
    elif lame is not None and nu is not None:
        shear = lame * (1 - 2 * nu) / (2 * nu)
        e_modul = 2 * shear * (1 + nu)

    return nu, lame, shear, e_modul

def bkd_log(x):
    backend_name = get_preferred_backend()
    if (backend_name=="tensorflow.compat.v1") or (backend_name=="tensorflow"):
        # return tf.math.log(x)
        return tf.math.log(tf.math.maximum(x, 1e-8))
    elif backend_name=="pytorch":
        return torch.log(torch.maximum(x, torch.tensor(1e-8, dtype=x.dtype, device=x.device)))

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

def matrix_determinant_3D(a11, a12, a13,
                          a21, a22, a23,
                          a31, a32, a33):
    # Calculate the determinant of the 3x3 matrix using cofactor expansion
    determinant = (
        a11 * (a22 * a33 - a23 * a32)
      - a12 * (a21 * a33 - a23 * a31)
      + a13 * (a21 * a32 - a22 * a31)
    )
    return determinant

def matrix_inverse_3D(a11, a12, a13,
                      a21, a22, a23,
                      a31, a32, a33):
    # Compute determinant
    det = (a11 * (a22 * a33 - a23 * a32)
         - a12 * (a21 * a33 - a23 * a31)
         + a13 * (a21 * a32 - a22 * a31))
    
    # if det == 0:
    #     raise ValueError("The matrix is singular and does not have an inverse.")

    # Compute cofactors and divide by determinant
    inv11 = (a22 * a33 - a23 * a32) / det
    inv12 = (a13 * a32 - a12 * a33) / det
    inv13 = (a12 * a23 - a13 * a22) / det

    inv21 = (a23 * a31 - a21 * a33) / det
    inv22 = (a11 * a33 - a13 * a31) / det
    inv23 = (a13 * a21 - a11 * a23) / det

    inv31 = (a21 * a32 - a22 * a31) / det
    inv32 = (a12 * a31 - a11 * a32) / det
    inv33 = (a11 * a22 - a12 * a21) / det

    return inv11, inv12, inv13, inv21, inv22, inv23, inv31, inv32, inv33


def deformation_gradient_2D(x, y):

    f_xx = dde.grad.jacobian(y, x, i=0, j=0) + 1
    f_yy = dde.grad.jacobian(y, x, i=1, j=1) + 1
    f_xy = dde.grad.jacobian(y, x, i=0, j=1)
    f_yx = dde.grad.jacobian(y, x, i=1, j=0)

    return f_xx, f_yy, f_xy, f_yx

def deformation_gradient_3D(x, y):
    # x: input coordinates (symbolic), shape (None, 3)
    # y: predicted displacement (u, v, w), shape (None, 3)

    # Diagonal terms (∂u_i/∂x_i + 1)
    f_xx = dde.grad.jacobian(y, x, i=0, j=0) + 1  # ∂u/∂x
    f_yy = dde.grad.jacobian(y, x, i=1, j=1) + 1  # ∂v/∂y
    f_zz = dde.grad.jacobian(y, x, i=2, j=2) + 1  # ∂w/∂z

    # Off-diagonal terms
    f_xy = dde.grad.jacobian(y, x, i=0, j=1)  # ∂u/∂y
    f_xz = dde.grad.jacobian(y, x, i=0, j=2)  # ∂u/∂z

    f_yx = dde.grad.jacobian(y, x, i=1, j=0)  # ∂v/∂x
    f_yz = dde.grad.jacobian(y, x, i=1, j=2)  # ∂v/∂z

    f_zx = dde.grad.jacobian(y, x, i=2, j=0)  # ∂w/∂x
    f_zy = dde.grad.jacobian(y, x, i=2, j=1)  # ∂w/∂y

    return f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy

def strain_energy_neo_hookean_2d(x, y):
    # deformation gradient
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)
    
    # construct C = F^T F
    C_xx = f_xx * f_xx + f_yx * f_yx
    C_xy = f_xx * f_xy + f_yx * f_yy
    C_yx = f_xy * f_xx + f_yy * f_yx
    C_yy = f_xy * f_xy + f_yy * f_yy
    
    if stress_state == "plane_strain":
        # First invarient
        f_zz = 1
        I_1 = C_xx + C_yy + f_zz*f_zz
        # determinant
        det_f = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)
    else:
        raise NotImplemented("Only plane-strain is implemented, thus, please switch to 'plane_strain'.")
        # It is quite challeging to implement the plane stress case: https://scicomp.stackexchange.com/questions/42177/finite-element-modelling-of-hyperelastic-material-under-2d-plane-strain-conditio
    W = 0.5 * shear * (I_1 - 3) - shear * bkd_log(det_f) + 0.5 * lame * bkd_log(det_f)**2

    return W

def strain_energy_neo_hookean_3d(x, y):
    # Deformation gradient (3x3)
    f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = deformation_gradient_3D(x, y)

    # Construct C = F^T F (right Cauchy-Green tensor)
    C_xx = f_xx * f_xx + f_yx * f_yx + f_zx * f_zx
    C_yy = f_xy * f_xy + f_yy * f_yy + f_zy * f_zy
    C_zz = f_xz * f_xz + f_yz * f_yz + f_zz * f_zz

    I_1 = C_xx + C_yy + C_zz  # first invariant of C

    # Determinant of F
    det_f = matrix_determinant_3D(
        f_xx, f_xy, f_xz,
        f_yx, f_yy, f_yz,
        f_zx, f_zy, f_zz
    )
    # print(bkd_log(det_f)[-1])
    # Strain energy
    W = 0.5 * shear * (I_1 - 3) - shear * bkd_log(det_f) + 0.5 * lame * bkd_log(det_f)**2

    return W

def second_piola_stress_tensor_2D(x, y):
    # deformation gradient
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)

    # Right Cauchy-Green tensor: C = F^T * F
    C_xx = f_xx**2 + f_yx**2
    C_xy = f_xx * f_xy + f_yx * f_yy
    C_yy = f_xy**2 + f_yy**2

    # Invert C (2x2 matrix)
    det_C = C_xx * C_yy - C_xy**2
    Cinv_xx =  C_yy / det_C
    Cinv_yy =  C_xx / det_C
    Cinv_xy = -C_xy / det_C
    Cinv_yx = Cinv_xy

    # Jacobian
    f_det = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)

    # Second Piola-Kirchhoff stress (compressible Neo-Hookean)
    s_xx = shear * (1 - Cinv_xx) + lame * bkd_log(f_det) * Cinv_xx
    s_yy = shear * (1 - Cinv_yy) + lame * bkd_log(f_det) * Cinv_yy
    s_xy = shear * (0 - Cinv_xy) + lame * bkd_log(f_det) * Cinv_xy
    s_yx = s_xy

    return s_xx, s_yy, s_xy, s_yx


def first_piola_stress_tensor_2D(x, y):
    s_xx, s_yy, s_xy, s_yx = second_piola_stress_tensor_2D(x, y)
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)

    # P = F * S
    p_xx = f_xx * s_xx + f_xy * s_yx
    p_yy = f_yx * s_xy + f_yy * s_yy
    p_xy = f_xx * s_xy + f_xy * s_yy
    p_yx = f_yx * s_xx + f_yy * s_yx

    return p_xx, p_yy, p_xy, p_yx


def cauchy_stress_2D(x, y):
    f_xx, f_yy, f_xy, f_yx = deformation_gradient_2D(x, y)
    p_xx, p_yy, p_xy, p_yx = first_piola_stress_tensor_2D(x, y)

    f_det = matrix_determinant_2D(f_xx, f_yy, f_xy, f_yx)

    # σ = (1/J) * P * F^T
    T_xx = (1 / f_det) * (p_xx * f_xx + p_xy * f_xy)
    T_xy = (1 / f_det) * (p_xx * f_yx + p_xy * f_yy)
    T_yx = (1 / f_det) * (p_yx * f_xx + p_yy * f_xy)
    T_yy = (1 / f_det) * (p_yx * f_yx + p_yy * f_yy)

    return T_xx, T_yy, T_xy, T_yx

def second_piola_stress_tensor_3D(x, y):
    # Deformation gradient
    f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = deformation_gradient_3D(x, y)

    # Right Cauchy-Green tensor C = F^T * F
    C_xx = f_xx**2 + f_yx**2 + f_zx**2
    C_yy = f_xy**2 + f_yy**2 + f_zy**2
    C_zz = f_xz**2 + f_yz**2 + f_zz**2
    C_xy = f_xx * f_xy + f_yx * f_yy + f_zx * f_zy
    C_xz = f_xx * f_xz + f_yx * f_yz + f_zx * f_zz
    C_yz = f_xy * f_xz + f_yy * f_yz + f_zy * f_zz

    # Inverse of C
    det_C = matrix_determinant_3D(C_xx, C_xy, C_xz,
                                  C_xy, C_yy, C_yz,
                                  C_xz, C_yz, C_zz)

    Cinv = matrix_inverse_3D(C_xx, C_xy, C_xz,
                             C_xy, C_yy, C_yz,
                             C_xz, C_yz, C_zz)

    # Jacobian
    f_det = matrix_determinant_3D(f_xx, f_xy, f_xz,
                                  f_yx, f_yy, f_yz,
                                  f_zx, f_zy, f_zz)

    # Stress components
    s_xx = shear * (1 - Cinv[0]) + lame * bkd_log(f_det) * Cinv[0]
    s_xy = shear * (0 - Cinv[1]) + lame * bkd_log(f_det) * Cinv[1]
    s_xz = shear * (0 - Cinv[2]) + lame * bkd_log(f_det) * Cinv[2]
    s_yx = shear * (0 - Cinv[3]) + lame * bkd_log(f_det) * Cinv[3]
    s_yy = shear * (1 - Cinv[4]) + lame * bkd_log(f_det) * Cinv[4]
    s_yz = shear * (0 - Cinv[5]) + lame * bkd_log(f_det) * Cinv[5]
    s_zx = shear * (0 - Cinv[6]) + lame * bkd_log(f_det) * Cinv[6]
    s_zy = shear * (0 - Cinv[7]) + lame * bkd_log(f_det) * Cinv[7]
    s_zz = shear * (1 - Cinv[8]) + lame * bkd_log(f_det) * Cinv[8]

    return s_xx, s_yy, s_zz, s_xy, s_yx, s_xz, s_zx, s_yz, s_zy

def first_piola_stress_tensor_3D(x, y):
    s_xx, s_yy, s_zz, s_xy, s_yx, s_xz, s_zx, s_yz, s_zy = second_piola_stress_tensor_3D(x, y)
    f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = deformation_gradient_3D(x, y)

    # P = F * S
    p_xx = f_xx * s_xx + f_xy * s_yx + f_xz * s_zx
    p_xy = f_xx * s_xy + f_xy * s_yy + f_xz * s_zy
    p_xz = f_xx * s_xz + f_xy * s_yz + f_xz * s_zz

    p_yx = f_yx * s_xx + f_yy * s_yx + f_yz * s_zx
    p_yy = f_yx * s_xy + f_yy * s_yy + f_yz * s_zy
    p_yz = f_yx * s_xz + f_yy * s_yz + f_yz * s_zz

    p_zx = f_zx * s_xx + f_zy * s_yx + f_zz * s_zx
    p_zy = f_zx * s_xy + f_zy * s_yy + f_zz * s_zy
    p_zz = f_zx * s_xz + f_zy * s_yz + f_zz * s_zz

    return p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy

def cauchy_stress_3D(x, y):
    f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = deformation_gradient_3D(x, y)
    p_xx, p_yy, p_zz, p_xy, p_yx, p_xz, p_zx, p_yz, p_zy = first_piola_stress_tensor_3D(x, y)

    f_det = matrix_determinant_3D(f_xx, f_xy, f_xz,
                                  f_yx, f_yy, f_yz,
                                  f_zx, f_zy, f_zz)

    # σ = (1/J) * P * F^T
    T_xx = (1 / f_det) * (p_xx * f_xx + p_xy * f_xy + p_xz * f_xz)
    T_xy = (1 / f_det) * (p_xx * f_yx + p_xy * f_yy + p_xz * f_yz)
    T_xz = (1 / f_det) * (p_xx * f_zx + p_xy * f_zy + p_xz * f_zz)

    T_yx = (1 / f_det) * (p_yx * f_xx + p_yy * f_xy + p_yz * f_xz)
    T_yy = (1 / f_det) * (p_yx * f_yx + p_yy * f_yy + p_yz * f_yz)
    T_yz = (1 / f_det) * (p_yx * f_zx + p_yy * f_zy + p_yz * f_zz)

    T_zx = (1 / f_det) * (p_zx * f_xx + p_zy * f_xy + p_zz * f_xz)
    T_zy = (1 / f_det) * (p_zx * f_yx + p_zy * f_yy + p_zz * f_yz)
    T_zz = (1 / f_det) * (p_zx * f_zx + p_zy * f_zy + p_zz * f_zz)

    return T_xx, T_yy, T_zz, T_xy, T_yx, T_xz, T_zx, T_yz, T_zy

    