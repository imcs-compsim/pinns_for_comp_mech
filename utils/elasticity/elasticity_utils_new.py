import deepxde as dde
import deepxde.backend as bkd

from utils.linalg.linalg_utils import transpose


def deformation_gradient(disp, coords, i=0, j=None):
    """Compute a (batch of) deformation gradient tensor(s) from a (batch of) 
    displacement field(s) and a (batch of) coordinate(s).
    
    Parameters
    ----------
    disp: Tensor
        The (batch of) displacement field(s).
    coords: Tensor
        The (batch of) coordinate(s).
    
    Returns
    -------
    _deformation_gradient: Tensor
        The (batch of) deformation gradient tensor(s).
    """
    _disp_grad = dde.grad.jacobian(disp, coords, i=i, j=j)
    _deformation_gradient = _disp_grad + bkd.lib.eye(bkd.shape(_disp_grad)[-1])
    return _deformation_gradient


def strain(disp_grad, linearize=False):
    """Compute a (batch of) strain tensor(s) from a (batch of) displacement gradient(s).
    
    Parameters
    ----------
    disp_grad: Tensor
        The (batch of) displacement gradient tensor(s).
    linearize: bool
        A flag indicating whether to use the linearized definition of the strain tensor.
    
    Returns
    -------
    _strain: Tensor
        The (batch of) strain tensor(s).
    """
    _disp_grad_T = transpose(disp_grad)
    _strain = 0.5 * (disp_grad + _disp_grad_T)
    if not linearize:
        _strain += 0.5 * bkd.matmul(_disp_grad_T, disp_grad)
    return _strain