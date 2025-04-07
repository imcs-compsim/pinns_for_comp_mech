import deepxde as dde
import deepxde.backend as bkd


def vector_jacobian(v, x, _j=None):
    r"""Compute the (batch of) gradient(s) from a (batch of) vector field(s) 
    :math:`\mathbf{v}` and a (batch of) coordinate(s) :math:`\mathbf{x}`:

    .. math::

            (\nabla \mathbf{v})_{i,j} = \frac{\partial v_i}{\partial x_j}
    
    Parameters
    ----------
    v: Tensor
        The (batch of) vector field(s).
    x: Tensor
        The (batch of) coordinate(s).
    _j: int
        The index of the coordinate with respect to which a derivative is 
        computed. If ``None``, derivatives are computed with respect to all 
        coordinates, i.e., a full jacobian is returned. If an integer is 
        provided, this function computes the partial derivative 
        :math:`\frac{\partial \mathbf{v}}{\partial x_j}`, corresponding to a 
        column vector of the jacobian (this is NOT the gradient).
    
    Returns
    -------
    _grad: Tensor
        The (batch of) jacobian(s).
    """
    assert bkd.ndim(v) == 2, \
        "vector_jacobian() requires the vector field to be a batch of rank 1 tensors (vectors)."
    assert bkd.ndim(x) == 2, \
        "vector_jacobian() requires the coordinates to be a batch of rank 1 tensors (vectors)."
    # If _j=None, we compute the gradients for the vector field component-wise 
    # and store the gradients of each component functions in a temporary list.
    # If an integer value for _j is provided, we compute the corresponding 
    # partial derivative of each component function (_j-th entry of the 
    # gradient) and store the partial derivatives in a temporary list.
    _grad_list = []
    for _i in range(bkd.shape(v)[-1]):
        _grad_i = dde.grad.jacobian(v, x, i=_i, j=_j)
        _grad_list.append(_grad_i)
    # consider the computed gradients as row vectors of the jacobian matrix 
    # and stack them accordingly
    _grad = bkd.stack(_grad_list, axis=1)
    return _grad


def tensor_divergence(T, x):
    r"""Compute the (batch of) divergence(s) from a (batch of) tensor field(s)
    :math:`\mathbf{T}` and a (batch of) coordinate(s) :math:`\mathbf{x}`:

    .. math::

           (\nabla \cdot \mathbf{T})_{i} = \frac{\partial T_{ij}}{\partial x_j}\mathbf{e}_i

    where we employ the summation convention, i.e., repeated indices on the
    right hand side imply a summation.

    Parameters
    ----------
    T: Tensor
        The (batch of) tensor field(s).
    x: Tensor
        The (batch of) coordinate(s).

    Returns
    -------
    _tensor_div: Tensor
        The (batch of) tensor divergences.
    """
    assert bkd.ndim(T) == 3, \
        "tensor_divergence() requires the tensor field to be a batch of rank 2 tensors (matrices)."
    assert bkd.ndim(x) == 2, \
        "tensor_divergence() requires the coordinates to be a batch of rank 1 tensors (vectors)."
    # we compute the corresponding partial derivative of each for the tensor 
    # field column and sum them directly in the result field
    _tensor_div = bkd.zeros_like(x)
    # iterate over the columns of the tensors
    for _j in range(bkd.shape(T)[-1]):
        # compute the partial derivative of the current column vector 
        # (this contains dT_ij/dx_j for all i)
        _col_grad = vector_jacobian(T[:,:,_j], x, _j)
        # the shape of _col_grad is (nbatch, 1, nx), so we reshape it
        _tensor_div += bkd.reshape(_col_grad, bkd.shape(x))  # has shape (nbatch, nx)

    return _tensor_div