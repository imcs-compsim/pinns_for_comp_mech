import deepxde.backend as bkd


def determinant(tensor):
    """Calculate the determinant of a given tensor.
    
    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensors for which a determinant is supposed to be computed.

    Returns
    -------
    The (batch of) tensor determinant(s).
    """
    assert bkd.ndim(tensor) == 3, "determinant() requires a batch of rank 2 tensors."
    return bkd.lib.linalg.det(tensor)

def identity(dim):
    """Generate an identity tensor of a given dimension.

    Parameters
    ----------
    dim: int
        The dimension of the identity tensor.

    Returns
    -------
    An identity tensor of the given dimension.
    """
    return bkd.lib.eye(dim)

def identity_like(tensor):
    """ Generate an identity tensor with the same shape as the input tensor.

    Parameters
    ----------
    tensor: Tensor
        A (batch of) tensor(s) to determine the shape of the identity tensor.

    Returns
    -------
    An identity tensor with the same shape as each tensor in the input batch.
    """
    assert bkd.ndim(tensor) == 3, "identity_like() requires a batch of rank 2 tensors."
    return identity(bkd.shape(tensor)[-1])

def inverse(tensor):
    """Calculates the inverse of a given tensor.

    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensor(s) to be inverted.

    Returns
    -------
    The (batch of) inverted tensor(s).
    """
    assert bkd.ndim(tensor) == 3, "inverse() requires a batch of rank 2 tensors."
    return bkd.lib.linalg.inv(tensor)

def outer_vec_mat_prod(vec, mat):
    r"""Calculates the outer product of a vector with a matrix.

    Given a vector :math:`\mathbf{v}\in\mathbb{R}^l` and a matrix 
    :math:`\mathbf{m}\in\mathbb{R}^{m\times n}`, this function computes the 
    outer product resulting in the tensor 
    :math:`\mathbf{T}\in\mathbb{R}^{l\times m\times n}` with components
    .. math::
            T_{ijk} = v_i m_{jk}.
    
    .. note::
            In theory this function should also account for a batched outer 
            product, but this functionality is not tested yet and thus disabled
            for now.

    Parameters
    ----------
    vec: Tensor
        The vector to be multiplied.
    mat: Tensor
        The matrix to be multiplied.
    
    Returns
    -------
    The (batch of) outer product(s) of the vector(s) with the matrix(s).
    """
    assert bkd.ndim(vec) == 1 or bkd.ndim(vec) == 2, "outer_vec_mat_prod() requires a (batch of) rank 1 tensors (vectors)."
    assert bkd.ndim(mat) == 2 or bkd.ndim(mat) == 3, "outer_vec_mat_prod() requires (a batch of) rank 2 tensors (matrices)."
    return bkd.lib.einsum("...i,...jk->...ijk", vec, mat)

def trace(tensor):
    """Calculates the trace of a given tensor.
    
    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensor(s) for which a trace is supposed to be computed.
    
    Returns
    -------
    The (batch of) tensor trace(s).
    """
    assert bkd.ndim(tensor) == 3, "trace() requires a batch of rank 2 tensors."
    return bkd.lib.einsum("...ii->...", tensor)

def transpose(tensor):
    """Calculates the transpose of a given tensor.

    Parameters
    ----------
    tensor: Tensor
        The (batch of) tensor(s) to be transposed.

    Returns
    -------
    The (batch of) transposed tensor(s).
    """
    assert bkd.ndim(tensor) == 3, "transpose() requires a batch of rank 2 tensors."
    return bkd.transpose(tensor, axes=[0, 2, 1])