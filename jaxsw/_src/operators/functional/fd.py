import typing as tp
import finitediffx as fdx
import jax
import jax.numpy as jnp
# from .padding import generate_forward_padding, generate_central_padding, generate_backward_padding
from jaxtyping import Array
import functools as ft
# TODO: Forward Different Init
# TODO: Backward Difference Init
# TODO: Mixed Difference Init
# TODO: Forward Difference


generate_finitediff_coeffs = fdx._src.utils.generate_finitediff_coeffs
generate_forward_offsets = fdx._src.utils._generate_forward_offsets
generate_central_offsets = fdx._src.utils._generate_central_offsets
generate_backward_offsets = fdx._src.utils._generate_backward_offsets


def difference_slicing(x: Array, axis: int, coeffs: tp.Sequence[int], offsets: tp.Iterable[int]) -> Array:
    
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)
    
    x = sum(
        coeff * sliced(ioffset - offsets[0], size + (ioffset - offsets[-1]))
        for ioffset, coeff in zip(offsets, coeffs)
    )
    return x


def central_difference(
        x: Array, 
        step_size: int, 
        axis: int, 
        derivative: int, 
        coeffs: tp.Iterable[int], 
        offsets: tp.Iterable[int], 
        padding: tp.Optional[tp.Iterable[int]]=None, 
        mode: str="edge"
    ) -> Array:
    
    if padding is not None:
        x = jnp.pad(x, pad_width=padding, mode=mode)
    
    # slicing with coeffficients
    x = difference_slicing(x, axis=axis, coeffs=coeffs, offsets=offsets)
    
    # derivative factor
    x = x / (step_size ** derivative)
    
    return x


def forward_difference(
        x: Array, 
        step_size: int, 
        axis: int, 
        derivative: int, 
        coeffs: tp.Sequence[int], 
        offsets: tp.Sequence[int], 
        padding: tp.Optional[tp.Sequence[int]]=None, 
        mode: str="edge"
    ) -> Array:
    
    if padding is not None:
        x = jnp.pad(x, pad_width=padding, mode=mode)
    
    # slicing with coeffficients
    x = difference_slicing(x, axis=axis, coeffs=coeffs, offsets=offsets)
    
    # derivative factor
    x = x / (step_size ** derivative)
    
    return x

def backward_difference(
        x: Array, 
        step_size: int, 
        axis: int, 
        derivative: int, 
        coeffs: tp.Sequence[int], 
        offsets: tp.Sequence[int], 
        padding: tp.Optional[tp.Sequence[int]]=None, 
        mode: str="edge"
    ) -> Array:
    
    if padding is not None:
        x = jnp.pad(x, pad_width=padding, mode=mode)
    
    # slicing with coeffficients
    x = difference_slicing(x, axis=axis, coeffs=coeffs, offsets=offsets)
    
    # derivative factor
    x = x / (step_size ** derivative)
    
    return x


# def fd_central_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
#     # get offsets and kernel
#     offsets = fdx._src.utils._generate_central_offsets(derivative=derivative, accuracy=accuracy)
#     kernel = fdx._src.utils.generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    
#     # add kernel dims
#     kernel = _add_kernel_dims(kernel, len(dx))
    
#     # add dx
#     kernel = kernel / dx[axis]
    
#     # get padding
#     padding = generate_central_padding(derivative=derivative, accuracy=accuracy)
    
#     return kernel, padding


# def fd_forward_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
#     # get offsets and kernel
#     offsets = fdx._src.utils._generate_forward_offsets(derivative=derivative, accuracy=accuracy)
#     kernel = fdx._src.utils.generate_finitediff_coeffs(offsets, derivative=derivative)
    
#     # add kernel dims
#     kernel = _add_kernel_dims(kernel, len(dx))
    
#     # add dx
#     kernel = kernel / dx[axis]
    
#     # get padding
#     padding = generate_forward_padding(derivative=derivative, accuracy=accuracy)
    
#     return kernel, padding


# def fd_backward_init(dx: tp.Iterable[float], axis: int=0, accuracy: int=2, derivative: int=1) -> tp.Tuple:
    
    
#     # get offsets and kernel
#     offsets = fdx._src.utils._generate_backward_offsets(derivative=derivative, accuracy=accuracy)
#     kernel = fdx._src.utils.generate_finitediff_coeffs(offsets, derivative=derivative)
    
#     # add kernel dims
#     kernel = _add_kernel_dims(kernel, len(dx))
    
#     # add dx
#     kernel = kernel / dx[axis]
    
#     # get padding
#     padding = generate_backward_padding(derivative=derivative, accuracy=accuracy)
    
#     return kernel, padding


# def _add_kernel_dims(kernel, ndim):
    
#     if ndim > 1:
#         for _ in range(ndim - 1):
#             kernel = rearrange(kernel, "... -> ... 1")
        
#         # move kernel to correct axis
#         kernel = np.moveaxis(kernel, -1, axis)
        
#     return kernel
    