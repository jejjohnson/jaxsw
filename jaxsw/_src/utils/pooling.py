import typing as tp
import kernex as kex
from jaxtyping import Array
import jax.numpy as jnp


def avg_pool_2d(
    u: Array, 
    kernel_size: tp.Iterable, 
    strides: tp.Iterable,
    padding: tp.Iterable
) -> Array:
    

    @kex.kmap(
        kernel_size=kernel_size, 
        strides=strides, 
        padding=padding
    )
    def apply_kernel(x):
        return jnp.mean(x)
    
    return apply_kernel(u)