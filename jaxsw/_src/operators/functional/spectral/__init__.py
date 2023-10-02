import typing as tp
import functools as ft
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
from jaxsw._src.utils.spectral import calculate_fft_freq
from jaxsw._src.fields.spectral import SpectralField
import math
from plum import dispatch


def fft_transform(u: Array, axis: int = -1, inverse: bool = False) -> Array:
    """the FFT transformation (forward and inverse)

    Args:
        u (Array): the input array to be transformed
        axis (int, optional): the axis to do the FFT transformation. Defaults to -1.
        scale (float, optional): the scaler value to rescale the inputs.
            Defaults to 1.0.
        inverse (bool, optional): whether to do the forward or inverse transformation.
            Defaults to False.

    Returns:
        u (Array): the transformation that maybe forward or backwards
    """
    if inverse:
        return jnp.fft.ifft(a=u, axis=axis)
    else:
        return jnp.fft.fft(a=u, axis=axis)


def spectral_difference(
    fu: Array, k_vec: Array, axis: int = 0, derivative: int = 1
) -> Array:
    """the difference method in spectral space

    Args:
        fu (Array): the array in spectral space to take the difference
        k_vec (Array): the parameters to take the finite difference
        axis (int, optional): the axis of the multidim array, u, to
            take the finite difference. Defaults to 0.
        derivative (int, optional): the number of derivatives to take.
            Defaults to 1.

    Returns:
        dfu (Array): the finite difference method
    """

    # reshape axis
    fu = jnp.moveaxis(fu, axis, -1)

    # do difference method
    dfu = fu * (1j * k_vec) ** float(derivative)

    # re-reshape axis
    dfu = jnp.moveaxis(dfu, -1, axis)

    return dfu


def difference(
    u: Array, k_vec: Array, axis: int = 0, derivative: int = 1, real: bool = True
) -> Array:
    """spectral difference from the real space

    Args:
        u (Array): the array in real space to do the difference
        k_vec (Array): the vector of frequencies
        axis (int, optional): the axis to do the difference.
            Defaults to 0.
        derivative (int, optional): the number of the derivatives to do.
            Defaults to 1.
        real (bool, optional): to return the array in real or complex.
            Defaults to True.

    Returns:
        du (Array): the resulting array with the derivatives
    """
    # forward transformation
    fu = fft_transform(u, axis=axis, inverse=False)

    # difference operator
    dfu = spectral_difference(fu, k_vec=k_vec, axis=axis, derivative=derivative)

    # inverse transformation
    du = fft_transform(dfu, axis=axis, inverse=True)

    # return real components
    if real:
        return jnp.real(du)
    else:
        return du


def difference_field(
    u: SpectralField, axis: int = 0, derivative: int = 1, real: bool = True
) -> SpectralField:
    # forward transformation
    fu = fft_transform(u.values, axis=axis, inverse=False)

    # difference operator
    dfu = spectral_difference(fu, k_vec=u.k_vec[axis], axis=axis, derivative=derivative)

    # inverse transformation
    du = fft_transform(dfu, axis=axis, inverse=True)

    # return real components
    if real:
        du = jnp.real(du)

    return SpectralField(values=du, domain=u.domain)


def elliptical_operator_2D(
    k_vec: tp.Iterable[Array],
    order: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Array:
    msg = "Error: the k_vec should be 2D"
    assert len(tuple(k_vec)) == 2, msg

    # expand each of the dimensions
    ks = [jnp.expand_dims(array, axis=i) for i, array in enumerate(k_vec)]

    # sum each of dimensions
    ksq = ft.reduce(lambda x, y: x ** (2 * order) + y ** (2 * order), ks)

    # reshape and add beta
    ksq = alpha * ksq.T + beta

    return ksq


def elliptical_inversion_2D(u: SpectralField) -> SpectralField:
    msg = "Error: the spectral field should be 2D"
    assert len(u.k_vec) == u.domain.ndim == 2, msg

    # calculate scalar quantity
    ksq = elliptical_operator_2D(u.k_vec)

    uh_values = jnp.fft.fftn(u[:], axes=(-2, -1)) / sum(u.domain.Nx)

    # do inversion
    invksq = 1.0 / ksq
    invksq = invksq.at[0, 0].set(1.0)

    uh_values = -invksq * uh_values

    u_values = jnp.real(sum(u.domain.Nx) * jnp.fft.ifftn(uh_values, axes=(-2, -1)))

    return SpectralField(values=u_values, domain=u.domain)


def laplacian_field(
    u: SpectralField, alpha: Array = 1.0, beta: Array = 0.0, order: float = 1.0
) -> SpectralField:
    # get laplacian vector
    ksq = elliptical_operator_2D(u.k_vec, alpha=alpha, beta=beta, order=order)
    # do fft
    uh = jnp.fft.fftn(u.values)
    uh_lap = ksq * uh
    # do inverse FFT
    u_lap = jnp.real(jnp.fft.ifftn(uh_lap))
    # initialize a new field
    return SpectralField(u_lap, u.domain)
