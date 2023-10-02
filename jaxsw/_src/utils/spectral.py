import math
import jax.numpy as jnp
from jaxtyping import Array


def calculate_fft_freq(Nx: int, Lx: float = 2.0 * math.pi) -> Array:
    """a helper function to generate 1D FFT frequencies

    Args:
        Nx (int): the number of points for the grid
        Lx (float): the distance for the points along the grid

    Returns:
        freq (Array): the 1D fourier frequencies
    """
    # return jnp.fft.fftfreq(n=Nx, d=Lx / (2.0 * math.pi * Nx))
    return (2 * math.pi / Lx) * jnp.fft.fftfreq(n=Nx, d=Lx / (Nx * 2.0 * math.pi))
