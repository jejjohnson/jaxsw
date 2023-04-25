import jax.numpy as jnp
from jaxtyping import Array

from jaxsw._src.utils.constants import OMEGA, R_EARTH


def beta_plane(lat: Array, omega: float = OMEGA, radius: float = R_EARTH) -> Array:
    """Beta-Plane Approximation from the mean latitude

    Equation:
        β = (1/R) 2Ω cosθ

    Args:
        lat (Array): the mean latitude [degrees]
        omega (float): the rotation (default=...)
        radius (float): the radius of the Earth (default=...)

    Returns:
        beta (Array): the beta plane parameter
    """
    lat = jnp.deg2rad(lat)
    return 2 * omega * jnp.cos(lat) / radius


def coriolis_param(lat: Array, omega: float = OMEGA) -> Array:
    """The Coriolis parameter

    Equation:
        f = 2Ω sinθ

    Args:
        lat (Array): the mean latitude [degrees]
        omega (Array): the roation (default=...)

    Returns:
        Coriolis (Array): the coriolis parameter
    """
    lat = jnp.deg2rad(lat)
    return 2.0 * omega * jnp.sin(lat)
