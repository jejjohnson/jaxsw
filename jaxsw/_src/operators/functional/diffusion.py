import finitediffx as fdx
from jaxtyping import Array


def diffusion_1D(
    u: Array,
    diffusivity: Array,
    step_size: Array,
    axis: int = 0,
    method: str = "central",
    accuracy: int = 1,
):
    """simple 2D diffusion scheme using central finite
    difference.

        Diffusion = ν ∂²u/∂x²

    Args:
        u (Array): the field
        diffusivity (Array): the field or constant for the diffusivity coefficient
        step_size (Array): the stepsize for the FD scheme
        axis (int, optional): the axis to operate the FD. Defaults to 0.
        method (str, optional): the method for FD. Defaults to "central".
        accuracy (int, optional): the accuracy for the FD scheme. Defaults to 1.

    Returns:
        Array: the RHS for the advection term
    """
    d2u_dx2 = fdx.difference(
        u,
        axis=axis,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
        derivative=2,
    )

    return diffusivity * d2u_dx2


def diffusion_2D(
    u: Array,
    diffusivity: Array,
    step_size: Array,
    method: str = "central",
    accuracy: int = 1,
):
    """simple 2D diffusion scheme using central finite
    difference.

        Diffusion = ν (∂²u/∂x² + ∂²u/∂y²)

    Args:
        u (Array): the field
        diffusivity (Array): the field or constant for the diffusivity coefficient
        step_size (Array): the stepsize for the FD scheme
        axis (int, optional): the axis to operate the FD. Defaults to 0.
        method (str, optional): the method for FD. Defaults to "central".
        accuracy (int, optional): the accuracy for the FD scheme. Defaults to 1.

    Returns:
        Array: the RHS for the advection term
    """

    u_lap = fdx.laplacian(u, method="central", accuracy=1, step_size=step_size)

    return diffusivity * u_lap


def diffusion_3D(
    u: Array,
    diffusivity: Array,
    step_size: Array,
    method: str = "central",
    accuracy: int = 1,
):
    """simple 2D diffusion scheme using central finite
    difference.

        Diffusion = ν (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)

    Args:
        u (Array): the field
        diffusivity (Array): the field or constant for the diffusivity coefficient
        step_size (Array): the stepsize for the FD scheme
        axis (int, optional): the axis to operate the FD. Defaults to 0.
        method (str, optional): the method for FD. Defaults to "central".
        accuracy (int, optional): the accuracy for the FD scheme. Defaults to 1.

    Returns:
        Array: the RHS for the advection term
    """
    d2u_dx2 = fdx.difference(
        u,
        axis=0,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
        derivative=2,
    )

    d2u_dy2 = fdx.difference(
        u,
        axis=1,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
        derivative=2,
    )

    d2u_dz2 = fdx.difference(
        u,
        axis=2,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
        derivative=2,
    )

    return diffusivity * (d2u_dx2 + d2u_dy2 + d2u_dz2)
