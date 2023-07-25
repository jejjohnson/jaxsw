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
        step_size=step_size[0],
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
    if u.ndim > 2:
        u = u[..., :1]

    return diffusion_3D(
        u=u,
        diffusivity=diffusivity,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
    )


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
    return diffusivity * fdx.laplacian(
        u, method=method, accuracy=accuracy, step_size=step_size
    )
