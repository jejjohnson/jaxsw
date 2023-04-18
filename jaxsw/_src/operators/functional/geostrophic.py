import typing as tp
from jaxtyping import Array
from jaxsw._src.utils.constants import GRAVITY
import finitediffx as fdx
from jaxsw._src.utils.dst_solver import inverse_elliptical_dst_solver
from jaxsw._src.boundaries.helmholtz import enforce_boundaries_helmholtz


def ssh_to_streamfn(ssh: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the ssh to stream function

    Eq:
        η = (g/f₀) Ψ

    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the stream function to ssh

    Eq:
        Ψ = (f₀/g) η

    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi


def streamfn_to_pvort(
    psi: Array, dx: Array, dy: Array, c1: float = 2.7, f0: float = 1e-5, **kwargs
) -> Array:
    """Calculates the potential vorticity to the streamfunction

    Eq:
        q = ∇²Ψ - (f₀²/c₁²) Ψ

    Args:
        psi (Array): The stream function
        dx (Array): the change in x
        dy (Array): the change in y
        f0 (Array): coriolis parameter at mean latitude
        c1 (Array): the ...

    Returns:
        q (Array): The potential vorticity

    """
    return fdx.laplacian(psi, step_size=(dx, dy), **kwargs) - (f0 / c1) ** 2 * psi


def pvort_to_streamfn(
    q: Array,
    psi_bcs: Array,
    dx: Array,
    dy: Array,
    f0: float = 1e-5,
    c1: float = 1.5,
    **kwargs
) -> Array:
    """Does the inverse problem for the PV and SF. We use a trick
    where we decompse the problem into interior and exterior points.
    For this, we use the Discrete Sine Transformation which
    is very fast for Dirichlet boundary conditions. This function
    works the best when dx=dy but we do leave it open...

    Eqn:
        Ψ = [∇² - f₀²/c₁²]⁻¹ q

    Args:
        q (Array): Potential Vorticity
        psi_bcs (Array): the psi where we enforce the boundaries.
        dx (Array|float): the change in x
        dy (Array|float): the change in y
        f0 (Array|float): the Coriolis parameter
        c1 (Array|float): the ...

    Returns:
        psi (Array): the stream function from the linear solver
    """

    beta = (f0 / c1) ** 2

    # calculate the interior potential vorticity
    psi_bcs = psi_bcs.at[1:-1, 1:-1].set(0.0)

    q_exterior = streamfn_to_pvort(psi_bcs, dx=dx, dy=dy, f0=f0, c1=c1, **kwargs)

    q_exterior = enforce_boundaries_helmholtz(q_exterior, psi_bcs, beta=beta)

    # remove interior influence
    q_interior = q[1:-1, 1:-1] - q_exterior[1:-1, 1:-1]

    # do inverse linear solver
    nx, ny = q.shape

    psi_interior = inverse_elliptical_dst_solver(
        q_interior, nx=nx, ny=ny, dx=dx, dy=dy, beta=beta
    )

    # add boundaries to the solution
    psi = psi_bcs.at[1:-1, 1:-1].set(psi_interior)

    return psi


def uv_velocity(psi: Array, dx: Array, dy: Array, **kwargs) -> tp.Tuple[Array, Array]:
    """Calculates the geostrophic velocities from the stream function.

    Eqn:
        u = -∂Ψ/∂y
        v =  ∂Ψ/∂x

    Args:
        psi (Array): the stream function
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all keyword arguments for the fdx.difference operator

    Return:
        u (Array): the meridonial velocty [m/s^2]
        v (Array): the zonal velocity [m/s^2]
    """

    u = -fdx.difference(psi, axis=1, step_size=dy, derivative=1, **kwargs)

    v = fdx.difference(psi, axis=0, step_size=dx, derivative=1, **kwargs)

    return u, v


def divergence(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the geostrophic divergence by using
    finite difference in the x and y direction for the
    u and v velocities respectively

    Eqn:
        ∇⋅[u,v] = ∂u/∂x + ∂v/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        div (Array): the geostrophic divergence
    """

    du_dx = fdx.difference(u, axis=0, step_size=dx, derivative=1, **kwargs)

    dv_dy = fdx.difference(v, axis=0, step_size=dy, derivative=1, **kwargs)

    return du_dx + dv_dy


def relative_vorticity(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        zeta (Array): the geostrophic vorticity
    """
    dv_dx = fdx.difference(v, axis=0, step_size=dx, derivative=1, **kwargs)

    du_dy = fdx.difference(u, axis=1, step_size=dy, derivative=1, **kwargs)

    return dv_dx - du_dy


def absolute_vorticity(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the absolute vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        |ζ| = ∂v/∂x + ∂u/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        zeta (Array): the geostrophic vorticity
    """
    dv_dx = fdx.difference(v, axis=0, step_size=dx, derivative=1, **kwargs)

    du_dy = fdx.difference(u, axis=1, step_size=dy, derivative=1, **kwargs)

    return dv_dx + du_dy


def kinetic_energy(u: Array, v: Array) -> Array:
    """Calculates the kinetic energy via an
    arbitrary magnitude of the u and v velocities

    Eqn:
        ke(u,v) = 0.5 * (u² + v²)

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity

    Returns:
        ke (Array): the geostrophic vorticity
    """
    return 0.5 * (u**2 + v**2)


def potential_energy(q: Array) -> Array:
    """Calculates the potential energy via an
    arbitrary magnitude of the potential vorticity

    Eqn:
        pq(q) = 0.5 (q²)

    Args:
        q (Array): the potential vorticity

    Returns:
        pe (Array): the potential energy
    """
    return 0.5 * q**2


def shear_strain(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the geostrophic vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        Sₛ = ∂v/∂x + ∂u/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        s_strain (Array): the geostrophic vorticity
    """
    dv_dx = fdx.difference(v, axis=0, step_size=dx, derivative=1, **kwargs)

    du_dy = fdx.difference(u, axis=1, step_size=dy, derivative=1, **kwargs)

    return dv_dx + du_dy


def tensor_strain(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the geostrophic vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        Sₙ = ∂u/∂x - ∂v/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        t_strain (Array): the geostrophic vorticity
    """
    du_dx = fdx.difference(u, axis=0, step_size=dy, derivative=1, **kwargs)

    dv_dy = fdx.difference(v, axis=1, step_size=dx, derivative=1, **kwargs)

    return du_dx - dv_dy


def strain(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the strain by using
    finite difference in the y and x direction for the
    u and v velocities respectively. Strain is the addition of the
    squared tensor strain and shear strain terms respectively.

    Eqn:
        σₛ = Sₙ² + Sₛ²

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        strain (Array): the geostrophic vorticity
    """
    t_strain = tensor_strain(u=u, v=v, dx=dx, dy=dy, **kwargs)

    s_strain = shear_strain(u=u, v=v, dx=dx, dy=dy, **kwargs)

    return t_strain**2 + s_strain**2


def okubo_weiss_param(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the Okubo-Weiss parameter by using
    finite difference in the y and x direction for the
    u and v velocities respectively. The Okubo-Weiss parameter
    is the difference between the strain and the divergence.

    Eqn:
        ow = σₛ - div(u)²

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        ow (Array): the Okubo-Weiss parameter
    """
    strain_magnitude = strain(u=u, v=v, dx=dx, dy=dy, **kwargs)

    div = divergence(u=u, v=v, dx=dx, dy=dy, **kwargs)

    return strain_magnitude - div**2
