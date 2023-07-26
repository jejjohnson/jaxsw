import typing as tp

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaxsw._src.operators.functional import elliptical as F_elliptical
from jaxsw._src.operators.functional import grid as F_grid


class PDEParams(tp.NamedTuple):
    f0: float = 9.375e-5  # coriolis (s^-1)
    beta: float = 1.754e-11  # coriolis gradient (m^-1 s^-1)
    tau0: float = 2.0e-5  # wind stress magnitude m/s^2
    y0: float = 2400000.0  # m
    a_2: float = 0.0  # laplacian diffusion coef (m^2/s)
    a_4: float = 5.0e11  # LR # 2.0e9 # HR
    bcco: float = 0.2  # boundary condition coef. (non-dim.)
    delta_ek: float = 2.0  # eckman height (m)

    @property
    def zfbc(self):
        return self.bcco / (1.0 + 0.5 * self.bcco)


def create_qgml_helmholtz_matrix(
    domain, heights, coriolis_param: float = 9.375e-05
) -> Array:
    # create coefficients
    alpha = 1 / coriolis_param**2
    beta = einops.repeat(heights, "Nz -> Nz 1 1")
    return F_elliptical.helmholtz_dst(
        nx=domain.Nx[0],
        ny=domain.Nx[1],
        dx=domain.dx[0],
        dy=domain.dx[1],
        alpha=alpha,
        beta=beta,
    )


def homogeneous_sol_layers(helmoltz_dst_mat, domain, A_mat):
    beta = einops.repeat(A_mat.lambd, "Nz -> Nz 1 1")
    # constant field
    num_layers = helmoltz_dst_mat.shape[0]
    constant_field = jnp.ones((num_layers, domain.nx, domain.ny)) / (
        domain.nx * domain.ny
    )

    s_solutions = jnp.zeros_like(constant_field)
    out = jax.vmap(F_elliptical.inverse_elliptic_dst, in_axes=(0, 0))(
        constant_field[:, 1:-1, 1:-1], helmoltz_dst_mat
    )
    s_solutions = s_solutions.at[:, 1:-1, 1:-1].set(out)

    homogeneous_sol = constant_field + s_solutions * beta

    return homogeneous_sol[:-1]


class Domain(eqx.Module):
    nx: int = eqx.static_field()
    ny: int = eqx.static_field()
    Lx: int = eqx.static_field()
    Ly: int = eqx.static_field()
    dx: float = eqx.static_field()
    dy: float = eqx.static_field()
    x: Array = eqx.static_field()
    y: Array = eqx.static_field()

    def __init__(self, nx, ny, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        X = jnp.linspace(0, Lx, nx)
        Y = jnp.linspace(0, Ly, ny)
        self.x, self.y = jnp.meshgrid(X, Y, indexing="ij")


class MLQGHeightMatrix(eqx.Module):
    heights: Array = eqx.static_field()
    reduced_gravities: Array = eqx.static_field()
    A: Array = eqx.static_field()
    lambd: Array = eqx.static_field()
    A_layer_2_mode: Array = eqx.static_field()
    A_mode_2_layer: Array = eqx.static_field()

    def __init__(self, heights: tp.List[float], reduced_gravities: tp.List[float]):
        # check input values
        assert len(heights) > 1 and len(reduced_gravities) > 1
        assert len(heights) - 1 == len(reduced_gravities)

        # create matrix
        A = create_qg_multilayer_mat(
            heights=heights, reduced_gravities=reduced_gravities
        )

        # create layer to mode matrices
        lambd, C_layer_2_mat, C_mat_2_layer = compute_layer_to_mode_matrices(A)

        self.heights = jnp.asarray(heights)
        self.reduced_gravities = jnp.asarray(reduced_gravities)
        self.A = jnp.asarray(A)
        self.lambd = jnp.asarray(lambd)
        self.A_layer_2_mode = jnp.asarray(C_layer_2_mat)
        self.A_mode_2_layer = jnp.asarray(C_mat_2_layer)

    def __call__(self):
        return self.A

    @property
    def Nz(self):
        return len(self.heights)


def create_qg_multilayer_mat(
    heights: tp.List[float], reduced_gravities: tp.List[float]
) -> np.ndarray:
    """Computes the Matrix that is used to connected a stacked
    isopycnal Quasi-Geostrophic model.

    Args:
        heights (tp.List[float]): the height for each layer
            Size = [Nx]
        reduced_gravities (tp.List[float]): the reduced gravities
            for each layer, Size = [Nx-1]

    Returns:
        np.ndarray: The Matrix connecting the layers, Size = [Nz, Nx]
    """
    num_heights = len(heights)

    # initialize matrix
    A = np.zeros((num_heights, num_heights))

    # top rows
    A[0, 0] = 1.0 / (heights[0] * reduced_gravities[0])
    A[0, 1] = -1.0 / (heights[0] * reduced_gravities[0])

    # interior rows
    for i in range(1, num_heights - 1):
        A[i, i - 1] = -1.0 / (heights[i] * reduced_gravities[i - 1])
        A[i, i] = (
            1.0 / heights[i] * (1 / reduced_gravities[i] + 1 / reduced_gravities[i - 1])
        )
        A[i, i + 1] = -1.0 / (heights[i] * reduced_gravities[num_heights - 2])

    # bottom rows
    A[-1, -1] = 1.0 / (heights[num_heights - 1] * reduced_gravities[num_heights - 2])
    A[-1, -2] = -1.0 / (heights[num_heights - 1] * reduced_gravities[num_heights - 2])
    return A


def compute_layer_to_mode_matrices(A):
    # eigenvalue decomposition
    lambd_r, R = jnp.linalg.eig(A)
    _, L = jnp.linalg.eig(A.T)

    # extract real components
    lambd, R, L = lambd_r.real, R.real, L.real

    # create matrices
    Cl2m = np.diag(1.0 / np.diag(L.T @ R)) @ L.T
    Cm2l = R
    # create diagonal matrix
    return lambd, -Cl2m, -Cm2l


def compute_alpha_matrix(C, hom_sol):
    Nz = C.shape[0]
    # extract top layer components
    M = (C[1:] - C[:-1])[: Nz - 1, : Nz - 1] * hom_sol.mean(axis=(1, 2))
    M_inv = np.linalg.inv(M)
    alpha_matrix = -M_inv @ (C[1:, :-1] - C[:-1, :-1])
    return alpha_matrix


def init_tau(domain, tau0: float = 2.0e-5):
    """
    Args
    ----
        tau0 (float): wind stress magnitude m/s^2
            default=2.0e-5"""
    # initial TAU
    tau = np.zeros((2, domain.nx, domain.ny))

    # create staggered coordinates (y-direction)
    y_coords = np.arange(domain.ny) + 0.5

    # create tau
    tau[0, :, :] = -tau0 * np.cos(2 * np.pi * (y_coords / domain.ny))

    return tau


def calculate_wind_forcing(tau, domain):
    # move from edges to nodes
    tau_x = F_grid.x_average_2D(tau[0])
    tau_y = F_grid.y_average_2D(tau[1])

    # calculate curl
    dF2dX = (tau_y[1:] - tau_y[:-1]) / domain.dx
    dF1dY = (tau_x[:, 1:] - tau_x[:, :-1]) / domain.dy
    curl_stagg = dF2dX - dF1dY

    # move from nodes to faces
    return F_grid.center_average_2D(curl_stagg)


def calculate_A_term(A, pressure):
    return -jnp.einsum("ij,jkl->ikl", A, pressure)


def calculate_beta_term(domain, params):
    return (params.beta / params.f0) * (domain.y - 0.5 * domain.Ly)


def laplacian(u, constant):
    delta_f = jnp.zeros_like(u)

    # calculate interior Laplacian
    out = laplacian_interior(u)
    delta_f = delta_f.at[..., 1:-1, 1:-1].set(out)
    # calculate Laplacian at the boundaries
    delta_f_bound = laplacian_boundaries(u, constant)
    return _apply_boundaries(delta_f, delta_f_bound)


def _apply_boundaries(u, bc):
    nx, ny = u.shape[-2:]

    # set the boundaries
    u = u.at[..., 0, 1:-1].set(bc[..., : ny - 2])
    u = u.at[..., -1, 1:-1].set(bc[..., ny - 2 : 2 * ny - 4])
    u = u.at[..., 0].set(bc[..., 2 * ny - 4 : nx + 2 * ny - 4])
    u = u.at[..., -1].set(bc[..., nx + 2 * ny - 4 : 2 * nx + 2 * ny - 4])
    return u


def laplacian_interior(u, constant=0.0):
    return (
        u[..., 2:, 1:-1]
        + u[..., :-2, 1:-1]
        + u[..., 1:-1, 2:]
        + u[..., 1:-1, :-2]
        - 4 * u[..., 1:-1, 1:-1]
    )


def laplacian_boundaries(u, constant=0.0):
    return constant * (
        jnp.concatenate(
            [u[..., 1, 1:-1], u[..., -2, 1:-1], u[..., 1], u[..., -2]], axis=-1
        )
        - jnp.concatenate(
            [u[..., 0, 1:-1], u[..., -1, 1:-1], u[..., 0], u[..., -1]], axis=-1
        )
    )


def pressure_to_vorticity(pressure, A, domain, params):
    # calculate A tearm
    Ap = calculate_A_term(A, pressure)

    # calculate beta term
    beta_term = calculate_beta_term(domain, params)

    # calculate laplacian
    lap_term = laplacian(pressure, params.zfbc) / (params.f0 * domain.dx) ** 2

    # calculate vorticity
    q = lap_term + Ap + beta_term
    return q


## discrete spatial differential operators
def det_jacobian(f, g):
    """Arakawa discretisation of Jacobian J(f,g).
    Scalar fields f and g must have the same dimension.
    Grid is regular and dx = dy."""
    dx_f = f[..., 2:, :] - f[..., :-2, :]
    dx_g = g[..., 2:, :] - g[..., :-2, :]
    dy_f = f[..., 2:] - f[..., :-2]
    dy_g = g[..., 2:] - g[..., :-2]
    return (
        (dx_f[..., 1:-1] * dy_g[..., 1:-1, :] - dx_g[..., 1:-1] * dy_f[..., 1:-1, :])
        + (
            (
                f[..., 2:, 1:-1] * dy_g[..., 2:, :]
                - f[..., :-2, 1:-1] * dy_g[..., :-2, :]
            )
            - (f[..., 1:-1, 2:] * dx_g[..., 2:] - f[..., 1:-1, :-2] * dx_g[..., :-2])
        )
        + (
            (g[..., 1:-1, 2:] * dx_f[..., 2:] - g[..., 1:-1, :-2] * dx_f[..., :-2])
            - (
                g[..., 2:, 1:-1] * dy_f[..., 2:, :]
                - g[..., :-2, 1:-1] * dy_f[..., :-2, :]
            )
        )
    ) / 12.0


def advection_term(q, p, params, domain):
    return (1.0 / (params.f0 * domain.dx * domain.dy)) * det_jacobian(q, p)


def diffusion_term(p, params, domain):
    return params.a_2 / params.f0**2 / domain.dx**4 * laplacian_interior(p)


def hyperdiffusion_term(p, params, domain):
    return (
        -(params.a_4 / params.f0**2)
        / domain.dx**6
        * laplacian_interior(laplacian(p, params.zfbc))
    )


def bottom_friction(p, params, domain, height_params):
    return (
        params.delta_ek
        / (2 * np.abs(params.f0) * domain.dx**2 * (-height_params.heights[-1]))
        * laplacian_interior(p[..., -1:, :, :])
    )


def rhs_pde(q, p, params, domain, A_mat, wind_forcing):
    # Calculate Determinant Jacobian
    rhs = advection_term(q, p, params, domain)

    # calculate Laplacian
    delta2_p = laplacian(p, params.zfbc)

    # Add Diffusion Term
    if params.a_2 != 0.0:
        rhs += diffusion_term(delta2_p, params, domain)

    # Add HyperDiffusion
    if params.a_4 != 0.0:
        rhs += hyperdiffusion_term(delta2_p, params, domain)

    # Add Wind Forcing
    rhs = rhs.at[..., 0:1, :, :].set(rhs[..., 0:1, :, :] + wind_forcing)

    # Add Bottom Friction
    bottom_term = bottom_friction(p, params, domain, A_mat)

    rhs = rhs.at[..., -1:, :, :].set(rhs[..., -1:, :, :] + bottom_term)

    return rhs


class QGARGS(eqx.Module):
    A_mat: MLQGHeightMatrix
    domain: Domain
    wind_forcing: Array = eqx.static_field()
    helmoltz_dst: Array = eqx.static_field()
    alpha_matrix: Array = eqx.static_field()
    homogeneous_sol: Array = eqx.static_field()


def rhs_time_step(q, p, params, args: QGARGS):
    # calculate advection - interior only
    dq_f0 = rhs_pde(
        q,
        p,
        params=params,
        domain=args.domain,
        A_mat=args.A_mat,
        wind_forcing=args.wind_forcing,
    )
    # pad - original domain
    dq_f0 = jnp.pad(dq_f0, ((0, 0), (1, 1), (1, 1)))

    # pressure
    rhs_helmholtz = jnp.einsum("ij,jkl->ikl", args.A_mat.A_layer_2_mode, dq_f0)
    dp_modes = jax.vmap(F_elliptical.inverse_elliptic_dst, in_axes=(0, 0))(
        rhs_helmholtz[:, 1:-1, 1:-1], args.helmoltz_dst
    )
    # pad - original domain
    dp_modes = jnp.pad(dp_modes, ((0, 0), (1, 1), (1, 1)))

    # ensure mass conservation
    dalpha = args.alpha_matrix @ dp_modes[..., :-1, :, :].mean((-2, -1))
    dalpha = einops.repeat(dalpha, "i -> i 1 1")
    dp_modes = dp_modes.at[..., :-1, :, :].set(
        dp_modes[..., :-1, :, :] + dalpha * args.homogeneous_sol
    )
    dp = jnp.einsum("ij,jkl->ikl", args.A_mat.A_mode_2_layer, dp_modes)

    # UPDATE VORTICITY @ BOUNDARIES
    delta_p_boundaries = laplacian_boundaries(
        dp / (params.f0 * args.domain.dx) ** 2, params.zfbc
    )
    dp_boundaries = jnp.concatenate(
        [dp[..., 0, 1:-1], dp[..., -1, 1:-1], dp[..., :, 0], dp[..., :, -1]], axis=-1
    )
    dq_f0_boundaries = delta_p_boundaries - args.A_mat.A @ dp_boundaries

    dq_f0 = _apply_boundaries(dq_f0, dq_f0_boundaries)

    return dq_f0, dp
