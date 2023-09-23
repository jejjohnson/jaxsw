import typing as tp

import einops
import equinox as eqx
import finitediffx as fdx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaxsw._src.domain.base import Domain
from jaxsw._src.domain.qg import LayerDomain
from jaxsw._src.operators.functional import elliptical as F_elliptical
from jaxsw._src.operators.functional.advection import det_jacobian


class PDEParams(eqx.Module):
    f0: float = 9.375e-5  # coriolis [s^-1]
    beta: float = 1.754e-11  # coriolis gradient [m^-1 s^-1]
    tau0: float = 2.0e-5  # wind stress magnitude m/s^2
    y0: float = 2_400_000.0  # [m]
    a_2: float = 0.0  # laplacian diffusion coef (m^2/s)
    a_4: float = 5.0e11  # LR # 2.0e9 # HR
    bcco: float = 0.2  # boundary condition coef. (non-dim.)
    delta_ek: float = 2.0  # eckman height [m]

    @property
    def zfbc(self):
        return self.bcco / (1.0 + 0.5 * self.bcco)


class QGState(tp.NamedTuple):
    q: Array
    p: Array


class QGARGS(eqx.Module):
    domain: Domain = eqx.static_field()
    layer_domain: LayerDomain = eqx.static_field()
    wind_forcing: Array = eqx.static_field()
    helmoltz_dst_mat: Array = eqx.static_field()
    alpha_matrix: Array = eqx.static_field()
    homogeneous_sol: Array = eqx.static_field()
    bc_fn: tp.Callable


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


def helmholtz_homogeneous_sol_multilayer(helmoltz_dst_mat, heights, domain):
    num_layers = len(heights)
    assert len(helmoltz_dst_mat.shape) == 3

    # inversion scheme (interior points)
    # (nabla_H - beta) s = L
    # constant field (where did we get this???)
    constant_field = jnp.ones((num_layers, domain.Nx[0], domain.Nx[1])) / (
        domain.Nx[0] * domain.Nx[1]
    )

    out = jax.vmap(F_elliptical.inverse_elliptic_dst, in_axes=(0, 0))(
        constant_field[:, 1:-1, 1:-1], helmoltz_dst_mat
    )

    # construct s (interior points)
    s_solutions = jnp.zeros_like(constant_field)
    s_solutions = s_solutions.at[:, 1:-1, 1:-1].set(out)

    # add the boundary: p = L + beta * s
    beta = einops.repeat(heights, "Nz -> Nz 1 1")
    homogeneous_sol = constant_field + beta * s_solutions

    # ignore last solution correponding to lambd = 0, i.e. Laplace equation
    return homogeneous_sol[:-1]


def compute_alpha_matrix(A_mode_2_layer, hom_sol):
    Nz = A_mode_2_layer.shape[0]
    # extract top layer components
    M = (A_mode_2_layer[1:] - A_mode_2_layer[:-1])[: Nz - 1, : Nz - 1] * hom_sol.mean(
        axis=(1, 2)
    )
    M_inv = np.linalg.inv(M)
    alpha_matrix = -M_inv @ (A_mode_2_layer[1:, :-1] - A_mode_2_layer[:-1, :-1])
    return alpha_matrix


def custom_boundaries(u, constant):
    # Top-Down (Channel)
    u = u.at[..., 0].set(constant * (u[..., 1] - u[..., 0]))
    u = u.at[..., -1].set(constant * (u[..., -2] - u[..., -1]))
    # Left-Right
    u = u.at[..., 0, 1:-1].set(constant * (u[..., 1, 1:-1] - u[..., 0, 1:-1]))
    u = u.at[..., -1, 1:-1].set(constant * (u[..., -2, 1:-1] - u[..., -1, 1:-1]))
    return u


def apply_boundaries(u, bc, constant=1.0):
    # Top-Down (Channel)
    u = u.at[..., 0].set(constant * bc[..., 0])
    u = u.at[..., -1].set(constant * bc[..., -1])
    # Left-Right
    u = u.at[..., 0, 1:-1].set(constant * bc[..., 0, 1:-1])
    u = u.at[..., -1, 1:-1].set(constant * bc[..., -1, 1:-1])
    return u


def pressure_to_vorticity(pressure, bc_fn: tp.Callable, A, params, domain):
    # Calculate Node Term
    Ap_term = -jnp.einsum("ij,jkl->ikl", A, pressure)
    # print_debug_quantity(Ap_term, "Ap_term")

    # =============================
    # calculate Laplacian (interior)
    lap_term = jnp.zeros_like(pressure)
    accuracy = 1
    fn = lambda x: fdx.laplacian(
        x, accuracy=accuracy, step_size=domain.dx, method="central"
    )
    lap_term = lap_term.at[..., 1:-1, 1:-1].set(jax.vmap(fn)(pressure)[..., 1:-1, 1:-1])
    lap_term /= params.f0**2  # scale factor

    lap_term = bc_fn(lap_term)  # apply boundary conditions

    # Calculate Beta Term
    beta_term = (params.beta / params.f0) * (domain.grid[..., 1] - 0.5 * domain.Lx[1])

    return Ap_term + lap_term + beta_term


def rhs_pde(
    q, p, bc_fn, params, layer_domain, domain, wind_forcing: tp.Optional[Array] = None
):
    rhs = jnp.zeros_like(p)

    # create Laplacian function
    laplacian = lambda x: fdx.laplacian(
        x, accuracy=1, step_size=domain.dx, method="central"
    )
    vectorized_laplacian = jax.vmap(laplacian)

    # laplacian
    lap_p = vectorized_laplacian(p)
    lap_p = custom_boundaries(lap_p, params.zfbc)

    # ADVECTION TERM (Interior)
    rhs = (1 / params.f0) * det_jacobian(q, p, domain.dx[0], domain.dx[1])

    # DIFFUSION TERM
    if params.a_2 != 0.0:
        rhs_diff = (params.a_2 / params.f0**2) * vectorized_laplacian(lap_p)
        rhs += rhs_diff[..., 1:-1, 1:-1]

    # HYPERDIFFUSION TERM
    if params.a_4 != 0.0:
        lap2_p = vectorized_laplacian(lap_p)
        lap2_p = custom_boundaries(lap2_p, params.zfbc)
        rhs_hyperdiff = -(params.a_4 / params.f0**2) * vectorized_laplacian(lap2_p)
        rhs += rhs_hyperdiff[..., 1:-1, 1:-1]

    # WIND FORCING
    if wind_forcing is not None:
        rhs = rhs.at[..., 0, :, :].set(rhs[..., 0, :, :] + wind_forcing)

    # BOTTOM FRICTION
    coeff = params.delta_ek / (2 * jnp.abs(params.f0) * (-layer_domain.heights[-1]))
    rhs_bottom = coeff * laplacian(p[..., -1, 1:-1, 1:-1])
    # print_debug_quantity(rhs_bottom[..., 1:-1, 1:-1], "BOTTOM FRICTION")
    rhs = rhs.at[..., -1, :, :].set(rhs[..., -1, :, :] + rhs_bottom)

    return rhs


def pde_time_step(p, q, params, args):
    # unpack state

    # RHS of PDE for Q (INTERIOR)
    dq_f0 = rhs_pde(
        q,
        p,
        bc_fn=args.bc_fn,
        params=params,
        layer_domain=args.layer_domain,
        domain=args.domain,
        wind_forcing=args.wind_forcing,
    )
    # pad for original domain
    dq_f0 = jnp.pad(dq_f0, ((0, 0), (1, 1), (1, 1)))

    # PRESSURE (INTERIOR)
    rhs_helmholtz = jnp.einsum("ij,jkl->ikl", args.layer_domain.A_layer_2_mode, dq_f0)
    dp_modes = jax.vmap(F_elliptical.inverse_elliptic_dst, in_axes=(0, 0))(
        rhs_helmholtz[:, 1:-1, 1:-1], args.helmoltz_dst_mat
    )

    dp_modes = jnp.pad(dp_modes, ((0, 0), (1, 1), (1, 1)))

    # ensure mass conservation
    dalpha = args.alpha_matrix @ dp_modes[..., :-1, :, :].mean((-2, -1))
    dalpha = einops.repeat(dalpha, "i -> i 1 1")

    dp_modes = dp_modes.at[..., :-1, :, :].set(
        dp_modes[..., :-1, :, :] + dalpha * args.homogeneous_sol
    )
    dp = jnp.einsum("ij,jkl->ikl", args.layer_domain.A_mode_2_layer, dp_modes)

    delta_p_boundaries = args.bc_fn(dp / (params.f0 * args.domain.dx[0]) ** 2)

    # apply boundaries
    dq_f0_boundaries = delta_p_boundaries - jnp.einsum(
        "ij,jkl->ikl", args.layer_domain.A, dp
    )

    dq_f0 = apply_boundaries(dq_f0, dq_f0_boundaries)

    return dp, dq_f0
