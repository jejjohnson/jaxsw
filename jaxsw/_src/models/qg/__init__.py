import typing as tp
import equinox as eqx 
from jaxsw._src.domain.base_v2 import Domain
import einops
import math
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
from jaxsw._src.domain.qg import create_qg_multilayer_mat, LayerDomain
from jaxsw._src.operators.functional.finitevol.difference import laplacian, laplacian_batch
from jaxsw._src.operators.functional.finitevol.geostrophic import gradient_perpendicular, divergence
from jaxsw._src.operators.functional.finitevol.interp import x_average_2D, y_average_2D, center_average_2D
from jaxsw._src.operators.functional.dst import helmholtz_dst, laplacian_dst, inverse_elliptic_dst, inverse_elliptic_dst_cmm
from jaxsw._src.masks import Mask
from jaxsw._src.operators.functional.interp import flux as F_flux
import numpy as np


import matplotlib.pyplot as plt
from loguru import logger
def plot_field(field, name=""):
    num_axis = len(field)
    fig, ax = plt.subplots(ncols=num_axis, figsize=(8, 2))
    fig.suptitle(name)
    for i in range(num_axis):
        pts = ax[i].pcolormesh(field[i], cmap="coolwarm")
        plt.colorbar(pts)

    plt.tight_layout()
    plt.show()
    
def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = jnp.min(quantity)
    max_ = jnp.max(quantity)
    mean_ = jnp.mean(quantity)
    median_ = jnp.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | {min_:.6e} | {mean_:.6e} | {median_:.6e} | {max_:.6e}"
    )

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
    
class DSTSolution(eqx.Module):
    homsol: Array = eqx.static_field()
    homsol_mean: Array = eqx.static_field()
    H_mat: Array = eqx.static_field()
    capacitance_matrix: tp.Optional[Array] = eqx.static_field()

    
def calculate_helmholtz_dst(
    domain: Domain, 
    layer_domain: LayerDomain,
    params: PDEParams
) -> Array:
    
    # get Laplacian dst transform
    # print(domain.Nx, domain.dx)
    L_mat = laplacian_dst(domain.Nx[0]-2, domain.Nx[1]-2, domain.dx[0], domain.dx[1])
    # print_debug_quantity(L_mat, "L_MAT")
    # get beta term
    lambda_sq = einops.rearrange(layer_domain.lambda_sq, "Nz -> Nz 1 1")
    beta = params.f0**2 * lambda_sq
    
    # calculate helmholtz dst
    H_mat = L_mat - beta
    
    return H_mat


def compute_homogeneous_solution(
    u: Array, 
    lambda_sq: Array,
    H_mat: Array
):
    
    # create constant field
    constant_field = jnp.ones_like(u)
    
    # get homogeneous solution
    sol = jax.vmap(inverse_elliptic_dst, in_axes=(0,0))(constant_field[:, 1:-1,1:-1], H_mat)
    
    print_debug_quantity(sol, "SOL")
    
    print_debug_quantity(lambda_sq, "LAMBDA SQ")
    # calculate the homogeneous solution
    homsol = constant_field + sol * lambda_sq
    
    return homsol




def calculate_potential_vorticity(
    psi: Array, 
    domain: Domain, 
    layer_domain: LayerDomain, 
    params: PDEParams,
    masks_psi=None,
    masks_q=None
) -> Array:
    
    
    # calculate laplacian [Nx,Ny] --> [Nx-2, Ny-2]
    psi_lap = laplacian_batch(psi, domain.dx)
    
    # pad 
    # [Nx-2,Ny-2] --> [Nx,Ny]
    psi_lap = jnp.pad(psi_lap, pad_width=((0,0),(1,1),(1,1)), mode="constant", constant_values=0.0)

    
    # calculate beta term in helmholtz decomposition
    beta_lap = params.f0**2 * jnp.einsum("lm,...mxy->...lxy", layer_domain.A, psi)
    
    q = psi_lap - beta_lap 
    
    # apply boundary conditions on psi
    if masks_psi:
        q *= masks_psi.values
        
    # [Nx,Ny] --> [Nx-1,Ny-1]
    q = center_average_2D(q)
        
    # calculate beta-plane
    y_coords = center_average_2D(domain.grid_axis[-1])
    f_y = params.beta * (y_coords - params.y0)
    
    q += f_y
    

    
    # apply boundary conditions on q
    if masks_q:
        q *= masks_q.values
    
    return q


def advection_rhs(
    q: Float[Array, "Nx-1 Ny-1"], 
    psi: Float[Array, "Nx Ny"], 
    dx: float, dy: float,
    num_pts: int=1,
    masks_u: Mask=None,
    masks_v: Mask=None
):
    """Calculates the advection term on the RHS of the Multilayer QG
    PDE. It assumes we use an arakawa C-grid whereby the potential 
    vorticity term is on the cell centers, the zonal velocity is on the
    east-west cell faces, and the meridional velocity is on the 
    north-south cell faces.
    
    Velocity:
        u, v = -∂yΨ, ∂xΨ
    Advection:
        u̅⋅∇ q = ∂x(uq) + ∂y(vq)
        
        
    This uses the conservative method which calculates the flux terms
    independently, (uq, vq) and then we calculate the partial 
    derivatives.
    
    Args:
        q (Array): the potential vorticity term on the cell centers
            Size = [Nx-1, Ny-1]
        psi (Array): the stream function on the cell vertices
            Size = [Nx, Ny]
    Returns:
        div_flux (Array): the flux divergence on the cell centers
            Size = [Nx-1, Ny-1]
    
    """
    
    # calculate velocities
    # [Nx,Ny] --> [Nx,Ny-1],[Nx-1,Ny]
    # u, v = ∂yΨ, ∂xΨ
    u, v = gradient_perpendicular(psi, dx, dy)
    # u = -∂yΨ
    u *= -1
    
    # u = (psi[...,:-1] - psi[...,1:]) / dy

    
    # np.testing.assert_array_almost_equal(u, u_)

    
    # check_field_shapes(u, name="u")
    # check_field_shapes(v, name="v")

    # take interior points of velocities
    # [Nx,Ny-1] --> [Nx-2,Ny-1]
    u_i = u[..., 1:-1, :]
    # [Nx-1,Ny] --> [Nx-1,Ny-2]
    v_i = v[..., 1:-1]
    
    # check_field_shapes(u_i, name="u_i")
    # check_field_shapes(v_i, name="v_i")
    
    # calculate flux terms
    # uq, vq
    # [Nx-1,Ny-1],[Nx-2,Ny-1] --> [Nx-2,Ny-1]
    
    
    
    if masks_u is not None:
#         # OPTION I - 1pt Flux (Standard Upwind Scheme)
#         q_flux_on_u = F_flux.tracer_flux_1pt_mask(
#             q=q, u=u_i, dim=0,
#             u_mask1=masks_u.distbound1[...,1:-1,:],
#         )
#         q_flux_on_v = F_flux.tracer_flux_1pt_mask(
#             q=q, u=v_i, dim=1,
#             u_mask1=masks_v.distbound1[...,1:-1],
            
#         )
        
#         # OPTION II - 3pt Flux 
#         # method - "linear" | "weno" | "wenoz"
#         q_flux_on_u = F_flux.tracer_flux_3pt_mask(
#             q=q, u=u_i, dim=0, method="wenoz",
#             u_mask1=masks_u.distbound1[...,1:-1,:],
#             u_mask2plus=masks_u.distbound2plus[...,1:-1,:], 
#         )
#         q_flux_on_v = F_flux.tracer_flux_3pt_mask(
#             q=q, u=v_i, dim=1, method="wenoz",
#             u_mask1=masks_v.distbound1[...,1:-1],
#             u_mask2plus=masks_v.distbound2plus[...,1:-1], 
            
#         )
        
        # OPTION III - 5pt Flux 
        # method - "linear" | "weno" | "wenoz"
        q_flux_on_u = F_flux.tracer_flux_5pt_mask(
            q=q, u=u_i, dim=0, method="wenoz",
            u_mask1=masks_u.distbound1[...,1:-1,:],
            u_mask2=masks_u.distbound2[...,1:-1,:], 
            u_mask3plus=masks_u.distbound3plus[...,1:-1,:],    
        )
        q_flux_on_v = F_flux.tracer_flux_5pt_mask(
            q=q, u=v_i, dim=1, method="wenoz",
            u_mask1=masks_v.distbound1[...,1:-1],
            u_mask2=masks_v.distbound2[...,1:-1], 
            u_mask3plus=masks_v.distbound3plus[...,1:-1],   
            
        )
        
    else:
        # # OPTION I - 1pt Flux (Standard Upwind Scheme)
        # q_flux_on_u = F_flux.tracer_flux_1pt(q=q, u=u_i, dim=0)
        # q_flux_on_v = F_flux.tracer_flux_1pt(q=q, u=v_i, dim=1)
        #OPTION II - 3pt Flux 
        #method - "linear" | "weno" | "wenoz"
        q_flux_on_u = F_flux.tracer_flux_3pt(q=q, u=u_i, dim=0, method="linear")
        q_flux_on_v = F_flux.tracer_flux_3pt(q=q, u=v_i, dim=1, method="linear")
        # # OPTION III - 5pt Flux 
        # # method - "linear" | "weno" | "wenoz"
        # q_flux_on_u = F_flux.tracer_flux_5pt(q=q, u=u_i, dim=0, method="wenoz")
        # q_flux_on_v = F_flux.tracer_flux_5pt(q=q, u=v_i, dim=1, method="wenoz")
        
        
        
    
    # check_field_shapes(q_flux_on_u, name="u_i")
    # check_field_shapes(q_flux_on_v, name="v_i")
    
    # pad arrays to comply with velocities (cell faces)
    # [Nx-2,Ny-1] --> [Nx,Ny-1]
    q_flux_on_u = jnp.pad(q_flux_on_u, pad_width=((1,1),(0,0)))
    # [Nx-1,Ny-2] --> [Nx-1,Ny]
    q_flux_on_v = jnp.pad(q_flux_on_v, pad_width=((0,0),(1,1)))
    
    # check_field_shapes(q_flux_on_u, name="u")
    # check_field_shapes(q_flux_on_v, name="v")
    
    # calculate divergence
    # [Nx,Ny-1] --> [Nx-1,Ny-1]
    div_flux = divergence(q_flux_on_u, q_flux_on_v, dx, dy)
    
    # check_field_shapes(div_flux, name="q")
    
    
    
    return - div_flux, u, v, q_flux_on_u, q_flux_on_v


def calculate_bottom_drag(
    psi: Array, 
    domain: Domain,
    H_z: float=1.0,
    delta_ek: float=2.0,
    f0: float=9.375e-05,
    masks_psi=None
) -> Array:
    """
    Equation:
        F_drag: (δₑf₀² / 2Hz)∇²ψN
    """
    
    # interior, vertex points on psi
    # [...,Nx,Ny] --> [...,Nx-2,Ny-2]
    omega = jax.vmap(laplacian, in_axes=(0,None))(psi, domain.dx)
    
    # pad interior psi points
    # [Nx,Ny] --> [Nx,Ny]
    omega = jnp.pad(omega, pad_width=((0,0),(1,1),(1,1)), mode="constant", constant_values=0.0)
    
    if masks_psi is not None:
        omega *= masks_psi.values
    
    # plot_field(omega)
    # pad interior, center points on q
    # [Nx,Ny] --> [Nx-1,Ny-1]
    omega = center_average_2D(omega)
    
    # calculate bottom drag coefficient
    bottom_drag_coeff = delta_ek / H_z * f0 / 2.0
    
    # calculate bottom drag
    bottom_drag = - bottom_drag_coeff * omega[-1]
    
    # plot_field(omega)
    return bottom_drag


def calculate_wind_forcing(
    domain: Domain, 
    H_0: float,
    tau0: float=0.08/1_000.0,
) -> Array:
    """
    Equation:
        F_wind: (τ₀ /H₀)(∂xτ−∂yτ)
    """
    
    Ly = domain.Lx[-1]
    
    # [Nx,Ny]
    y_coords = domain.grid_axis[-1]
    
    # center coordinates, cell centers
    # [Nx,Ny] --> [Nx-1,Ny-1]
    y_coords_center = center_average_2D(y_coords)
    
    
    # calculate tau
    # analytical form! =]
    curl_tau = - tau0 * 2 * math.pi/Ly * jnp.sin(2 * math.pi * y_coords_center/Ly)
    
    # print_debug_quantity(curl_tau, "CURL TAU")
    
    wind_forcing = curl_tau / H_0
    
    
    
    return wind_forcing

def add_forces(q):
    forces = jnp.zeros_like(q)
    forces = forces.at[0].set(forces[0] + wind_forcing)
    forces = forces.at[-1].set(forces[-1] + bottom_drag)
    return forces


def qg_rhs(
    q: Array, 
    psi: Array, 
    params: PDEParams,
    domain: Domain,
    layer_domain: LayerDomain,
    dst_sol: DSTSolution,
    wind_forcing: Array,
    bottom_drag: Array,
    masks=None,
) -> Array:
    
    # use psi for the boundary conditions
    # TODO
    
    # psi = compute_psi_from_q( 
    #     q=q,
    #     params=params,
    #     domain=domain,
    #     layer_domain=layer_domain,
    #     dst_sol=dst_sol
    # )
    
    # # calculate potential vorticity
    # q  = calculate_potential_vorticity(
    #     psi, domain, layer_domain,
    #     params=params,
    #     masks_psi=masks.psi,
    #     masks_q=masks.q,
    # )
    
    # calculate advection
    fn = jax.vmap(advection_rhs, in_axes=(0,0,None,None,None,None,None))

    dq, u, v, q_flux_on_u, q_flux_on_v = fn(
        q, psi, domain.dx[-2], domain.dx[-1], 3, masks.u, masks.v)
    # print_debug_quantity(u, "U")
    # print_debug_quantity(v, "V")
    # print_debug_quantity(q_flux_on_u, "Q FLUX U")
    # print_debug_quantity(q_flux_on_v, "Q FLUX V")
    # print_debug_quantity(dq, "DIV_FLUX")
    
    bottom_drag = calculate_bottom_drag(
        psi=psi, 
        domain=domain,
        H_z=layer_domain.heights[-1],
        delta_ek=params.delta_ek,
        f0=params.f0,
        masks_psi=masks.psi
    )
    

    
    # add forces duh
    # print_debug_quantity(wind_forcing, "WIND FORCING") 
    # print_debug_quantity(bottom_drag, "BOTTOM DRAG") 
    forces = jnp.zeros_like(dq)
    forces = forces.at[0].set(wind_forcing)
    forces = forces.at[-1].set(bottom_drag)
    # print_debug_quantity(forces, "FORCES")
    dq += forces
    
    # multiply by mask
    dq *= masks.q.values
    
    
    # print_debug_quantity(dq, "DQ") 
    
    # dpsi = compute_psi_from_q( 
    #     q=q,
    #     params=params,
    #     domain=domain,
    #     layer_domain=layer_domain,
    #     dst_sol=dst_sol
    # )
    
    # get interior points (cell verticies interior)
    # [Nx-1,Ny-1] --> [Nx-2,Ny-2]
    dq_i = center_average_2D(dq)

    # calculate helmholtz rhs
    # [Nx-2,Ny-2]
    helmholtz_rhs = jnp.einsum("lm, ...mxy -> ...lxy", layer_domain.A_layer_2_mode, dq_i)
    
    # print_debug_quantity(helmholtz_rhs, "HELMHOLTZ RHS") 
    # print_debug_quantity(dst_sol.H_mat, "HELMHOLTZ MAT") 
    
    # print_debug_quantity(helmholtz_rhs, "HELMHOLTZ RHS")
    # solve elliptical inversion problem
    # [Nx-2,Ny-2] --> [Nx,Ny]
    if dst_sol.capacitance_matrix is not None:
        # print_debug_quantity(dst_sol.capacitance_matrix, "CAPACITANCE MAT") 
        dpsi_modes = inverse_elliptic_dst_cmm(
            rhs=helmholtz_rhs, 
            H_matrix=dst_sol.H_mat,
            cap_matrices=dst_sol.capacitance_matrix,
            bounds_xids=masks.psi.irrbound_xids,
            bounds_yids=masks.psi.irrbound_yids,
            mask=masks.psi.values
        )
    else:
        dpsi_modes = jax.vmap(inverse_elliptic_dst, in_axes=(0,0))(helmholtz_rhs, dst_sol.H_mat)
    
    # Add homogeneous solutions to ensure mass conservation
    # [Nx,Ny] --> [Nx-1,Ny-1]
    # plot_field(dpsi_modes, "DPSI MODES")
    # print_debug_quantity(dpsi_modes, "DPSI_MODES") 
    dpsi_modes_i = center_average_2D(dpsi_modes)
    
    dpsi_modes_i_mean = einops.reduce(dpsi_modes_i, "... Nx Ny -> ... 1 1", reduction="mean")

    # print_debug_quantity(dpsi_modes_i_mean, "DPSI MODES MEAN") 
    # [Nz] / [Nx,Ny] --> [Nx,Ny] 
    alpha = -  dpsi_modes_i_mean / dst_sol.homsol_mean
    
    # print_debug_quantity(alpha, "ALPHAAA") 
    # print_debug_quantity(dst_sol.homsol_mean, "HOMSOL MEAN") 
    
    # [Nx,Ny]
    dpsi_modes += alpha * dst_sol.homsol
    
    # print_debug_quantity(dpsi_modes, "DPSI_MODES") 
    
    # [Nx,Ny]
    dpsi = jnp.einsum("lm , ...mxy -> lxy",layer_domain.A_mode_2_layer, dpsi_modes)
    
    # plot_field(dpsi, "DPSI")
    # print_debug_quantity(dpsi, "DPSI") 
    return dq, dpsi


def compute_psi_from_q(
    q: Array,
    params: PDEParams,
    domain: Domain,
    layer_domain: LayerDomain,
    dst_sol: DSTSolution,
    mask_psi: Mask=None,
) -> Array:
    
    # calculate beta-plane
    y_coords = center_average_2D(domain.grid_axis[-1])
    
    f_y = params.beta * (y_coords - params.y0)
    
    elliptical_rhs = center_average_2D(q - f_y)
    
    helmholtz_rhs = jnp.einsum("LM, ...MXY->...LXY", layer_domain.A_layer_2_mode, elliptical_rhs)
                               
    if dst_sol.capacitance_matrix is not None:
        # print_debug_quantity(dst_sol.capacitance_matrix, "CAPACITANCE MAT") 
        psi_modes = inverse_elliptic_dst_cmm(
            rhs=helmholtz_rhs, 
            H_matrix=dst_sol.H_mat,
            cap_matrices=dst_sol.capacitance_matrix,
            bounds_xids=mask_psi.irrbound_xids,
            bounds_yids=mask_psi.irrbound_yids,
            mask=mask_psi.values
        )
    else:
        psi_modes = jax.vmap(inverse_elliptic_dst, in_axes=(0,0))(helmholtz_rhs, dst_sol.H_mat)
        
        
    psi_modes_i = center_average_2D(psi_modes)
    psi_modes_i_mean = einops.reduce(psi_modes_i, "... Nx Ny -> ... 1 1", reduction="mean")

    # print_debug_quantity(dpsi_modes_i_mean, "DPSI MODES MEAN") 
    # [Nz] / [Nx,Ny] --> [Nx,Ny] 
    alpha = -  psi_modes_i_mean / dst_sol.homsol_mean
    
    # [Nx,Ny]
    psi_modes += alpha * dst_sol.homsol
    
    # print_debug_quantity(dpsi_modes, "DPSI_MODES") 
    
    # [Nx,Ny]
    psi = jnp.einsum("lm , ...mxy -> lxy", layer_domain.A_mode_2_layer, psi_modes)
    
    return psi