import os
import autoroot
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.config import config
import numpy as np
import numba as nb
import pandas as pd
import equinox as eqx
import finitediffx as fdx
import diffrax as dfx
import xarray as xr
import einops
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange, repeat, reduce
from tqdm import tqdm, trange
from jaxtyping import Array, Float
import typing as tp
import einops
config.update("jax_enable_x64", True)

from jaxsw._src.operators.functional import advection as F_adv
from jaxsw._src.operators.functional import geostrophic as F_geos
from jaxsw._src.operators.functional import cgrid as F_cgrid
from jaxsw._src.operators.functional import grid as F_grid
from jaxsw._src.operators.functional import interp as F_interp
from jaxsw._src.boundaries.helmholtz import enforce_boundaries_helmholtz
from jaxsw._src.models import qg as F_qg
import jaxsw._src.domain.base_v2 as Domain
from jaxsw._src.domain.qg import create_qg_multilayer_mat, LayerDomain
from jaxsw._src.masks import Mask
from jaxsw._src.operators.functional.finitevol.interp import x_average_2D, y_average_2D, center_average_2D
from jaxsw._src.operators.functional.interp import flux as F_flux
from jaxsw._src.operators.functional.dst import (
    helmholtz_dst, 
    laplacian_dst, 
    inverse_elliptic_dst, 
    dstI2D, dstI1D,
    inverse_elliptic_dst_cmm,
    helmholtz_fn,
    compute_capacitance_matrices
)



def plot_field(field):
    num_axis = len(field)
    fig, ax = plt.subplots(ncols=num_axis, figsize=(8, 2))

    for i in range(num_axis):
        pts = ax[i].imshow(field[i].T, origin="lower", cmap="coolwarm")
        plt.colorbar(pts)

    plt.tight_layout()
    plt.show()

def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = np.min(quantity)
    max_ = np.max(quantity)
    mean_ = np.mean(quantity)
    median_ = np.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | {min_:.6e} | {mean_:.6e} | {median_:.6e} | {max_:.6e}"
    )


# Low Resolution
# Nx, Ny = 128, 128
# Medium Resolution
Nx, Ny = 256, 256
# High Resolution
# Nx, Ny = 769, 961

# Lx, Ly = 3840.0e3, 4800.0e3
Lx, Ly = 5_120.0e3, 5_120.0e3

dx, dy = Lx / Nx, Ly / Ny

xy_domain = Domain.Domain(
    xmin=(0.0,0.0), 
    xmax=(Lx,Ly),
    Lx=(Lx,Ly),
    Nx=(Nx, Ny), dx=(dx, dy)
)

params = F_qg.PDEParams(y0=0.5 * Ly)


# # octogonal domain
# domain_type = "octogonal"
# mask = np.ones((Nx-1,Ny-1))
# for i in range(Nx//4):
#     for j in range(Ny//4):
#         if i+j < min(Nx//4, Ny//4):
#             mask[i,j] = 0.
#             mask[i,-1-j] = 0.
#             mask[-1-i,j] = 0.
#             mask[-1-i,-1-j] = 0.
            
# masks = Mask.init_mask(mask, variable="q")

domain_type = "rectangular"

mask = jnp.ones((Nx,Ny))
mask = mask.at[0].set(0.0)
mask = mask.at[-1].set(0.0)
mask = mask.at[:,0].set(0.0)
mask = mask.at[:,-1].set(0.0)

masks = Mask.init_mask(mask, variable="psi")


# heights
# heights = [350.0, 750.0, 2900.0]
heights = [400.0, 1_100.0, 2_600.0]

# reduced gravities
reduced_gravities = [0.025, 0.0125]

# initialize layer domain
layer_domain = LayerDomain(heights, reduced_gravities, correction=False)
print_debug_quantity(layer_domain.A_layer_2_mode, "CL2M")
print_debug_quantity(layer_domain.A_mode_2_layer, "CM2L")

# from jaxsw._src.operators.functional import elliptical as F_elliptical
H_mat = F_qg.calculate_helmholtz_dst(xy_domain, layer_domain, params)


psi0 = jnp.zeros(shape=(layer_domain.Nz,) + xy_domain.Nx)
# psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_000y_360d.npy")[0]
lambda_sq = params.f0**2 *einops.rearrange(layer_domain.lambda_sq, "Nz -> Nz 1 1")
homsol = F_qg.compute_homogeneous_solution(
    psi0, 
    lambda_sq=lambda_sq,
    H_mat=H_mat
)
print_debug_quantity(homsol, "HOMSOL")

# CALCULATE CAPCITANCE MATRIX
if domain_type == "octogonal":
    cap_matrices = compute_capacitance_matrices(
        H_mat, 
        masks.psi.irrbound_xids,
        masks.psi.irrbound_yids
    )
else:
    cap_matrices = None


# calculate homogeneous solution
homsol_i = center_average_2D(homsol) * masks.q.values

homsol_mean = einops.reduce(homsol_i, "Nz Nx Ny -> Nz 1 1", reduction="mean")
print_debug_quantity(homsol_mean, "HOMSOL MEAN")

dst_sol = F_qg.DSTSolution(
    homsol=homsol, 
    homsol_mean=homsol_mean, 
    H_mat=H_mat,
    capacitance_matrix=cap_matrices
)

# PV
q = F_qg.calculate_potential_vorticity(
    psi0, xy_domain, layer_domain, 
    params=params,
    masks_psi=masks.psi, 
    masks_q=masks.q
)
print_debug_quantity(psi0, "PSI0")
print_debug_quantity(q, "Q0")

fn = jax.vmap(F_qg.advection_rhs, in_axes=(0,0,None,None,None,None,None))

div_flux = fn(
    q, psi0, xy_domain.dx[-2],xy_domain.dx[-1], 3, masks.u, masks.v
    # q, psi0, xy_domain.dx[-2],xy_domain.dx[-1], 1, None, None,
)

bottom_drag = F_qg.calculate_bottom_drag(
    psi=psi0, 
    domain=xy_domain, 
    H_z=layer_domain.heights[-1],
    f0=params.f0, 
    masks_psi=masks.psi
)

wind_forcing = F_qg.calculate_wind_forcing(
    domain=xy_domain,
    H_0=layer_domain.heights[0],
    tau0=0.08/1_000.0,
)

class State(eqx.Module):
    q: Array
    psi: Array
    
    
def vector_field(t: float, state: State, args) -> State:
    
    dq, dpsi = F_qg.qg_rhs(
        q=state.q, psi=state.psi,
        domain=xy_domain, params=params,
        layer_domain=layer_domain,
        dst_sol=dst_sol,
        wind_forcing=wind_forcing,
        bottom_drag=bottom_drag,
        masks=masks
    )
    
    state = eqx.tree_at(lambda x: x.q, state, dq)
    state = eqx.tree_at(lambda x: x.psi, state, dpsi)
    
    return state


def time_step(state: State, dt: float) -> State:
    
    
    
    # 1st order time derivative (Euler)
    state_new = vector_field(0, state, None)
    
    
    # extract new state
    dq0 = state_new.q
    dpsi0 = state_new.psi
    
    # do time step
    q = state.q + dt * dq0
    psi = state.psi + dt * dpsi0
    
    # update state
    state = eqx.tree_at(lambda x: x.q, state, q)
    state = eqx.tree_at(lambda x: x.psi, state, psi)
    
    # ===============
    # 2nd order?
    # ===============
    state_new = vector_field(0, state, None)
    
    
    # extract new state
    dq1 = state_new.q
    dpsi1 = state_new.psi
    
    # do time step
    q = state.q + (dt/4.0) * (dq1 - 3.0*dq0)
    psi = state.psi + (dt/4.0) * (dpsi1 - 3.0*dpsi0)
    
    # update state
    state = eqx.tree_at(lambda x: x.q, state, q)
    state = eqx.tree_at(lambda x: x.psi, state, psi)
    
    
    # ===============
    # 3rd order?
    # ===============
    state_new = vector_field(0, state, None)
    
    # extract new state
    dq2 = state_new.q
    dpsi2 = state_new.psi
    
    # do time step
    q = state.q + (dt/12.0) * (8.0 * dq2 - dq1 - dq0)
    psi = state.psi + (dt/12.0) * (8.0* dpsi2 - dpsi1 - dpsi0)
    
    # update state
    state = eqx.tree_at(lambda x: x.q, state, q)
    state = eqx.tree_at(lambda x: x.psi, state, psi)
    
    
    
    return state

state_t0 = State(q=q, psi=psi0)


jitted_time_step = jax.jit(time_step)

psi0 = jnp.zeros(shape=(layer_domain.Nz,) + xy_domain.Nx)
# psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_000y_360d.npy")[0]

q = F_qg.calculate_potential_vorticity(
    psi0, xy_domain, layer_domain, 
    params=params,
    masks_psi=masks.psi, 
    masks_q=masks.q
)

state = State(q=q, psi=psi0)



dt = 4_000

_ = jitted_time_step(state, dt)

tmin = 0.0
num_days = 10*360
tmax = pd.to_timedelta(num_days, unit="days").total_seconds()
num_save = 20

ts = jnp.arange(tmin, tmax+dt, dt)





for t in tqdm(ts):
    state = jitted_time_step(state, dt)




    
plot_field(state.psi)
plot_field(state.q)
n_years = num_days/365

output_dir = "/Users/eman/code_projects/data/qg_runs"
fname = os.path.join(output_dir, f'psi_{n_years:.3f}y_{num_days:.2f}d_{domain_type}.npy')
np.save(fname, np.asarray(state.psi).astype('float32'))

fname = os.path.join(output_dir, f'q_{n_years:.3f}y_{num_days:.2f}d_{domain_type}.npy')
np.save(fname, np.asarray(state.q).astype('float32'))