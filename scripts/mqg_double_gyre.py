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
from tqdm.notebook import tqdm, trange
from jaxtyping import Array, Float
import typing as tp
import einops

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


a_4 = 5.0e11  # 2.0e9 #
params = F_qg.PDEParams()



# Low Resolution
Nx, Ny = 97, 121
# High Resolution
# Nx, Ny = 769, 961

# Lx, Ly = 3840.0e3, 4800.0e3
Lx, Ly = 5_120.0e3, 5_120.0e3

x_domain = Domain.init_domain_1d(Lx, Nx)
y_domain = Domain.init_domain_1d(Ly, Ny)
x_domain, y_domain
xy_domain = x_domain * y_domain




mask = jnp.ones((Nx,Ny))
mask = mask.at[0].set(0.0)
mask = mask.at[-1].set(0.0)
mask = mask.at[:,0].set(0.0)
mask = mask.at[:,-1].set(0.0)

masks = Mask.init_mask(mask, variable="psi")


# heights
heights = [350.0, 750.0, 2900.0]

# reduced gravities
reduced_gravities = [0.025, 0.0125]

# initialize layer domain
layer_domain = LayerDomain(heights, reduced_gravities)

# from jaxsw._src.operators.functional import elliptical as F_elliptical
H_mat = F_qg.calculate_helmholtz_dst(xy_domain, layer_domain, params)


psi0 = jnp.zeros(shape=(layer_domain.Nz,) +xy_domain.Nx)

homsol = F_qg.compute_homogeneous_solution(
    psi0, 
    lambda_sq=einops.rearrange(layer_domain.lambda_sq, "Nz -> Nz 1 1"),
    H_mat=H_mat
)

# calculate homogeneous solution
homsol_i = center_average_2D(homsol) * masks.q.values

homsol_mean = einops.reduce(homsol_i, "Nz Nx Ny -> Nz 1 1", reduction="mean")

dst_sol = F_qg.DSTSolution(homsol=homsol, homsol_mean=homsol_mean, H_mat=H_mat)

# PV
q = F_qg.calculate_potential_vorticity(
    psi0, xy_domain, layer_domain, 
    params=params,
    masks_psi=masks.psi, 
    masks_q=masks.q
)

fn = jax.vmap(F_qg.advection_rhs, in_axes=(0,0,None,None,None,None,None))

div_flux = fn(
    q, psi0, xy_domain.dx[-2],xy_domain.dx[-1], 1, masks.u, masks.v
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




dq, dpsi = F_qg.qg_rhs(
    q=q, 
    psi=psi0, 
    domain=xy_domain,
    params=params, 
    layer_domain=layer_domain,
    dst_sol=dst_sol, 
    wind_forcing=wind_forcing,
    bottom_drag=bottom_drag,
    masks=masks
)




