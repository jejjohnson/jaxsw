import jax.numpy as jnp
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.operators.functional import grid as F_grid
import equinox as eqx
from jaxtyping import Array
from jaxsw._src.models.sw import Params, State


def enforce_boundaries(u: Array, component: str = "h", periodic: bool = False):
    if periodic:
        u = u.at[0, :].set(u[-2, :])
        u = u.at[-1, :].set(u[1, :])
    if component == "h":
        return u
    elif component == "u":
        return u.at[-2, :].set(jnp.asarray(0.0))
    elif component == "v":
        return u.at[:, -2].set(jnp.asarray(0.0))
    else:
        msg = f"Unrecognized component: {component}"
        msg += "\nNeeds to be h, u, or v"
        raise ValueError(msg)


class LinearShallowWater2D(DynamicalSystem):
    @staticmethod
    def boundary_f(state: State, component: str = "h"):
        if component == "h":
            return state
        elif component == "u":
            u = state.u.at[-2, :].set(jnp.asarray(0.0))
            return eqx.tree_at(lambda x: x.u, state, u)
        elif component == "v":
            v = state.v.at[:, -2].set(jnp.asarray(0.0))
            return eqx.tree_at(lambda x: x.v, state, v)
        else:
            msg = f"Unrecognized component: {component}"
            msg += "\nNeeds to be h, u, or v"
            raise ValueError(msg)

    @staticmethod
    def equation_of_motion(t: float, state: State, args) -> State:
        """2D Linear Shallow Water Equations

        Equation:
            ∂h/∂t + H (∂u/∂x + ∂v/∂y) = 0
            ∂u/∂t - fv = - g ∂h/∂x - ku
            ∂v/∂t + fu = - g ∂h/∂y - kv
        """

        # apply boundary conditions
        h: Array = enforce_boundaries(state.h, "h")
        u = enforce_boundaries(state.u, "u")
        v = enforce_boundaries(state.v, "v")
        # update state
        state = eqx.tree_at(lambda x: x.u, state, u)
        state = eqx.tree_at(lambda x: x.v, state, v)
        state = eqx.tree_at(lambda x: x.h, state, h)

        # apply RHS
        h_rhs, u_rhs, v_rhs = equation_of_motion(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        return state


def equation_of_motion(state: State, params: Params):
    h, u, v = state.h, state.u, state.v

    domain = params.domain

    # enforce boundaries
    h = enforce_boundaries(h, "h")
    v = enforce_boundaries(v, "v")
    u = enforce_boundaries(u, "u")

    # pad boundaries with edge values
    h_node = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    h_node = enforce_boundaries(h_node, "h")

    # PLANETARY VORTICITY
    planetary_vort = params.coriolis_param(domain)[1:-1, 1:-1]

    # ################################
    # HEIGHT Equation
    # ∂h/∂t = - H (∂u/∂x + ∂v/∂y)
    # ################################

    # finite difference
    # u --> h | top edge --> cell node | right edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    du_dx = F_grid.difference(
        u, step_size=domain.dx[0], axis=0, accuracy=1, method="left"
    )
    du_dx = du_dx[:-1, 1:-1]

    # v --> h | right edge --> cell node | top edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    dv_dy = F_grid.difference(
        v, step_size=domain.dx[1], axis=1, accuracy=1, method="right"
    )
    dv_dy = dv_dy[1:-1, :-1]

    # print("H_RHS")
    h_rhs = jnp.zeros_like(h)
    h_rhs = h_rhs.at[1:-1, 1:-1].set(-params.depth * (du_dx + dv_dy))

    # #############################
    # U VELOCITY
    #  ∂u/∂t = fv - g ∂h/∂x
    # #############################
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    v_on_u = planetary_vort * F_grid.interp_center(v)[1:, :-1]

    # H --> U
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    dhdx_on_u = (
        -params.gravity
        * F_grid.difference(h, axis=0, step_size=domain.dx[0], method="right")[1:, 1:-1]
    )

    u_rhs = jnp.zeros_like(h)
    u_rhs = u_rhs.at[1:-1, 1:-1].set(v_on_u + dhdx_on_u)

    # #############################
    # V - VELOCITY
    # ∂v/∂t = - fu - g ∂h/∂y
    # #############################
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    u_on_v = -planetary_vort * F_grid.interp_center(u)[:-1, 1:]

    # H --> U
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    dhdy_on_v = (
        -params.gravity
        * F_grid.difference(h, axis=1, step_size=domain.dx[1], method="right")[1:-1, 1:]
    )

    v_rhs = jnp.zeros_like(h)
    v_rhs = v_rhs.at[1:-1, 1:-1].set(u_on_v + dhdy_on_v)

    return h_rhs, u_rhs, v_rhs
