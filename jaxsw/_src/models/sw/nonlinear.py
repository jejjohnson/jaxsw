import jax.numpy as jnp
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.operators.functional import grid as F_grid
from jaxtyping import Array
import equinox as eqx
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


class ShallowWater2D(DynamicalSystem):
    @staticmethod
    def boundary_f(state, component: str = "h"):
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
    def equation_of_motion(t: float, state: State, args: Params):
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

    h = enforce_boundaries(h, "h", False)
    v = enforce_boundaries(v, "v", False)
    u = enforce_boundaries(u, "u", False)

    # pad
    # print("Ghost")
    h_node = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    h_node = enforce_boundaries(h_node, "h", False)

    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    h_on_u = F_grid.interp(h_node, axis=0, method="linear")[1:, 1:-1]

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    h_on_v = F_grid.interp(h_node, axis=1, method="linear")[1:-1, 1:]

    # hu, hv (interior only)
    # print("HU --> U,V")
    uh_on_u = jnp.zeros_like(h)
    vh_on_v = jnp.zeros_like(h)
    uh_on_u = uh_on_u.at[1:-1, 1:-1].set(h_on_u * u[1:-1, 1:-1])
    vh_on_v = vh_on_v.at[1:-1, 1:-1].set(h_on_v * v[1:-1, 1:-1])

    uh_on_u = enforce_boundaries(uh_on_u, "h", False)
    vh_on_v = enforce_boundaries(vh_on_v, "h", False)

    # finite difference
    # u --> h | top edge --> cell node | right edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    duh_dx = F_grid.difference(
        uh_on_u, step_size=domain.dx[0], axis=0, accuracy=1, method="right"
    )[:-1, 1:-1]

    # v --> h | right edge --> cell node | top edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    dvh_dy = F_grid.difference(
        vh_on_v, step_size=domain.dx[1], axis=1, accuracy=1, method="right"
    )[1:-1, :-1]

    # print("H_RHS")
    h_rhs = jnp.zeros_like(h)
    h_rhs = h_rhs.at[1:-1, 1:-1].set(-(duh_dx + dvh_dy))

    # planetary and relative vorticity
    planetary_vort = params.coriolis_param(domain)[1:-1, 1:-1]

    # relative vorticity
    # v --> q | right edge --> cell face | top edge --> cell node
    # [Nx+2,Nx+2] --> [Nx+1,Nx+2] --> [Nx,Ny]
    dv_dx = F_grid.difference(
        v, axis=0, step_size=domain.dx[0], accuracy=1, method="right"
    )[1:, 1:-1]
    # u --> q | top edge --> cell face | right edge --> cell node
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    du_dy = F_grid.difference(
        u, axis=1, step_size=domain.dx[1], accuracy=1, method="right"
    )[1:-1, 1:]

    relative_vort = dv_dx - du_dy

    # calculate potential vorticity
    # h --> q | cell node --> cell face | cell face --> cell node
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    h_on_vort = F_grid.interp_center(h_node, method="linear")[1:, 1:]

    # print("Potential VORTICITY")
    # [Nx+2,Ny+2]
    potential_vort = jnp.zeros_like(h)
    potential_vort = potential_vort.at[1:-1, 1:-1].set(
        (planetary_vort + relative_vort) / h_on_vort
    )

    # enforce boundaries
    potential_vort = enforce_boundaries(potential_vort, "h", False)

    # flux on v (top edge) ---> vort (cell center)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny+2]
    vh_on_q = F_grid.interp(vh_on_v, axis=0, method="linear")[1:]
    # PV: [Nx+2,Ny+2] --> [Nx,Ny+1], [Nx,Ny+1] --> [Nx,Ny]
    adv_rhs = F_grid.interp(potential_vort[1:-1] * vh_on_q, axis=1, method="linear")

    # PV: [Nx,Ny+1] --> [Nx,Ny]
    adv_rhs = adv_rhs[:, :-1]

    # u --> h | top edge --> cell node | right edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    u2_on_h = F_grid.interp(u**2, axis=0, method="linear")[:-1, 1:-1]
    # v --> h | right edge --> cell node | top edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    v2_on_h = F_grid.interp(v**2, axis=1, method="linear")[1:-1, :-1]

    # kinetic energy
    ke_on_h = jnp.zeros_like(h)
    ke_on_h = ke_on_h.at[1:-1, 1:-1].set(0.5 * (u2_on_h + v2_on_h))

    # enforce boundary conditions
    ke_on_h = enforce_boundaries(ke_on_h, "h", False)

    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    dke_on_u = -F_grid.difference(ke_on_h, step_size=domain.dx[0], axis=0)[1:, 1:-1]

    # pressure work
    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    p_work = (
        -params.gravity * F_grid.difference(h, step_size=domain.dx[0], axis=0)[1:, 1:-1]
    )

    u_rhs = jnp.zeros_like(h)
    u_rhs = u_rhs.at[1:-1, 1:-1].set(adv_rhs + p_work + dke_on_u)

    # flux on u (top edge) ---> vort (cell center)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx+2,Ny]
    uh_on_q = F_grid.interp(uh_on_u, axis=1, method="linear")[:, 1:]
    # PV: [Nx+2,Ny+2] --> [Nx+1,Ny+2], [Nx+1,Ny+2] --> [Nx,Ny]
    adv_rhs = -F_grid.interp(potential_vort[:, 1:-1] * uh_on_q, axis=0, method="linear")

    # PV: [Nx+1,Ny] --> [Nx,Ny]
    adv_rhs = adv_rhs[:-1]

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    dke_on_v = -F_grid.y_difference_2D_(ke_on_h[1:-1, 1:], step_size=domain.dx[1])

    # pressure work
    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    p_work = (
        -params.gravity * F_grid.difference(h, axis=1, step_size=domain.dx[1])[1:-1, 1:]
    )

    v_rhs = jnp.zeros_like(h)
    v_rhs = v_rhs.at[1:-1, 1:-1].set(adv_rhs + p_work + dke_on_v)

    return h_rhs, u_rhs, v_rhs
