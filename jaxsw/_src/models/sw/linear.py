import jax.numpy as jnp
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.operators.functional import grid as F_grid
import equinox as eqx
import finitediffx as fdx
from jaxtyping import Array
from jaxsw._src.models.sw import Params, State


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
    def equation_of_motion_h(state: State, params: Params) -> Array:
        """
        Equation:
             ∂h/∂t + H (∂u/∂x + ∂v/∂y) = 0
        """
        # parse state container
        h, u, v = state.h, state.u, state.v

        # parse params container
        depth, domain = params.depth, params.domain

        # create empty matrix
        h_rhs = jnp.zeros_like(h)

        # create RHS
        du_dx = fdx.difference(
            u, axis=0, accuracy=1, method="backward", step_size=domain.dx[0]
        )
        dv_dy = fdx.difference(
            v, axis=1, accuracy=1, method="backward", step_size=domain.dx[1]
        )

        # set the interior points only
        h_rhs = h_rhs.at[1:-1, 1:-1].set(
            -depth * (du_dx[1:-1, 1:-1] + dv_dy[1:-1, 1:-1])
        )
        return h_rhs

    @staticmethod
    def equation_of_motion_u(state: State, params: Params) -> State:
        """Equation of Motion for the u-component

        Equation:
            ∂u/∂t = fv - g ∂h/∂x
        """
        # parse state and params
        h, u, v = state.h, state.u, state.v
        gravity, domain = params.gravity, params.domain

        coriolis = params.coriolis_param(domain)

        u_rhs = jnp.zeros_like(u)

        v_avg = F_grid.center_average_2D(v[1:, :-1], padding="valid")
        v_avg *= coriolis[1:-1, 1:-1]

        dh_dx = fdx.difference(
            h, axis=0, accuracy=1, method="forward", step_size=domain.dx[0]
        )
        dh_dx *= -gravity

        u_rhs = u_rhs.at[1:-1, 1:-1].set(v_avg + dh_dx[1:-1, 1:-1])

        return u_rhs

    @staticmethod
    def equation_of_motion_v(state: State, params: Params) -> Array:
        """Equation of motion for v-component
        Equation:
            ∂v/∂t = - fu - g ∂h/∂y
        """
        # parse state and parameters
        h, u, v = state.h, state.u, state.v
        gravity, domain = params.gravity, params.domain

        coriolis = params.coriolis_param(domain)

        v_rhs = jnp.zeros_like(v)

        u_avg = F_grid.center_average_2D(u[:-1, 1:], padding="valid")
        u_avg *= -coriolis[1:-1, 1:-1]

        dh_dy = fdx.difference(
            h, axis=1, accuracy=1, method="forward", step_size=domain.dx[1]
        )
        dh_dy *= -gravity

        v_rhs = v_rhs.at[1:-1, 1:-1].set(u_avg + dh_dy[1:-1, 1:-1])
        return v_rhs

    @staticmethod
    def equation_of_motion(t: float, state: State, args) -> State:
        """2D Linear Shallow Water Equations

        Equation:
            ∂h/∂t + H (∂u/∂x + ∂v/∂y) = 0
            ∂u/∂t - fv = - g ∂h/∂x - ku
            ∂v/∂t + fu = - g ∂h/∂y - kv
        """

        # apply boundary conditions
        state = LinearShallowWater2D.boundary_f(state, "h")
        state = LinearShallowWater2D.boundary_f(state, "u")
        state = LinearShallowWater2D.boundary_f(state, "v")

        # apply RHS
        h_rhs = LinearShallowWater2D.equation_of_motion_h(state, args)
        v_rhs = LinearShallowWater2D.equation_of_motion_v(state, args)
        u_rhs = LinearShallowWater2D.equation_of_motion_u(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        # # apply boundary conditions
        # state = LinearShallowWater2D.boundary_f(state, "h")
        # state = LinearShallowWater2D.boundary_f(state, "u")
        # state = LinearShallowWater2D.boundary_f(state, "v")
        return state
