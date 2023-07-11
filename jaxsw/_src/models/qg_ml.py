# import typing as tp
# import numpy as np

# # import jax.scipy as jsp
# import jax.numpy as jnp
# import equinox as eqx
# from jaxtyping import Array


# # def homogeneous_sol_layers(helmoltz_dst_mat, domain):
# #     # constant field
# #     num_layers = helmoltz_dst_mat.shape[0]
# #     constant_field = jnp.ones((num_layers, domain.size[0], domain.size[1])) / (
# #         domain.size[0] * domain.size[1]
# #     )

# #     s_solutions = jnp.zeros_like(constant_field)
# #     out = jax.vmap(F_elliptical.inverse_elliptic_dst, in_axes=(0, 0))(
# #         constant_field[:, 1:-1, 1:-1], helmoltz_dst_mat
# #     )
# #     s_solutions = s_solutions.at[:, 1:-1, 1:-1].set(out)

# #     homogeneous_sol = constant_field + s_solutions * beta

# #     return homogeneous_sol[:-1]
