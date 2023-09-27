import finitediffx as fdx
from jaxsw._src.fields.finitediff import FDField


def difference(
    u: FDField,
    axis: int = 0,
    derivative: int = 1,
    accuracy: int = 5,
    method: str = "backward",
) -> FDField:
    # do finite difference
    u_values = fdx.difference(
        u.values,
        step_size=u.domain.dx[axis],
        axis=axis,
        derivative=derivative,
        accuracy=accuracy,
        method=method,
    )
    return FDField(values=u_values, domain=u.domain)
