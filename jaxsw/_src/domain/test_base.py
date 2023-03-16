import typing as tp
import pytest
from jaxsw._src.domain.base import Domain


class DemoDomain(tp.NamedTuple):
    xmin: tp.Iterable[float]
    xmax: tp.Iterable[float]
    dx: tp.Iterable[float]


def test_1d_domain():
    demo = DemoDomain(xmin=(0.0,), xmax=(2.0,), dx=(0.1,))
    domain = Domain(xmin=demo.xmin, xmax=demo.xmax, dx=demo.dx)

    assert domain.ndim == 1
    assert domain.size == (21,)
    assert domain.grid.shape == (21, 1)
    assert domain.cell_volume == 0.1


def test_2d_domain():
    demo = DemoDomain(xmin=(0.0, 0.0), xmax=(2.0, 10.0), dx=(0.1, 0.5))
    domain = Domain(xmin=demo.xmin, xmax=demo.xmax, dx=demo.dx)

    assert domain.ndim == 2
    assert domain.size == (21, 21)
    assert domain.grid.shape == (21, 21, 2)
    assert domain.cell_volume == 0.05
