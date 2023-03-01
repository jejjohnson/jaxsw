import pytest
from ..domain.base import Domain
from .utils import check_discretization, DiscretizationError



def test_passes():
    
    # alternative way
    xmin = 0.0
    xmax = 2.0
    nx = 21
    domain1 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    

    xmin = 0.0
    xmax = 2.0
    nx = 21
    domain2 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    try:
        check_discretization(domain1, domain2)
    except DiscretizationError as exc:
        assert False, f"domain1,domain 2 raised an exception {exc}"


def test_fails():
    # alternative way
    xmin = 0.0
    xmax = 2.0
    nx = 21
    domain1 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    

    xmin = 0.0
    xmax = 1.0
    nx = 21
    domain2 = Domain.from_numpoints(xmin=(xmin,), xmax=(xmax,), N=(nx,))
    with pytest.raises(DiscretizationError):
        check_discretization(domain1, domain2)
    