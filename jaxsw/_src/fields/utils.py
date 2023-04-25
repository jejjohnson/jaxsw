from ..domain.base import Domain


class DiscretizationError(Exception):
    def __init__(self, d1, d2):
        self.message = f"Mismatched spatial discretizations\n{d1}\n{d2}"
        super().__init__(self.message)


def check_discretization(d1: Domain, d2: Domain):
    assert type(d1) == type(d2)

    if d1.dx != d2.dx or d1.xmin != d2.xmin or d1.xmax != d2.xmax:
        print(d1, d2)
        raise DiscretizationError(d1, d2)
