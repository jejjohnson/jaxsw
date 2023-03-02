import typing as tp
import equinox as eqx
import jax.numpy as jnp
from ..domain.base import Domain
from jaxtyping import Array, Float
from .utils import check_discretization

# TODO: add power operation
# TODO: add negative operation

class Field(eqx.Module):
    values: Float
    domain: Domain
    
    @classmethod
    def init_from_fn(cls, domain: Domain, fn: tp.Callable, *kwargs):
        
        values = fn(domain.grid, *kwargs)
        
        return cls(values=values, domain=domain)
    
    def replace_values(self, values):
        return self.__class__(values, self.domain)
    
    def binop(self, other, fn: tp.Callable):

        # check discretization
        check_discretization(self.domain, other.domain)
        values = fn(self.values, other.values)
        return Field(values, self.domain)
    
    def __getitem__(self, idx):
        return self.values[idx]

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)
    
    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)
    
#     def __bool__(self, other):
#         return binop(self, other, lambda x, y: bool(x))

#     def __neg__(self):
#         return binop(self, other, lambda x, y: 

    # def __pow__(self, other):
    #     return Field(self.values**2, self.domain)

#     def __rpow__(self, other):
#         return binop(self, other, 

    def __truediv__(self, other):
        return self.binop(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self.binop(other, lambda x, y: y / x)



    