import typing as tp
from jaxsw._src.fields.base import Field
from jaxsw._src.utils.spectral import calculate_fft_freq
from .utils import check_discretization

# from jaxsw._src.operators.functional import spectral as F_spectral  # calculate_fft_freq
from functools import cached_property


class SpectralField(Field):
    @cached_property
    def k_vec(self):
        k_vec = [
            calculate_fft_freq(Nx=Nx, Lx=Lx)
            for Nx, Lx in zip(self.domain.Nx, self.domain.Lx)
        ]
        return k_vec

    def binop(self, other, fn: tp.Callable):
        # check discretization
        check_discretization(self.domain, other.domain)
        values = fn(self.values, other.values)
        return SpectralField(values, self.domain)
