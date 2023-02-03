# A Shallow Water Model written in JAX

## Motivation

Sea surface height is a gateway variable to other important ocean properties, e.g. sea surface temperature, geostrophic currents.
There are many massive models that attempt to model this, e.g. NEMO, MOM6, MITGCM. However they are very expensive and
quite difficult to run. So there are many small models that are useful approximations, e.g. Quasi-Geostrophic and Shallow Water.
This repo attempts to have. What makes this different from the tons and tons of different implementations is that we
will be using JAX. Having a differentiable model is important because it allows us to:

* Learn some of the hyperparameters if necessary
* Embed this in machine learning models where differentiability is needed

**Why Not PyTorch?**

We could easily just use PyTorch. However, there are some advantanges to JAX over other languages like PyTorch and TensorFlow:

* Familiar Numpy-Like API which is nice for newcomers in the scientific community
* CPU/GPU/TPU capabilities with minimal code changes
* Gradient Operators instead of storing in tensors

---
## Applications

This library will be relatively general but this will be a development platform for the following applications:

* Surrogate Models
* Data Assimilation

---
## Main Components

Without making it too complicated, we settled on a few key objects that the package will comprise of.

**Grid Object**

This will be the object to define the grids where all of the variables live.

**Boundary Conditions Object**

This will be the object which handles all of the boundary conditions.

**Integrators**

This will be the object which handles all of the integration steps.

**Operators**

This will be a suite of functions for different gradient calculations and combined operations for well-known equations.

---
## Installation

```bash
conda env create -f environments/jax_linux_cpu.yaml
```

---
## Contributions



---
## Acknowledgements

* [`qg_utils`](https://github.com/bderembl/qgutils) - useful functions for dealing with QG equations
* [`jaxdf`](https://github.com/ucl-bug/jaxdf) - Nice API for defining operators for PDEs.
* [`jax-cfd`](https://github.com/google/jax-cfd) - Nice API for defining PDEs
* [`invobs-data-assimilation`](https://github.com/googleinterns/invobs-data-assimilation) - Nice API for Dynamical Systems
* [`MASSH`](https://github.com/leguillf/MASSH) - The differentiable QG and SW models applied to sea surface height interpolation.
* [`qgm_pytorch`](https://github.com/louity/qgm_pytorch) - Quasi-Geostrophic Model in PyTorch
* [`QGNet`](https://github.com/redouanelg/qgsw-DI/blob/master/QGNET/QG_PyTorch.ipynb) - QG implementation in PyTorch with convolutions.