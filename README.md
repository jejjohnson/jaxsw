# Simple Ocean Models in JAX

## Motivation

Sea surface height is a gateway variable to other important ocean properties, e.g. sea surface temperature, geostrophic currents.
There are many massive models that attempt to model this, e.g. NEMO, MOM6, MITGCM. 
However they are very expensive and quite difficult to run. So there are many small models that are useful approximations, e.g. Quasi-Geostrophic and Shallow Water.
This repo attempts to showcase how we can use some modern tools to construct dynamical systems for PDEs.

What makes this different from the tons and tons of different implementations is that we
will be using JAX. 
JAX is basically numpy on steroids because the API is very similar but we also get some of the modern toolsets along with speed.
Most importantly, JAX is differentiable.
Having a differentiable model is important because it allows us to:

* Learn some of the hyperparameters if necessary
* Embed this in machine learning models where differentiability is needed

**Why Not PyTorch?**

We could easily just use PyTorch. However, there are some advantanges to JAX over other languages like PyTorch and TensorFlow:

* Familiar Numpy-Like API which is nice for newcomers in the scientific community
* CPU/GPU/TPU capabilities with minimal code changes
* Gradient Operators instead of storing the transformations in the tensors
* Functional-like language which is easier to read for newcomers
* Auto-Vectorization so we can easily parallize the operators for multiple dimensions without code changes (note: TensorFlow has this)
* JIT compilation speeds up the code by a lot (note: both PyTorch and TensorFlow has this)

---
## Applications

This library will be relatively general but this will be a development platform for the following applications:

* Generate Simulations
* Surrogate Models
* Data Assimilation

---
## Main Components

Without making it too complicated, we settled on a few key objects that the package will comprise of.

**Domain**

This will be the object to define the grids where all of the fields live. It will be easy to access the coordinates, boundaries, grids and cell volumes. We don't need to store the grid all of the time, instead we just generate it as we see fit.

**Operators**

This will be a suite of functions for different gradient calculations and combined operations for well-known equations. We will primarily focus on finite difference operators with the `finiteDiffX` package. At a later date, we can introduce spectral and finite volume methods.

**Integrators**

We will use the `diffrax` package to do the time integration. We'll use the method-of-lines technique to formulate all of our PDEs to calculate the RHS of the equation for the state at $t$. Then we can propagate them through the time integrator to get the state at $t+1$.

**Params, State & Equations of Motion**

We will have a general API for how we can keep store parameters, initialize states and pass thew both through the equation of motion.


**Configs**

We will use the `hydra` package to keep track of the configurations and to initialize parameters for experiments.

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