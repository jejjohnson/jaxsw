# Summary


In this tutorial, I go step-by-step through the original [12 Steps to Navier-Stokes]() that was created by Lorena Barber.
However, I use the API from this library which uses many other libraries as the backbone, e.g. [FiniteDiffX](), [Diffrax](), [jaxdf](), and [kernex]().


## 1D Problems



[**Linear Convection**](1.1_linear_convection.ipynb)

$$
\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0
$$ (eq:linear_convection_1d)


[**Diffusion**](1.2_diffusion_1d.ipynb)

$$
\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}
$$  (eq:diffusion_1d)


[**Advection Diffusion**](1.3_burgers_1d.ipynb)

$$
\begin{aligned}
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} &= \nu\frac{\partial^2 u}{\partial x^2}
\end{aligned}
$$ (eq:burgers_1d)




## 2D Problems


[**Linear Convection**](2.1_linear_convection_2d.ipynb)

$$
\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} + c\frac{\partial u}{\partial y} = 0
$$ (eq:linear_convection_2d)


[**Nonlinear Convection**](2.2_nonlinear_convection_2d.ipynb)

$$
\begin{aligned}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} &= 0 \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} &= 0 
\end{aligned}
$$ (eq:nonlinear_convection_2d)


[**Diffusion**](2.3_diffusion_2d.ipynb)

$$
\begin{aligned}
\frac{\partial u}{\partial t} &= \nu\frac{\partial^2 u}{\partial x^2} + \nu\frac{\partial^2 u}{\partial y^2}
\end{aligned}
$$ (eq:2d_diffusion)


[**Burgers Equation**](2.4_burgers_2d.ipynb)


$$
\begin{aligned}
\frac{\partial u}{\partial t} &+ 
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = 
\nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
\frac{\partial v}{\partial t} &+ 
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = 
\nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
\end{aligned}
$$ (eq:burgers_2d)


## Elliptical Equations


[**Laplace's Equation**](3.1_laplace.ipynb)

$$
\begin{aligned}
\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = 0
\end{aligned}
$$ (eq:laplace_2d)


[**Poisson's Equation**](3.2_poisson.ipynb)

$$
\begin{aligned}
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = \nabla u = b
\end{aligned}
$$ (eq:poisson_2d)


**Navier-Stokes** (**TODO**)

$$
\begin{aligned}
\frac{\partial u}{\partial t} &+ 
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = 
- \frac{1}{\rho}\frac{\partial p}{\partial x} +
\nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
\frac{\partial v}{\partial t} &+ 
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = 
- \frac{1}{\rho}\frac{\partial p}{\partial y} +
\nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) \\
\frac{\partial^2 p}{\partial x^2} &+ \frac{\partial^2 p}{\partial y^2}  =
-\rho\left( \frac{\partial u}{\partial x}\frac{\partial u}{\partial x} +
2 \frac{\partial u}{\partial y}\frac{\partial v}{\partial x} +
\frac{\partial v}{\partial y}\frac{\partial v}{\partial y}\right)
\end{aligned}
$$ (eq:ns_2d)