# Summary

In this section, we look at how we can solve the QG equations using elements from this package.


## QG Equations

We are going to be using the formulation that is described in the Q-GCM model. The manual can be found [here](). We write the multi-layer QG equations in terms of the vorticity term, $q$, and the stream function term, $\psi$. We consider the stream function and the potential vorticity to be $N_Z$ stacked  isopycnal layers.


$$
\partial_t q_k + (u_kq_k)_x + (v_kq_k)_y = F_k + D_k
$$ (eq:qg_form_adv)

where the $F_k$ and $D_k$ are forcing terms for each layer, $k$.  The vorticity term is defined as

$$
q = 
\frac{1}{f_0} \boldsymbol{\nabla}_H^2\psi -
f_0\mathbf{A}\psi + \beta(y-y_0)+ \tilde{\mathbf{D}}
$$ (eq:qg_vorticity)

where $\tilde{D}$ is the dynamic topography and $\beta$ is the $\beta$-plane approximation. The term that links each of the layers together, $\mathbf{A}$, is a tri-diagonal matrix that can be written as

$$
\mathbf{A} =
\begin{bmatrix}
\frac{1}{H_1 g_1'} & \frac{-1}{H_1 g_2'} & \ldots & \ldots & \ldots  \\
\frac{-1}{H_2 g_1'} & \frac{1}{H_1}\left(\frac{1}{g_1'} + \frac{1}{g_2'} \right) & \frac{-1}{H_2 g_2'} & \ldots & \ldots  \\
\ldots & \ldots & \ldots & \ldots & \ldots \\
\ldots & \ldots & \frac{-1}{H_{n-1} g_{n-2}'} & \frac{1}{H_{n-1}}\left(\frac{1}{g_{n-2}'} + \frac{1}{g_{n-1}'} \right) & \frac{-1}{H_{n-1} g_{n-2}'}  \\
\ldots & \ldots& \ldots & \frac{-1}{H_n g_{n-1}'} & \frac{1}{H_n g_{n-1}'}   \\
\end{bmatrix}
$$ (eq:qg_A)


---
### Jacobian Form

We can also write this using the deterimant Jacobian formulation

$$
\partial_t q + \det\boldsymbol{J}\left(q, \psi \right) = 
BE + \frac{A_2}{f_0}\boldsymbol{\nabla}_H^4\psi -
\frac{A_4}{f_0}\boldsymbol{\nabla}_H^6\psi
$$ (eq:qg_form_detjac)

where the determinant Jacobian is defined as:

$$
\det\boldsymbol{J}\left( q, \psi \right) = 
\partial_x q \partial_y\psi - 
\partial_y q \partial_x\psi
$$ (eq:qg_detjac)

---
### Example Forcing + Diffusion Terms

The formulation used [in]() included the following terms

$$
\begin{aligned}
\text{Forcing}: && 
\boldsymbol{F} &= 
BE\\
\text{Diffusion}: && 
\boldsymbol{D} &= 
\frac{A_2}{f_0}\boldsymbol{\nabla}_H^4\psi -
\frac{A_4}{f_0}\boldsymbol{\nabla}_H^6\psi
\end{aligned}
$$


where B is a matrix containing the coefficients of the forcing terms and $e$ is the entrainment vector.

In the paper [[Thiry et al., 2023](https://doi.org/10.22541/essoar.167397445.54992823/v1)], they use the following method

$$
\begin{aligned}
\text{Hyperviscosity}: && 
\boldsymbol{D_1} &= 
-a_4\boldsymbol{\nabla}_H^6\psi\\
\text{Wind Forcing}: && 
\boldsymbol{F} &= 
\frac{\tau_0}{\rho_0H_1}\left[\partial_x\tau_y - \partial_y\tau_x, 0\cdots,0\right]\\
\text{Bottom Drag}: && 
\boldsymbol{D_2} &= 
\frac{\delta_{ek}}{2H_{N_Z}}
\left[0,\cdots,0,\Delta\psi_N\right]
\end{aligned}
$$


---
### Sea Surface Height

My of my applications are related to Sea Surface Height (SSH) and the QG model is a decent estimation. For example, the formulation that was used here was taken from the Q-GCM which is a fully-fledged ocean model which works for small-medium scale problems over specific regions.

SSH is linked to the QG equations via the stream function which we can write this as:

$$
\psi = \frac{g}{f_0}\eta
$$ (eq:qg_ssh_streamfunction)

This adds some additional interpretation how the vorticity term can be interpreted when dealing with the SSH over the globe.

$$
q_l = 
\underbrace{\boldsymbol{\nabla}_H \psi_l}_{\text{Dynamical}} +
\underbrace{(\mathbf{A}\psi)_k}_{\text{Thermal}} +
\underbrace{f_k}_{\text{Planetary}}
$$ (eq:qg_ssh_vorticity)

We also have . See {cite}`BFNQG` for more information about this term.


---
## Navier-Stokes Relation

This paragraph was largely borrowed from the following paper {cite}`FNOUNET`

We introduce vorticity, $\omega$, as the curl of the flow vorticity, i.e. $\omega = \nabla \times \boldsymbol{v}$.
This results in the incompressible 2D Navier-Stokes equations in the **scalar vorticity stream function form**.

$$
\begin{aligned}
\frac{\partial \omega}{\partial t} + 
\frac{\partial \psi}{\partial y}\frac{\partial \omega}{\partial x} +
\frac{\partial \psi}{\partial x}\frac{\partial \omega}{\partial y}&=
\frac{1}{\text{Re}}
\left(
    \frac{\partial^2\omega}{\partial\omega^2} + 
    \frac{\partial^2\omega}{\partial\omega^2}
\right), \hspace{10mm}
\left(\frac{\partial^2\psi}{\partial x^2} + \frac{\partial^2\psi}{\partial y^2}\frac{}{}\right) = -\omega
\end{aligned}
$$

where the *streamfunction* is defined as $v_x = \frac{\partial\psi}{\partial y}$ and $u_y = - \frac{\partial\psi}{\partial x}$.



## Sea Surface Height Edition

$$
\begin{aligned}
\partial_t q &= - \det\boldsymbol{J}(\psi,q) - \beta\partial_x\psi \\
\psi &= \frac{g}{f_0}\eta \\
q &= \nabla^2 \psi - \frac{f_0^2}{c_1^2}\psi \\
\psi &= \frac{f_0}{g}\eta \\
u &= -\partial_y\psi \\
v &= \partial_x\psi \\
f &= 2\Omega\sin\theta_0 + \frac{1}{R}2\Omega\cos\theta_0 y \\
f_0 &= \mathcal{E}[f] \\
L_R &= \frac{c_1}{f_0}
\end{aligned}
$$ (eq:qg_parts)



