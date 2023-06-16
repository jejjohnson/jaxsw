# Summary

In this section, we look at how we can solve the QG equations using elements from this package.

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
$$ (eq:qg_full)



