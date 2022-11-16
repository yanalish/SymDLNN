"""Functions for Double Pendulum system dynamics
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial


def lagrangian(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    # the lagrangian of a double pendulum
    t1, t2 = q     # theta 1 and theta 2
    w1, w2 = q_dot # omega 1 and omega 2

    # kinetic energy (T)
    T1 = 0.5 * m1 * (l1 * w1)**2
    T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + 2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    T = T1 + T2

    # potential energy (V)
    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2

    return T - V


def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    # Given state x return x_dot *ANALYTICALLY*
    # ie compute the x_dot from equations of motion analytically
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    a2 = (l1 / l2) * jnp.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - (g / l1) * jnp.sin(t1)
    f2 = (l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return jnp.stack([w1, w2, g1, g2])


def f_autograd(lagrangian, state, t=None):
    # Given state x return x_dot *Via AUTOGRAD*
    # Name should be changed to f_autograd
    # ie compute the x_dot from equations of motion via autograd
    q, q_t = jnp.split(state, 2)
    q = q % (2*jnp.pi)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t)) @ (jax.grad(lagrangian, 0)(q, q_t) 
        - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])


def solve_lagrangian_autograd(lagrangian, initial_state, **kwargs):
    # Returns the outputs of the odeint(dx/dt, x0, kwargs) with the kwargs for time 
    ## Can be combined possibly for cleaner code
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(f_autograd, lagrangian), initial_state, **kwargs)
    return f(initial_state)


# Double pendulum dynamics via the rewritten Euler-Lagrange
@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=1, m2=1, l1=1, l2=1, g=9.8):
    L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    return solve_lagrangian_autograd(L, initial_state, t=times, rtol=1e-10, atol=1e-10)


@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

