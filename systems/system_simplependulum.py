"""Functions for Simple Pendulum system dynamics
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial

def lagrangian_simplepend(q, q_dot, m=1, l=1, g=9.8):
    # the lagrangian of a simple pendulum
    T = 0.5 * m*(l**2) * (q_dot)**2
    V = m * g *l*(1-jnp.cos(q)) 

    return (T - V)[0]

def f_analytical(state, t=0, m=1, l=1, g=9.8):
    # Given state x return x_dot *ANALYTICALLY*
    # ie compute the x_dot from equations of motion analytically
    q, qt = jnp.split(state, 2)
    qtt =-g/l*jnp.sin(q)
    return jnp.stack([qt, qtt])

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
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(f_autograd, lagrangian), initial_state, **kwargs)
    return f(initial_state)

@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m=1, l=1, g=9.8):
    L = partial(lagrangian_simplepend,m=m, l=l, g=g)
    return solve_lagrangian_autograd(L, initial_state, t=times, rtol=1e-10, atol=1e-10)

@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

