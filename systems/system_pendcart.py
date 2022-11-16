"""Functions for Pendulum on a cart 
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial

m = 1
M = 1
l = 1
alpha0 = m * l**2
beta0 = m * l
gamma0 = M + m
D0 = -9.81 * m * l

def lagrangian_pendcart(q, q_dot, alpha=alpha0, beta=beta0, gamma=gamma0, D=D0):
    # A.M. Bloch, 2nd ed, p42, (1.9.23)
    # q[0] = s, q[1] = phi
    L = 0.5*(alpha*q_dot[1]**2 + 2*beta*jnp.cos(q[1]) *
             q_dot[0]*q_dot[1]+gamma*q_dot[0]**2) + D*jnp.cos(q[1])
    return L

def hamiltonian_pendcart(q, pm, alpha=alpha0, beta=beta0, gamma=gamma0, D=D0):
    H = 0.5*(alpha*pm[1]**2 + 2*beta*
        pm[0]*pm[1]+gamma*pm[0]**2) - D*jnp.cos(q[1])
    return H

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
    # Can be combined possibly for cleaner code
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(f_autograd, lagrangian), initial_state, **kwargs)
    return f(initial_state)

# Pendulum on a cart via the rewritten Euler-Lagrange
@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, alpha=alpha0, beta=beta0, gamma=gamma0, D=D0):
    L = partial(lagrangian_pendcart, alpha=alpha, beta=beta, gamma=gamma, D=D)
    return solve_lagrangian_autograd(L, initial_state, t=times, rtol=1e-10, atol=1e-10)


