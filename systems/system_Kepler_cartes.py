"""Functions for KEPLER problem
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial


def Kepler_lagrangian_cartes(q, qdot, m1 = 6*10**24, m2 = 100, gamma = 6.673*10**(-26)):
    q1, q2 = np.split(q,2)
    q1_dot, q2_dot = jnp.split(qdot,2)
    L = (0.5) * m2 *(q1_dot**2 + q2_dot**2) + gamma*m1*m2/jnp.sqrt(q1**2 + q2**2)
    return L.squeeze(-1)


def f_autograd(lagrangian, state, t=None):
    # Given state x return x_dot *Via AUTOGRAD*
    q, qt = jnp.split(state, 2)
    qtt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, qt)) @ (jax.grad(lagrangian, 0)(q, qt) 
        - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, qt) @ qt))
    return jnp.hstack([qt, qtt])


def solve_lagrangian_autograd(lagrangian, initial_state, **kwargs):
    # Returns the outputs of the odeint(dx/dt, x0, kwargs) with the kwargs for time 
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(f_autograd, lagrangian), initial_state, **kwargs)
    return f(initial_state)


@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times):
    L = partial(Kepler_lagrangian_cartes)
    return solve_lagrangian_autograd(L, initial_state, t=times, rtol=1e-10, atol=1e-10)


def hamiltonian_Kepler_cart(q, pm, m1 = 6*10**24, m2 = 1, gamma = 6.673*10**(-26)):
    H = (0.5) * m2 *(pm[0]**2 + pm[1]**2) - gamma*m1*m2/jnp.sqrt(q[0]**2 + q[1]**2)
    return H