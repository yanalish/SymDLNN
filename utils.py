"""
Various utility functions helpful to individual cases
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial # reduces arguments to function by making some subset implicit

def normalize_cp(state):
    # wrap generalized coordinates to [-pi, pi] for CP (cart pendulum)
    # DEPRECATED
    return jnp.array((state[0], (state[1]+np.pi)%(2*np.pi)-np.pi, state[2], state[3]))

def normalize_dp(state):
    # wrap generalized coordinates to [-pi, pi] for DP (double pendulum)
    # DEPRECATED
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

def rk4_step(f, x, t, h):
    """one step of runge-kutta integration
    
    Args:
        f (functional): Typically the equations of motion applied to the lagrangian
        x (nd.array): the state
        t (float): time value (0.0 most often used)
        h (float): the time step
    """
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def infnorm(a, b):
    """Computes the infinity norm between two input tensors a and b
    """
    row_maxes = []
    for i in range(len(a[0])):
        row_max = max(jnp.abs(a[:, i] - b[:, i]))
        row_maxes.append(row_max)
    return max(row_maxes)

def f2c_trajectory(proportion, stepsize, N, solver, discrete_lagrangian, q0, p0):
    """Integrates trajectory at integrator_ss, 
    and produces an N length output made stepsize stepsize
    """
    # Produce fine-grained trajectory
    integrator_ss = stepsize / proportion
    lengthN = (proportion) * N
    fine_trajectory, _ = solver(discrete_lagrangian, integrator_ss, q0, p0, lengthN)
    # Return an output as if steps were taken every stepsize step size
    coarse_trajectory = []
    
    for i in range(0, lengthN+1, proportion):
        coarse_trajectory.append(fine_trajectory[i,:])
    coarse_trajectory = np.array(coarse_trajectory)
    
    return coarse_trajectory