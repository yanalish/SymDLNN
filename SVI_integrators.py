import jax
import jax.numpy as jnp
from utility_funcs import newton_solver, newton_solver_pinv,secant_method
from SVI_funcs import SVI_funcs
import scipy
import jaxopt
import numpy as np
from NewtonLAX import NewtonLAX
from tqdm import tqdm
from functools import partial
from jax import lax

from jax.config import config
config.update("jax_enable_x64", True)
from tqdm import tqdm

def SVI_solver_TQs(discrete_lagrangian, stepsize, q0,p0, N):
    """SVI integrator in phase space -  determines both configuration and momentum in time
    Uses our implementation of Newton Rhapson to find the roots.
    """
    D1Ldk, _, D2Ldk = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ldk = jax.jit(D2Ldk)

    x0 = jnp.concatenate((q0,p0))
    result = [x0] 
    times = jnp.arange(N+1) * stepsize
    qk = q0
    pk = p0
    for i in tqdm(range(1, N+1)): # find values from q1 and p1 to qn and pn (not pythonic N)
        def temp(q_k1):
            return pk + D1Ldk(qk, q_k1, stepsize)
        
        q_k1 = newton_solver(temp, qk)
        p_k1 = D2Ldk(qk, q_k1, stepsize)
        
        x_k1 = jnp.concatenate((q_k1,p_k1))
        qk = q_k1
        pk = p_k1
        result.append(x_k1)

    return jnp.array(result), times  


def SVI_solver_TQs_NewtonLax(discrete_lagrangian, stepsize, q0, p0, N):
    """SVI integrator in phase space -  determines both configuration and momentum in time
    Uses our implementation of Newton Rhapson to find the roots.
    """
    D1Ldk, _, D2Ldk = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ldk = jax.jit(D2Ldk)

    x0 = jnp.vstack((q0,p0))
    result = [x0] 
    times = jnp.arange(N+1) * stepsize
    qk = q0
    pk = p0
    x_result = jnp.zeros((N+1, 2*q0.shape[0]))
    jax.lax.dynamic_update_slice(x_result, x0, (0,0))

    def temp(q_k1, pk, qk, stepsize):
        return pk + D1Ldk(qk, q_k1, stepsize)

    for i in range(1, N+1): # find values from q1 and p1 to qn and pn (not pythonic N)
        kwargs = {'pk':x_result[i-1,q0.shape[0]:2*q0.shape[0]], 'qk':x_result[i-1,0:q0.shape[0]], 'stepsize':stepsize}
        srf = jaxopt.ScipyRootFinding(method='hybr', jit=False, implicit_diff_solve=None, has_aux=False, optimality_fun=partial(temp,**kwargs))

        q_k1 = srf.run(x_result[i-1,0:q0.shape[0]])[0]
        p_k1 = D2Ldk(qk, q_k1, stepsize)
        
        x_k1 = jnp.vstack((q_k1,p_k1))
        jax.lax.dynamic_update_slice(x_result, x_k1, (i,0))

    return jnp.array(x_result), times  



def find_q1_scipyroot(D1Ldk, q0, p0, stepsize,method = "hybr"):
    """Called initial_condition() in some references
    Solves for q1 based on the initial conditions [q0,p0]

    To find q1, one needs to solve
    p0 + D1Ldk(Ld, q0, q1, stepsize) = 0

    We do this via jax wrapper of scipy root.
    """
    def temp(q1):
        return p0 + D1Ldk(q0, q1, stepsize)
    srf = jaxopt.ScipyRootFinding(method=method, implicit_diff_solve=None, has_aux=False, optimality_fun=temp)
    q1 = srf.run(q0)[0]
    return q1


def SVI_solver_Q_scipyroot(discrete_lagrangian, stepsize, q0, p0, N, method="hybr"):
    """SVI integrator on the configurational manifold- determines only the configuration in time
    but initializes based on both configuration and omentum initial condition. 

    Uses jax wrapper of scipy root to solve for the unknowns.
    """

    D1Ldk, D2Ld1k, _ = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ld1k = jax.jit(D2Ld1k)

    q1 = find_q1_scipyroot(D1Ldk, q0, p0, stepsize,method)

    q1_k = q0
    qk = q1

    result = [q0, q1] #q0, q1 
    times = jnp.arange(N+1) * stepsize
    for i in tqdm(range(2, N)): # find values from q2 to qn (not pythonic N) ##CHECK N RANGE, NEWT rha
        
        # As the optimality func, given x it has to have an output with the same pytree structure
        def discrete_EOM(qk_1):
            return D1Ldk(qk, qk_1, stepsize) + D2Ld1k(q1_k, qk, stepsize)

        srf = jaxopt.ScipyRootFinding(method=method, implicit_diff_solve=None, has_aux=False, optimality_fun=discrete_EOM)
        qk1 = srf.run(qk)[0]

        q1_k = qk
        qk = qk1
        result.append(qk)

    return jnp.array(result).T, times


def SVI_solver_Q_noinitstep_NewtLAX(discrete_lagrangian, stepsize, q0, q1, N):
    """SVI integrator on the configurational manifold- determines only the configuration in time
    but initializes based on q0 and q1.

    """

    D1Ldk, D2Ld1k, _ = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ld1k = jax.jit(D2Ld1k)

    q1_k = q0
    qk = q1
    minstepsize=1e-14
    maxiter=1000
    result = [q0, q1] #q0, q1 
    times = jnp.arange(N+1) * stepsize
    for i in tqdm(range(2, N+1)): # find values from q2 to qn (not pythonic N) ##CHECK N RANGE, NEWT rha
        
        # As the optimality func, given x it has to have an output with the same pytree structure
        def discrete_EOM(qk_1):
            return D1Ldk(qk, qk_1, stepsize) + D2Ld1k(q1_k, qk, stepsize)

        qk1 = NewtonLAX(discrete_EOM,2*qk-q1_k,minstepsize,maxiter)
        q1_k = qk
        qk = qk1
        result.append(qk)

    return jnp.array(result).T, times


def SVI_solver_Q_noinitstep_scipyroot(discrete_lagrangian, stepsize, q0, q1, N, method):
    """SVI integrator on the configurational manifold- determines only the configuration in time
    but initializes based on q0 and q1. 

    """

    D1Ldk, D2Ld1k, _ = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ld1k = jax.jit(D2Ld1k)

    q1_k = q0
    qk = q1
    result = [q0, q1] #q0, q1 
    times = jnp.arange(N+1) * stepsize
    for i in tqdm(range(2, N+1)): # find values from q2 to qn (not pythonic N)
        
        # As the optimality func, given x it has to have an output with the same pytree structure
        def discrete_EOM(qk_1):
            return D1Ldk(qk, qk_1, stepsize) + D2Ld1k(q1_k, qk, stepsize)

        srf = jaxopt.ScipyRootFinding(method=method, implicit_diff_solve=None, has_aux=False, optimality_fun=discrete_EOM)
        qk1 = srf.run(qk)[0]
        # qk1 = NewtonLAX(discrete_EOM,2*qk-q1_k,minstepsize,maxiter)
        q1_k = qk
        qk = qk1
        result.append(qk)

    return jnp.array(result).T, times

def SVI_solver_Q_noinitstep_scipyroot2(discrete_lagrangian, stepsize, q0, q1, N, method):
    """SVI integrator on the configurational manifold- determines only the configuration in time
    but initializes based on q0 and q1. 
    
    """

    D1Ldk, D2Ld1k, _ = SVI_funcs(discrete_lagrangian)
    D1Ldk = jax.jit(D1Ldk)
    D2Ld1k = jax.jit(D2Ld1k)

    q1_k = q0
    qk = q1
    result = [q0, q1] #q0, q1 
    times = jnp.arange(N+1) * stepsize
    for i in tqdm(range(2, N+1)): # find values from q2 to qn (not pythonic N) ##CHECK N RANGE, NEWT rha
        
        # As the optimality func, given x it has to have an output with the same pytree structure
        def discrete_EOM(qk_1):
            return D1Ldk(qk, qk_1, stepsize) + D2Ld1k(q1_k, qk, stepsize)

        #srf = jaxopt.ScipyRootFinding(method=method, implicit_diff_solve=None, has_aux=False, optimality_fun=discrete_EOM)
        #qk1 = srf.run(2*qk-q1_k)[0]
        qk1 = NewtonLAX(discrete_EOM,qk,10**(-10),200)
        #qk1 = newton_solver(discrete_EOM, 2*qk-q1_k)
        q1_k = qk
        qk = qk1
        result.append(qk)

    return jnp.array(result).T, times