import jax
from jax import jacrev
import jax.numpy as jnp
import numpy as np
import scipy

def newton_solver(func, x0, tol=1e-6, maxiter=100):
    """Solves the multi-variable problem with multi-variable optimization method
    known as Newton's method

    Returns the solution to an equation of the form f(x_vec) = 0 

    Args:
        func (function): the F(x_vec) funciton
        x0 (nd.array): initial guess at solution
        tol (float): gives the tolerance for the accuracy of the solution
        maxiter (int): sets the maximum number of iterations the solver attempts
            to find a solution

    Returns:
        x1 (nd.array): the solvers solution (same shape as x0)   
    """
    def func_jacobian(x):
        return jacrev(func)(x)
    #print(np.linalg.cond(-func_jacobian(x0)))
    delta_x = jnp.linalg.inv(-func_jacobian(x0)) @ func(x0)
    x1 = x0 + delta_x
    x0 = x1
    iteration = 0

    while jnp.linalg.norm(delta_x) >= tol and iteration < maxiter:
        #print(jnp.linalg.norm(delta_x))
        delta_x = jnp.linalg.inv(-func_jacobian(x0)) @ func(x0)
        x1 = x0 + delta_x
        x0 = x1
        #print('NewtRha iter= ',iteration, ' with norm delta_x= ', jnp.linalg.norm(delta_x))
        iteration +=1 

    return x1


def newton_solver_pinv(func, x0, tol=1e-7, maxiter=100):
    """Solves the multi-variable problem with multi-variable optimization method
    known as Newton's method

    Returns the solution to an equation of the form f(x_vec) = 0 

    Args:
        func (function): the F(x_vec) funciton
        x0 (nd.array): initial guess at solution
        tol (float): gives the tolerance for the accuracy of the solution
        maxiter (int): sets the maximum number of iterations the solver attempts
            to find a solution

    Returns:
        x1 (nd.array): the solvers solution (same shape as x0)   
    """
    x00 =x0
    func_jacobian = jax.jacfwd(func)
    delta_x = jnp.linalg.pinv(-func_jacobian(x0)) @ func(x0)
    x1 = x0 + delta_x
    x0 = x1
    iteration = 0

    while jnp.linalg.norm(delta_x) >= tol and iteration < maxiter:
        delta_x = jnp.linalg.pinv(-func_jacobian(x0)) @ func(x0)
        x1 = x0 + delta_x
        x0 = x1
        print('NewtRha iter= ',iteration, ' with norm delta_x= ', jnp.linalg.norm(delta_x))
        iteration +=1 

    return x1

def Newton_Rhapson_second_version(func,jacob_func,x0,atol,rtol,maxiter):
    delta_x = scipy.linalg.solve(-jacob_func(x0),func(x0))
    x1 = x0 + delta_x
    x0 = x1
    iterr = 0
    while np.linalg.norm(delta_x)>= 5*10**(-6) and iterr<maxiter:
        #print(np.linalg.norm(delta_x))
        delta_x = scipy.linalg.solve(-jacob_func(x0),func(x0))
        x1 = x0 + delta_x
        x0 = x1
        iterr = iterr+1

    return x1

def seperate_q(E,q_orig,dof_s,dof):
    """Change of coordinates frame of reference for the configiration 
    from the singlerate coordinate system to the multirate one."""

    q_multi = np.matmul(np.linalg.inv(E),q_orig)
    qs = q_multi[0:dof_s]
    qf = q_multi[dof_s:dof]
    return q_multi,qs,qf

def seperate_qdot(E,qdot_orig,dof_s,dof):
    """Change of coordinates frame of reference for the velocity
    from the singlerate coordinate system to the multirate one."""

    (qdot_multi,qdots,qdotf) = seperate_q(E,qdot_orig,dof_s,dof)
    return qdot_multi,qdots,qdotf

def seperate_pm(E,pm_orig,dof_s,dof):
    """Change of coordinates frame of reference for the momentum
    from the singlerate coordinate system to the multirate one."""

    pm_multi = np.matmul(E.transpose(),pm_orig)
    ps = pm_multi[0:dof_s]
    pf = pm_multi[dof_s:dof]
    return pm_multi,ps,pf

def seperate_q_inv(E,qs, qf ,dof_s,dof):
    """Change of coordinates frame of reference for the configiration 
    from the multirate coordinate system to the singlerate one."""

    q_orig = jnp.matmul(E,jnp.concatenate((qs,qf)))
    return q_orig

def seperate_qdot_inv(E,qdots, qdotf ,dof_s,dof):
    """Change of coordinates frame of reference for the velocity
    from the multirate coordinate system to the singlerate one."""

    qdot_orig = seperate_q_inv(E,qdots,qdotf,dof_s,dof)
    return qdot_orig

def seperate_pm_inv(E,ps, pf ,dof_s,dof):
    """Change of coordinates frame of reference for the momentum 
    from the multirate coordinate system to the singlerate one."""

    pm_orig = seperate_q_inv(jnp.linalg.inv((E.transpose())),ps,pf,dof_s,dof)
    return pm_orig    

def secant_method(x0,x1,func,tol,maxiter):
    x2 = x1 - func(x1)*(x1-x0)/(func(x1)-func(x0))
    x1 = x2
    x0 = x1
    iteration = 0
    print(jnp.linalg.norm(x2-x1))
    while jnp.linalg.norm(x2-x1) >= tol and iteration < maxiter:
        x2 = x1 - func(x1)*(x1-x0)/(func(x1)-func(x0))
        x1 = x2
        x0 = x1
        iteration = 0
        print('Secant iter= ',iteration, ' with norm x2-x1= ', jnp.linalg.norm(x2-x1))
        iteration +=1 

    return x1


