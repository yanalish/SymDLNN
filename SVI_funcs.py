from jax import grad

def SVI_funcs(Ldk_SVI):
    """Creates the needed gradient functions based on a given discrete Lagrangian

    Args:
        Ldk_SVI (function) - a singlerate discrete approximation of the Lagrangian

    Returns:
        D1Ldk (function) - gradient of Ldk wrt its first input (aka qk)
        D2Ld1k (function)- gradient of Ld1k wrt its second input (aka qk)
        D2Ldk (function)- gradient of Ldk wrt its second input (aka qk1)
        
    """
    def D2Ld1k(q1k,qk,h): 
        return grad(Ldk_SVI,argnums=1)(q1k,qk,h)

    def D2Ldk(qk,qk1,h): 
        return grad(Ldk_SVI, argnums=1)(qk,qk1,h) #or D2Ld1k(qk,qk1,h)
    
    def D1Ldk(qk,qk1,h):  
        return grad(Ldk_SVI, argnums=0)(qk,qk1,h)

    return D1Ldk, D2Ld1k, D2Ldk