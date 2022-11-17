from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

# Computes Hamiltonian to a Lagrangian L
def Hamiltonian_from_L(L):
	
	def Hamiltonian(q,qdot):
		p2 = jax.grad(L,argnums=1)(q,qdot)
		return jnp.dot(qdot,p2) - L(q,qdot)
		
	return Hamiltonian


# Applies backward error analysis to a continuous Lagrangian
# version 2d
def L_BEA1d(L,h):

	def LBEA1(y):
		# modified Lagrangian obtained by variational backward error analysis

		Lmi = lambda y: L(y[0],y[1])
		q  = y[0]
		qd = y[1]
		L_0 = Lmi(y)

		# compute jet
		L_0, gradL = jax.value_and_grad(Lmi)(y)
		HessL = jax.hessian(Lmi)(y)

		# assign variables compatible with formulas spit out by Mathematica
		L10 = gradL[0]
		L01 = gradL[1]
		L20 = HessL[0,0]
		L11 = HessL[0,1]
		L02 = HessL[1,1]

		# compute higher order correction term
		L_2 = (-(L20*qd**2) + (L10 - L11*qd)**2/L02)/24.

		# modified Lagrangian truncated to second order
		L2 = L_0 + h**2*L_2

		return L2
		
	return LBEA1

	

# Applies backward error analysis to a continuous Lagrangian
# version 2d
def L_BEA2d(L,h):

	def LBEA2(y):

		Lmi = lambda y: L(y[:2],y[2:])

		y1d = y[2]
		y2d = y[3]

		L_0 = Lmi(y)

		# compute jet
		gradL = jax.grad(Lmi)(y)
		HessL = jax.hessian(Lmi)(y)

		# assign variables compatible with formulas spit out by Mathematica
		L1000 = gradL[0]
		L0100 = gradL[1]
		L0010 = gradL[3]
		L0001 = gradL[4]

		L2000 = HessL[0,0]
		L0200 = HessL[1,1]
		L0020 = HessL[2,2]
		L0002 = HessL[3,3]

		L1100 = HessL[0,1]
		L1010 = HessL[0,2]
		L1001 = HessL[0,3]
		L0110 = HessL[1,2]
		L0101 = HessL[1,3]
		L0011 = HessL[2,3]

		# compute higher order correction terms
		L_1 = 0
		L_2 = -1/24*(2*L0011*(L0100 - L1001*y1d - L0101*y2d)*(-L1000 + L1010*y1d + L0110*y2d) + L0002*(-L1000 + L1010*y1d + L0110*y2d)**2 + L0020*(L0100**2 + L1001**2*y1d**2 - L0002*L2000*y1d**2 + 2*L0101*L1001*y1d*y2d - 2*L0002*L1100*y1d*y2d + L0101**2*y2d**2 - L0002*L0200*y2d**2 - 2*L0100*(L1001*y1d + L0101*y2d)) + L0011**2*(L2000*y1d**2 + y2d*(2*L1100*y1d + L0200*y2d)))/(L0011**2 - L0002*L0020)

		# compute recovered Lagrangian truncated to order 0,2
		Lrecov2 = L_0 + h**2*L_2
		return Lrecov2
		
	return LBEA2
	

# for contour plots
# evaluate function f:R^2->R over 2d mesh

def MeshF(f,spc,nn):
	# evaluate f on mesh
	# example input
	# nn = (10,10)
	# spc = ((-1.3, 1.3), (-1., 1.4))

	qq1    = jnp.linspace(spc[0][0],spc[0][1],nn[0])
	qq2 = jnp.linspace(spc[1][0],spc[1][1],nn[1])
	qqq1,qqq2 = jnp.meshgrid(qq1,qq2)
	f_vec = jnp.vectorize(f)

	return qq1,qq2,f_vec(qqq1,qqq2)





