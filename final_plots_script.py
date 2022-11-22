
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

fh1 = os.path.join("results_final", "A", "LNN", "results_dict.pkl")
with open(fh1, 'rb') as handle:
    results_dict_LNN_A = pickle.load(handle)

fh2 = os.path.join("results_final", "A", "DLNN", "results_dict.pkl")
with open(fh2, 'rb') as handle:
    results_dict_DLNN_A = pickle.load(handle)

fh3 = os.path.join("results_final", "A", "SymDLNN", "results_dict.pkl")
with open(fh3, 'rb') as handle:
    results_dict_SymDLNN_A = pickle.load(handle)

fh4 = os.path.join("results_final", "C", "LNN", "results_dict.pkl")
with open(fh4, 'rb') as handle:
    results_dict_LNN_C = pickle.load(handle)

fh5 = os.path.join("results_final", "C", "DLNN", "results_dict.pkl")
with open(fh5, 'rb') as handle:
    results_dict_DLNN_C = pickle.load(handle)

fh6 = os.path.join("results_final", "C", "SymDLNN", "results_dict.pkl")
with open(fh6, 'rb') as handle:
    results_dict_SymDLNN_C = pickle.load(handle)   

###############PLOT FOR TRAJECTORY WITH NOISE
ground_truth_A = results_dict_LNN_A['train_trajectory_extra']
LNN_predict_A = results_dict_LNN_A['q_model_SVI_Q_q0_q1_scipy_extra2']
DLNN_predict_A = results_dict_DLNN_A['q_model_SVI_Q_q0_q1_scipy_extra2']
SymDLNN_predict_A = results_dict_SymDLNN_A['q_model_SVI_Q_q0_q1_scipy_extra2']
ground_truth_energy_A = results_dict_LNN_A['Horig_qorig_vorigCD_extra']
LNN_predict_energy_Hlearnt_qlearnt_A = results_dict_LNN_A['Horig_qlearnt_vlearntCD_extra']
DLNN_predict_energy_Hbea_qlearnt_A = results_dict_DLNN_A['Hbea_qlearnt_vlearntCD_extra']
SymDLNN_predict_energy_Hbea_qlearnt_A = results_dict_SymDLNN_A['Hbea_qlearnt_vlearntCD_extra']
LNN_predict_energy_I_Lorig_qlearnt_A = results_dict_LNN_A['p_Lorig_qlearnt_vlearntCD_extra'][0,:]
DLNN_predict_energy_I_Lorig_qlearnt_A = results_dict_DLNN_A['p_Lorig_qlearnt_vlearntCD_extra'][0,:]
SymDLNN_predict_energy_Ilearnt_qlearnt_A = results_dict_SymDLNN_A['I_LdNNsym_p_Ldlearnt_qlearnt_new_extra']


N_A = 200
N_A_aftersim = 300

ground_truth_C = results_dict_LNN_C['train_trajectory_extra']
LNN_predict_C = results_dict_LNN_C['q_model_SVI_Q_q0_q1_scipy_extra2']
DLNN_predict_C = results_dict_DLNN_C['q_model_SVI_Q_q0_q1_scipy_extra2']
SymDLNN_predict_C = results_dict_SymDLNN_C['q_model_SVI_Q_q0_q1_scipy_extra2']
#ground_truth_energy_C = results_dict_LNN_C['Horig_qorig_vorigCD_extra']
LNN_predict_energy_Hlearnt_qlearnt_C = results_dict_LNN_C['Horig_qlearnt_vlearntCD_extra']
DLNN_predict_energy_Hbea_qlearnt_C = results_dict_DLNN_C['Hbea_qlearnt_vlearntCD_extra']
SymDLNN_predict_energy_Hbea_qlearnt_C = results_dict_SymDLNN_C['Hbea_qlearnt_vlearntCD_extra']

LNN_predict_energy_I_Lorig_qlearnt_C = results_dict_LNN_C['I_Lorig_qlearnt_vlearntCD_extra']
DLNN_predict_energy_I_Lorig_qlearnt_C = results_dict_DLNN_C['I_Lorig_qlearnt_vlearntCD_extra']
SymDLNN_predict_energy_Ilearnt_qlearnt_C = results_dict_SymDLNN_C['I_LdNNsym_p_Ldlearnt_qlearnt_new_extra']

N_C = 50
N_C_aftersim = 200

import numpy as np
timesteps_A = np.zeros((N_A_aftersim+1))
for k in range (1,N_A_aftersim+1):
    timesteps_A[k] = timesteps_A[k-1]+0.01

timesteps_C = np.zeros((N_C_aftersim+1))
for k in range (1,N_C_aftersim+1):
    timesteps_C[k] = timesteps_C[k-1]+0.1
###########################################################################################################

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
#pl.suptitle('Recration and prediction of the trajectory for the Kepler example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(ground_truth_C[:,0],ground_truth_C[:,1], 'C0',label="Ground truth")
pl.plot(LNN_predict_C[0:N_C,0],LNN_predict_C[0:N_C,1],'C4', label="LNN recreation")
pl.plot(DLNN_predict_C[0:N_C,0],DLNN_predict_C[0:N_C,1],'C2', label="dLNN recreation")
pl.plot(SymDLNN_predict_C[0:N_C,0],SymDLNN_predict_C[0:N_C,1],'r', label="SymdLNN recreation")
pl.plot(LNN_predict_C[N_C:N_C_aftersim,0],LNN_predict_C[N_C:N_C_aftersim,1],'C4--', label="LNN prediction")
pl.plot(DLNN_predict_C[N_C:N_C_aftersim,0],DLNN_predict_C[N_C:N_C_aftersim,1],'C2--', label="dLNN prediction")
pl.plot(SymDLNN_predict_C[N_C:N_C_aftersim,0],SymDLNN_predict_C[N_C:N_C_aftersim,1],'r--', label="SymdLNN prediction")
pl.xlabel('x')
pl.xlim([-1,1.5])
pl.ylabel('y')
pl.ylim([-1,1.5])
#pl.legend(loc="upper right")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C4', label="LNN recreation")
pl.plot(0,'C2', label="DLNN recreation")
pl.plot(0,'r', label="SymDLNN recreation")
pl.plot(0,'C4--', label="LNN prediction")
pl.plot(0,'C2--', label="DLNN prediction")
pl.plot(0,'r--', label="SymDLNN prediction")
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')

#pl.legend(loc="upper right")
ax = pl.subplot(gs[1, 0]) # row 0, col 1
pl.plot(timesteps_C,ground_truth_C[:,0], 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_C[0:N_C,0],'C4', label="LNN recreation")
pl.plot(timesteps_C[0:N_C],DLNN_predict_C[0:N_C,0],'C2', label="dLNN recreation")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_C[0:N_C,0],'r', label="SymdLNN recreation")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_C[N_C:N_C_aftersim,0],'C4--', label="LNN prediction")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_C[N_C:N_C_aftersim,0],'C2--', label="dLNN prediction")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_C[N_C:N_C_aftersim,0],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel('x')
pl.ylim([-1.5,2])
#pl.legend(loc="upper right")


ax = pl.subplot(gs[1,1]) # row 1, span all columns
pl.plot(timesteps_C,ground_truth_C[:,1], 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_C[0:N_C,1],'C4', label="LNN recreation")
pl.plot(timesteps_C[0:N_C],DLNN_predict_C[0:N_C,1],'C2', label="dLNN recreation")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_C[0:N_C,1],'r', label="SymdLNN recreation")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_C[N_C:N_C_aftersim,1],'C4--', label="LNN prediction")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_C[N_C:N_C_aftersim,1],'C2--', label="dLNN prediction")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_C[N_C:N_C_aftersim,1],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel('y')
pl.ylim([-1.5,1.5])
pl.tight_layout(rect=[0.02, 0.02, 1, 0.95])

plt.savefig('results/singletraj_TRAJpred_C.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_TRAJpred_C.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()
####################################################################################################################

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
#pl.suptitle('Recration and prediction of the trajectory for the Cart-pendulum example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(ground_truth_A[:,0],ground_truth_A[:,1], 'C0.',label="Ground truth")
pl.plot(LNN_predict_A[0:N_A,0],LNN_predict_A[0:N_A,1],'C4', label="LNN recreation")
pl.plot(DLNN_predict_A[0:N_A,0],DLNN_predict_A[0:N_A,1],'C2', label="dLNN recreation")
pl.plot(SymDLNN_predict_A[0:N_A_aftersim,0],SymDLNN_predict_A[0:N_A_aftersim,1],'r', label="SymdLNN recreation")
pl.plot(LNN_predict_A[N_A:N_A_aftersim,0],LNN_predict_A[N_A:N_A_aftersim,1],'C4--', label="LNN prediction")
pl.plot(DLNN_predict_A[N_A:N_A_aftersim,0],DLNN_predict_A[N_A:N_A_aftersim,1],'C2--', label="dLNN prediction")
pl.plot(SymDLNN_predict_A[N_A:N_A_aftersim,0],SymDLNN_predict_A[N_A:N_A_aftersim,1],'r--', label="SymdLNN prediction")
pl.xlabel('s')
pl.ylabel(r'$\phi$')
#pl.legend(loc="upper right")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0.',label="Ground truth")
pl.plot(0,'C4', label="LNN recreation")
pl.plot(0,'C2', label="DLNN recreation")
pl.plot(0,'r', label="SymDLNN recreation")
pl.plot(0,'C4--', label="LNN prediction")
pl.plot(0,'C2--', label="DLNN prediction")
pl.plot(0,'r--', label="SymDLNN prediction")
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')

#pl.legend(loc="upper right")
ax = pl.subplot(gs[1, 0]) # row 0, col 1
pl.plot(timesteps_A,ground_truth_A[0:N_A_aftersim+1,0], 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_A[0:N_A,0],'C4', label="LNN recreation")
pl.plot(timesteps_A[0:N_A],DLNN_predict_A[0:N_A,0],'C2', label="dLNN recreation")
pl.plot(timesteps_A[0:N_A_aftersim],SymDLNN_predict_A[0:N_A_aftersim,0],'r', label="SymdLNN recreation")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_A[N_A:N_A_aftersim,0],'C4--', label="LNN prediction")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_A[N_A:N_A_aftersim,0],'C2--', label="dLNN prediction")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_A[N_A:N_A_aftersim,0],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel('s')
#pl.legend(loc="upper right")


ax = pl.subplot(gs[1,1]) # row 1, span all columns
pl.plot(timesteps_A,ground_truth_A[0:N_A_aftersim+1,1], 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_A[0:N_A,1],'C4', label="LNN recreation")
pl.plot(timesteps_A[0:N_A],DLNN_predict_A[0:N_A,1],'C2', label="dLNN recreation")
pl.plot(timesteps_A[0:N_A_aftersim],SymDLNN_predict_A[0:N_A_aftersim,1],'r', label="SymdLNN recreation")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_A[N_A:N_A_aftersim,1],'C4--', label="LNN prediction")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_A[N_A:N_A_aftersim,1],'C2--', label="dLNN prediction")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_A[N_A:N_A_aftersim,1],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel(r'$\phi$')
pl.tight_layout(rect=[0.02, 0.02, 1, 0.95])

plt.savefig('results/singletraj_TRAJpred_A.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_TRAJpred_A.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()

# ground_truth_energy_C = results_dict_LNN_C['Horig_qorig_vorigCD']
# LNN_predict_energy_Horig_qlearnt_C = results_dict_LNN_C['Horig_qlearnt_vlearntCD']
# DLNN_predict_energy_Hbea_qlearnt_C = results_dict_DLNN_C['Hbea_qlearnt_vlearntCD']
# SymDLNN_predict_energy_Hbea_qlearnt_C = results_dict_SymDLNN_C['Hbea_qlearnt_vlearntCD']

##############################################################################################################

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
pl.suptitle('          Recration and prediction of the energy for the Kepler example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_energy_Hlearnt_qlearnt_C[0:N_C]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_Hlearnt_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# pl.title('Recration and prediction of the symmetry for the Kepler example',y=0.92)

ax = pl.subplot(gs[1, :]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
#pl.plot(timesteps_C[0:N_C],LNN_predict_energy_Hlearnt_qlearnt_C[0:N_C]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
#pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_Hlearnt_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax = pl.subplot(gs[0, 1])

pl.plot(0, 'C0.',label="Ground truth")
pl.plot( 0,'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(0,'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(0,'r', label="SymdLNN recreation Hbea q learnt")
pl.plot( 0,'C4--', label="LNN recreation Hleanrt q learnt")
pl.plot(0,'C2--', label="dLNN recreation Hbea qlearnt")
pl.plot(0 ,'r--', label="SymdLNN recreation Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
pl.tight_layout(rect=[0.05, 0.05, 1.1, 0.95])#[left, bottom, right, top]

plt.savefig('results/singletraj_Hpred_C.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_Hpred_C.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()




##############################################################################################################

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
pl.suptitle('Recration and prediction of the energy for the cart-pendulum example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_Hlearnt_qlearnt_A[0:N_A]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_Hlearnt_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# pl.title('Recration and prediction of the symmetry for the Kepler example',y=0.92)

ax = pl.subplot(gs[1, :]) # row 0, col 0
#pl.plot(timesteps_A[0:300],ground_truth_energy_A[:]- ground_truth_energy_A[0]*np.ones((300)), 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_Hlearnt_qlearnt_A[0:N_A]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_Hlearnt_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
#pl.plot(timesteps_A[100:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[100:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((100)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax = pl.subplot(gs[0, 1])

pl.plot(0, 'C0.',label="Ground truth")
pl.plot( 0,'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(0,'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(0,'r', label="SymdLNN recreation Hbea q learnt")
pl.plot( 0,'C4--', label="LNN recreation Hleanrt q learnt")
pl.plot(0,'C2--', label="dLNN recreation Hbea qlearnt")
pl.plot(0 ,'r--', label="SymdLNN recreation Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
pl.tight_layout(rect=[0.05, 0.05, 1.1, 0.95])#[left, bottom, right, top]

plt.savefig('results/singletraj_Hpred_A.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_Hpred_A.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()


############################


# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
pl.suptitle('Recration and prediction of the I for the cart-pendulum example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- DLNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Ilearnt_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_I_Lorig_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('I-I0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# pl.title('Recration and prediction of the symmetry for the Kepler example',y=0.92)

ax = pl.subplot(gs[1, :]) # row 0, col 0
pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0',label="Ground truth")
#pl.plot(timesteps_A[0:300],ground_truth_energy_A[:]- ground_truth_energy_A[0]*np.ones((300)), 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- DLNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Ilearnt_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_I_Lorig_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_I_Lorig_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('I-I0')
#pl.legend(loc="upper left")
pl.ylim([-0.5,0.1])
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax = pl.subplot(gs[0, 1])

pl.plot(0, 'C0.',label="Ground truth")
pl.plot( 0,'C4', label="LNN recreation Iorig qlearnt ")
pl.plot(0,'C2', label="dLNN recreation Iorig qlearnt")
pl.plot(0,'r', label="SymdLNN recreation Ilearnt q learnt")
pl.plot( 0,'C4--', label="LNN recreation Iorig q learnt")
pl.plot(0,'C2--', label="dLNN recreation Iorig qlearnt")
pl.plot(0 ,'r--', label="SymdLNN recreation Ilearnt qlearnt ")
pl.xlabel('time')
pl.ylabel('I-I0')
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
pl.tight_layout(rect=[0.05, 0.05, 1.1, 0.95])#[left, bottom, right, top]

plt.savefig('results/singletraj_Ipred_A.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_Ipred_A.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()




# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
pl.suptitle('Recration and prediction of the I for the Kepler example',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Ilearnt_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('I-I0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# pl.title('Recration and prediction of the symmetry for the Kepler example',y=0.92)

ax = pl.subplot(gs[1, :]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
#pl.plot(timesteps_C[0:N_C],LNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Ilearnt_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
#pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('time')
pl.ylabel('I-I0')
#pl.legend(loc="upper left")
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax = pl.subplot(gs[0, 1])

pl.plot(0, 'C0.',label="Ground truth")
pl.plot( 0,'C4', label="LNN recreation Iorig qlearnt ")
pl.plot(0,'C2', label="dLNN recreation Iorig qlearnt")
pl.plot(0,'r', label="SymdLNN recreation Ilearnt q learnt")
pl.plot( 0,'C4--', label="LNN recreation Iorig q learnt")
pl.plot(0,'C2--', label="dLNN recreation Iorig qlearnt")
pl.plot(0 ,'r--', label="SymdLNN recreation Ilearnt qlearnt ")
pl.xlabel('time')
pl.ylabel('H-H0')
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
pl.tight_layout(rect=[0.05, 0.05, 1.1, 0.95])#[left, bottom, right, top]

plt.savefig('results/singletraj_Ipred_C.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_Ipred_C.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()



# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)
import seaborn as sns

pl.figure()
#sns.set_theme()
#pl.suptitle('Noisy Kepler example: recration and prediction of H and I errors ',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_energy_Hlearnt_qlearnt_C[0:N_C]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_Hlearnt_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel('H-H0')
#pl.legend(loc="lower left")
pl.ylim([-0.01,0.04])
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C4', label="LNN   "+r'$H^{true}_k-H^{true}_0$'+', RP')
pl.plot(0,'C2', label="DLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',RP')
pl.plot(0,'C4--', label="LNN   "+r'$H^{true}_k-H^{true}_0$'+', PP')
pl.plot(0,'C2--', label="DLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=10,frameon=False) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')


ax = pl.subplot(gs[1, 0]) # row 0, col 0
pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_C[0:N_C],LNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Ilearnt_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C_aftersim-N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel(r'I_k-I_0')
#pl.legend(loc="lower left")
pl.ylim([-5,11])
pl.tight_layout(rect=[0.05, 0.05, 1, 1.2])#

ax = pl.subplot(gs[1, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
pl.plot(0,'C2--', label="DLNN  "+r'$I^{true}_k-I^{true}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=10,frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')

plt.savefig('results/singletraj_HIpred_C.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_HIpred_C.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()

####################################################################################################

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)
import seaborn as sns

pl.figure()
#sns.set_theme()
#pl.suptitle('Noisy Kepler example: recration and prediction of H and I errors ',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0.',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_Hlearnt_qlearnt_A[0:N_A]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_Hlearnt_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel('H-H0')
#pl.legend(loc="lower left")
pl.ylim([-0.008,0.055])
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0.',label="Ground truth")
pl.plot(0,'C4', label="LNN   "+r'$H^{true}_k-H^{true}_0$'+', RP')
pl.plot(0,'C2', label="DLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',RP')
pl.plot(0,'C4--', label="LNN   "+r'$H^{true}_k-H^{true}_0$'+', PP')
pl.plot(0,'C2--', label="DLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=10,frameon=False) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')


ax = pl.subplot(gs[1, 0]) # row 0, col 0
pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0',label="Ground truth")
pl.plot(timesteps_A[0:N_A],LNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- DLNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Ilearnt_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_I_Lorig_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A_aftersim-N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")

pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel(r'I_k-I_0')
#pl.legend(loc="lower left")
pl.ylim([-2,5])
pl.tight_layout(rect=[0.05, 0.05, 1, 1.2])#

ax = pl.subplot(gs[1, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
pl.plot(0,'C2--', label="DLNN  "+r'$I^{true}_k-I^{true}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=10,frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')

plt.savefig('results/singletraj_HIpred_A.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_HIpred_A.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()


# # Create 2x2 sub plots
# gs = gridspec.GridSpec(3, 3)

# pl.figure()
# #pl.suptitle('recration and prediction of symmetry errors ',y=0.92)
# ax = pl.subplot(gs[1, :]) # row 0, col 0
# pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
# pl.plot(timesteps_C[0:N_C],LNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
# pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_I_Lorig_qlearnt_C[0:N_C]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
# pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Ilearnt_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_I_Lorig_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_I_Lorig_qlearnt_C[0]*np.ones((N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_C[0]*np.ones((N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
# pl.xlabel('discrete time '+ r'$k \Delta t$')
# #pl.ylabel('I-I0 for Kepler')
# #pl.legend(loc="upper left")
# ax.title.set_text('Kepler example')
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
# #pl.legend(loc="lower left")

# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# ax = pl.subplot(gs[0, :]) # row 0, col 0
# pl.plot(timesteps_A[0:N_A_aftersim],np.zeros((N_A_aftersim)), 'C0',label="Ground truth")
# pl.plot(timesteps_A[0:N_A],LNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
# pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_I_Lorig_qlearnt_A[0:N_A]- DLNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
# pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Ilearnt_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_I_Lorig_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_I_Lorig_qlearnt_A[0]*np.ones((N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Ilearnt_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Ilearnt_qlearnt_A[0]*np.ones((N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
# pl.xlabel('discrete time '+ r'$k \Delta t$')
# #pl.legend(loc="upper left")
# ax.title.set_text('Pendulum on a cart example')
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]


# ax = pl.subplot(gs[2, 0]) # row 0, col 0
# pl.plot(0, 'C0',label="Ground truth")
# pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# #pl.plot(0,'r', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
# # pl.plot(0,'C4--', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# # pl.plot(0,'C2--', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# # pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')

# ax.set_yticklabels([])
# ax.set_xticklabels([])
# pl.legend(loc="upper right")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# pl.legend(frameon=False)
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])


# ax = pl.subplot(gs[2, 1]) # row 0, col 0
# # pl.plot(0, 'C0',label="Ground truth")
# # pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# # pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# pl.plot(0,'r', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
# pl.plot(0,'C4--', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# #pl.plot(0,'C2--', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# #pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# pl.legend(loc="upper right")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# pl.legend(frameon=False)
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# ax = pl.subplot(gs[2, 2]) # row 0, col 0
# #pl.plot(0, 'C0',label="Ground truth")
# #pl.plot(0,'C4', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# #pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
# #pl.plot(0,'r', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
# # pl.plot(0,'C4--', label="LNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# pl.plot(0,'C2--', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', PP')
# pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# pl.legend(loc="upper right")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# pl.legend(frameon=False)
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])


# plt.show()
# plt.savefig('results/singletraj_Ipred_AC.png',bbox_inches = 'tight')
# plt.savefig('results/singletraj_Ipred_AC.eps', format = 'eps', bbox_inches = 'tight')
# plt.close()



# # Create 2x2 sub plots
# gs = gridspec.GridSpec(3, 2)

# pl.figure()
# pl.suptitle('recration and prediction of energy errors ',y=0.92)
# ax = pl.subplot(gs[1, :]) # row 0, col 0
# pl.plot(timesteps_C[0:N_C_aftersim],np.zeros((N_C_aftersim)), 'C0',label="Ground truth")
# pl.plot(timesteps_C[0:N_C],LNN_predict_energy_Hlearnt_qlearnt_C[0:N_C]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4', label="LNN recreation Hlearnt qlearnt ")
# pl.plot(timesteps_C[0:N_C],DLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'C2', label="dLNN recreation Hbea qlearnt")
# pl.plot(timesteps_C[0:N_C],SymDLNN_predict_energy_Hbea_qlearnt_C[0:N_C]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'r', label="SymdLNN recreation Hbea q learnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],LNN_predict_energy_Hlearnt_qlearnt_C[N_C:N_C_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_C[0]*np.ones((N_C)),'C4--', label="LNN prediction Hleanrt q learnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],DLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- DLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'C2--', label="dLNN prediction Hbea qlearnt")
# pl.plot(timesteps_C[N_C:N_C_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_C[N_C:N_C_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_C[0]*np.ones((N_C)),'r--', label="SymdLNN prediction Hbea qlearnt ")
# pl.xlabel('discrete time '+ r'$k \Delta t$')
# pl.ylabel('H-H0 for Kepler')
# #pl.legend(loc="upper left")
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

# ax = pl.subplot(gs[0, :]) # row 0, col 0
# pl.plot(timesteps_A[0:N_A],LNN_predict_energy_Hlearnt_qlearnt_A[0:N_A]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A)),'C4', label="LNN recreation Hlearnt qlearnt ")
# pl.plot(timesteps_A[0:N_A],DLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2', label="dLNN recreation Hbea qlearnt")
# pl.plot(timesteps_A[0:N_A],SymDLNN_predict_energy_Hbea_qlearnt_A[0:N_A]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'r', label="SymdLNN recreation Hbea q learnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],LNN_predict_energy_Hlearnt_qlearnt_A[N_A:N_A_aftersim]- LNN_predict_energy_Hlearnt_qlearnt_A[0]*np.ones((N_A)),'C4--', label="LNN prediction Hleanrt q learnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],DLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- DLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'C2--', label="dLNN prediction Hbea qlearnt")
# pl.plot(timesteps_A[N_A:N_A_aftersim],SymDLNN_predict_energy_Hbea_qlearnt_A[N_A:N_A_aftersim]- SymDLNN_predict_energy_Hbea_qlearnt_A[0]*np.ones((N_A)),'r--', label="SymdLNN prediction Hbea qlearnt ")
# pl.xlabel('discrete time '+ r'$k \Delta t$')
# pl.ylabel('H-H0 for cartpend')
# #pl.legend(loc="upper left")
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]


# ax = pl.subplot(gs[2, 0]) # row 0, col 0
# pl.plot(0, 'C0',label="Ground truth")
# pl.plot(0,'C2', label="dLNN recreation Hbea qlearnt")
# pl.plot(0,'r', label="SymdLNN recreation Hbea qlearnt")
# # pl.plot(0,'C2--', label="dLNN prediction Hbea qlearnt")
# # pl.plot(0,'r--', label="SymdLNN prediction Hbea qlearnt ")
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# pl.legend(loc="upper right")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# pl.legend(frameon=False)
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# ax = pl.subplot(gs[2, 1]) # row 0, col 0
# # pl.plot(0, 'C0',label="Ground truth")
# # pl.plot(0,'C2', label="dLNN recreation Hbea qlearnt")
# # pl.plot(0,'r', label="SymdLNN recreation Hbea q learnt")
# pl.plot(0,'C2--', label="dLNN prediction Hbea qlearnt")
# pl.plot(0,'r--', label="SymdLNN prediction Hbea qlearnt ")
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# pl.legend(loc="upper right")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axis('off')
# pl.legend(frameon=False)
# pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])
# plt.show()
# plt.savefig('results/singletraj_Hpred_AC.png',bbox_inches = 'tight')
# plt.savefig('results/singletraj_Hpred_AC.eps', format = 'eps', bbox_inches = 'tight')
# plt.close()
