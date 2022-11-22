
import matplotlib.pyplot as plt
import pickle
import os

fh2 = os.path.join("results_final", "E", "DLNN", "results_dict.pkl")
with open(fh2, 'rb') as handle:
    results_dict_DLNN = pickle.load(handle)

fh3 = os.path.join("results_final", "E", "SymDLNN", "results_dict.pkl")
with open(fh3, 'rb') as handle:
    results_dict_SymDLNN = pickle.load(handle)

###############PLOT FOR TRAJECTORY WITH NOISE
ground_truth = results_dict_DLNN['train_trajectory_extra']
DLNN_predict = results_dict_DLNN['q_model_SVI_Q_q0_q1_scipy_extra2']
SymDLNN_predict = results_dict_SymDLNN['q_model_SVI_Q_q0_q1_scipy_extra2']

import numpy as np
timesteps = np.zeros((201))
for k in range (1,201):
    timesteps[k] = timesteps[k-1]+0.1


import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
#pl.suptitle('Recration and prediction of the trajectory with noise',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(ground_truth[0:200,0],ground_truth[0:200,1], 'C0',label="Ground truth")
pl.plot(DLNN_predict[0:50,0],DLNN_predict[0:50,1],'C2', label="dLNN recreation")
pl.plot(DLNN_predict[50:200,0],DLNN_predict[50:200,1],'C2--', label="dLNN prediction")
pl.plot(SymDLNN_predict[0:50,0],SymDLNN_predict[0:50,1],'r', label="SymdLNN recreation")
pl.plot(SymDLNN_predict[50:200,0],SymDLNN_predict[50:200,1],'r--', label="SymdLNN prediction")
pl.xlabel('x')
pl.ylabel('y')
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]


ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C2', label="DLNN recreation")
pl.plot(0,'C2--', label="DLNN prediction")
pl.plot(0,'r', label="SymDLNN recreation")
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
pl.plot(timesteps[0:200],ground_truth[0:200,0], 'C0',label="Ground truth")
pl.plot(timesteps[0:50],DLNN_predict[0:50,0],'C2', label="dLNN recreation")
pl.plot(timesteps[0:50],SymDLNN_predict[0:50,0],'r', label="SymdLNN recreation")
pl.plot(timesteps[50:200],DLNN_predict[50:200,0],'C2--', label="dLNN prediction")
pl.plot(timesteps[50:200],SymDLNN_predict[50:200,0],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel('x')


ax = pl.subplot(gs[1,1]) # row 1, span all columns
pl.plot(timesteps[0:200],ground_truth[0:200,1], 'C0',label="Ground truth")
pl.plot(timesteps[0:50],DLNN_predict[0:50,1],'C2', label="dLNN recreation")
pl.plot(timesteps[0:50],SymDLNN_predict[0:50,1],'r', label="SymdLNN recreation")
pl.plot(timesteps[50:200],DLNN_predict[50:200,1],'C2--', label="dLNN prediction")
pl.plot(timesteps[50:200],SymDLNN_predict[50:200,1],'r--', label="SymdLNN prediction")
pl.xlabel('discrete time '+ r'$k \Delta t$')
pl.ylabel('y')
pl.tight_layout(rect=[0.02, 0.02, 1, 0.95])

plt.savefig('results/q0q1plot_Kepler_noise.png',bbox_inches = 'tight')
plt.savefig('results/q0q1plot_Kepler_noise.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()

###########################################################
DLNN_predict_energy_Hbea_qlearnt = results_dict_DLNN['Hbea_qlearnt_vlearntCD_extra']
SymDLNN_predict_energy_Hbea_qlearnt = results_dict_SymDLNN['Hbea_qlearnt_vlearntCD_extra']


DLNN_predict_energy_I_Lorig_qlearnt = results_dict_DLNN['I_Lorig_qlearnt_vlearntCD_extra']
SymDLNN_predict_energy_Ilearnt_qlearnt = results_dict_SymDLNN['I_LdNNsym_p_Ldlearnt_qlearnt_new_extra']

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)
import seaborn as sns

pl.figure()
#sns.set_theme()
#pl.suptitle('Noisy Kepler example: recration and prediction of H and I errors ',y=0.92)
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot(timesteps[0:200],np.zeros((200)), 'C0',label="Ground truth")
pl.plot(timesteps[0:50],DLNN_predict_energy_Hbea_qlearnt[0:50]- DLNN_predict_energy_Hbea_qlearnt[0]*np.ones((50)),'C2', label="dLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(timesteps[0:50],SymDLNN_predict_energy_Hbea_qlearnt[0:50]- SymDLNN_predict_energy_Hbea_qlearnt[0]*np.ones((50)),'r', label="SymdLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(timesteps[50:200],DLNN_predict_energy_Hbea_qlearnt[50:200]- DLNN_predict_energy_Hbea_qlearnt[0]*np.ones((150)),'C2--', label="dLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(timesteps[50:200],SymDLNN_predict_energy_Hbea_qlearnt[50:200]- SymDLNN_predict_energy_Hbea_qlearnt[0]*np.ones((150)),'r--', label="SymdLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', PP')
pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel('H-H0')
#pl.legend(loc="lower left")
pl.ylim([-0.02,0.05])
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]

ax = pl.subplot(gs[0, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C2', label="DLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',RP')
pl.plot(0,'C2--', label="DLNN  "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$H^{VBEA}_k-H^{VBEA}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=11,frameon=False) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')


ax = pl.subplot(gs[1, 0]) # row 0, col 0
pl.plot(timesteps[0:200],np.zeros((200)), 'C0',label="Ground truth")
pl.plot(timesteps[0:50],DLNN_predict_energy_I_Lorig_qlearnt[0:50]- DLNN_predict_energy_I_Lorig_qlearnt[0]*np.ones((50)),'C2', label="dLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(timesteps[0:50],SymDLNN_predict_energy_Ilearnt_qlearnt[0:50]- SymDLNN_predict_energy_Ilearnt_qlearnt[0]*np.ones((50)),'r', label="SymdLNN   "+r'$I^{NN}_k-I^{NN}_0$'+', RP')
pl.plot(timesteps[50:200],DLNN_predict_energy_I_Lorig_qlearnt[50:200]- DLNN_predict_energy_I_Lorig_qlearnt[0]*np.ones((150)),'C2--', label="dLNN  "+r'$I^{true}_k-I^{true}_0$'+', PP')
pl.plot(timesteps[50:200],SymDLNN_predict_energy_Ilearnt_qlearnt[50:200]- SymDLNN_predict_energy_Ilearnt_qlearnt[0]*np.ones((150)),'r--', label="SymdLNN    "+r'I$^{NN}_k-I^{NN}_0$'+', PP')
pl.xlabel('discrete time '+ r'$k \Delta t$')
#pl.ylabel(r'I_k-I_0')
#pl.legend(loc="lower left")
pl.ylim([-15,15])
pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#

ax = pl.subplot(gs[1, 1]) # row 0, col 0
pl.plot(0, 'C0',label="Ground truth")
pl.plot(0,'C2', label="DLNN   "+r'$I^{true}_k-I^{true}_0$'+', RP')
pl.plot(0,'r', label="SymDLNN   "+r'$I^{NN}_k-I^{NN}_0$'+',RP')
pl.plot(0,'C2--', label="DLNN  "+r'$I^{true}_k-I^{true}_0$'+',PP')
pl.plot(0,'r--', label="SymDLNN    "+r'$I^{NN}_k-I^{NN}_0$'+',PP')
#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]
ax.set_yticklabels([])
ax.set_xticklabels([])
pl.legend(loc="upper left",fontsize=11,frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')


#pl.tight_layout(rect=[0.05, 0.05, 1, 0.95])#[left, bottom, right, top]


# ax = pl.subplot(gs[2, 0]) # row 0, col 0
# pl.plot(0, 'C0',label="Ground truth")
# pl.plot(0,'C2', label="dLNN recreation result")
# pl.plot(0,'r', label="SymdLNN based recreation result")
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
# pl.plot(0,'C2--', label="dLNN prediction Hbea/Iorig qlearnt")
# pl.plot(0,'r--', label="SymdLNN prediction Hbea/Ilearnt qlearnt ")
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
plt.savefig('results/singletraj_HIpred_E.png',bbox_inches = 'tight')
plt.savefig('results/singletraj_HIpred_E.eps', format = 'eps', bbox_inches = 'tight')
plt.show()
plt.close()
