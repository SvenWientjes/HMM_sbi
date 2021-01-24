########################################################################################
# Function that wraps a Hidden Markov Model generator into a specific structure that
# accepts 1800 observations of 3 features
### Parameter Inputs:
# mus (SxM): Means of the features organized by occupied state
# sds (SxM): Standard deviations of the features organized by occupied state
# TPM (SxS): Transition probability matrix. S(t) at rows, S(t+1) at columns
### Code alterations:
# // Within the call to the HMM_gen() function, N can be changed to directly accomodate 
# for datasets with different number of observations
# // The indexing of the parameters from input theta can be changed to accomodate for
# different nr of states and features. This needs to be fixed before parsing to sbi
TPM_trans = np.zeros(6)
idxje = 0
for i in range(3):
    TPM_trans[idxje] = np.log(TPM[i,1]/(1-TPM[i,1]-TPM[i,2]))
    idxje+=1
    TPM_trans[idxje] = np.log(TPM[i,2]/(1-TPM[i,1]-TPM[i,2]))
    idxje+=1

theta = np.array([0.1, 1.1, 2.1, 2.1, 0.1, 1.1, -3,  -2,  -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
theta = np.concatenate((theta, TPM_trans))

def HMM_wrap_sbi(theta):
    from HMM_gen import HMM_gen
    from math import exp
    import numpy as np
    mus = np.reshape(theta[0:3*3], (3,3))
    sds = np.reshape(theta[3*3:2*3*3], (3,3))
    
    TPM = np.zeros((3,3))
    TPM[0,0] = 1/(1+exp(theta[2*3*3])+exp(theta[2*3*3+1]))
    TPM[0,1] = exp(theta[2*3*3])/(1+exp(theta[2*3*3])+exp(theta[2*3*3+1]))
    TPM[0,2] = exp(theta[2*3*3+1])/(1+exp(theta[2*3*3])+exp(theta[2*3*3+1]))
    TPM[1,0] = 1/(1+exp(theta[2*3*3+2])+exp(theta[2*3*3+3]))
    TPM[1,1] = exp(theta[2*3*3+2])/(1+exp(theta[2*3*3+2])+exp(theta[2*3*3+3]))
    TPM[1,2] = exp(theta[2*3*3+3])/(1+exp(theta[2*3*3+2])+exp(theta[2*3*3+3]))
    TPM[2,0] = 1/(1+exp(theta[2*3*3+4])+exp(theta[2*3*3+5]))
    TPM[2,1] = exp(theta[2*3*3+4])/(1+exp(theta[2*3*3+4])+exp(theta[2*3*3+5]))
    TPM[2,2] = exp(theta[2*3*3+5])/(1+exp(theta[2*3*3+4])+exp(theta[2*3*3+5]))
    
    return(HMM_gen(mus=mus, sds=sds, TPM=TPM, delta=[1,0,0], N=1800))




