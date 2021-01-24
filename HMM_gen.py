########################################################################################
# Function that generates a DataSet following a Hidden Markov Model structure with 
# Gaussian distributed features
### Parameter Inputs:
# mus (SxM): Means of the features organized by occupied state
# sds (SxM): Standard deviations of the features organized by occupied state
# TPM (SxS): Transition probability matrix. S(t) at rows, S(t+1) at columns
# N   (int): How many data points to generate
# delta (S): Vector of probabilities of the starting state. Fix to e.x. [0,1,0,0] to always start in state 2.

def HMM_gen(mus, sds, TPM, delta, N):
    import numpy as np
    from numpy.random import choice
    from numpy.random import normal
    HMC = np.zeros(N)                                   # Initialize hidden markov chain
    HMC[0] = choice(range(0, mus.shape[0]), 1, p=delta) # Select first state from initial distribution
    
    for i in range(1,N):
        HMC[i] = choice(range(0, mus.shape[0]), 1, p=TPM[HMC[i-1].astype(int),])
    
    X = np.zeros([N,mus.shape[1]])
    for i in range(0,N):
        X[i,] = normal(mus[HMC[i].astype(int),], sds[HMC[i].astype(int),])
        
    return(X)

def HMM_wrap_sbi(theta):
    from HMM_gen import HMM_gen
    import numpy as np
    from math import exp
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
    
    return(HMM_gen(mus=mus, sds=sds, TPM=TPM, delta=[1,0,0], N=1800).flatten())