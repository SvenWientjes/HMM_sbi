########################################################################################
# Function that generates a DataSet following a Hidden Markov Model structure with 
# Gaussian distributed features
### Parameter Inputs:
# mus (SxM): Means of the features organized by occupied state
# sds (SxM): Standard deviations of the features organized by occupied state
# TPM (SxS): Transition probability matrix. S(t) at rows, S(t+1) at columns
# N   (int): How many data points to generate
# delta (S): Vector of probabilities of the starting state. Fix to e.x. [0,1,0,0] to always start in state 2.
import numpy as np
from numpy.random import choice
from numpy.random import normal

mus = np.array([[0.1, 1.1, 2.1],
                [2.1, 0.1, 1.1],
                [-3,  -2,  -1 ]])
sds = np.array([[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5,0.5,0.5]])
TPM = np.array([[0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9]])
delta = [1,0,0]
N = 1800

def HMM_gen(mus, sds, TPM, delta, N):
    HMC = np.zeros(N)                                   # Initialize hidden markov chain
    HMC[0] = choice(range(0, mus.shape[0]), 1, p=delta) # Select first state from initial distribution
    
    for i in range(1,N):
        HMC[i] = choice(range(0, mus.shape[0]), 1, p=TPM[HMC[i-1].astype(int),])
    
    X = np.zeros([N,mus.shape[1]])
    for i in range(0,N):
        X[i,] = normal(mus[HMC[i].astype(int),], sds[HMC[i].astype(int),])
        
    return(X)
