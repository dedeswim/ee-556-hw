import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from projL1 import projL1
from time import time


def projNuc(Z, kappa):
    assert kappa > 0, "Radius s must be strictly positive (%d <= 0)" % kappa
    
    # Compute the SVD
    u, s, v = np.linalg.svd(Z)
    
    # Compute nuclear norm
    sigma = np.diag(s)
    nuclear_norm = np.linalg.norm(sigma, ord=1)
    
    # check if v is already a solution
    if nuclear_norm <= kappa:
        # Nuclear norm is <= kapp
        return Z
    
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # compute l1-projected s matrix
    s_l1 = projL1(s)

    # Compute and return projected matrix
    return u @ s_l1 @ v.T


data = scipy.io.loadmat('./dataset/ml-100k/ub_base')  # load 100k dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float).toarray()
kappa = 5000

tstart = time()
Z_proj = projNuc(Z, kappa)
elapsed = time() - tstart
print('proj for 100k data takes {} sec'.format(elapsed))

# NOTE: This one can take few minutes!
data = scipy.io.loadmat('./dataset/ml-1m/ml1m_base')  # load 1M dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float).toarray()
kappa = 5000

tstart = time()
Z_proj = projNuc(Z, kappa)
elapsed = time() - tstart
print('proj for 1M data takes {} sec'.format(elapsed))
