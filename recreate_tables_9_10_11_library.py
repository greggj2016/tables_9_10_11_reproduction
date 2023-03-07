import numpy as np
from copy import deepcopy as COPY
from scipy.stats import norm
from tqdm import tqdm
import pdb

def asd_cutoff(X, a):
    mu = np.mean(X)
    sig = np.std(X)
    inds1 = (X >= mu - a*sig)
    inds2 = (X <= mu + a*sig)
    inds_to_keep = np.logical_and(inds1, inds2)
    return(inds_to_keep)

def mad_cutoff(X, a):
    med = np.median(X)
    MAD = (1/norm.ppf(3/4))*np.median(np.abs(med - X))
    inds1 = (X >= med - a*MAD)
    inds2 = (X <= med + a*MAD)
    inds_to_keep = np.logical_and(inds1, inds2)
    return(inds_to_keep)

def iqr_cutoff(X):
    Q1, Q3 = np.percentile(X, [25, 75])
    cIQR = 1.5*(Q3 - Q1)
    inds1 = (X >= Q1 - cIQR)
    inds2 = (X <= Q3 + cIQR)
    inds_to_keep = np.logical_and(inds1, inds2)
    return(inds_to_keep)

def T2_thresh_data(X, T, cutoff):

    all_inds = np.arange(len(X))
    inds = np.arange(len(X))
    for i in range(T): inds = inds[cutoff(X[inds])]
    outlier_inds = (np.isin(all_inds, inds) == False)
    return(outlier_inds)

def mean_shift_process(X, k, l):
    X_intermediate = COPY(X)
    X_new = COPY(X_intermediate)
    inds = [z for z in range(len(X))]
    for iteration in range(l):
        for i in tqdm(inds):
            dists = np.sum((X_new[i] - X_new)**2, axis = 1)**(1/2)
            knns = X_new[np.argsort(dists)[1:(k+1)]]
            X_intermediate[i] = np.mean(knns, axis = 0)
        X_new = COPY(X_intermediate)
    return(X_new)

def MOD(X, k, l):
    X_new = mean_shift_process(X, k, l)
    MOD_scores = np.sum((X - X_new)**2, axis = 1)**(1/2)
    return(MOD_scores)
