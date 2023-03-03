import numpy as np
from copy import deepcopy as COPY
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
    MAD = 1.4826*np.median(np.abs(med - X))
    inds1 = med - a*MAD
    inds2 = med + a*MAD
    inds_to_keep = np.logical_and(inds1, inds2)
    return(inds_to_keep)

def iqr_cutoff(X):
    Q1, Q3 = np.percentile(X, [25, 75])
    cIQR = 1.5*(Q3 - Q1)
    inds1 = Q1 - cIQR
    inds2 = Q3 + cIQR
    inds_to_keep = np.logical_and(inds1, inds2)
    return(inds_to_keep)

def thresh_data(X, a, T):

    inds = np.arange(len(X))
    inds_asd = COPY(inds)
    inds_mad = COPY(inds)
    inds_iqr = COPY(inds)
    for i in range(T):

        inds_asd = inds_asd[asd_cutoff(X[inds_asd], a)]
        inds_mad = inds_mad[mad_cutoff(X[inds_mad], a)]
        inds_iqr = inds_iqr[iqr_cutoff(X[inds_iqr])]

    outliers_asd = (np.isin(inds, inds_asd) == False)
    outliers_mad = (np.isin(inds, inds_mad) == False)
    outliers_iqr = (np.isin(inds, inds_iqr) == False)
    return([outliers_asd, outliers_mad, outliers_iqr])

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
