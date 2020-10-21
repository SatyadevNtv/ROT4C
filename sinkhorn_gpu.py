"""
This module implements sinkhorn for batch inputs.

NOTE: Some of the checks are skipped towards the benefit
of computation time. These are OK since the maxiters is generally
in orders of 10 and the entropic regularizer is fine-tuned before
the run of full algorithm. Checkout the helper `sinkhornAutoTune` in `utils.py`
for automatic (prior to full run) tuning.
"""


# Toggle this for GPU usage
import numpy as cp

def sinkhorn_fixedCost(batch_a, batch_b, M, reg, stopThr, numItermax):
    """
    Sinkhorn Knopp when Cost matrix is common across datapoints
    """
    batch_a = cp.array(batch_a)
    batch_b = cp.array(batch_b)
    batch = batch_a.shape[0]
    classes = batch_a.shape[1]

    u = cp.ones((batch, classes)) / classes
    v = cp.ones((batch, classes)) / classes

    K = cp.empty(M.shape, dtype=M.dtype)
    cp.divide(cp.array(M), -reg, out=K)
    cp.exp(K, out=K)

    for cpt in range(numItermax):
        KtransposeU = cp.einsum('ij,bi->bj', K, u, dtype='double')
        v = cp.divide(batch_b, KtransposeU)
        u = 1. / ((1. / batch_a) * cp.einsum('ij,bj->bi', K, v, dtype='double'))

    gamma = cp.einsum('bi,ij,bj->bij', u, K, v, dtype='double')
    loss = cp.sum(cp.einsum('ijk,jk->i', gamma, M, dtype='double'))
    return loss, gamma, u


def sinkhorn_fixedMarginals(a, b, M, reg, stopThr, numItermax):
    """
    Sinkhorn Knopp when marginals are common across datapoints
    """
    batch = M.shape[0]
    u = cp.ones((batch, a.shape[0])) / a.shape[0]
    v = cp.ones((batch, b.shape[0])) / b.shape[0]

    K1 = cp.exp(-M/reg)
    for cpt in range(numItermax):
        KtransposeU = cp.einsum('bij,bi->bj', K1, u, dtype='double')
        v = cp.divide(b, KtransposeU)
        u = 1. / ((1. / a) * cp.einsum('bij,bj->bi', K1, v, dtype='double'))

    gamma = cp.einsum('bi,bij,bj->bij', u, K1, v, dtype='double')
    return gamma


def sinkhorn(batch_a, batch_b, M, reg, stopThr, numItermax):
    """
    Sinkhorn Knopp
    """
    batch_a = cp.array(batch_a)
    batch_b = cp.array(batch_b)
    batch = batch_a.shape[0]
    classes_a = batch_a.shape[1]
    classes_b = batch_b.shape[1]

    u = cp.ones((batch, classes_a)) / classes_a
    v = cp.ones((batch, classes_b)) / classes_b

    K1 = cp.exp(-M/reg)
    for cpt in range(numItermax):
        KtransposeU = cp.einsum('bij,bi->bj', K1, u, dtype='double')
        v = cp.divide(batch_b, KtransposeU)
        u = 1. / ((1. / batch_a) * cp.einsum('bij,bj->bi', K1, v, dtype='double'))

    gamma = cp.einsum('bi,bij,bj->bij', u, K1, v, dtype='double')
    return gamma, u
