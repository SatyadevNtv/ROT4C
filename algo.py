# Toggle the numpy calls to `cupy` calls for GPU usage
import numpy as np
from numpy import inf

import sinkhorn_gpu as helper
from ot.bregman import sinkhorn

from IPython import embed

def gradLW(x, dLdH, pred, m, num_features):
    """
    Gradient of loss function w.r.t weights.
    """
    batch = x.shape[0]
    predGPU = np.array(pred)
    xGPU = np.array(x)
    dLdHGPU = np.array(dLdH)
    return np.einsum('bi,bj->ji', ((predGPU * dLdHGPU.reshape(batch, m)) - predGPU * np.einsum('bi,bi->b', predGPU, dLdHGPU.reshape(batch, m), dtype='double').reshape(batch, 1)), xGPU.reshape(batch, num_features), dtype='double')


def gradLH(u, m, batch_size, sinkhorn_reg):
    """
    Gradient of loss function w.r.t h(x).
    """
    ones = np.ones(m)
    if 0.0 in u:
        # TODO Can improve
        non_zero_ids = np.where(u > 0)
        lower_bound = min(1e-15, u[non_zero_ids].min())
        zero_ids = np.where(u == 0)
        u[zero_ids] = lower_bound
    dLdH = np.log(u)
    dLdH = dLdH - ((np.log(u).dot(ones)/m).reshape(batch_size, 1) * ones.reshape(1, m))
    dLdH = dLdH * sinkhorn_reg
    return dLdH


class W22():
    """
    Implements W22 loss function's gradient
    """

    def __init__(self, C, args, num_features, num_classes):
        self.options = args
        self.C = C
        self.m, self.n = num_classes, num_classes
        self.num_features = num_features


    def gradient(self, x, pred, gt):
        batch_size = pred.shape[0]
        loss, _, u  = helper.sinkhorn_fixedCost(
            np.array(pred),
            np.array(gt),
            np.array(self.C),
            self.options.sinkhorn_reg,
            1e-9,
            self.options.sinkhorn_iters
        )
        dLdH = gradLH(u, self.m, batch_size, self.options.sinkhorn_reg)
        return loss, gradLW(x.numpy().reshape(x.shape[0], x.shape[1], 1), dLdH.reshape(batch_size, self.m, 1), pred, self.m, self.num_features)


class RbOT():
    """
    Implements RbOT loss function's
    """

    def __init__(self, P, args, num_features, num_classes):
        self.options = args
        self.P = P
        self.m, self.n = num_classes, num_classes
        self.num_features = num_features

        if self.options.reg_type == "pnorm":
            self.k = self.options.pnorm_k
            assert self.k in [1, 2, 4]
            self.grad_gamma = self._egradv_elem_pnorm_gpu
            self.batch_grad_gamma = self._batch_egradv_elem_pnorm_gpu
        elif self.options.reg_type == "kl":
            if self.options.minit == "eye":
                self.M0 = np.eye(self.options.r).reshape(self.options.r**2)
            elif self.options.minit == "ones":
                self.M0 = np.outer(np.ones(self.options.r), np.ones(self.options.r)).reshape(self.options.r**2)
            else:
                raise ValueError
            self.tikhR = self.options.tikhr
            self.grad_gamma = self._egradv_kl_gpu
            self.batch_grad_gamma = self._batch_egradv_kl_gpu
        elif self.options.reg_type == "ds":
            # For DS, M0 is being used as 11^\top.
            self.tikhR = self.options.tikhr
            self.grad_gamma = self._egradv_doublyStochastic_gpu
            self.batch_grad_gamma = self._batch_egradv_doublyStochastic_gpu
        else:
            raise ValueError

        self.get_vgamma = self.get_vgamma_gpu
        self.batch_get_vgamma = self.batch_get_vgamma_gpu
        self.gamma0 = np.outer(np.ones(self.m)/self.m, np.ones(self.n)/self.n)


    def batch_get_vgamma_gpu(self, batch_size, gamma):
        gammaVecGpu = np.asarray(gamma.reshape(batch_size, self.m*self.n))
        res = np.einsum('bi,di->bd', gammaVecGpu, self.P, dtype='double')
        return res


    def _batch_egradv_elem_pnorm_gpu(self, batch_size, gamma, vgamma):
        v2k = vgamma**(2*self.k - 1)
        return 2*self.k * np.einsum('id,bd->bi', self.P.T, v2k, dtype='double').reshape(batch_size, self.m, self.n)


    def get_vgamma_gpu(self, gamma):
        return np.einsum('di,i->d', self.P, gamma.reshape(self.m*self.n), dtype='double')

    def _egradv_elem_pnorm_gpu(self, vgamma):
        v2k = vgamma**(2*self.k - 1)
        return 2*self.k * np.einsum('id,d->i', self.P.T, v2k, dtype='double').reshape(self.m, self.n)


    def _batch_egradv_kl_gpu(self, batch_size, gamma, vgamma):
        Mopt = self.M0 * np.exp(vgamma/self.tikhR)
        return np.einsum('id,bd->bi', self.P.T, Mopt, dtype='double').reshape(batch_size, self.m, self.n)

    def _egradv_kl_gpu(self, vgamma):
        Mopt = self.M0 * np.exp(vgamma/self.tikhR)
        return np.einsum('id,d->i', self.P.T, Mopt, dtype='double').reshape(self.m, self.n)


    def _batch_egradv_doublyStochastic_gpu(self, batch_size, gamma, vgamma):
        mu, nu = np.ones(self.options.r)/self.options.r, np.ones(self.options.r)/self.options.r
        Mopt = helper.sinkhorn_fixedMarginals(mu, nu, -vgamma.reshape(batch_size, self.options.r, self.options.r), reg=self.tikhR, stopThr=1e-9, numItermax=self.options.sinkhorn_iters)
        return np.einsum('id,bd->bi', self.P.T, Mopt.reshape(batch_size, self.options.r**2), dtype='double').reshape(batch_size, self.m, self.n)


    def _egradv_doublyStochastic_gpu(self, vgamma):
        mu, nu = np.ones(self.options.r)/self.options.r, np.ones(self.options.r)/self.options.r
        Mopt = sinkhorn(mu, nu, -vgamma.get().reshape(self.options.r, self.options.r), reg=self.tikhR, numItermax=self.options.sinkhorn_iters)
        return np.einsum('id,d->i', self.P.T, Mopt.reshape(self.options.r**2), dtype='double').reshape(self.m, self.n)


    def gradient(self, x, pred, gt):
        batch_size = pred.shape[0]

        ## FW ##
        gamma_hat = None
        maxiters = self.options.fw_iters
        # For more iters, we can optimize by truncating the gt by excluding 0's
        # as the optimal transport plan to that element would be 0 anyways.
        max_n = 0
        nz_ids_list = []
        for i in range(batch_size):
            nz_ids_list.append(gt[i].nonzero()[0])
            max_n = max(len(nz_ids_list[-1]), max_n)
        tr_gt = np.zeros((batch_size, max_n))
        for i in range(batch_size):
            tr_gt[i][list(range(len(nz_ids_list[i])))] = gt[i][nz_ids_list[i]]

        iters = 0
        while iters < maxiters:
            if iters == 0:
                vgamma = self.get_vgamma(self.gamma0)
                grad = self.grad_gamma(vgamma) + self.options.sinkhorn_reg*(1 + np.log(np.array(self.gamma0)))

                _, gamma_hat, u = helper.sinkhorn_fixedCost(pred, gt, grad, self.options.fw_mu, 1e-9, self.options.sinkhorn_iters)

                gamma = gamma_hat
            else:
                vgamma = self.batch_get_vgamma(batch_size, gamma)
                # Ignore warnings raised here.
                grad = self.batch_grad_gamma(batch_size, gamma, vgamma) + self.options.sinkhorn_reg*(1 + np.log(np.array(gamma)))
                # Truncate
                tr_grad = np.zeros((batch_size, self.m, max_n))
                for i in range(batch_size):
                    tr_grad[i][:, list(range(len(nz_ids_list[i])))] = grad[i][:, nz_ids_list[i]]
                tr_gamma_hat, u = helper.sinkhorn(pred, tr_gt, tr_grad, self.options.fw_mu, 1e-9, self.options.sinkhorn_iters)

                # Retrieve
                gamma_hat = np.zeros((batch_size, self.m, self.n))
                for i in range(batch_size):
                    gamma_hat[i][:, nz_ids_list[i]] = tr_gamma_hat[i][:, list(range(len(nz_ids_list[i])))]

                beta = 2/(2+iters)
                # Reuse `gamma_hat` for memory optimization
                np.subtract(gamma_hat, gamma, out=gamma_hat)
                np.multiply(beta, gamma_hat, out=gamma_hat)
                np.add(gamma, gamma_hat, out=gamma)
            iters+=1

        if self.options.reg_type == "pnorm":
            Mopt = 2*self.k * self.batch_get_vgamma(batch_size, gamma)**(2*self.k - 1)
        elif self.options.reg_type == "kl":
            Mopt = self.M0 * np.exp(self.batch_get_vgamma(batch_size, gamma)/self.tikhR)
        elif self.options.reg_type == "ds":
            mu_tmp, nu_tmp = np.ones(self.options.r)/self.options.r, np.ones(self.options.r)/self.options.r
            Mopt = helper.sinkhorn_fixedMarginals(mu_tmp, nu_tmp, -self.batch_get_vgamma(batch_size, gamma).reshape(batch_size, self.options.r, self.options.r), reg=self.tikhR, stopThr=1e-9, numItermax=self.options.sinkhorn_iters)
            Mopt = Mopt.reshape(batch_size, self.options.r**2)
        else:
            raise ValueError
        CStar = np.einsum('ds,bd->bs', self.P, Mopt, dtype='double').reshape(batch_size, self.m, self.n)
        loss = np.sum(gamma.reshape(batch_size, self.m, self.n) * CStar)

        CStar += self.options.sinkhorn_reg * (1 + np.log(gamma))
        CStar[np.where(CStar == -inf)] = 0
        nVec = np.array([len(nz_ids) for nz_ids in nz_ids_list])
        dLdH = (np.einsum('bmn,bn->bm', CStar, np.ones((batch_size, self.n)), dtype='double') - (1/self.m) * ( np.ones((batch_size, self.m)) ) * np.einsum('bm,bmn,bn->b', np.ones((batch_size, self.m)), CStar, np.ones((batch_size, self.n)), dtype='double')[:, np.newaxis])/nVec[:, np.newaxis]

        return loss, gradLW(x.numpy().reshape(x.shape[0], x.shape[1], 1), dLdH.reshape(batch_size, self.m, 1), pred, self.m, self.num_features)
