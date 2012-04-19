from __future__ import division

import numpy as np
import scipy as sp

import scipy.linalg

class MoGMDP(object):
    """Mixture of Gaussians MDP."""

    def __init__(self, mu0, Sigma0, A, B, Sigma, w, y, L, M):
        # the initial state model.
        self.mu0 = np.array(mu0, ndmin=1)
        self.Sigma0 = np.array(Sigma0, ndmin=2)

        # the transition model.
        self.A = np.array(A, ndmin=2)
        self.B = np.array(B, ndmin=2)
        self.Sigma = np.array(Sigma, ndmin=2)

        # the reward model.
        self.w = np.array(w, ndmin=1)
        self.y = np.array(y, ndmin=2)
        self.L = np.array(L, ndmin=3)
        self.M = np.array(M, ndmin=3)

        # state/action dimensions.
        self.nx, self.na = self.B.shape
        self.nr = self.w.size

        # cache cholesky decompositions of our matrices.
        self.cholSigma,  _ = sp.linalg.cho_factor(self.Sigma)
        self.cholSigma0, _ = sp.linalg.cho_factor(self.Sigma0)

        self.cholL = self.L.copy()
        for Li in self.cholL:
            sp.linalg.cho_factor(Li, overwrite_a=True)

    def unpack_policy(self, theta):
        theta = np.array(theta)
        ell = self.nx * self.na
        K = theta[0:ell].reshape(self.na, self.nx)
        m = theta[ell:-1]
        sigma = theta[-1]
        return MoGPolicy(K, m, sigma)

class MoGPolicy(object):
    def __init__(self, K, m, sigma):
        self.K = np.array(K, ndmin=2)
        self.m = np.array(m, ndmin=1)
        self.sigma = sigma
        self.na, self.nx = self.K.shape

    def dlogpi(self, x, u):
        n = x.shape[0]
        A = u - np.dot(x, self.K.T) - self.m.reshape(1,-1)
        dK = self.sigma**(-2) * oprod(A, x).reshape(n,-1)
        dm = self.sigma**(-2) * A
        ds = self.sigma**(-3) * sum(A**2, axis=1).reshape(n,-1) - self.na/self.sigma
        return c_[dK, dm, ds]

class ZTransition(object):
    def __init__(self, model, policy):
        self.m = np.r_[np.zeros(model.nx), policy.m]
        self.F = np.r_[
            np.c_[model.A, model.B],
            np.c_[np.dot(policy.K, model.A), np.dot(policy.K, model.B)]]

        KSigma = np.dot(policy.K, model.Sigma)
        self.Sigma = np.r_[
            np.c_[model.Sigma, KSigma.T],
            np.c_[KSigma, np.dot(KSigma, policy.K.T) + np.eye(model.na) * (policy.sigma**2)]]

        cholSigma, _ = sp.linalg.cho_factor(self.Sigma)
        cholSigmaI_F = sp.linalg.solve_triangular(cholSigma, self.F, trans=1)
        cholSigmaI_m = sp.linalg.solve_triangular(cholSigma, self.m, trans=1)

        self.SigmaI = sp.linalg.cho_solve((cholSigma, False), np.eye(model.nx+model.na))
        self.SigmaI_F = sp.linalg.solve_triangular(cholSigma, cholSigmaI_F)
        self.SigmaI_m = sp.linalg.solve_triangular(cholSigma, cholSigmaI_m)
        self.F_SigmaI_F = np.dot(cholSigmaI_F.T, cholSigmaI_F)
        self.m_SigmaI_m = np.dot(cholSigmaI_m.T, cholSigmaI_m)
        self.logDetSigma = 2*np.sum(np.log(np.diag(cholSigma)))

class ForwardMessage(object):
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma

        # get the cholesky (upper triangular) of the covariance.
        cholSigma, _ = sp.linalg.cho_factor(self.Sigma)
        cholSigmaI_mu = sp.linalg.solve_triangular(cholSigma, self.mu, trans=1)

        # cache some values.
        self.SigmaI = sp.linalg.cho_solve((cholSigma, False), np.eye(mu.size))
        self.SigmaI_mu = sp.linalg.solve_triangular(cholSigma, cholSigmaI_mu)
        self.cholSigma = cholSigma
        self.logDetSigma = 2*np.sum(np.log(np.diag(cholSigma)))
        self.mu_SigmaI_mu = np.dot(cholSigmaI_mu.T, cholSigmaI_mu)

    @classmethod
    def init(cls, model, policy):
        KSigma0 = np.dot(policy.K, model.Sigma0)
        mu = np.r_[model.mu0, np.dot(policy.K, model.mu0) + policy.m]
        Sigma = np.r_[
            np.c_[model.Sigma0, KSigma0.T],
            np.c_[KSigma0, np.dot(KSigma0, policy.K.T) + np.eye(model.na) * (policy.sigma**2)]]
        return cls(mu, Sigma)

    def get_next(self, trans):
        mu = np.dot(trans.F, self.mu) + trans.m
        Sigma = np.dot(np.dot(trans.F, self.Sigma), trans.F.T) + trans.Sigma
        return self.__class__(mu, Sigma)

class BackwardMessage(object):
    def __init__(self, c, mu, Omega):
        self.c = c
        self.mu = mu
        self.Omega = Omega

    @classmethod
    def init(cls, model):
        c, mu, Omega = map(np.zeros, (model.w.shape, model.y.shape, model.L.shape))
        for i in xrange(c.size):
            cholLI_y = sp.linalg.solve_triangular(model.cholL[i], model.y[i], trans=1)
            cholLI_M = sp.linalg.solve_triangular(model.cholL[i], model.M[i], trans=1)

            c[i] = -2*np.log(model.w[i]) + np.dot(cholLI_y.T, cholLI_y) + \
                    np.log(np.linalg.det(2 * np.pi * model.L[i]))
            # c[i] = np.dot(cholLI_y.T, cholLI_y)
            mu[i] = np.dot(cholLI_M.T, cholLI_y)
            Omega[i] = np.dot(cholLI_M.T, cholLI_M)

        return cls(c, mu,  Omega)

    def get_next(self, trans):
        c, mu, Omega = map(np.zeros, (self.c.shape, self.mu.shape, self.Omega.shape))

        for i in xrange(c.size):
            cholO, _ = sp.linalg.cho_factor(self.Omega[i] + trans.SigmaI)
            cholOI_vt = sp.linalg.solve_triangular(cholO, self.mu[i] + trans.SigmaI_m, trans=1)
            cholOI_SigmaI_F = sp.linalg.solve_triangular(cholO, trans.SigmaI_F, trans=1)

            c[i] = self.c[i] + trans.logDetSigma + 2*np.sum(np.log(np.diag(cholO))) \
                             - np.dot(cholOI_vt.T, cholOI_vt) + trans.m_SigmaI_m

            mu[i] = np.dot(trans.SigmaI_F.T, sp.linalg.solve_triangular(cholO, cholOI_vt) - trans.m)
            Omega[i] = trans.F_SigmaI_F - np.dot(cholOI_SigmaI_F.T, cholOI_SigmaI_F)

        return self.__class__(c, mu, Omega)

#===================================================================================================
# MESSAGE HANDLERS/SOLVERS.
#===================================================================================================

def get_alphabeta(alpha, beta, i):
    cholS, _ = sp.linalg.cho_factor(beta.Omega[i] + alpha.SigmaI)
    logDetS = 2*np.sum(np.log(np.diag(cholS)))

    m = sp.linalg.cho_solve((cholS, False), beta.mu[i] + alpha.SigmaI_mu)
    S = sp.linalg.cho_solve((cholS, False), np.eye(cholS.shape[0]))

    v = sp.linalg.solve_triangular(alpha.cholSigma, m, trans=1)
    w = np.exp(-0.5 * (
        + beta.c[i] + logDetS - np.dot(np.dot(m.T, beta.Omega[i]), m) - np.dot(v.T,v)
        + alpha.logDetSigma + alpha.mu_SigmaI_mu))

    return (w,m,S)

def get_jtheta(model, policy, gamma, H):
    trans = ZTransition(model, policy)
    alpha = ForwardMessage.init(model, policy)
    beta = BackwardMessage.init(model)
    jtheta = 0.0

    for n in xrange(H+1):
        for i in xrange(model.nr):
            w, _, _ = get_alphabeta(alpha, beta, i)
            jtheta += gamma**n * w
        alpha = alpha.get_next(trans)
    return jtheta

def get_expectations(policy, mu, Sigma):
    # convert these intro matrices (while not copying) just so that they're 
    # easier to work with. it doesn't really use any additional memory anyways.
    z = np.asmatrix(mu).T
    S = np.asmatrix(Sigma)
    zz = np.asmatrix(Sigma + z*z.T)

    # given the mean and covariance get the sufficient statistics, i.e. single
    # state/action expectations, expectations of state/action products, and
    # finally the covariances (respectively).
    nx = policy.nx
    x, u = z[0:nx], z[nx:]
    xx, uu, ux = zz[0:nx,0:nx], zz[nx:,nx:], zz[nx:,0:nx]
    covx, covu, covxu = S[0:nx,0:nx], S[nx:,nx:], S[0:nx,nx:]

    # finally, compute CC, the expectation of (u-Kx-m)'(u-Kx-m)
    mt = u - policy.K*x
    Kt = policy.K*covxu
    St = covu - Kt.T - Kt + policy.K*covx*policy.K.T
    CC = np.trace(St) + mt.T*mt - 2*np.dot(policy.m, mt) + np.dot(policy.m, policy.m)

    # return everything, note it's still a matrix.
    return (x,u,xx,uu,ux,CC)

def get_messages(model, policy, H):
    trans = ZTransition(model, policy)
    A = [ForwardMessage.init(model, policy)]
    B = [BackwardMessage.init(model)]
    for k in xrange(H):
        A.append(A[-1].get_next(trans))
        B.append(B[-1].get_next(trans))
    return A, B

def get_gradient(model, policy, gamma, H):
    A, B = get_messages(model, policy, H)
    J, dJ = 0, 0

    for k in xrange(H+1):
        for n in xrange(k+1):
            for i in xrange(model.nr):
                # grab the mixture model.
                (w,mu,Sigma) = get_alphabeta(A[n], B[k-n], i)
                (x,u,xx,uu,ux,CC) = get_expectations(policy, mu, Sigma)

                # compute the partial derivatives.
                dK = (policy.sigma**-2) * np.array(ux - policy.K*xx - policy.m*x.T)
                dm = (policy.sigma**-2) * np.array(u - policy.K*x - policy.m)
                ds = (policy.sigma**-1) * np.array(policy.sigma**-2 * CC - model.na)

                # put together the gradient.
                w  *= (gamma ** k)
                J  += w if (n == k) else 0
                dJ += w * np.r_[dK.flatten(), dm.flatten(), ds.flatten()]

    return J, dJ

# def get_moments(model, policy, gamma, H):
#     A, B = get_messages(model, policy, H)
#     Z, ZZ = 0, 0
# 
#     for k in xrange(H+1):
#         for n in xrange(k+1):
#             for i in xrange(model.nr):
#                 w, mu, Sigma = get_alphabeta(A[n], B[k-n], i)
#                 Z  += w * gamma**k * mu
#                 ZZ += w * gamma**k * (Sigma + np.outer(mu, mu))
# 
#     return Z, ZZ

#===================================================================================================
# New code to just compute the moments.
#===================================================================================================

def mvn_pdf(y, mu, Sigma):
    d = len(y)
    cholSigma, _ = sp.linalg.cho_factor(Sigma)
    diff = sp.linalg.solve_triangular(cholSigma, y-mu, trans=1, overwrite_b=True)
    tmp = np.sum(diff**2) + 2*np.sum(np.log(np.diag(cholSigma))) + d * np.log(2*np.pi)
    return np.exp(-0.5*tmp)

def get_moments(model, policy, gamma, H):
    trans = ZTransition(model, policy)

    forward = [ForwardMessage.init(model, policy)]
    for k in xrange(H):
        forward.append(forward[-1].get_next(trans))

    # for k in reversed(xrange(H+1)):
        
    k = H
    alpha = forward[k]
    y, M, L = model.y[0], model.M[0], model.L[0]
        
    # get the innovation r, i.e. the difference between the "observed" y 
    # and the prediction, and the cholesky of the innovation covariance 
    # cholS.
    r = y - np.dot(M, alpha.mu)
    cholS = sp.linalg.cho_factor(np.dot(M, np.dot(alpha.Sigma, M.T)) + L, overwrite_a=True)
        
    K = sp.linalg.cho_solve(cholS, np.dot(M, alpha.Sigma), overwrite_b=True).T
    mu = alpha.mu + np.dot(K, r)
    Sigma = np.dot(np.eye(y.size) - np.dot(K, M), alpha.Sigma)
        
    # NOTE! This appears to be off by a factor of 2pi, at least in the case when L is one. This is probably because here I'm assuming a true, normalized Gaussian, but I should really figure out what's going on and fix it.
    # c = mvn_pdf(y, np.dot(M, mu), MSM + L)
    # jtheta += gamma**k * c

    return mu, Sigma

#===================================================================================================
# HELPER CODE.
#===================================================================================================

def oprod(x, y):
    """A 'vectorized' version of the outer product."""
    nx = x.shape[1]
    ny = y.shape[1]
    return \
        tile(x.reshape(-1,nx, 1), (1, 1,ny)) * \
        tile(y.reshape(-1, 1,ny), (1,nx, 1))
