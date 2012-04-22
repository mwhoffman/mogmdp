from __future__ import division

import numpy as np
import scipy as sp
import scipy.linalg

from collections import namedtuple

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

#===================================================================================================
# Code to get the state/action transition and initial "state" models.
#===================================================================================================

ZModel = namedtuple('ZModel', 'mu0, Sigma0, F, m, Sigma')

def get_zmodel(model, policy):
    ISigma = np.eye(model.na) * policy.sigma**2
    KSigma = np.dot(policy.K, model.Sigma)
    KSigma0 = np.dot(policy.K, model.Sigma0)

    mu0 = np.r_[model.mu0, np.dot(policy.K, model.mu0) + policy.m]
    Sigma0 = np.r_[np.c_[model.Sigma0, KSigma0.T],
                   np.c_[KSigma0, np.dot(KSigma0, policy.K.T) + ISigma]]

    F = np.r_[np.c_[model.A, model.B], np.c_[np.dot(policy.K, model.A), np.dot(policy.K, model.B)]]
    m = np.r_[np.zeros(model.nx), policy.m]
    Sigma = np.r_[np.c_[model.Sigma, KSigma.T],
                  np.c_[KSigma, np.dot(KSigma, policy.K.T) + ISigma]]

    return ZModel(mu0, Sigma0, F, m, Sigma)

#===================================================================================================
# Code to compute the moments and the gradient.
#===================================================================================================

def kalman_predict(zmodel, mu, Sigma):
    mu = np.dot(zmodel.F, mu) + zmodel.m
    Sigma = np.dot(np.dot(zmodel.F, Sigma), zmodel.F.T) + zmodel.Sigma
    return (mu, Sigma)

def kalman_update(model, mu, Sigma):
    y, M, L = model.y[0], model.M[0], model.L[0]

    # FIXME: I'm currently just using the first component, of the reward
    # model, and I'm ignoring the weights w, i.e. just assuming they're one.

    # get the innovation r, i.e. the difference between the "observed" y
    # and the prediction; the cholesky of the innovation covariance cholS;
    # and the log-determinant of this covariance.
    d = model.nx + model.na
    r = y - np.dot(M, mu)
    cholS, _ = sp.linalg.cho_factor(np.dot(M, np.dot(Sigma, M.T)) + L, overwrite_a=True)
    logDetS = 2*np.sum(np.log(np.diag(cholS)))

    # get the Kalman gain and get the updated mean/variance.
    K = sp.linalg.cho_solve((cholS, False), np.dot(M, Sigma), overwrite_b=True).T
    mu = mu + np.dot(K, r)
    Sigma = np.dot(np.eye(d) - np.dot(K, M), Sigma)

    # this is the "likelihood" term, which corresponds to the reward. since
    # we're using unnormalized Gaussians we don't include the 2*pi term, and we
    # subtract off the normalizing constant involving L.
    c = np.sum(sp.linalg.solve_triangular(cholS, r, trans=1)**2)
    c += logDetS - np.log(np.linalg.det(L))
    c = np.exp(-0.5*c)

    return mu, Sigma, c

def get_forward(model, policy, H):
    # get the Z-model from the model/policy and initialize the forward messages.
    zmodel = get_zmodel(model, policy)
    forward = [None] * (H+1)
    forward[0] = (zmodel.mu0, zmodel.Sigma0)

    # run the forward pass up to the end-horizon.
    for k in xrange(H):
        forward[k+1] = kalman_predict(zmodel, *forward[k])

    return forward, zmodel

def get_moments(model, policy, gamma, H):
    forward, zmodel = get_forward(model, policy, H)

    # get the first components.
    mu_fwd, Sigma_fwd = forward[H]
    mu_hat, Sigma_hat, c = kalman_update(model, mu_fwd, Sigma_fwd)

    # initialize the components we'll be using for the backward pass.
    c *= gamma**H
    mu = c*mu_hat
    Omega = c*np.outer(mu_hat, mu_hat) + c*Sigma_hat

    # initialize the accumulators.
    J, Js, Z, ZZ = c, c, mu.copy(), Omega.copy()

    for n in reversed(xrange(H)):
        # these are the components of the forward messages and the backward
        # messages respectively, i.e. p(z_n) and p(z_n|y_n).
        mu_fwd, Sigma_fwd = forward[n]
        mu_hat, Sigma_hat, c = kalman_update(model, mu_fwd, Sigma_fwd)
        c *= gamma**n

        # get the components in order to do the smoothing step.
        tmp = np.dot(zmodel.F, Sigma_fwd)
        P = np.dot(tmp, zmodel.F.T) + zmodel.Sigma
        G = sp.linalg.solve(P, tmp, sym_pos=True, overwrite_b=True).T

        # do the smoothing for the summation of the first moment.
        mu_rev = mu_fwd - np.dot(G, np.dot(zmodel.F, mu_fwd) + zmodel.m)
        Gmu = np.dot(G, mu)
        mu = c*mu_hat + J*mu_rev + Gmu

        # do the smoothing for the summation of the second moment, for which we
        # get the covariance for free.
        a = np.outer(Gmu, mu_rev)
        Omega_hat = np.outer(mu_hat, mu_hat) + Sigma_hat
        Omega_rev = np.outer(mu_rev, mu_rev) + Sigma_fwd - np.dot(G, np.dot(P, G.T))
        Omega = c*Omega_hat + J*Omega_rev + np.dot(G, np.dot(Omega, G.T)) + a + a.T

        J += c
        Js += J
        Z += mu
        ZZ += Omega

    return J, Js, Z, ZZ

def get_gradient(model, policy, gamma, H):
    # get the expected return and the first/second moments in the joint
    # state/action space.
    J, Js, Z, ZZ = get_moments(model, policy, gamma, H)

    # get the moments we're actually interested in.
    nx = policy.nx
    X, U = Z[0:nx], Z[nx:]
    XX, UU, UX = ZZ[0:nx,0:nx], ZZ[nx:,nx:], ZZ[nx:,0:nx]

    # and get the expectation of the inner product of C with itself, where C is
    # the action minus the "predicted-action".
    M = np.c_[-policy.K, np.eye(model.na)]
    A = np.r_[np.c_[np.inner(policy.K, policy.K), -policy.K.T], M]
    CC = np.trace(np.dot(A, ZZ)) + J*np.inner(policy.m, policy.m) - 2*np.inner(policy.m, np.dot(M, Z))

    dK = (policy.sigma**-2) * UX - np.dot(policy.K, XX) - np.outer(policy.m, X)
    dm = (policy.sigma**-2) * U - np.dot(policy.K, X) - Js*policy.m
    ds = (policy.sigma**-3) * CC - Js*model.na*policy.sigma**-1

    return J, np.r_[dK.flatten(), dm.flatten(), ds]
