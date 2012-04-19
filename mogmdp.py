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

class ForwardMessage(object):
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma

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

#===================================================================================================
# MESSAGE HANDLERS/SOLVERS.
#===================================================================================================

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

#===================================================================================================
# New code to just compute the moments.
#===================================================================================================

def get_moments(model, policy, gamma, H):
    trans = ZTransition(model, policy)
    jtheta = 0.0

    forward = [ForwardMessage.init(model, policy)]
    for k in xrange(H):
        forward.append(forward[-1].get_next(trans))

    for k in reversed(xrange(H+1)):
        alpha = forward[k]
        y, M, L = model.y[0], model.M[0], model.L[0]
        
        # FIXME: I'm currently just using the first component, of the reward 
        # model, and I'm ignoring the weights w, i.e. just assuming they're one.
                
        # get the innovation r, i.e. the difference between the "observed" y 
        # and the prediction; the cholesky of the innovation covariance cholS; 
        # and the log-determinant of this covariance.
        d = y.size
        r = y - np.dot(M, alpha.mu)
        cholS, _ = sp.linalg.cho_factor(np.dot(M, np.dot(alpha.Sigma, M.T)) + L, overwrite_a=True)
        logDetS = 2*np.sum(np.log(np.diag(cholS)))
    
        K = sp.linalg.cho_solve((cholS, False), np.dot(M, alpha.Sigma), overwrite_b=True).T
        mu = alpha.mu + np.dot(K, r)
        Sigma = np.dot(np.eye(d) - np.dot(K, M), alpha.Sigma)

        # this is the "likelihood" term, which corresponds to the reward. since
        # we're using unnormalized Gaussians we don't include the 2*pi term, and we
        # subtract off the normalizing constant involving L.
        c = np.sum(sp.linalg.solve_triangular(cholS, r, trans=1)**2)
        c += logDetS - np.log(np.linalg.det(L))
        c = np.exp(-0.5*c)
        
        jtheta += gamma**k * c

    return jtheta
    # return c, mu, Sigma
