from __future__ import division

import numpy as np
import scipy as sp
import scipy.linalg
import lbfgsb

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
        policy = MoGPolicy.zero(self.nx, self.na)
        policy.update(np.asarray(theta))
        return policy

class MoGPolicy(object):
    def __init__(self, K, m, sigma, copy=True):
        self.K = np.array(K, ndmin=2, copy=copy)
        self.m = np.array(m, ndmin=1, copy=copy)
        self.sigma = sigma
        self.na, self.nx = self.K.shape
        self.nd = self.K.size + self.m.size + 1

    @classmethod
    def zero(cls, nx, na):
        return cls(np.zeros((na,nx)), np.zeros(na), 0.0)

    def dlogpi(self, x, u):
        n = x.shape[0]
        A = u - np.dot(x, self.K.T) - self.m.reshape(1,-1)
        dK = self.sigma**(-2) * oprod(A, x).reshape(n,-1)
        dm = self.sigma**(-2) * A
        ds = self.sigma**(-3) * sum(A**2, axis=1).reshape(n,-1) - self.na/self.sigma
        return np.c_[dK, dm, ds]

    def copy(self):
        return MoGPolicy(self.K, self.m, self.sigma)

    def pack(self):
        return np.r_[self.K.flatten(), self.m.flatten(), self.sigma]

    def update(self, dtheta):
        ell = self.nx * self.na
        self.K += dtheta[0:ell].reshape(self.na, self.nx)
        self.m += dtheta[ell:-1]
        self.sigma += dtheta[-1]

#===================================================================================================
# Code to get the state/action transition and initial "state" models.
#===================================================================================================

class ZModel(object):
    def __init__(self, model, policy):
        ISigma = np.eye(model.na) * policy.sigma**2
        KSigma = np.dot(policy.K, model.Sigma)
        KSigma0 = np.dot(policy.K, model.Sigma0)

        self.mu0 = np.r_[model.mu0, np.dot(policy.K, model.mu0) + policy.m]
        self.Sigma0 = np.r_[np.c_[model.Sigma0, KSigma0.T],
                            np.c_[KSigma0, np.dot(KSigma0, policy.K.T) + ISigma]]

        self.F = np.r_[np.c_[model.A, model.B],
                       np.c_[np.dot(policy.K, model.A), np.dot(policy.K, model.B)]]
        self.m = np.r_[np.zeros(model.nx), policy.m]
        self.Sigma = np.r_[np.c_[model.Sigma, KSigma.T],
                           np.c_[KSigma, np.dot(KSigma, policy.K.T) + ISigma]]

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

def kalman_smooth(zmodel, J, mu, Omega, mu_fwd, Sigma_fwd, mu_hat, Sigma_hat, c):
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

    return mu, Omega

def get_forward(model, policy, H):
    # get the Z-model from the model/policy and initialize the forward messages.
    zmodel = ZModel(model, policy)
    forward = [None] * (H+1)
    forward[0] = (zmodel.mu0, zmodel.Sigma0)

    # run the forward pass up to the end-horizon.
    for k in xrange(H):
        forward[k+1] = kalman_predict(zmodel, *forward[k])

    return forward, zmodel

def get_zmoments(model, policy, gamma, H):
    forward, zmodel = get_forward(model, policy, H)
    do_init = True

    for n in reversed(xrange(H+1)):
        try:
            # these are the components of the forward messages and the backward
            # messages respectively, i.e. p(z_n) and p(y_n|z_n).
            mu_fwd, Sigma_fwd = forward[n]
            mu_hat, Sigma_hat, c = kalman_update(model, mu_fwd, Sigma_fwd)
            c *= gamma**n
        except np.linalg.LinAlgError:
            do_init = True
            continue

        if do_init:
            mu = c*mu_hat
            Omega = c*np.outer(mu_hat, mu_hat) + c*Sigma_hat
            J, Js, Z, ZZ = 0, 0, 0, 0
            my_init, do_init = n, False
        else:
            try:
                mu, Omega = kalman_smooth(zmodel, J, mu, Omega,
                                          mu_fwd, Sigma_fwd,
                                          mu_hat, Sigma_hat, c)
            except np.linalg.LinAlgError:
                do_init = True
                continue

        # the sum of all components from n to the horizon H.
        J += c; Js += J; Z += mu; ZZ += Omega

    # if my_init < H:
    #     print 'WARNING: truncated horizon to %d (rather than %d)' % (n, H)

    return J, Js, Z, ZZ

def get_jtheta(model, policy, gamma, H):
    forward, _ = get_forward(model, policy, H)
    J = 0.0
    for n in reversed(xrange(H)):
        _, _, c = kalman_update(model, *forward[n])
        J += gamma**n * c
    return J

def get_moments(model, policy, gamma, H):
    # get the expected return and the first/second moments in the joint
    # state/action space.
    J, Js, Z, ZZ = get_zmoments(model, policy, gamma, H)

    # get the moments we're actually interested in.
    nx = policy.nx
    X, U = Z[0:nx], Z[nx:]
    XX, UU, UX = ZZ[0:nx,0:nx], ZZ[nx:,nx:], ZZ[nx:,0:nx]

    # and get the expectation of the inner product of C with itself, where C is
    # the action minus the "predicted-action".
    M = np.c_[-policy.K, np.eye(model.na)]
    A = np.r_[np.c_[np.inner(policy.K, policy.K), -policy.K.T], M]
    CC = np.trace(np.dot(A, ZZ)) + Js*np.inner(policy.m, policy.m) - 2*np.inner(policy.m, np.dot(M, Z))

    return J, Js, X, U, XX, UU, UX, CC

def get_gradient(model, policy, gamma, H):
    # get the moments we're interested in.
    J, Js, X, U, XX, UU, UX, CC = get_moments(model, policy, gamma, H)

    dK = (policy.sigma**-2) * (UX - np.dot(policy.K, XX) - np.outer(policy.m, X))
    dm = (policy.sigma**-2) * (U - np.dot(policy.K, X) - Js*policy.m)
    ds = (policy.sigma**-1) * (CC*policy.sigma**-2 - Js*model.na)

    return J, np.r_[dK.flatten(), dm.flatten(), ds]

def get_emstep(model, policy, gamma, H):
    # get the moments we're interested in.
    J, Js, X, U, XX, UU, UX, CC = get_moments(model, policy, gamma, H)

    K = scipy.linalg.solve(XX, UX - np.outer(policy.m, X), sym_pos=True, overwrite_b=True)
    m = (U - np.dot(policy.K, X)) / Js
    s = np.sqrt(CC / Js / model.na)

    return J, np.r_[(K-policy.K).flatten(), (m-policy.m).flatten(), s-policy.sigma]

def solve_mogmdp_em(model, policy, gamma, H, maxfun=100):
    ns = np.arange(maxfun+1, dtype=np.int32)
    xs = np.empty((maxfun+1, policy.nd), np.float64)
    fs = np.empty(maxfun+1, np.float64)

    policy = policy.copy()
    for i in xrange(maxfun):
        xs[i] = policy.pack()
        fs[i], dtheta = get_emstep(model, policy, gamma, H)
        policy.update(dtheta)
    xs[maxfun] = policy.pack()
    fs[maxfun] = get_jtheta(model, policy, gamma, H)

    return fs[-1], xs[-1], dict(numevals=ns, theta=xs, jtheta=fs)

def solve_mogmdp(model, policy, gamma, H, sigma_min=0.1, em=False):
    # get the initial parameter vector and any bounds.
    theta0 = policy.pack()
    bounds = [(None,None) for i in xrange(len(theta0)-1)] + [(sigma_min,None)]

    # get the objective function.
    fun = get_emstep if em else get_gradient
    obj = lambda theta: map(np.negative, fun(model, model.unpack_policy(theta), gamma, H))

    # solve the problem.
    J, theta, info = lbfgsb.lbfgsb(obj, theta0, bounds)

    # just rename the info elements to be more policy-gradient-y.
    info[ 'theta'] = info['x']; del info['x']
    info['jtheta'] = info['f']; del info['f']
    info['dtheta'] = info['g']; del info['g']

    info['jtheta'] *= -1
    info['dtheta'] *= -1

    return -J, theta, info