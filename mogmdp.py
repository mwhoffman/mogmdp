"""
Module to analytically solve MDPs using an EM-based algorithm for models with
linear-Gaussian transitions and rewards which are a mixture-of-Gaussians.
"""

from __future__ import division

import numpy as np
import scipy as sp
import scipy.linalg
import lbfgsb

__all__ = ['MoGMDP', 'MoGPolicy', 'solve_mogmdp', 'solve_mogmdp_em']

__author__ = "Matthew W. Hoffman"
__copyright__ = "Copyright (c) 2012, M.W.Hoffman <hoffmanm@cs.ubc.ca>"

#===================================================================================================
# classes for the mixture-of-Gaussians model, the corresponding policy, and the
# transition model in the joint state/action space.
#===================================================================================================

class MoGMDP(object):
    """
    Model for the mixture-of-Gaussians MDP. This class basically just acts as a
    structure containing the model parameters: (mu0, Sigma0) for the initial
    state distribution; (A, B, Sigma) for the transition distribution; and
    (w, y, L, M) for the reward model.

    See the paper for more details.
    """

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

        # Cache the cholesky decompositions. NOTE: these are upper triangular!
        self.cholSigma0 = sp.linalg.cholesky(self.Sigma0)
        self.cholSigma  = sp.linalg.cholesky(self.Sigma)
        self.cholL = [sp.linalg.cho_factor(Li)[0] for Li in self.L]
        self.logDetL = [2*np.sum(np.log(np.diag(cholL))) for cholL in self.cholL]

    def sample_init(self, n):
        return self.mu0 + np.dot(np.random.normal(size=(n, self.nx)), self.cholSigma0)

    def sample_next(self, x, u):
        z = np.c_[x, u]
        r = 0.0
        for w, y, M, cholL in zip(self.w, self.y, self.M, self.cholL):
            a = sp.linalg.solve_triangular(cholL, (y - np.dot(z, M.T)).T, trans=1)
            r += w * np.exp(-0.5 * np.sum(a**2, axis=0))
        xn = np.dot(x, self.A.T) + np.dot(u, self.B.T)
        xn += np.dot(np.random.normal(size=x.shape), self.cholSigma)
        return xn, r

    def unpack_policy(self, theta):
        """Return a policy object parameterized by the vector `theta`."""
        policy = MoGPolicy.zero(self.nx, self.na)
        policy.update(np.asarray(theta))
        return policy

class MoGPolicy(object):
    """
    Policy object for the mixture-of-Gaussians MDP. Like the MDP object this
    acts as a structure for the policy parameters (K, m, sigma).
    """
    def __init__(self, K, m, sigma, copy=True):
        self.K = np.array(K, ndmin=2, copy=copy)
        self.m = np.array(m, ndmin=1, copy=copy)
        self.sigma = sigma
        self.na, self.nx = self.K.shape
        self.nd = self.K.size + self.m.size + 1

    @classmethod
    def zero(cls, nx, na):
        """
        Class method to create a new policy with all parameters set to zero.
        This is mostly useful for creating an empty policy and then adding to
        the parameter vector using `update()`.
        """
        return cls(np.zeros((na,nx)), np.zeros(na), 0.0, copy=False)

    def sample(self, x):
        return self.m + np.dot(x, self.K.T) + np.random.normal(size=x.shape, scale=self.sigma)

    def dlogpi(self, x, u):
        """
        Evaluate the immediate policy-gradient on n state/action pairs,
        collected in the vectors `x` and `u`. These should be arrays of size
        (n,nx) and (n,nu) respectively.
        """
        n = x.shape[0]
        A = u - np.dot(x, self.K.T) - self.m.reshape(1,-1)
        dK = self.sigma**(-2) * oprod(A, x).reshape(n,-1)
        dm = self.sigma**(-2) * A
        ds = self.sigma**(-3) * sum(A**2, axis=1).reshape(n,-1) - self.na/self.sigma
        return np.c_[dK, dm, ds]

    def copy(self):
        """Return a copy of the policy object."""
        return MoGPolicy(self.K, self.m, self.sigma)

    def pack(self):
        """Return the policy parameters as a vector."""
        return np.r_[self.K.flatten(), self.m.flatten(), self.sigma]

    def update(self, dtheta):
        """Update the policy in-place given a vector in parameter space."""
        ell = self.nx * self.na
        self.K += dtheta[0:ell].reshape(self.na, self.nx)
        self.m += dtheta[ell:-1]
        self.sigma += dtheta[-1]

class ZModel(object):
    """
    Given a mixture-of-Gaussians MDP model and a policy object we can construct
    a linear-Gaussian model in the joint state/action space. This class acts a
    struct containing this model with initial states parameterized by
    (mu0, Sigma0) and transitions (F, m, Sigma).
    """
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
# Code to compute the moments, i.e. E[Z], E[ZZ'] and E[C'C].
#===================================================================================================

def kalman_predict(zmodel, (mu, Sigma)):
    """
    Given a state/action space transition model and a Gaussian in this space
    parameterized by (mu, Sigma) return an updated Gaussian corresponding to
    the prediction step of the Kalman filter.
    """
    mu = np.dot(zmodel.F, mu) + zmodel.m
    Sigma = np.dot(np.dot(zmodel.F, Sigma), zmodel.F.T) + zmodel.Sigma
    return (mu, Sigma)

def kalman_update(model, gamma, (mu_fwd, Sigma_fwd)):
    """
    Given the reward model and a Gaussian in the state/action space return
    (mu, Sigma, c) corresponding to the update step of the Kalman filter. Here
    c corresponds to the "likelihood" of the observation, i.e. the expected
    reward.
    """
    d = model.nx + model.na
    mu, Omega, c = 0, 0, 0

    for (y,M,L,logDetL) in zip(model.y, model.M, model.L, model.logDetL):
        # get the innovation r, i.e. the difference between the "observed" y
        # and the prediction; the cholesky of the innovation covariance cholS;
        # and the log-determinant of this covariance.
        r = y - np.dot(M, mu_fwd)
        cholS, _ = sp.linalg.cho_factor(np.dot(M, np.dot(Sigma_fwd, M.T)) + L, overwrite_a=True)
        logDetS = 2*np.sum(np.log(np.diag(cholS)))

        # get the Kalman gain and get the updated mean/variance.
        K = sp.linalg.cho_solve((cholS, False), np.dot(M, Sigma_fwd), overwrite_b=True).T
        mu_ = mu_fwd + np.dot(K, r)
        Omega_ = np.dot(np.eye(d) - np.dot(K, M), Sigma_fwd) + np.outer(mu_, mu_)

        # this is the "likelihood" term, which corresponds to the reward. since
        # we're using unnormalized Gaussians we don't include the 2*pi term, and we
        # subtract off the normalizing constant involving L.
        c_ = np.sum(sp.linalg.solve_triangular(cholS, r, trans=1)**2) + logDetS - logDetL
        c_ = gamma * np.exp(-0.5*c_)

        mu += c_ * mu_
        Omega += c_ * Omega_
        c += c_

    return mu, Omega, c

def kalman_smooth(model, zmodel, J, mu, Omega, gamma, (mu_fwd, Sigma_fwd), init):
    """
    This performs Kalman smoothing for the mixture-of-Gaussians MDP. Returns a
    tuple (mu, Omega) corresponding to the smoothed sum of moments.
    """
    # get the first/second moments at the current horizon.
    mu_til, Omega_til, c = kalman_update(model, gamma, (mu_fwd, Sigma_fwd))

    if init:
        mu, Omega = mu_til, Omega_til

    else:
        # get the components in order to do the smoothing step.
        tmp = np.dot(zmodel.F, Sigma_fwd)
        P = np.dot(tmp, zmodel.F.T) + zmodel.Sigma
        G = sp.linalg.solve(P, tmp, sym_pos=True, overwrite_b=True).T

        # do the smoothing for the summation of the first moment.
        mu_rev = mu_fwd - np.dot(G, np.dot(zmodel.F, mu_fwd) + zmodel.m)
        Gmu = np.dot(G, mu)
        mu = mu_til + J*mu_rev + Gmu

        # do the smoothing for the summation of the second moment, for which we
        # get the covariance for free.
        a = np.outer(Gmu, mu_rev)
        Omega_rev = np.outer(mu_rev, mu_rev) + Sigma_fwd - np.dot(G, np.dot(P, G.T))
        Omega = Omega_til + J*Omega_rev + np.dot(G, np.dot(Omega, G.T)) + a + a.T

    return mu, Omega, c

def get_zmoments(model, policy, gamma, H):
    """
    Given a model, policy, discount factor gamma, and horizon H, return the
    necessary moments in the joint state/action space. This will return
    (J, Js, Z, ZZ) corresponding to the expected reward, sum of expected rewards
    at each time-horizon, and the first/second moments.
    """
    # initialize the z-space transition model and grab the forward messages.
    zmodel = ZModel(model, policy)
    forward = [(zmodel.mu0, zmodel.Sigma0)]
    for k in xrange(H):
        forward.append(kalman_predict(zmodel, forward[k]))

    init = True
    J, Js, Z, ZZ = 0, 0, 0, 0
    mu, Omega = None, None

    for n in reversed(xrange(H+1)):
        try:
            mu, Omega, c = kalman_smooth(model, zmodel, J, mu, Omega, gamma**n, forward[n], init)
            init = False

        except np.linalg.LinAlgError:
            init = True
            J, Js, Z, ZZ = 0, 0, 0, 0
            continue

        # the sum of all components from n to the horizon H.
        J += c; Js += J; Z += mu; ZZ += Omega

    return J, Js, Z, ZZ

def get_moments(model, policy, gamma, H):
    """
    Get the necessary moments in the individual state and action spaces.
    Returns a tuple (J, Js, X, U, XX, UU, UX, CC) corresponding the the
    expected reward, the sum of expected rewards at each time-horizon, and the
    necessary moments.
    """
    # get the moments we're actually interested in.
    nx = policy.nx
    J, Js, Z, ZZ = get_zmoments(model, policy, gamma, H)
    X, U = Z[0:nx], Z[nx:]
    XX, UU, UX = ZZ[0:nx,0:nx], ZZ[nx:,nx:], ZZ[nx:,0:nx]

    # and get the expectation of the inner product of C with itself, where C is
    # the action minus the "predicted-action".
    M = np.c_[-policy.K, np.eye(model.na)]
    A = np.r_[np.c_[np.inner(policy.K, policy.K), -policy.K.T], M]
    CC = np.trace(np.dot(A, ZZ)) + Js*np.inner(policy.m, policy.m) - 2*np.inner(policy.m, np.dot(M, Z))

    return J, Js, X, U, XX, UU, UX, CC


#===================================================================================================
# Code to solve MoGMDP problem using either the plain EM algorithm or the
# better performing gradient-based algorithms. This includes functions to
# compute the gradients/steps, all of which are based on the moment
# calculations from the previous section.
#===================================================================================================

def get_jtheta(model, policy, gamma, H):
    """
    Get the expected reward of the model and policy using discount factor gamma
    and time-horizon H.
    """
    zmodel = ZModel(model, policy)
    forward = (zmodel.mu0, zmodel.Sigma0)
    J = 0.0
    for n in xrange(H):
        _, _, c = kalman_update(model, gamma**n, forward)
        J += c
        forward = kalman_predict(zmodel, forward)
    return J

def get_gradient(model, policy, gamma, H):
    """
    Evaluate the gradient of the expected reward under some given policy and
    return a tuple (J, g) containing the reward and this gradient.
    """
    # get the moments we're interested in.
    J, Js, X, U, XX, UU, UX, CC = get_moments(model, policy, gamma, H)

    dK = (policy.sigma**-2) * (UX - np.dot(policy.K, XX) - np.outer(policy.m, X))
    dm = (policy.sigma**-2) * (U - np.dot(policy.K, X) - Js*policy.m)
    ds = (policy.sigma**-1) * (CC*policy.sigma**-2 - Js*model.na)

    return J, np.r_[dK.flatten(), dm.flatten(), ds]

def get_emstep(model, policy, gamma, H):
    """
    Evaluate the next iteration of the EM algorithm given some policy and
    return a tuple (J, d) corresponding to the expected reward along with the
    step taken by EM.
    """
    # get the moments we're interested in.
    J, Js, X, U, XX, UU, UX, CC = get_moments(model, policy, gamma, H)

    K = scipy.linalg.solve(XX, UX - np.outer(policy.m, X), sym_pos=True, overwrite_b=True)
    m = (U - np.dot(policy.K, X)) / Js
    s = np.sqrt(CC / Js / model.na)

    return J, np.r_[(K-policy.K).flatten(), (m-policy.m).flatten(), s-policy.sigma]

def solve_mogmdp_em(model, policy, gamma, H, maxfun=100):
    """
    Solve the model using EM using some initial policy, discount factor gamma,
    and a time-horizon of H. Returns a tuple (theta, jtheta, info) where info
    is a dictionary containing extra information (i.e. per-iteration, etc.).
    """
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

    return xs[-1], fs[-1], dict(numevals=ns, theta=xs, jtheta=fs)

def solve_mogmdp(model, policy, gamma, H, sigma_min=0.1, em=False):
    """
    Solve the model using gradient-based methods using some initial policy,
    discount factor gamma, and a time-horizon of H. Returns a tuple (theta,
    jtheta, info) where info is a dictionary containing extra information (i.e.
    per-iteration, etc.).

    If `em` is True this will use a pseudo-gradient given by the EM steps,
    otherwise it will directly optimize using the gradient returned by EM.
    """
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

    return theta, -J, info
