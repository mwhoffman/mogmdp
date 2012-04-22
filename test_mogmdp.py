import numpy as np
import scipy as sp
import scipy.linalg
import itertools

import mogmdp

def get_moment(model, policy, gamma, H, tau):
    forward, zmodel = mogmdp.get_forward(model, policy, H)

    # get the "final" component.
    mu_fwd, Sigma_fwd = forward[H]
    mu_hat, Sigma_hat, c = mogmdp.kalman_update(model, mu_fwd, Sigma_fwd)

    for n in itertools.islice(reversed(xrange(H)), tau):
        mu_fwd, Sigma_fwd = forward[n]

        # get the components in order to do the smoothing step.
        tmp = np.dot(zmodel.F, Sigma_fwd)
        P = np.dot(tmp, zmodel.F.T) + zmodel.Sigma
        G = sp.linalg.solve(P, tmp, sym_pos=True, overwrite_b=True).T

        mu_hat = mu_fwd + np.dot(G, mu_hat) - np.dot(G, np.dot(zmodel.F, mu_fwd) + zmodel.m)
        Sigma_hat = Sigma_fwd + np.dot(G, np.dot(Sigma_hat - P, G.T))

    return mu_hat, Sigma_hat, c

params = {}
params['mu0'] = -5.0
params['Sigma0'] = 0.5
params['A'] = 1.0
params['B'] = 1.0
params['Sigma'] = 0.3
params['y'] = [3,0]
params['M'] = 3*np.eye(2)
params['L'] = 2*np.eye(2)
params['w'] = 1.0

model = mogmdp.MoGMDP(**params)
policy = mogmdp.MoGPolicy(-0.8, 0.0, 1.0)

H = 20
gamma = 0.95
J, Js, Z, ZZ = 0, 0, 0, 0

d = model.nx
M = np.c_[-policy.K, np.eye(model.na)]
A = np.r_[np.c_[np.inner(policy.K, policy.K), -policy.K.T], M]

for n in xrange(H+1):
    for k in xrange(n, H+1):
        mu_hat, Sigma_hat, c = get_moment(model, policy, 0.95, k, k-n)
        c *= gamma**k
        z, zz = mu_hat, (Sigma_hat + np.outer(mu_hat, mu_hat))

        Z += c*z
        ZZ += c*zz
        Js += c
        J += c if (k==n) else 0

for (a,b) in zip((J, Js, Z, ZZ), mogmdp.get_moments(model, policy, gamma, H)):
    assert np.allclose(a, b)
