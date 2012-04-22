from __future__ import division

import numpy as np
import mogmdp

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
trans = mogmdp.ZTransition(model, policy)

forward = [mogmdp.ForwardMessage.init(model, policy)]
for k in xrange(20):
    forward.append(forward[-1].get_next(trans))

H = 20
gamma = 0.95
J, Js, Z, ZZ = 0, 0, 0, 0

d = model.nx
M = np.c_[-policy.K, np.eye(model.na)]
A = np.r_[np.c_[np.inner(policy.K, policy.K), -policy.K.T], M]

for n in xrange(H+1):
    for k in xrange(n, H+1):
        c, mu_hat, Sigma_hat = mogmdp.get_moment(model, policy, 0.95, k, k-n)
        c *= gamma**k
        z, zz = mu_hat, (Sigma_hat + np.outer(mu_hat, mu_hat))

        Z += c*z
        ZZ += c*zz
        Js += c
        J += c if (k==n) else 0

for (a,b) in zip((J, Js, Z, ZZ), mogmdp.get_moments(model, policy, gamma, H)):
    assert np.allclose(a, b)