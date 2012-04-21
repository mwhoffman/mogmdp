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
J, Z, ZZ = 0, 0, 0

for n in xrange(H+1):
    for k in xrange(n, H+1):
        c, mu, Sigma = mogmdp.get_moment(model, policy, 0.95, k, k-n)
        c *= gamma**k
        Z += c*mu
        ZZ += c*(Sigma + np.outer(mu, mu))
        J += c if (k==n) else 0

for (a,b) in zip((J, Z, ZZ), mogmdp.get_moments(model, policy, gamma, H)):
    assert np.allclose(a, b)