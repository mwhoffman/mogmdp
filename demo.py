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

H = 20
theta = [-0.8, 0.2, 1.0]
gamma = 0.95

model = mogmdp.MoGMDP(**params)
gradient = mogmdp.get_gradient(model, model.unpack_policy(theta), gamma, H)

print gradient