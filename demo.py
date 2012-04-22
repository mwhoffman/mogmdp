from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
import mogmdp

params = {}
params['mu0'] = -5.0
params['Sigma0'] = 0.5
params['A'] = 1.0
params['B'] = 1.0
params['Sigma'] = 0.3
params['w'] = 1.0
params['y'] = [3.,0.]
params['M'] = 3*np.eye(2)
params['L'] = 2*np.eye(2)

H = 40
gamma = 0.95
sigma_min = 0.1

model = mogmdp.MoGMDP(**params)

# construct the contours.
X, Y = np.meshgrid(np.linspace(-2,0,20), np.linspace(-.5,3,20))
J = np.empty(X.shape)
for i, (K, m) in enumerate(zip(X.flat, Y.flat)):
	J.flat[i] = mogmdp.get_jtheta(model, model.unpack_policy([K, m, sigma_min]), gamma, H)

pl.contour(X, Y, J)

