from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
import mogmdp
import lbfgsb

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
theta0 = [-1.5, 0.1, 1.0]

model = mogmdp.MoGMDP(**params)

# construct the contours.
X, Y = np.meshgrid(np.linspace(-2,0,20), np.linspace(-.5,3,20))
J = np.empty(X.shape)
for i, (K, m) in enumerate(zip(X.flat, Y.flat)):
    J.flat[i] = mogmdp.get_jtheta(model, model.unpack_policy([K, m, sigma_min]), gamma, H)

# get the objective function.
fun = lambda theta: mogmdp.get_gradient(model, model.unpack_policy(theta), gamma, H)
obj = lambda theta: map(np.negative, fun(theta))

bounds = [(None,None) for i in xrange(len(theta0)-1)] + [(sigma_min,None)]
_, _, info = lbfgsb.lbfgsb(obj, theta0, bounds)

pl.contour(X, Y, J)
pl.plot(info['x'][:,0], info['x'][:,1], 'r-', lw=2)
