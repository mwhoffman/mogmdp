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

H = 50
gamma = 0.95
sigma_min = 0.1
theta0 = [-1.8, -0.3, 1.0]

model = mogmdp.MoGMDP(**params)
policy0 = model.unpack_policy(theta0)

# construct the contours.
X, Y = np.meshgrid(np.linspace(-2,0,20), np.linspace(-.5,3,20))
J = np.empty(X.shape)
for i, (K, m) in enumerate(zip(X.flat, Y.flat)):
    J.flat[i] = mogmdp.get_jtheta(model, model.unpack_policy([K, m, sigma_min]), gamma, H)

_, _, info_gem = mogmdp.solve_mogmdp(model, policy0, gamma, H)
_, _, info_pem = mogmdp.solve_mogmdp(model, policy0, gamma, H, em=True)
_, _, info_em  = mogmdp.solve_mogmdp_em(model, policy0, gamma, H, maxfun=50)

pl.figure()
pl.subplot(2,2,1)
pl.contour(X, Y, J)
pl.plot(info_gem['theta'][:,0], info_gem['theta'][:,1], 'r-', lw=2)
pl.title('LBFGS')

pl.subplot(2,2,2)
pl.contour(X, Y, J)
pl.plot(info_pem['theta'][:,0], info_pem['theta'][:,1], 'r-', lw=2)
pl.title('LBFGS-EM')

pl.figure()
pl.subplot(2,2,3)
pl.contour(X, Y, J)
pl.plot(info_em['theta'][:,0], info_em['theta'][:,1], 'r-', lw=2)
pl.title('EM')

pl.figure()
pl.plot(info_gem['numevals'], info_gem['jtheta'], lw=2, label='LBFGS')
pl.plot(info_pem['numevals'], info_pem['jtheta'], lw=2, label='LBFGS-EM')
pl.plot(info_em ['numevals'], info_em ['jtheta'], lw=2, label='vanilla EM')
pl.legend(loc='lower right')
pl.xlabel('function evaluations')
pl.ylabel('expected reward')
pl.title('performance of EM-based optimizers')

