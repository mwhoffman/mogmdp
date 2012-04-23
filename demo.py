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

_, _, info_g = mogmdp.solve_mogmdp(model, policy0, gamma, H)
_, _, info_e = mogmdp.solve_mogmdp(model, policy0, gamma, H, em=True)
_, _, info_m = mogmdp.solve_mogmdp_em(model, policy0, gamma, H)

pl.figure()
pl.contour(X, Y, J)
pl.plot(info_g['theta'][:,0], info_g['theta'][:,1], 'r-', lw=2)
pl.title('LBFGS')

pl.figure()
pl.contour(X, Y, J)
pl.plot(info_e['theta'][:,0], info_e['theta'][:,1], 'r-', lw=2)
pl.title('LBFGS-EM')

pl.figure()
pl.contour(X, Y, J)
pl.plot(info_m['theta'][:,0], info_m['theta'][:,1], 'r-', lw=2)
pl.title('EM')

pl.figure()
pl.plot(info_g['numevals'], info_g['jtheta'], lw=2, label='LBFGS')
pl.plot(info_e['numevals'], info_e['jtheta'], lw=2, label='LBFGS-EM')
pl.plot(info_m['numevals'], info_m['jtheta'], lw=2, label='EM')
pl.legend(loc='lower right')

