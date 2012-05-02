from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import mogmdp

# construct the model.
params = {}
params['mu0'] = -5.0
params['Sigma0'] = 0.5
params['A'] = 1.0
params['B'] = 1.0
params['Sigma'] = 0.3
params['w'] = [1.0, -.5]
params['y'] = [[3.,0.], [2.,0.]]
params['M'] = [3*np.eye(2), np.eye(2)]
params['L'] = [2*np.eye(2), np.eye(2)]

# construct parameters under which we'll optimize the model.
H = 50
gamma = 0.95
sigma_min = 0.001
theta = [-.8, -0.0, 1.0]

# create the actual model/policy objects.
model = mogmdp.MoGMDP(**params)
policy = model.unpack_policy(theta)

# construct the contours.
X, Y = np.meshgrid(np.linspace(-2,0,20), np.linspace(-.5,3,20))
J = np.empty(X.shape)
for i, (K, m) in enumerate(zip(X.flat, Y.flat)):
    J.flat[i] = mogmdp.get_jtheta(model, model.unpack_policy([K, m, sigma_min]), gamma, H)

# solve the problem using a few different methods.
_, _, info_gem = mogmdp.solve_mogmdp(model, policy, gamma, H, sigma_min)
_, _, info_pem = mogmdp.solve_mogmdp(model, policy, gamma, H, sigma_min, em=True)
_, _, info_em  = mogmdp.solve_mogmdp_em(model, policy, gamma, H, maxfun=20)

titles = ['LBFGS', 'LBFGS-EM', 'vanilla EM']
infos = [info_gem, info_pem, info_em]

# plot the paths taken.
pl.figure(figsize=(12,4))
for i, (title, info) in enumerate(zip(titles, infos)):
    pl.subplot(1,3,i)
    pl.contour(X, Y, J)
    pl.plot(info['theta'][:,0], info['theta'][:,1], 'r-', lw=2)
    pl.title(title)

# plot the performance results.
pl.figure()
for title, info in zip(titles, infos):
    pl.plot(info['numevals'], info['jtheta'], lw=2, label=title)
pl.legend(loc='lower right')
pl.xlabel('function evaluations')
pl.ylabel('expected reward')
pl.title('performance of EM-based optimizers')

