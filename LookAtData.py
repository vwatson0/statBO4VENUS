# coding = utf8!


import numpy as np
import datetime
import time
import statistics
import matplotlib.pyplot as plt


#from bayes_opt import BayesianOptimization, UtilityFunction
#from bayes_opt.logger import JSONLogger
#from bayes_opt.event import Events
#from bayes_opt.util import load_logs

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy as sp
from scipy.stats import norm
data = np.loadtxt('nextpoint_1750201244')

t = data[:, 0]
p = data[:, 1]
v = data[:, 2]
I = data[:, 3]
sig = data[:, 4]

scaleV = [25, 75]
scaleP = [8.6, 9.6]

Niter = 10

def InvExpectedImp(Param, gpr, nu, Ymax) :
    mean, std = gpr.predict([Param], return_std=True)
    a = mean - Ymax - nu
    z = a / std
    return -(a * norm.cdf(z) + std * norm.pdf(z))  # Bug fix 1 bad inverse of the function


# Gathering initial data (feed with random eval or something clever)
Xdata = data[0:Niter, 1:3] # normalize !!!!!!!!!!!!!!!!!!!!!!!!!!
Xdata[:, 0] = (Xdata[:, 0] - scaleP[0])/(scaleP[1]-scaleP[0])
Xdata[:, 1] = (Xdata[:, 1] - scaleV[0])/(scaleV[1]-scaleV[0])

normP = (p - scaleP[0])/(scaleP[1]-scaleP[0])
normV = (v - scaleV[0])/(scaleV[1]-scaleV[0])

Ydata = data[0:Niter, 3]
# initializing kernel
Klen = np.ones(2) * .1

kernel = Matern(length_scale=Klen, nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
gpr.fit(Xdata, Ydata)

#Create grid and look at Acq + pin next query

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros(np.shape(X))
Z2 = np.zeros(np.shape(X))
for k in range(len(X)):
    for l in range(len(X[0])):
        Z[k, l] = InvExpectedImp([X[k, l], Y[k, l]], gpr, .5, max(Ydata))
    Z2[k] = gpr.predict(np.asarray([X[k], Y[k]]).T)
CntIterRndSearch = 0
MaxIterForAcqOptInit = 1000
RunRndSearch = True
rndSearch = []
resSearch = []
while RunRndSearch:
    InitParam = np.random.rand(2) # random generation in normalized parameter space
    if abs(InvExpectedImp(InitParam, gpr, .5, max(Ydata))) > 1e-100: # bug 3 fix value close to 0 -> fail
        RunRndSearch = False
        rndSearch.append(InitParam)
        resSearch.append(InvExpectedImp(InitParam, gpr, .5, max(Ydata)))
    else:
        CntIterRndSearch += 1
        rndSearch.append(InitParam)
        resSearch.append(0)
    if CntIterRndSearch >= MaxIterForAcqOptInit:
        RunRndSearch = False
        print('Failed to find proper initialization for optimizing the acquisition function')
    # Modification 5 June 2025 end | vwatson ###########################################################

    res = sp.optimize.minimize(InvExpectedImp, InitParam, args=(gpr, .5, max(Ydata)), bounds = {(0, 1.),(0, 1.)}, method = 'Nelder-Mead') # bug 2 fix reg grad failed
    nextX = res.x #

print(rndSearch)
print(resSearch)
print('next suggested point', nextX)
print('Nextpoint normalized coord: ', normP[Niter], normV[Niter])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(nextX[0], nextX[1], InvExpectedImp(nextX, gpr, .5, max(Ydata)), color = 'green')
ax.scatter(normP[Niter], normV[Niter], InvExpectedImp(np.asarray([normP[Niter], normV[Niter]]), gpr, .5, max(Ydata)), color = 'red')
ax.set_xlabel('Pressure N')
ax.set_ylabel('Voltage N')
ax.set_zlabel('AcQ')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, Z2)
ax2.scatter(Xdata[:, 0], Xdata[:, 1], Ydata, color = 'red')
ax2.set_xlabel('Pressure N')
ax2.set_ylabel('Voltage N')
ax2.set_zlabel('Ib')




plt.show()



