#!/bin/env python3

import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime
import time
import statistics
import telnetlib
import os

#from bayes_opt import BayesianOptimization, UtilityFunction
#from bayes_opt.logger import JSONLogger
#from bayes_opt.event import Events
#from bayes_opt.util import load_logs

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy as sp
from scipy.stats import norm


import warnings

warnings.filterwarnings("ignore")

import venus_data_utils.venusplc as venusplc
        
venus = venusplc.VENUSController(read_only=False)

############################################
# set safe parameters.  Must be done each time!
#############################################
runsafe = 0  # set to 1 for testing, 0 for actual running
pressurewatch = 6.6e-8
pressurestop = 6.3e-8
toomanylowpressure = 12

############## stuff to set up faster Ammeter
measurementFrequency = 1000

def sendCommand(connection,command):
    connection.write((command+'\n').encode('ascii'))

def setupSystem(verbose=0):
    IP = "10.10.100.75"
    port = 5024

    # Connect to Ammeter
    if verbose:   print('attempt to connect...')
    connection = telnetlib.Telnet(IP,port,timeout = 3)
    output = connection.read_until(b'\n')
    if verbose:   print('connected.  Output: ',output,'\nResetting system')
    
    # Reset System
    sendCommand(connection,"*rst")
    if verbose:   print('reset')
    time.sleep(2)
    if verbose:   print('waited 2 seconds, setting up current reading')

    # Setting up reading settings
    sendCommand(connection,':sens:func "curr"')
    sendCommand(connection,':sens:curr:rang:auto on')
    sendCommand(connection,':sens:curr:nplc:auto off')

    # Set integration time in terms of wall frequency: MeasTime*^60Hz
    nplc = 1./measurementFrequency*60.0   
    sendCommand(connection,':sens:curr:nplc '+str(nplc))   

    # turn on input switch
    sendCommand(connection,':inp on')

    return connection

def getCurrent(connection):
    sendCommand(connection,":meas:curr?")
    return float(connection.read_until(b'\n').decode("ascii")[-15:-2])

connection = setupSystem(verbose=0)

################ done setting up faster Ammeter

def savesource(injpressure, numlowpressure):
    if runsafe:
        print('savesource: set bias to 15, 18 to 500, 28 to 3000')
    else:
        venus.write({'bias_v' : 15.0})
    f=open('sourcesaves', 'a')
    f.write('%d %.3e %d\n' % (time.time(), injpressure, numlowpressure))
    f.close()

def checkpressure(numlowpressure):
    # returns whether killed off and numlowpressure
    injpressure = venus.read(['inj_mbar'])
    if injpressure<pressurewatch:
        numlowpressure = numlowpressure + 1
        if injpressure < pressurestop:
            time.sleep(0.37)
            injpressure = venus.read(['inj_mbar'])
            if injpressure < pressurestop:   # make a catch to give two consecutive fails be the limit
                savesource(injpressure,numlowpressure)
                numlowpressure = 0
                return(1,0)
        if numlowpressure > toomanylowpressure:
            savesource(injpressure,numlowpressure)
            numlowpressure = 0
            return(1,0)
        return(0,numlowpressure)
     
    else:
        return(0,0)

def writeoutput(outfile):
    outfile.write("%.3f %.3f\n" % (time.time(),getCurrent(connection)*1e6))

def setbalzer(balzernum,bvalue,lowaim,lowok):
    if runsafe:
        print('setting balzer'+balzernum+' to '+bvalue)
    else:
        balzername = 'gas_balzer_'+str(int(balzernum))
        venus.write({balzername:lowaim})
        while(venus.read({balzername})>lowok):
            time.sleep(0.4)
        venus.write({balzername:bvalue})
        time.sleep(4)
    return(bvalue)

readvars = venus.read_vars()

squared = lambda x: 0.5*(20*x)**2
BEAM_CURR_STD = 30

tstart = time.time()
fname = 'data/bayes_'+str(int(tstart))
outfile = open(fname,'w')
#logger = JSONLogger(path="logs/log_"+str(int(tstart))+".log")

countblackboxes = 0

#def black_box_function(balzer1N,balzer5N,biasN,p18N):
def black_box_function(settingsN):
    # settings 0:balzer1N, 1:biasN
    global countblackboxes
    numlowpressure = 0
    killoff = 0

    [balzer1,bias]=setnumbers(settingsN[0],settingsN[1])
    if runsafe: print('from setnumbers: ',[balzer1,bias])
    time_bbox = time.time()
    twait = 0.37

    #print(f'setting balzer to {balzer1:.2f} and bias to {bias:.2f}')
    balzer1val = setbalzer(1,balzer1,8.4,8.5)
    venus.write({"bias_v":bias})
        
    # changes done...wait for ~180 seconds (250 measurements)
    #   now wait 410 = 295 seconds
    #for j in range(800):  DST
    for j in range(100):
        writeoutput(outfile)
        time.sleep(twait)
        killoff,numlowpressure = checkpressure(numlowpressure)
        if killoff:
            print('in killoff')
            break
    
    tmeasstart = time.time()
    v_list = []
    if killoff == 0:
        for j in range(30):
            v_list.append(getCurrent(connection)*1e6)
    if killoff:
        v_list = []
        for j in range(30):
            v_list.append(0.2)
    v_mean = sum(v_list) / (1.*len(v_list))
    #if v_mean
    v_std = statistics.stdev(v_list)
    rel_std = v_std / v_mean
    print(f'{balzer1:5.2f} + {bias:6.2f}: {v_mean:7.2f} {rel_std*100:6.2f}%')
    g = open('data/nextpoint_'+str(int(tstart)),'a')
    g.write('%i '%(time_bbox))
    g.write('%6.2f %6.2f '%(balzer1,bias))
    g.write('%7.2f %7.2f %7.2f\n'%(v_mean,v_std,rel_std*100))
    g.close()
    #instability_cost = squared(rel_std)*BEAM_CURR_STD
    instability_cost = squared(rel_std)*BEAM_CURR_STD
    ffff = open('data/info_'+str(int(tstart)),'a')
    a=ffff.write(f'bbox {countblackboxes:d} after 30 measurements: average={v_mean:.2f} std = {v_std:.2f}, dt[min] = {(time.time()-time_bbox)/60.0}\n')
    ffff.close()
    countblackboxes = countblackboxes+1
    tst_str = str(int(time.time()))

    #return v_mean - instability_cost
    if rel_std< .05:
        return v_mean
    else:
        return .10
        #return v_mean * (1 - instability_cost/10.)


def denormalize(scalenum):
    return scalenum[0]+scalenum[2]*(scalenum[1]-scalenum[0])

def setnumbers(balzer1N,biasN):
    scale_balzer1 = [8.6,9.6,balzer1N]
    scale_bias = [25,75,biasN]
    scales = [scale_balzer1, scale_bias]
    values = []
    for scale in scales:
        values.append(denormalize(scale))
    return values



######## Modifications VWatson May21 25 - Tune anisotropic kernel length and Exploration/Exploitation bias



def InvExpectedImp(Param, gpr, nu, Ymax):

    mean, std = gpr.predict([Param], return_std = True)
    a = mean - Ymax - nu
    z = a / std
    return -(a * norm.cdf(z) + std * norm.pdf(z))


# Gathering initial data (feed with random eval or something clever)
Xdata = np.array([[ 0.2, 0.7],
                  [ 0.6, 0.5],
                  [ 0.2, 0.4],
                  [ 1.0, 0.8]])

# len(Xdata) blackboxcalls to get Ydata
ninit = len(Xdata)
Ydata = np.zeros(ninit)
for i in range(ninit):
    Ydata[i] = black_box_function(np.array([Xdata[i,0],Xdata[i,1]]))
XdataSpaceDimension = 2
maxIter = 30   # DST vary this
ExpBias = .5
# initializing kernel
Klen = np.ones(XdataSpaceDimension) * .1

Cov = np.zeros([maxIter,XdataSpaceDimension])

kname = 'data/klen_'+str(int(tstart))
kk = open(kname,'w')

for cptr in range(maxIter):

    # Setting up Anisotropic Kernel function
    kernel = Matern(length_scale=Klen, nu=2.5)
    # Initializing regressor
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    # Fitting the data
    gpr.fit(Xdata, Ydata)

    # Initializing the standard optimizer for Acq function
    # Modification 5 June 2025 start | vwatson ###########################################################
    MaxIterForAcqOptInit = 1000 # number max of iterations to ind an area where the acquisition function is computable
    CntIterRndSearch = 0 # way out when impossible to find a solution

    Ymax = np.max(Ydata)
    
    RunRndSearch = True
    while RunRndSearch:
        InitParam = np.random.rand(XdataSpaceDimension) # random generation in normalized parameter space
        if abs(InvExpectedImp(InitParam, gpr, ExpBias, Ymax)) > 1E-100 :
            RunRndSearch = False
        else:
            CntIterRndSearch += 1
        if CntIterRndSearch >= MaxIterForAcqOptInit:
            RunRndSearch = False
            print('Failed to find proper initialization for optimizing the acquisition function')
    # Modification 5 June 2025 end | vwatson ###########################################################

    res = sp.optimize.minimize(InvExpectedImp, InitParam, args=(gpr, ExpBias, Ymax), bounds = {(0,1.),(0,1.)},method = 'Nelder-Mead')
    nextX = res.x
    nextY = black_box_function(np.array([nextX[0],nextX[1]]))

    # adding the new point to the dataset
    Xdata = np.concatenate((Xdata,[nextX]))
    Ydata = np.concatenate((Ydata,[nextY]))

    # randomly modifying the Kernel length along dimensions where Cov is too small

    for kV in range(len(Cov[cptr])) :
        Cov[cptr, kV] = np.abs(np.cov(Xdata[:, kV], Ydata)[0, 1])  # need to check data orientation
        if cptr > 20 :
            if np.polyfit(Cov[cptr - 20 : cptr, kV], np.arange(20), 1)[0] < (60./1.)/100. :  
                # Tresh = (expectedYmax/DeltaX)/100 Empirically
                Klen[kV] = np.random.rand() * .30 + .01

        # tests
        if cptr > maxIter * .7 :  # can be adjusted
            ExpBias = -5
    for i in range(len(Klen)):
        kk.write("%e " % (Klen[i]))
    kk.write("\n")

with open('data/cov_'+str(int(tstart)),'w') as cc:
    shpe = np.shape(Cov)
    for j in range(shpe[1]):
        for i in range(shpe[0]):
            cc.write("%e " % (Cov[i,j]))
        cc.write("\n")


outfile.close()
kk.close()
