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

with open('alpha','r') as f:
    alphaval = float(f.readline())
with open('alpha','w') as f:
    f.write('%e'%(alphaval/4.0))


############################################
# set safe parameters.  Must be done each time!
#############################################
runsafe = 0  # set to 1 for testing, 0 for actual running
pressurewatch = 6.6e-8
pressurestop = 6.3e-8
toomanylowpressure = 12

############## stuff to set up faster Ammeter
measurementFrequency = 1000


class KFparAlt :
    def __init__(self, cLin=10E-3, cMeas=10E+3) :
        ''' Initialize:
        Create object with values for R and Q.
        X[0] with the first measure
        '''
        self.X = np.zeros(2)  # X[0] contains the current estimate of the Beam mean X[1] is the estimate of the slope
        self.PX = np.zeros([2, 2])  # Covariance Matrix of X (Calculated by the filter)
        self.Sig = np.zeros(2)  # Sig[0] contains the current estimate of Beam std
        self.PS = np.zeros([2, 2])  # Covariance Matrix of Sig (Calculated by the filter)
        self.Q = np.ones(
            [2, 2]) * cLin  # relative confidence in the linear dynamic if increased less noise slower convergence
        self.R = np.array(
            [[cMeas]])  # relative confidence in the measurement if increased faster convergence more noise
        self.F = np.array([[1., 1.], [0, 1.]])  # self dynamic of the system
        # [[link measure to state estimate, link fist order to state estimate],[link state to 1st order, propagate 1st order]]
        self.H = np.array([1.,
                           0])  # link from the state space to the measures space (here the transformation from the measure to X[0] is 1)
        # and we do not measure directly dx/dt but deduce it with F

    def EstimateState(self, measure, deltaT) :
        # extracting current values from the filter object

        PoldX = self.PX
        PoldSig = self.PS
        oldx = self.X
        oldSig = self.Sig

        F = self.F
        Q = self.Q
        R = self.R
        H = self.H
        F[0, 1] = deltaT  # updating F

        # predictions
        xPred = np.dot(F,
                       oldx)  # predicting the state xPred[k,0] = xEst[k-1,0] + (dx/dt)_estimate | xPred[k,1] = (dx/dt)_est
        pPred = np.dot(np.dot(F, PoldX),
                       F.T) + Q  # Covariance matrix of the prediction (the bigger, the less confident we are)

        SigPred = np.dot(F, oldSig)  # Same thing but with standard deviation of the beam current
        SigpPred = np.dot(np.dot(F, PoldSig), F.T) + Q

        Inow = measure

        # updates

        y = Inow - np.dot(H,
                          xPred)  # Calculating the innovation (diff between measure and prediction in the measure space)
        S = np.dot(np.dot(H, pPred),
                   H.T) + R  # Calculating the Covariance of the measure (the bigger the less confident in the measure)

        K = np.dot(np.array([np.dot(PoldX, H.T)]).T, np.linalg.inv(S))  # Setting the Kalman optimal gain
        newX = xPred + np.dot(K, np.atleast_1d(y)).T  # Estimating the state at this instant
        PnewX = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), pPred)  # Covariance matrix of the state

        # same steps followed for the standard deviation
        y = np.sqrt((Inow - newX[0]) ** 2) - np.dot(H, SigPred)  # Innovation of the standard deviation
        # this is an additional drawer to the Kalman filter and it is rather uncommon to estimate another variable that is
        # statistically dependent there may be better solutions, but this one works
        S = np.dot(np.dot(H, SigpPred), H.T) + R
        K = np.dot(np.array([np.dot(PoldSig, H.T)]).T, np.linalg.inv(S))
        newSig = (SigPred + np.dot(K, np.atleast_1d(y)).T)
        PnewSig = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), SigpPred)

        # Updating the Kalman filter object
        self.PX = PnewX
        self.X = newX
        self.Sig = newSig
        self.PS = PnewSig


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
fname = 'data3/bayes_'+str(int(tstart))
outfile = open(fname,'w')
#logger = JSONLogger(path="logs/log_"+str(int(tstart))+".log")

countblackboxes = 0

maxsettletime = 180.0
minsettletime = 15.0

#def black_box_function(balzer1N,balzer5N,biasN,p18N):
def black_box_function(settingsN):
    # settings 0:balzer1N, 1:biasN, 2: P18
    global countblackboxes
    numlowpressure = 0
    killoff = 0

    [balzer1,bias,p18]=setnumbers(settingsN[0],settingsN[1],settingsN[2])
    if runsafe: print('from setnumbers: ',[balzer1,bias,p18])
    time_bbox = time.time()
    twait = 0.37

    #print(f'setting balzer to {balzer1:.2f} and bias to {bias:.2f}')
    venus.write({"k18_fw":int(p18)})
    venus.write({"bias_v":bias})
    balzer1val = setbalzer(2,balzer1,4.1,4.2)
        
    # changes done...wait for ~180 seconds (250 measurements)
    #   now wait 410 = 295 seconds
    timedout = False 
    tstartwatch = time.time()

    v_list = []
    KF = KFparAlt(.01, 10000.) # Lower first number for higher filtering
    MinSample2Move = 100 # Mod vwatson KF June5 | Min number of sample before check for settled
    NsampleTestSlope = 10 # Mod vwatson KF June5 | number of sample on which we test the slope for settled < MinSample2Move
    slope = [] # Mod vwatson KF June5 | storing slope values
    SettleSlope = 5./10. #1E-6 # Mod vwatson KF June5 | max acceptable slope for a settled system
    CntrNeverSettle = 0 # Mod vwatson KF June5 | escape loop if never settled

    v_list.append(getCurrent(connection) * 1e6)
    oldtime = time.time() # Mod vwatson KF June5 | storing deference time
    KF.X[0] = v_list[len(v_list)-1]# Mod vwatson KF June5 | first measure initialize KF
    settled = False # Mod vwatson KF June5 | initialize system as not settled

    announce = 'settled: '
    while not settled and not killoff and not timedout:
#dst    for j in range(800):
#dst        writeoutput(outfile)
#dst        time.sleep(twait)
#dst        killoff,numlowpressure = checkpressure(numlowpressure)
#dst        if killoff:
#dst            break
            #time.sleep(twait)     # dst: always wait a little
            killoff,numlowpressure = checkpressure(numlowpressure)    #dst: watch for low pressure
            tloop = time.time()
            iloop = 0
            while time.time()-tloop < 0.37:
               v_list.append(getCurrent(connection)*1e6)
               newtime = time.time() # Mod vwatson KF June5 | grabbing current time
               outfile.write("%.3f %.3f\n" % (newtime,v_list[-1]))    # dst: write time and current
               KF.EstimateState(v_list[len(v_list)-1], newtime-oldtime) # Mod vwatson KF June5
               oldtime = newtime # Mod vwatson KF June5 | updating reference time
               slope.append(KF.X[1]) # Mod vwatson KF June5 | storing slope value
               iloop = iloop + 1
            CntrNeverSettle += 1 # Mod vwatson KF June5 | counting loop
            #print(f'{CntrNeverSettle:5d} {v_list[-1]:10.2f} {KF.X[0]:10.2f} {KF.X[1]:10.4f} {iloop:3d}')
            #if len(v_list) > MinSample2Move: # Mod vwatson KF June5 | waited enougth to test the slope ?
            if newtime-tstartwatch > minsettletime:
                if np.all(np.abs(slope[len(slope)-NsampleTestSlope-1:len(slope)-1]) < SettleSlope):# Mod vwatson KF June5 | is abs(slope) small enough (check code)
                    settled = True # Mod vwatson KF June5
            if newtime-tstartwatch > maxsettletime:   # dst
#dst            if CntrNeverSettle > MaxIter:# Mod vwatson KF June5 | maxed time system never settled
#dst                settled = True # Mod vwatson KF June5
                timedout = True
                print('System could not settle') # Mod vwatson KF June5
        # Exteacting the data # Mod vwatson KF June5
    v_mean = KF.X[0] # Mod vwatson KF June5
    v_std = KF.Sig[0] # Mod vwatson KF June5
    outfile.flush()

    if killoff:
       v_mean = 0.2
       v_std = 2.0
       announce = 'killed : '
    if timedout:
       v_mean = 0.3
       v_std = 0.3
       announce = 'timeout: '
    rel_std = v_std / v_mean

    print(f'{announce} {balzer1:5.2f} + {bias:6.2f} + {p18:6.1f}: {v_mean:7.2f} {rel_std*100:6.2f}% tstep={time.time()-tstartwatch:.2f}')
    g = open('data3/nextpoint_'+str(int(tstart)),'a')
    g.write('%i '%(time_bbox))
    g.write('%6.2f %6.2f %6.2f '%(balzer1,bias,p18))
    g.write('%7.2f %7.2f %7.2f\n'%(v_mean,v_std,rel_std*100))
    g.close()
    #instability_cost = squared(rel_std)*BEAM_CURR_STD
    instability_cost = squared(rel_std)*BEAM_CURR_STD
    ffff = open('data3/info_'+str(int(tstart)),'a')
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

def setnumbers(balzer1N,biasN,p18N):
    scale_balzer1 = [4.3,5.0,balzer1N]
    scale_bias = [25,75,biasN]
    scale_p18 = [200,800,p18N]
    scales = [scale_balzer1, scale_bias, scale_p18]
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
Xdata = np.array([[ 0.2, 0.7, 0.3],
                  [ 0.6, 0.5, 0.2],
                  [ 0.2, 0.4, 0.6],
                  [ 1.0, 0.8 ,0.1]])

# len(Xdata) blackboxcalls to get Ydata
ninit = len(Xdata)
Ydata = np.zeros(ninit)
for i in range(ninit):
    Ydata[i] = black_box_function(np.array([Xdata[i,0],Xdata[i,1],Xdata[i,2]]))
XdataSpaceDimension = 3
maxIter = 100   # DST vary this
ExpBias = .5
# initializing kernel
Klen = np.ones(XdataSpaceDimension) * .1

Cov = np.zeros([maxIter,XdataSpaceDimension])

kname = 'data3/klen_'+str(int(tstart))
kk = open(kname,'w')

succeedname = 'data3/succeed_'+str(int(tstart))
ssf = open(succeedname,'w')

with open('data3/alpha_'+str(int(tstart)),'w') as aaaa:
    aaaa.write("%e"%(alphaval))


for cptr in range(maxIter):

    # Setting up Anisotropic Kernel function
    kernel = Matern(length_scale=Klen, nu=2.5)
    # Initializing regressor
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha = alphaval)
    # Fitting the data
    gpr.fit(Xdata, Ydata)

    # Initializing the standard optimizer for Acq function
    # Modification 5 June 2025 start | vwatson ###########################################################
    MaxIterForAcqOptInit = 4000 # number max of iterations to ind an area where the acquisition function is computable
    CntIterRndSearch = 0 # way out when impossible to find a solution

    Ymax = np.max(Ydata)

    RunRndSearch = True
    while RunRndSearch:
        InitParam = np.random.rand(XdataSpaceDimension) # random generation in normalized parameter space
        if abs(InvExpectedImp(InitParam, gpr, ExpBias, Ymax)) >  1E-100:
            RunRndSearch = False
            ssf.write("1\n")
        else:
            CntIterRndSearch += 1
        if CntIterRndSearch >= MaxIterForAcqOptInit:
            RunRndSearch = False
            print('Failed to find proper initialization for optimizing the acquisition function')
            ssf.write("0\n")

    # Modification 5 June 2025 end | vwatson ###########################################################

    res = sp.optimize.minimize(InvExpectedImp, InitParam, args=(gpr, ExpBias, Ymax), bounds = {(0,1.),(0,1.),(0,1.)}, method = 'Nelder-Mead')
    nextX = res.x
    nextY = black_box_function(np.array([nextX[0], nextX[1], nextX[2]]))

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

        # tescts
        if cptr > maxIter * .7 :  # can be adjusted
            ExpBias = -5
    for i in range(len(Klen)):
        kk.write("%e " % (Klen[i]))
    kk.write("\n")

with open('data3/cov_'+str(int(tstart)),'w') as cc:
    shpe = np.shape(Cov)
    for j in range(shpe[1]):     #DST3: Should this loop over shpe[2], too???
        for i in range(shpe[0]):
            cc.write("%e " % (Cov[i,j]))
        cc.write("\n")


outfile.close()
kk.close()
ssf.close()
