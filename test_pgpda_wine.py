# -*- coding: utf-8 -*-
'''
Created on 18 oct. 2014

@author: mfauvel
'''
from pgpda import *
from accuracy_index import *
import time
import scipy.io as sio
data = sp.loadtxt('wine.data',delimiter=',')
Y = data[:,0]
X = data[:,1:]
X = standardize(X)[0]

x = X[0::2,:]
y = Y[0::2] 
xt = X[1::2,:]
yt = Y[1::2]

model=PGPDA()
model.model = 'M4'
sig_r = 2.0**sp.arange(-4,4)
threshold_r = sp.linspace(0.85,0.9999,10)
dc_r =sp.arange(5,30,10)

sig,threshold,err=model.cross_validation(x, y,sig_r=sig_r,dc_r=dc_r,threshold_r=threshold_r)
model.train(x,y)
yp=model.predict(xt,x,y)
conf = CONFUSION_MATRIX()
conf.compute_confusion_matrix(yp,yt)
print conf.Kappa
print conf.OA

# M0
# 0.93182688625
# 0.955056179775

# M1 
# 0.948987390141
# 0.966292134831

# M2
# 0.948987390141
# 0.966292134831

# M3 
# 0.965854594284
# 0.977528089888

# M4
# 0.861262665627
# 0.910112359551

# M5
# 0.948987390141
# 0.966292134831

# M6
# 0.684583579445
# 0.797752808989
