"""
A script to do a cell-wise random forest for classifying prcp/no prcp
This script "pickles" a model for each of the cells
This version of the script uses out of bag data for validation to inform early stopping.
The data is 1980-2010 on a slightly lower section of the PNW (south of US-Canada border at the 49th parallel)"""

#K. Wheelan 7.16.19                                                                

import tensorflow as tf
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
import pickle

import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/scripts/StatFns.py")
from StatFns import *

start, end = int(sys.argv[3]),int(sys.argv[4]) #start and end cell of the range to be modeled
location  = sys.argv[1] #where to save the pickled models
sourceData = sys.argv[2] #data with which to train the models
pnw24 = xr.open_mfdataset(sourceData)

#Separate even and odd years for training and testing/validation                  
even, odd = evenOdd(pnw24)
trainData = even.to_dataframe().reset_index()
period = 16*365 + 8 #8 leap years: 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008                

#Scale data on 4th root scale
for axis in list('abcdefghijklmnopqrstuvwxy')+['Prcp']:
    trainData[axis]=trainData[axis]**.25

#Set up random forest with its parameters
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 40,  n_jobs = -1, bootstrap = True, max_features = 'log2', oob_score=True, warm_start = False)

#Training and calculating for validation data       

for cell in range(start-1,end-1):
    #do a regression for every cell in the area                               
                                                                                
    X = trainData.drop(['Prcp','time','latitude','longitude','time-copy'],axis=1).values[(cell)*period:(cell+1)*period,]
    y = trainData.Prcp.values[(cell)*period:(cell+1)*period,].reshape(-1,1)
    #Remove empty cells (water)
    ynotna = [i for i in range(y.shape[0]) if not np.isnan(y[i])]
    y = y[ynotna]
    y[y>0] = 1 #Make output binary (prcp/no prcp)
    X = X[ynotna]
    if y.shape[0] >  0:   #Only fit model in areas with obs data
        rf.fit(X,y)
        # save the model to disk                                                
        filename = location +'/'+str(cell)+'_RF.sav'
        pickle.dump(rf, open(filename, 'wb'))
    
        

