
"""A script to run a seperate random forest on each cell in the obs data in the PNW with 24 neighbors as input                                                       
This script "pickles" a model for each of the cells

This version of the script uses some random subset of the odd years as validation data.
The random forest stops training when the validation scores (OOB scores) decrease too much.

K. Wheelan 7.8.19"""                                                                

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

start, end = int(sys.argv[3]),int(sys.argv[4]) #start and end of the chunk of cells to be predicted
location  = sys.argv[1] #where to save the preds
source_data = sys.argv[2] #where the validation data is
pnw24 = xr.open_mfdataset(source_data)

#Separate even and odd years for training and testing/validation                  
even, odd = evenOdd(pnw24)
trainData = even.to_dataframe().reset_index()

period = 16*365 + 8 #8 leap years: 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008          

#Scale data on 4th root scale                                                                                                                              
for axis in list('abcdefghijklmnopqrstuvwxy')+['Prcp']:
#    trainData[axis]=trainData[axis]**.25      
    pass

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 40,  n_jobs = -1, bootstrap = True, max_features = 'log2', oob_score=True, warm_start = False)


#Training and calculating for validation data       

for cell in range(start-1,end-1):
    #do a regression for every cell in the area                               
                                                                                
    X = trainData.drop(['Prcp','time','time-copy','latitude','longitude'],axis=1).values[(cell)*period:(cell+1)*period,]
    y = trainData.Prcp.values[(cell)*period:(cell+1)*period,].reshape(-1,1)
    #error_rate = []


    ynotna = [i for i in range(y.shape[0]) if not np.isnan(y[i])]
    y = y[ynotna]
    X = X[ynotna]
    if y.shape[0] >  0:   
        #Only fit model in areas with obs data
                                                                 
        #Use out-of-bag samples to optimize model and avoid over-fitting     
        min_samples_split = 500
        oob_max = (0,0)
        oob_score = (0,0)
        while(oob_score[1] > (oob_max[1] - .0125) and min_samples_split > 1):
            rf.set_params(min_samples_split=min_samples_split)
            rf.fit(X,y)
            oob_score = (min_samples_split, rf.oob_score_)
            if oob_score[1] > oob_max[1]:
                oob_max = oob_score
            min_samples_split //= 2
            #error_rate.append(oob_score)
            
        #Re-run best model
        if not oob_max == oob_score:
            rf.set_params(min_samples_split=oob_max[0])
            rf.fit(X,y)
        # save the model to disk                                                
        filename = location +'/'+str(cell)+'_RF.sav'
        pickle.dump(rf, open(filename, 'wb'))
    
        

