"""
A script to predict prcp every day in the PNW in some validation year.
Using cell-wise RFs

K. Wheelan 7.3.19                                                     
"""
           
import tensorflow as tf
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import pickle
import sys

import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/scripts/StatFns.py")
from StatFns import *

#Which cells to evaluate
start, end = int(sys.argv[4]),int(sys.argv[5])
model_location = sys.argv[1]
year = sys.argv[2]
partialLocation = sys.argv[3]

#Where to write the final data                                                 
netCDFName = partialLocation+'/yr'+str(year)+'/cells_'+str(start)+'-'+str(end)+'.nc'

#prepping validation data
inputData = xr.open_mfdataset('/glade/scratch/kwheelan/datasets/future_climate/PNW_'+str(year)+'_prepped.nc')
valData = inputData.to_dataframe().reset_index()
valData['Predictions']= 999.00

#period length of TRAINING data from creating the models
period = 16*365 + 8 #8 leap years: 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008 

for cell in range(start-1, end-1):
#All cells is range(valData.shape[0]//inputData.time.shape[0])
    filename = model_location+'/'+str(cell)+'_RF.sav'
    try:
        #grab the saved model from this cell
        loaded_model = pickle.load(open(filename, 'rb'))
        row = list(range(cell*inputData.time.shape[0], (cell+1)*inputData.time.shape[0]))
        valData.Predictions[row] = loaded_model.predict(valData.drop(['Prcp', 'latitude',
                                'longitude','Predictions', 'time'], axis=1).values[row,])
    except: pass
#    if(cell%25 == 0):                                                                                                
       # print("cell: ", str(cell))

#write final data                                                              

x = valData.set_index(['latitude','longitude','time']).drop(list('abcdefghijklnopqrstuvwxy'),axis=1).to_xarray()
#x['Predictions'] = x.Predictions**4
x.to_netcdf(netCDFName)
