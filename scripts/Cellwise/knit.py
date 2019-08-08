"""A script to knit together all the predictions for the cells from cell-wise random forests.
Requires as input a folder containing predictions (scratch/kwheelan/datasets/partialData)
Run the prediction script first to generate this data

K. Wheelan 7.5.19                                                                                                                              
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

fileDestination = sys.argv[1] #where to save the final output
valYear = sys.argv[2] #The year the predictions came from
in_data = sys.argv[3] #The dataset for predictons
partialDataSpot = sys.argv[4] #where the cellwise predictions are saved

#getting validation data                                                        
inputData = xr.open_dataset(in_data).sel(time = slice(str(valYear)+'-01-01',str(valYear)+'-12-31'))                                                                
#inputData = xr.open_dataset(sys.argv[2])
valData = inputData.to_dataframe().reset_index()
valData['Predictions']= 999.00

for i in range(3200//100): #this should match with the chunk size being saved by the prediction script (100 cells right now)
    period = inputData.time.shape[0]
    start, end = 100*i, 100*(i+1) #start and end cell of the chunk
    fileName = partialDataSpot + "/yr"+str(valYear)+"/cells_"+str(start+1)+"-"+str(end+1)+".nc"
    data = xr.open_dataset(fileName).to_dataframe().reset_index()
    #Load the predictions into a larger dataframe for all cells
    dates = list(range(start*period, end*period))
    valData.Predictions[dates] = data.Predictions[dates]

#write final data                                                                 
x = valData.set_index(['latitude','longitude','time']).drop(list('abcdefghijklnopqrstuvwxy'),axis=1).to_xarray()

x = maskWater(x.to_dataframe(), "Predictions").to_xarray() #drop values for areas that cover water
x.to_netcdf(fileDestination+'/preds_'+valYear+'_run6.nc') #save to disk
