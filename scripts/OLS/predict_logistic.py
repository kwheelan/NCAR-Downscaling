"""A file to predict the probability of prcp given cellwise logistic models.

K.Wheelan 7.22.19
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.linear_model import LogisticRegression

import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/scripts/StatFns.py")
from StatFns import *

import sys

test = sys.argv[1] #dataset for test data
betas_loc = sys.argv[2] #where the coeffients are saved
final_dest = sys.argv[3] #where to save the final predictions

def addBias(ds):
    """Prepares an xarray ds for cell-wise lin reg"""
    df = ds.to_dataframe().reset_index().dropna()
    data = df.drop(["Prcp", 'latitude', 'longitude', 'time', 'time-copy', 'Prcp_scaled']+list('abcdefghijklmnopqrstuvwxy'),axis=1).values
    m, n = data.shape
    df_plus_bias = np.c_[np.ones((m,1)),data]
    return df, df_plus_bias

#open the file with the coefficients
betas = pd.read_csv(betas_loc)
betas.drop('Unnamed: 0', axis=1, inplace = True)

test['m'] = test.m **.25 #scale data on fourth root scale
pnw_test = test.to_dataframe().reset_index().dropna()

inputData = pnw_test
beta_input = betas
period = 16*365 + 8 #8 leap years

valData = prepFullData(inputData) #check this function

#Validation data as input
col = 5
valObs = np.matrix(valData[:,[0,col]], dtype = np.float32) #data
betaMatrix = np.matrix(beta_input, dtype = np.float32)

#At each time-step, weight each input by the appropriate coefficients for each cell
#This yields the log-odds
log_odds = [ float(np.matmul( valObs[cell_num + period_num*betaMatrix.shape[1],], betaMatrix[:,cell_num])) 
              for period_num in range(valObs.shape[0]//betaMatrix.shape[1])
              for cell_num in range(betaMatrix.shape[1]) ]


#Set up dataframe to save the predictions
valData = pd.DataFrame(valData[:,(1,2,3,4,5)], columns = ['latitude', 'longitude','time','Prcp', 'current'])
valData['log_odds'] = log_odds #just calculated predictions for log odds
#convert from log-odds to raw probability
valData['cell_preds'] = 1/(1 + np.exp(-valData.log_odds))

#convert to xarray for easy plotting
preds = valData.set_index(['time','latitude', 'longitude']).to_xarray()

#write final preds to disk
preds.to_netcdf(final_dest)
