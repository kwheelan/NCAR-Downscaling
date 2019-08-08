
#Predicting prcp using cellwise linear models
#This is not parallelized, but it probably should be and easily could be

#K. Wheelan 7.22.19

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

test = xr.open_dataset(sys.argv[1]) #the test data
beta_file = sys.argv[2] #where the coefficients are saved
file_destination = sys.argv[3] #where to save the final predictions

#read in coefficients
betas = pd.read_csv(beta_file)
betas.drop('Unnamed: 0', axis=1, inplace = True)

def prepFullData(inputData):
    """Preps data for analysis on all time periods"""
    valData = inputData.sort_values(by = ['time', 'latitude', 'longitude']).drop(list('abcdefghijklnopqrstuvwxy'),axis=1).reset_index()
    valData = valData.dropna().drop(['index'],axis=1)
    m, n = valData.shape
    return np.c_[np.ones((m,1)),valData]

test['m'] = test.m **.25 #scale data on fourth root scale
pnw_test = test.to_dataframe().reset_index().dropna()

inputData = pnw_test
beta_input = betas
period = 16*365 + 8 #8 leap years

valData = prepFullData(inputData)

#Validation data as input
col = 5
valObs = np.matrix(valData[:,[0,col]], dtype = np.float32) #data
betaMatrix = np.matrix(beta_input, dtype = np.float32)

#At each time-step, weight each input by the appropriate coefficients for each cell
#cell_preds is a list of all the predicted prcp values at each cell and time-step
cell_preds = [ float(np.matmul( valObs[cell_num + period_num*betaMatrix.shape[1],], betaMatrix[:,cell_num])) 
              for period_num in range(valObs.shape[0]//betaMatrix.shape[1])
              for cell_num in range(betaMatrix.shape[1]) ]


#Set up dataframe to save the predictions
valData = pd.DataFrame(valData[:,(1,2,3,4,5)], columns = ['latitude', 'longitude','time','Prcp', 'current'])
valData['cell_preds'] = cell_preds #just calculated predictions

#convert to xarray for easy plotting
preds = valData.set_index(['time','latitude', 'longitude']).to_xarray()
preds['cell_preds']=preds.cell_preds**4

#save to disk
preds.to_netcdf(file_destination)
