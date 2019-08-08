"""Runs a cell-wise logistic regression on all cells in the PNW.
Saves a file with all the coefficients for easy prediction.

I've implemented this using linear algebra operations on TensorFlow for prediction because it's faster
to just save the coefficients than to save the individual models and then reopen them for prediction.

K.Wheelan 7.22.19"""


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

trainData = sys.argv[1] #training data location
betas_loc = sys.argv[2] #where to save the coefficients from the regression

pnw = xr.open_mfdataset(trainData)
trainData, testData = evenOdd(pnw)

#Scaling all the imputs to fourth root
trainData['Prcp_scaled'] = trainData.Prcp**.25
trainData['m_scaled'] = trainData.m**.25
trainData['Prcp'] = trainData.Prcp**.25
trainData['m'] = trainData.m**.25

def addBias(ds):
    """Prepares an xarray ds for cell-wise logistic reg"""
    df = ds.to_dataframe().reset_index().dropna()
    data = df.drop(["Prcp", 'latitude', 'longitude', 'time', 'time-copy', 'Prcp_scaled']+list('abcdefghijklmnopqrstuvwxy'),axis=1).values
    m, n = data.shape
    #Adding a row of ones in the X matrix for the intercepts
    df_plus_bias = np.c_[np.ones((m,1)),data]
    return df, df_plus_bias

df, df_plus_bias = addBias(trainData)

period = 16*365 + 8 #8 leap years: 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008
betasdf = pd.DataFrame(index = range(0,2)) #make an empty df for storing coefficients

#prepare binary outputs for training the data (precip vs no precip)
y_bin = df.Prcp
y_bin[y_bin > 0] = 1

#Save location for pickled models
location = '/glade/scratch/kwheelan/models/OLS/'

glm = LogisticRegression()

#Training

for cell in range(df.shape[0]//period):
    #do a regression for every cell in the area
    X = df_plus_bias[(cell)*period:(cell+1)*period,]
    y = y_bin.values[(cell)*period:(cell+1)*period,].reshape(-1,1)
    glm.fit(X,y)
    #save coefficients
    betasdf[str(cell),] = glm.coef_[0]
    if(cell%100 == 0): print("cell: ", str(cell))
  
#filename = '/glade/scratch/kwheelan/datasets/OLS/Log_betas.txt'
betasdf.to_csv(betas_loc)

