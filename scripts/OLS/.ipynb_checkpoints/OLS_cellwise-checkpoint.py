"""A script to run a cell-wise linear regression for prcp data.

K. Wheelan 7.16.19"""

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

pnw = xr.open_mfdataset("/glade/scratch/kwheelan/datasets/PNW_1979-2010.nc")
trainData, testData = evenOdd(pnw)
trainData['Prcp_scaled'] = trainData.Prcp**.25
trainData['m_scaled'] = trainData.m**.25

#split into even/odd?
def addBias(ds):
    """Prepares an xarray ds for cell-wise lin reg"""
    df = ds.to_dataframe().reset_index().dropna()
    data = df.drop(["Prcp", 'latitude', 'longitude', 'time', 'time-copy', 'Prcp_scaled']+list('abcdefghijklmnopqrstuvwxy'),axis=1).values
    m, n = data.shape
    df_plus_bias = np.c_[np.ones((m,1)),data]
    return df, df_plus_bias

df, df_plus_bias = addBias(trainData)

period = 16*365 + 8 #8 leap years: 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008
betasdf = pd.DataFrame(index = range(0,2))

#Training

for cell in range(df.shape[0]//period):
    #do a regression for every cell in the area
    X = tf.constant(df_plus_bias[(cell)*period:(cell+1)*period,], dtype = tf.float32, name = "X")
    y = tf.constant(df.Prcp_scaled.values[(cell)*period:(cell+1)*period,].reshape(-1,1), dtype=tf.float32, name = "y")
    XT = tf.transpose(X)
    #compute the coefficients using linear algebra
    betas = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)
    with tf.Session() as sess:
        betasdf[str(cell)] = betas.eval()
    if(cell%100 == 0): print("cell: ", str(cell))

betasdf.to_csv(r'/glade/scratch/kwheelan/datasets/OLS/OLS_betas.txt')
