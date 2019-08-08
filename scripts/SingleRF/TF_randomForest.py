#A simple script to run a random forest using TensorFlow

#K. Wheelan 7.3.19

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/scripts/StatFns.py")
from StatFns import *

tf.reset_default_graph()

#all daily data
path="/glade/scratch/kwheelan/datasets/PNW_allyears.nc"
#just 5 years
ds = xr.open_mfdataset(path).sel(time = slice("1994-01-01", "1999-12-31"))

pnw_zoom = ds.sel(latitude = slice(45, 46), longitude = slice(-124, -123))
pnw_zoom = pnw_zoom.to_dataframe().reset_index().dropna()

(X_train, X_test, Y_train, Y_test) = getTrainTest(ds.to_dataframe().reset_index())

def fix(data):
    return data.astype(np.float32).values

X_train, Y_train = fix(X_train), fix(Y_train)
ix = [i for i in range(len(Y_train)) if not np.isnan(Y_train[i])]
X_train, Y_train = X_train[ix,], Y_train[ix]


from tensorflow.contrib.tensor_forest.python import tensor_forest

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
  num_classes=1, num_features=8,regression=True,
  num_trees=20, max_nodes=10000)

est = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(
    params, model_dir="/glade/scratch/kwheelan/models/tf_rf_run1")

est.fit(x=X_train, y=Y_train)

path="/glade/scratch/kwheelan/datasets/PNW_1993.nc"
ds93 = xr.open_mfdataset(path)
pnw93 = ds93.to_dataframe().reset_index()
pnw93_zoom = ds93.sel(latitude = slice(45, 46), longitude = slice(-124, -123)).to_dataframe().reset_index()

inputData = pnw93

valData = inputData.groupby(['latitude', 'longitude']).mean().reset_index().dropna()
valData = fix(valData.drop(['Prcp'],axis=1))

y_out = list(est.predict(valData))

preds93 = inputData.groupby(['latitude', 'longitude']).mean().reset_index().dropna()

preds93['preds'] = [pred['scores'] for pred in y_out]

preds93 = preds93.set_index(['latitude','longitude']).to_xarray()
preds93.to_netcdf('/glade/scratch/kwheelan/datasets/tf_pnw_preds93.nc')
