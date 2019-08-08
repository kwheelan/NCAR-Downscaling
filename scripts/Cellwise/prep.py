"""
This script preps input data by adding values for 24 neighboring RCM cells for each given obs cell.
As written right now, it preps non-leap years.

This is written to prep the future/historical WRF runs from the GCM, but can easily be adjusted to prep obs. There are also saved prepped datasets.

K.Wheelan July 2019
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

#specify year of data to prep
import sys
yr= sys.argv[1]
save_location = sys.argv[2]

future = xr.open_mfdataset('/glade/scratch/kwheelan/datasets/rawdata/future_wrf.nc')
#Picks an arbitrary non-leap year to grab the obs grid cell dimensions for future/historical WRF
#replace this with a line openning the obs if you want to prep obs
x = xr.open_dataset('/glade/scratch/kwheelan/datasets/PNW_1979-2010.nc').sel(time=slice('1982-01-01','1982-12-31'))


#grabbbing RCM data
f = future.sel(time=slice(str(yr)+'-01-01',str(yr)+'-12-31'))
x['time'] = f.time #matching time dimensions
test = prep_data24(future ,x) #using prep_data24 function from DataFns module

#Saves output data
test.to_netcdf(save_location+'/'+str(yr)+'_prepped.nc')
