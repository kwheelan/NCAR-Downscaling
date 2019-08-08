#K.Wheelan 7.12.19                                                                                                                        
#Much of the U-net code comes from https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277                                             
#A script to make a U-net convolutional neural network for downscalings                                                                   

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
from os.path import join, expanduser
from sklearn.preprocessing import StandardScaler
from ipywidgets import interact
import ipywidgets as widgets
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Conv2DTranspose, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, Dropout
from keras.layers import SpatialDropout2D
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, roc_auc_score

import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/scripts/StatFns.py")
from StatFns import *

import sys

location = sys.argv[1]
final_file = sys.argv[2]

# Create a Tensorflow session that only allocates GPU memory as needed                                                                    
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
K.set_session(session)

#Getting the data for training                                                                                                            
DATA_ROOT = '/glade/scratch/kwheelan/datasets/'
ds = xr.open_dataset(join(DATA_ROOT, 'PNW_w_elevation.nc'))

#Using a fourth root to make the distribution more Gaussian
ds['Prcp'] = ds.Prcp**.25
ds['m']= ds.m **.25

#Using even years to train, odd to test                                                                                                   
train, test = evenOdd(ds)

#training input data                                                                                                                      
train['elev'] = train.a*0 + train.elevation                                                                                       

#for elevation as an input uncomment      
#X_train = xr.concat([train.transpose().m, train.transpose().elev], dim='var').transpose()                              

#no elevation uncomment below
train = train.transpose().expand_dims({'var': "Prcp"}, axis=0).transpose().m

#training output data                                                                                                  
Y_train = train.transpose().expand_dims({'var': "Prcp"}, axis=0).transpose().Prcp.fillna(0)

# Build U-Net model                                                                                                                       
learning_rate = 0.001
#for using elevation:
#inputs = Input((40, 80, 2))
#for no elevation:
inputs = Input((40,80,1)))
act_fn = 'relu'

c1 = Conv2D(10, (4, 4), activation=act_fn, kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(10, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(20, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(20, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(40, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(40, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(40, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(40, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(1, 1)) (c4)

c5 = Conv2D(80, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(80, (2, 2), activation=act_fn, kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(40, (2, 2), strides=(1, 1), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(40, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(40, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation=act_fn, kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='linear') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
opt = Adam(lr=learning_rate)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)                                                                                                                                                                                                             
model.compile(loss='mean_squared_error', optimizer=opt)

# Fit model using early stopping and save best model
earlystopper = EarlyStopping(patience=5, verbose=1) 
checkpointer = ModelCheckpoint(location, verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=15, epochs=50, callbacks=[earlystopper, checkpointer])

#Load best model and use to predict training data                                                                                         
model = load_model(location)
preds = model.predict(X_train, batch_size=15)
train['preds'] = xr.DataArray(preds, dims = ['time', 'latitude', 'longitude', 'var'])
train['preds'] = train.preds**4 #raise to fourth power
train['preds'] = train.preds.where(train.Prcp > -1000, np.nan)

#write predictions for train data
train.sel(var=0).to_netcdf(final_file)

