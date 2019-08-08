#Some stat functions

#K. Wheelan 6.11.19

"""
A collection of functions to help with various statistical tasks.
"""

__all__ = ['weight','getTrainTest','getTrainTestBinary','getTrainTestLog','boxCox','qqTrans', 'plotPredicted','maskWater', 'makePrcpBinary', 'posPrcpDF','df_toXarray', 'plotPrcp_2layer', 'plotResiduals', 'makeSingleAnim', 'makeDoubleAnim']

import imp
my_file = imp.load_source('DataFns', "/glade/u/home/kwheelan/scripts/DataFns.py")
from DataFns import *

import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation

from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer #may be outdated

def getTrainTest(df, random_state = 56):
    """Returns (data, testdata, labels, testlabels).
    Requires pandas df because of sklearn's train_test_split method."""
    X = df.drop(['time','Prcp'], axis = 1)
    y = df['Prcp']
    return  train_test_split(X, y, test_size=0.3, random_state=random_state)

def getTrainTestLog(data):
    """Returns (data, testdata, labels, testlabels). Does log of prcp.
        Note that this doesn't work when we include 0 prcp values."""
    (prData, prData_test, prData_labs, prData_test_labs) = data
    prData['current'] = np.log(prData['current'])
    prData_test['current'] = np.log(prData_test['current'])
    prData_labs, prData_test_labs = np.log(prData_labs), np.log(prData_test_labs)
    return (prData, prData_test, prData_labs, prData_test_labs)

def getTrainTestBinary(df, makebin = True, threshold = 0):
    """determines test/train sets for predicting binary rain/no rain.
    Y sets are binary and X sets are all predictors."""
    if not type(df) == pd.core.frame.DataFrame:
        df = df.to_dataframe().reset_index()
    if makebin:
        df = makePrcpBinary(df, thresh = threshold)
    X_train, X_test, Y_train, Y_test = getTrainTest(df)
    Y_train = X_train['prcp_binary']
    X_train = X_train.drop('prcp_binary', axis=1)
    return (X_train, X_test, Y_train, Y_test)

def boxCox(data):
    """Transforms the data to a nearly Gaussian dist using Box-Cox parametric method."""
    (X_train, X_test, Y_train, Y_test) = data
    bc = PowerTransformer(method='box-cox')
    X_trans_bc = bc.fit(X_train).transform(X_test)
    Y_trans_bc = bc.fit(Y_train).transform(Y_test)
    
def qqTrans(data):
    """Does a quantile-quantile transformation to approximate a Gaussian distribution.
    Input xarray or pandas df. Output the 4 sets fo data."""
    (X_train, X_test, Y_train, Y_test) = data
    qt = QuantileTransformer(output_distribution='normal', random_state=56)
    X_trans_test = qt.fit(X_train).transform(X_test)
    X_trans_train = qt.fit_transform(X_train)
    return (X_trans_train, X_trans_test, Y_train, Y_test)

def weight(obs):
    """Weights neighboring cells by their distance from the point.
    Input is xarray oject containing observation grid;
    returns xarray object with RCM cells weighted by distance to current obs cell."""
    obs_ds = obs.copy()
    #calculate distances
    S = _m(obs_ds['latitude'])
    W = _m(obs_ds['longitude'])
    N, E = 0.5 -S, 0.5 - W
    NE,NW = pythag(N,E), pythag(N,W)
    SE, SW = pythag(S,E), pythag(S,W)
    for lab in ['N','NE','E','SE','S','SW','W','NW']:
        #multiply distances by existing values
        obs_ds[lab]*=eval(lab)
    return obs_ds

def _m(n):
    """A helper method for prep_data() to find distances"""
    return np.round((n-.25)%.5,3)

def plotPredicted(pred_df, regObj, colTitle, plotTitle=None, maskWater = False, vmax=12,vmin=0):
    """Plots predicted Prcp for some regression"""
    if not (type(pred_df) == pd.core.frame.DataFrame):
        pred_df = pred_df.to_dataframe().dropna()
    pred_df[colTitle] = regObj.predict(pred_df.reset_index().filter(items = ['latitude', 'longitude', 'N', 'NE', "E", "SE", "S","SW","W","NW", "current"]))
    if maskWater: 
        pred_df = maskWater(pred_df, colTitle)
    pred_df.to_xarray()[colTitle].mean(dim='time').plot(cmap = "jet", vmax = vmax, vmin=vmin)
    if not plotTitle is None: 
        plt.title(plotTitle)
    return pred_df

def maskWater(df, colName):
    """Masks non-land parts of the grid"""
    #This is faster using the xarray .where() command (as opposed to converting
    #to a Pandas DF, but the where() seems to sometimes act strangely...
    df[colName][df.Prcp.isnull()] = np.nan
    return df

def makePrcpBinary(ds, thresh=0):
    """Adds a predictor variable with a 1 if there was any prcp and a 0 
    if there was less than (or equal to) the threshhold amount of prcp."""
    ds['prcp_binary'] = (ds['Prcp']>thresh).astype(int)
    return ds

def posPrcpDF(df, thresh):
    """Returns a subset of the datafame (pandas df or xarray) with only 
    the positive Prcp data. Returns pandas df."""
    if not type(df) == pd.core.frame.DataFrame:
        df = df.to_dataframe().reset_index()
    df = makePrcpBinary(df, thresh)
    return df[df['prcp_binary']>0].drop('prcp_binary', axis=1)
        
def df_toXarray(df):
    """Convert a pandas dataframe to x-array object"""
    return df.reset_index().set_index(['time', 'latitude','longitude']).to_xarray()

def plotPrcp_2layer(predictions, plotTitle = "Average Predicted Precipitation", thresh=0.5):
    """Plots the predicted prcp averages for a given threshold level.
    Input a predictions xarray."""
    predictions['final_pred'] = predictions['preds']
#this sometimes does not work as desired... 
    predictions['final_pred'] = predictions.final_pred.where(predictions.class_preds>thresh, 0)
#    predictions['final_pred'] = predictions.final_pred.where(predictions.preds > -1000, np.nan) #mask water
    predictions.final_pred.mean(dim='time').plot(cmap='jet',vmin=0,vmax=12)
    plt.title(plotTitle +  ", Threshold = "+ str(thresh))
    return predictions

def plotResiduals(predictions, predColTitle, cmap = 'jet',  vmin=-1, vmax=1, plotTitle = "Residual Plot"):
    """Plot residuals from predictions"""
#Actually this plots average over the residuals (bias)
    predictions['residuals'] = predictions['Prcp'] - predictions[predColTitle]
    df_toXarray(predictions).residuals.mean(dim='time').plot(cmap=cmap,vmin=vmin,vmax=vmax)
    plt.title(plotTitle)

class UpdateQuad(object):
    """Initialized with xarray object and colName for variable to be plotted.
        Code for this class modified from https://groups.google.com/forum/#!topic/xarray/P6B1FsvkLME."""
    def __init__(self, data, ax, colName, vmin = 0, vmax = 8):
        self.ax = ax
        self.data = data[colName]
        self.quad = data[colName][0].plot(cmap = "jet", vmax = vmax, vmin = vmin, ax = self.ax)

    def init(self):
        self.quad.set_array(np.asarray([]))
        return self.quad

    def __call__(self, i):
        # data at time i
        ti = self.data[i]
        self.quad.set_array(ti.data.ravel())
        return self.quad

class comboQuad (object):
    """A class to combine the animations on one plot"""
    def __init__(self, ud1, ud2):
        self.ud1 = ud1
        self.ud2 = ud2
    
    def __call__(self, i):
        self.ud1(i)
        self.ud2(i)
#        plt.title("Observed and "+colName)

def saveAnim(anim, filePath):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filePath)

def makeSingleAnim(ds, colName, filePath=None, frames = 365, vmin = 0, vmax = 8):
    """Makes videos of daily prcp in the dataset. Takes xarray 
        object ds and string colName for the var to be plotted over time.
        Saves animation to filePath. Note that it cannot overwrite existing files."""
    fig, ax1 = plt.subplots()
    ud1 = UpdateQuad(ds, ax1, colName, vmin, vmax)
    anim = animation.FuncAnimation(fig, ud1, init_func=ud1.init,frames=frames, blit=False)
    saveAnim(anim, filePath)

def makeDoubleAnim(ds, colName1, colName2, filePath=None, frames = 365, vmin = 0, vmax = 8):
    """Makes videos of daily prcp in the dataset. Takes xarray 
        object ds and string colName (ex. Prcp) for the var to be plotted over time.
        Saves animation to filePath. Note that it cannot overwrite existing files."""
    fig, (ax1,ax2) = plt.subplots(1,2)
    ud1 = UpdateQuad(ds, ax1, colName1, vmin, vmax)
    ud2 = UpdateQuad(ds, ax2, colName2, vmin, vmax)
    ud3 = comboQuad(ud1, ud2)
    anim = animation.FuncAnimation(fig, ud3, init_func=ud1.init,frames=frames, blit=False)
    saveAnim(anim, filePath)
