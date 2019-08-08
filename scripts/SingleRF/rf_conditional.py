import imp
imp.load_source('DataFns', "/glade/u/home/kwheelan/dataPrep/StatFns.py")
from DataFns import *

imp.load_source('StatFns', "/glade/u/home/kwheelan/dataPrep/StatFns.py")
from StatFns import *

import warnings
warnings.filterwarnings('ignore')

import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import interp

from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer #may be outdated                                                                        

#all daily data for 6 years in the PNW
path="/glade/scratch/kwheelan/datasets/PNW_allyears.nc"                                                                
ds = xr.open_mfdataset(path).sel(time = slice("1994-01-01", "1999-12-31"))

#daily data in the PNW for 1993 only
path="/glade/scratch/kwheelan/datasets/PNW_1993.nc"
ds93 = xr.open_mfdataset(path)
pnw_93 = ds93.to_dataframe().reset_index()

#Spliting into Training and Testing Data
(X_train, X_test, Y_train, Y_test) = getTrainTestBinary(ds)

# Training the Model for Binary Classification
forest_class = RandomForestRegressor(n_estimators=10, n_jobs = -1)
forest_class.fit(X_train, Y_train)

#get prcp probability predicitons
classifications93 = plotPredicted(ds93, forest_class, "class_preds", plotTitle = "Average Predicted Probability of Prcp, 1994-1999", maskWater = True, vmax=1, vmin=0)

#Training the Model for Regressing on Cells with >0 Prcp
thresh = 0 #threshold for mm of obs prcp per day to count as prcp

#filter data so we only have positive prcp values
pnw_prcp = posPrcpDF(ds, thresh)

#Get a new training and testing split among the pos prcp df
X_train, X_test, Y_train, Y_test = getTrainTest(pnw_prcp)

#Training the Random Forest conditioned on predicted prcp
forest = RandomForestRegressor(n_estimators=15, bootstrap = False, n_jobs = -1, max_features = 0.5, random_state=56)
forest.fit(X_train, Y_train)

#Get predictions for prcp levels
predictions93 = plotPredicted(classifications93, forest, "preds", plotTitle = "Average Predicted Prcp, 1993", maskWater = True, vmax=12, vmin=0)

#plot the predictions given some threshold only above which it rains
rain_prob_thresh = 0.65
plotPrcp_2layer(predictions93, "Average Predicted Precipitation, 1993", thresh=rain_prob_thresh)

predictions93.to_xarray().to_netcdf('/glade/scratch/kwheelan/predictions93_thresh65.nc')
