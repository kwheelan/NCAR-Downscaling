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

#all daily data
path="/glade/scratch/kwheelan/datasets/PNW_allyears.nc"
#just 6 years
ds = xr.open_mfdataset(path).sel(time = slice("1994-01-01", "1999-12-31"))

ds_wt = weight(ds)
pnw_wt = ds_wt.to_dataframe().reset_index()

path="/glade/scratch/kwheelan/datasets/PNW_1993.nc"
ds93 = xr.open_mfdataset(path)
pnw_93 = ds93.to_dataframe().reset_index()

#Making a dataset of only positive prcp values

#Threshold = 0
thresh = 0
np.random.seed(100)
pnw = makePrcpBinary(ds, thresh).to_dataframe().reset_index()
pnw_prcp = pnw[pnw['prcp_binary']>0].drop('prcp_binary', axis=1)
(X_train, X_test, Y_train, Y_test) = getTrainTest(pnw_prcp)

#Grid Search for RF

forest = RandomForestRegressor(random_state = 56, n_jobs = -1)

param_grid = [
    {'n_estimators': [5, 10]},
    {'max_features': [.5,.75,'auto'], 'bootstrap': [True,False]},
]

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring = 'neg_mean_squared_error')

grid_search.fit(X_train, Y_train)

#or directly
print(grid_search.best_estimator_)

