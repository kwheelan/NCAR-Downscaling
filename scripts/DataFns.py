#Katrina Wheelan
#6.7.19

"""Various custom functions to help process predicted and observed precipitation data."""

__all__ = ['_loadCordex', '_loadObs', '_loadCordexI','add_prCell', 'dropNeighbors','getPr', 'coords_toXY', 'xy_toCoords', 'printCompare','prep_data', 'pythag', 'add_elevCell', 'evenOdd', 'prep_data24']

import xarray as xr
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np
import pyproj
import sklearn

CORDEX_HOME = "/glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44/eval"
OBS_HOME = "/glade/p/ral/hap/common_data/Maurer_w_MX_CA/pr"
CORDEX_I = "/glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44i/eval"

def _loadCordexI(year=1999, home=CORDEX_I):
    """Loads up RCM data on lat-lon grid."""
    file = "day/pr_NAM-44i_ECMWF-ERAINT_evaluation_r1i1p1_NCAR-WRF_v3.5.1_day_"+str(year)+"0101-"+str(year)+"1231.nc"
    path = os.path.join(home,file)
    return xr.open_mfdataset(path)

def _loadCordex(year=1999, home=CORDEX_HOME):
    """Loads up RCM data on x-y grid."""
    file = "day/pr_NAM-44_ECMWF-ERAINT_evaluation_r1i1p1_NCAR-WRF_v3.5.1_day_" + str(year) + "0101-" + str(year) + "1231.nc"
    path = os.path.join(home,file)
    return xr.open_mfdataset(path)
    
def _loadObs(year=1999, home=OBS_HOME):
    """Loads up Maurer observed data."""
    file = "gridded_obs.daily.Prcp." + str(year) + ".nc"
    path = os.path.join(home,file)
    return xr.open_mfdataset(path)

def dropNeighbors(df):
    """Drops neighboring predictors."""
    return df.drop(['N','E','S','W','NE','NW','SE','SW'], axis = 1)

def xy_toCoords(coords, ds):
    """input Cartesian x and y and return (lat,lon) from Cordex"""
    (x_c, y_c) = coords
    step = 50000
    #round to nearest whole step
    x_c, y_c = round(x_c/step)*step, round(y_c/step)*step
    lat = float(ds.sel(x=x_c, y=y_c)['lat'].values)
    lon = float(ds.sel(x=x_c, y=y_c)['lon'].values)
    return (lat,lon)
    
def coords_toXY(coords):
    """Finds the cell that corresponds to given coordinates."""
    (lat,lon) = coords
    offset = (3676573.4012608817, 3460579.79798922)
    #check ellps and datum values 
    lambert=pyproj.Proj("+proj=lcc +lat_1=35 +lat_2=60 +lat_0=46 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 + units=m +no_defs")
    x,y = lambert(lon, lat)
    #adjusts to center (0,0)
    return (x+offset[0], y+offset[1])

def check(coords, df):
    """input (x,y) and check to see if it matches projection"""
    (x,y) = coords
    latlon = xy_toCoords(coords, lower)
    est = coords_toXY(latlon)
    print("Input: (x,y)=" + str(coords))
    print("Coordinates: (lat, lon) = " + str(latlon))
    print("Estimated XY: (x,y) = " + str(est))

def getObsCellCoords(coords, df):
    """finds recorded lat and lon for cell containing given point"""
    if coords[0] < (np.min(df.latitude.values) - (1/16)) or coords[0] > (np.max(df.latitude.values) + (1/16)):
        return None
    if coords[1] < (np.min(df.longitude.values) - (1/16)) or coords[1] > (np.max(df.longitude.values) + (1/16)):
        return None
    lat = df.sel(bound=0).sel(latitude = coords[0], longitude = coords[1], method = "nearest")['latitude'].values
    lon = df.sel(bound=0).sel(latitude = coords[0], longitude = coords[1], method = "nearest")['longitude'].values
    return (float(lat), float(lon))

def getObsCell(coords, df):
    """Returns data for cell containing given point"""
    latlon = getObsCellCoords(coords, df)
    if latlon is None:
        return None
    return df.sel(bound=0).sel(latitude = latlon[0], longitude = latlon[1])
    
def getPr(coords,date, df, data, shift = (0, 0)):
    """return percipitation value for given parameters"""
    if data in ["pred", "p"]:
        return float(getPredCell(coords, df, shift).sel(time=date)['pr'].values)
    return float(getObsCell(coords, df).sel(time=date)['Prcp'].values)

def getPredCell(coords, df, shift = (0,0)):
    """Get to RCM cell from (lat, lon) coordinates"""
    step = 50000
    (x,y) = coords_toXY(coords)
    x,y = x+shift[0], y+shift[1]
    if( x<np.min(df['x']) or x>np.max(df['x']) ): 
        x=None
    if( y<np.min(df['y']) or y>np.max(df['y']) ):
        y=None
    lower_bounds = ( (x//step)*step, (y//step)*step )
    return df.sel(bnds=0).sel(x = lower_bounds[0], y = lower_bounds[1])

def avgPr(coords, df, data="pred"):
    """Returns average precipitation"""
    if data in ["pred", "p"]:
        return float(getPredCell(coords, df)['pr'].mean(dim="time").values)
    return float(getObsCell(coords, df)['Prcp'].mean(dim="time").values)

def printCompare(coords, obs_df, cordex_df):
    """Compares precipitation observed and predicted at given gridcell and date."""
    #observed precip is in mm/d = millimeters per day, but predicted precip is in kg m-2 s-1 = mm/s
    p_pr, o_pr = avgPr(coords, cordex_df, data="p"), avgPr(coords, obs_df, data="o")
    p_pr*=86400 # 24*3600 = 86400 s/d
    print("Predicted (mm/day): " + str(p_pr) + "\nObserved (mm/day): " + str(o_pr))

def pythag(a, b):
    """Helper method for weighting the cells by distance."""
    return np.sqrt(a*a + b*b)
 
def prep_data(mod_ds, obs_ds):
    """add N,NE,E,SE,S,SW,W,NW, and current RCM cells to obs_ds xarray object using mod_ds xarray object.
    mod_ds must be an xarray object (not yet unit corrected) and must be in the lon-lat grid."""
    #These labs and shifts can be altered to include more/fewer neighboring cells. 
    labs = ['N','NE','E','SE','S','SW','W','NW',"current"]
    shifts = [(.5,0),(.5,.5),(0,.5),(-.5,.5),(-.5,0),(-.5,-.5),(0,-.5),(.5,.5),(0,0)]
    cells = zip(labs,shifts)
    for lab, shift in cells:
        #add a new variable for a neighboring cell
        obs_ds = add_prCell(mod_ds,obs_ds,lab,shift)
    return obs_ds

def prep_data24(mod_ds, obs_ds):
    """add 24 neighbprs and current RCM cells to obs_ds xarray object us\
ing mod_ds xarray object.                                                           
    mod_ds must be an xarray object and must be in the lon\
-lat grid."""
    #These labs and shifts can be altered to include more/fewer neighboring cells. 
    labs = list('abcdefghijklmnopqrstuvwxy')
    #cells go left to right, top to bottom
    shifts = [(x/2-1,-(y/2-1)) for y in range(5) for x in range(5)]
    cells = zip(labs,shifts)
    for lab, shift in cells:
        #add a new variable for a neighboring cell                                  
        obs_ds = add_prCell(mod_ds,obs_ds,lab,shift)
    return obs_ds


def add_prCell(mod_ds, obs_ds, var_name, shift=(0,0)):
    """Adds a new column to the observed xarray dataset with a Cordex cell. 
    Returns the altered observed dataset. mod_ds is Cordex xarray object (lat-lon grid),
    and obs_ds is observed xarray object to be altered. var_name is the name of the new column in obs_ds."""
    #creates a new structure with the appropriate Cordex cells
    new = mod_ds.sel(lat=obs_ds['latitude']-shift[0], lon=obs_ds['longitude']-shift[1], time=obs_ds['time'], method="nearest")
    new = new.drop('lon').drop('lat')
    #adding 'new' to obs_ds dataset
    obs_ds[var_name] = ({'time':'time','latitude':'latitude','longitude':'longitude'},new.orog)
    obs_ds[var_name] = 24*3600*obs_ds[var_name] #converting to mm/day
    return obs_ds

def add_elevCell(mod_ds, obs_ds, var_name, shift=(0,0)):
    """Adds a new column to the observed xarray dataset with an RCM cell.                                                 
    Returns the altered observed dataset. mod_ds is Cordex xarray object (lat-lon grid),                                    
    and obs_ds is observed xarray object to be altered. var_name is the name of the new column in obs_ds."""
    #creates a new structure with the appropriate RCM cells                
    new = mod_ds.sel(lat=obs_ds['latitude']-shift[0], lon=obs_ds['longitude']-shift[1], method="nearest")
    new = new.drop('lon').drop('lat')
    #adding 'new' to obs_ds dataset                                                                                        
    obs_ds[var_name] = ({'latitude':'latitude','longitude':'longitude'},new.orog)
    return obs_ds

def evenOdd(ds):
    """input xarray dataset
    returns even and odd year datasets"""
    dcopy = ds.copy()
    dcopy['time-copy'] = dcopy['time']
    #classify years as even or odd
    dcopy['time'] =  pd.DatetimeIndex(dcopy.time.values).year%2 == 0
    even = dcopy.sel(time = True)
    odd = dcopy.sel(time = False)
    even['time'],odd['time'] = even['time-copy'],odd['time-copy']
    return even, odd
