In this folder are scripts for cellwise random forests:

prep.py  - This script 'preps' data for later analysis by getting the RCM data on a finer grid and by adding the neighboring cells as variables (necessary for running the random forests).

class_cw_savemodels.py - This trains a separate random forest on each fine grid cell that predicts probability of precipitation. The models are pickled and saved individually.

saveModels_validation.py - This trains individual models for each cell for predicting precipiation values.

predict_cellwiseRF.py - This makes predictions for precip based on the input data and on the saved models. Run the same script for probability and intenisty predictions (the pickled models should be saved in different locations).

knit.py - This script combines all the chunks of predictions for cells into a single dataframe, which it saves as a netCDF output file. 
