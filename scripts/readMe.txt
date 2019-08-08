This folder contains various Python scripts and a few "modules" of functions

Modules:

     DataFns.py - various data-related functions that can be loaded into the other scripts as a module. Look at the __all__ at the top of the script to see all the functions that are available. Most of these aren't that useful, but I kept all of them. 

     StatFns.py - various other functions that can also be lodaded into the other scripts as a module. 

Folders:

     Cellwise - Contains various scripts for doing cellwise random forests for probability and for intensity of precipitation. Also contains scripts for predictions. 

     NeuralNet - Contains scripts for running a U-net on the data.

     OLS - Contains various scripts for doing cellwise linear and logistic regressions as well as scripts for predicting new values from saved models.

     SingleRF - scripts for running a single random forest over the entire domain (as opposed to cellwise random forests). This also includes a script for a grid hyperparameter search.	
