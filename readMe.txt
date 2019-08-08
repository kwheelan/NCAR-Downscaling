Katrina Wheelan, August 2019

This is a repository with the scripts and output data from the 2019 summer SIParCS project on statistical downscaling using machine learning. The three models (linear, random forest, and convolutional neural network) each have precipitation predictions as well as the scripts and data used to train the models. All models are training/evaluated using WRF initialized with Era-Interim and Maurer observed precipitation data from 1980-2010. 

Folders in this repository:

  *scripts* - The folder containing the Python scripts that fit the models and make predictions.
  
  *shellScripts* - The bash scripts to run on Casper to fit the models and make predictions. These shell scripts call the python scripts in the scripts folder and run them in parallel.
  
  *animations* - Several animations of the outputs of the predictions from the models.
  
  *input_data* - 'prepped' data used to train the models. All files are netCDFs.
  
  *FinalModelPreds* - Datasets containing the predictions for odd years from 1980-2010 from models trained using the even years in the same period. Predictions are saved as NetCDF files.
  
  
Data References:

       Mearns, L.O., et al., 2017: The NA-CORDEX dataset, version 1.0. NCAR Climate Data Gateway, Boulder CO, accessed 3 June 2019, https://doi.org/10.5065/D6SJ1JCH.

      Maurer, E.P., A.W. Wood, J.C. Adam, D.P. Lettenmaier, and B. Nijssen, 2002, A Long-Term Hydrologically-Based Data Set of Land Surface Fluxes and States for the Conterminous United States, J. Climate 15(22), 3237-3251.

  
