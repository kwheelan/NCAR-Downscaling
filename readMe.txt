Katrina Wheelan, August 2019

--------------------------------------------------------------

This is a repository with the scripts from the 2019 summer SIParCS project on statistical downscaling using machine learning. The three models (linear, random forest, and convolutional neural network) each have the scripts to train the models and to make predictions. All models are training/evaluated using WRF initialized with Era-Interim and Maurer observed precipitation data from 1980-2010. 

Folders in this repository:

  *scripts* - The folder containing the Python scripts that fit the models and make predictions.
  
  *shellScripts* - The bash scripts to run on Casper to fit the models and make predictions. These shell scripts call the python scripts in the scripts folder and run them in parallel.
  
  *notebooks* - This contains some Jupyter notebooks with some animations/plots, including a notebook to read in and combine the probability and intensity datasets.
  
---------------------------------------------------------------
  
Finding the training data/predictions:
   
   The data files are mostly saved as netCDFs and are too large to upload to GitHub. All the prediction data can be replicated using the scripts in this repository. Here are the paths to the data saved in GLADE storage:
   
   *Original data* -
        Maurer (1980-1999) - /glade/p/ral/hap/common_data/Maurer_w_MX_CA/pr
        Maurer (2000-2010) - /glade/p/ral/hap/common_data/Maurer_met_full
        elevation data for Maurer - /glade/p/ral/hap/common_data/geophysical/maurer.125_topo/ldas_official.dem.xyz
        WRF (1980-2010) - /glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44i/eval/pr*
        WRF historical (1976-2005) - /glade/collections/cdg/work/cordex/esgf/wrf/mpi-esm-lr/nam-44i/hist/day/pr*
        WRF future (2070-2099) - /glade/collections/cdg/work/cordex/esgf/wrf/mpi-esm-lr/nam-44i/rcp85/day/pr*
        elevation data for WRF - /glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44/eval/fx/orog*
        
   *Prepped data* - (Data for only the Pacific Northwest, unit adjusted, and with the proper input variables)
        WRF and Maurer data with 24 neighboring cells for Pacific Northwest only - /glade/work/kwheelan/datasets/1979-2010.nc
        Same as the above but with Maurer elevation data - /glade/work/kwheelan/datasets/PNW_w_elevation.nc
        Only odd years (for validation) - /glade/work/kwheelan/PNW_oddYears.nc
        Prepped WRF historical period (1976-2005) - /glade/work/kwheelan/PNW_past_prepped.nc
        Prepped WRF future period (2070-2099) - /glade/work/kwheelan/PNW_future_prepped.nc
        
        (Note that the columns labeled with letters represent RCM cells. There are 25 letters--a through y--that represent the 25 cells around the RCM cell containing the lat-lon point for a fine resolution obs cell. These cells are labeled left to right, top to bottom. So the column labeled 'm' is the RCM cell containing the obs cell, 'n' is the cell directly to the right, etc.)
        
        a | b | c | d | e
        f | g | h | i | j
        k | l |*m*| n | o
        p | q | r | s | t
        u | v | w | x | y
        
    *Prediction data* - 
        OLS predictions for odd years (static threshold = 0.5) - /glade/work/kwheelan/FinalModelsPreds/OLS_log_oddyrs.nc
        Same as above, but bias corrected - /glade/work/kwheelan/FinalModelsPreds/OLS_log_oddyrs_scaled.nc
        Cellwise random forest predictions (static threshold = 0.5) - /glade/work/kwheelan/FinalModelsPreds/Cellwise_RF_allOddYears.nc
        Unet odd year predictions - /glade/work/kwheelan/FinalModelsPreds/Unet_oddyr_preds.nc
        Historical WRF predictions using cellwise random forests (1976-2005) - /glade/work/kwheelan/FinalModelsPreds/historical-WRF.nc
        Future WRF predictions using cellwise random forests (2070-2099) - /glade/work/kwheelan/FinalModelsPreds/future-WRF.nc
        Annual domain-wide averages for each model and each odd year (1980-2010) - /glade/work/kwheelan/FinalModelsPreds/YearlyAvgs.csv
   
----------------------------------------------------------------
  
Data References:

       Mearns, L.O., et al., 2017: The NA-CORDEX dataset, version 1.0. NCAR Climate Data Gateway, Boulder CO, accessed 3 June 2019, https://doi.org/10.5065/D6SJ1JCH.

      Maurer, E.P., A.W. Wood, J.C. Adam, D.P. Lettenmaier, and B. Nijssen, 2002, A Long-Term Hydrologically-Based Data Set of Land Surface Fluxes and States for the Conterminous United States, J. Climate 15 (22), 3237-3251.

  
