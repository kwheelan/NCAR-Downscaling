Katrina Wheelan, August 2019

This is a repository with the scripts from the 2019 summer SIParCS project on statistical downscaling using machine learning. The three models (linear, random forest, and convolutional neural network) each have the scripts used to train the models and make predictions. All models are training/evaluated using WRF initialized with Era-Interim and Maurer observed precipitation data from 1980-2010. 

Folders in this repository:

  *scripts* - The folder containing the Python scripts that fit the models and make predictions.
  
  *shellScripts* - The bash scripts to run on Casper to fit the models and make predictions. These shell scripts call the python scripts in the scripts folder and run them in parallel.
  
---------------------------------------------------------------
  
Finding the training data/predictions:
   
   The data files are mostly saved as netCDF and are too large to upload to GitHub. All the data can be replicated using the scripts in this repository. Here are the paths to the data saved in GLADE storage:
   
   *Original data* -
        Maurer (1980-1999) - /glade/p/ral/hap/common_data/Maurer_w_MX_CA/pr
        Maurer (2000-2010) - /glade/p/ral/hap/common_data/Maurer_met_full
        elevation data for Maurer - /glade/p/ral/hap/common_data/geophysical/maurer.125_topo/ldas_official.dem.xyz
        WRF (1980-2010) - /glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44i/eval/pr*
        WRF historical (1976-2005) - /glade/collections/cdg/work/cordex/esgf/wrf/mpi-esm-lr/nam-44i/hist/day/pr*
        WRF future (2070-2099) - /glade/collections/cdg/work/cordex/esgf/wrf/mpi-esm-lr/nam-44i/rcp85/day/pr*
        elevation data for WRF - /glade/collections/cdg/work/cordex/esgf/wrf/era-int/nam-44/eval/fx/orog*
        
   *Prepped data* - (Data for only the Pacific Northwest, unit adjusted, and with the proper input variables)
        /glade/work/kwheelan/datasets/1979-2010.nc - 
        
   
----------------------------------------------------------------
  
Data References:

       Mearns, L.O., et al., 2017: The NA-CORDEX dataset, version 1.0. NCAR Climate Data Gateway, Boulder CO, accessed 3 June 2019, https://doi.org/10.5065/D6SJ1JCH.

      Maurer, E.P., A.W. Wood, J.C. Adam, D.P. Lettenmaier, and B. Nijssen, 2002, A Long-Term Hydrologically-Based Data Set of Land Surface Fluxes and States for the Conterminous United States, J. Climate 15(22), 3237-3251.

  
