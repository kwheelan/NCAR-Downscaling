#!/bin/bash -l                                                                  
#SBATCH --job-name=OLS                                                   
#SBATCH --account=p48500028                                                     
#SBATCH --ntasks=1                                                             
#SBATCH --cpus-per-task=8                                                       
#SBATCH --time=00:10:00                                                         
#SBATCH --output=out.job.%j                                     
#SBATCH --mem=300G

#A script to predict precip intensity for some data based on existing models

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

#where the script is located
cd /glade/u/home/kwheelan/scripts/OLS

TESTDATA=/glade/scratch/kwheelan/datasets/PNW_oddYears.nc #data to predict using
BETAS=/glade/scratch/kwheelan/datasets/OLS/OLS_betas_nonzero.txt #where the coefficients are stored
LOCATION=/glade/scratch/kwheelan/datasets/OLS_odd_preds_nonzero.nc #where to save the predictions

time(
python predict_OLS.py $TESTDATA $BETAS $LOCATION
)
