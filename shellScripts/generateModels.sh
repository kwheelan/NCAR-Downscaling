#!/bin/bash -l
#SBATCH --job-name=OpenMP_job
#SBATCH --account=p48500028
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=modelGeneration.output.%j                                              
#SBATCH --mem=300G

#a script to fit and save random forest models
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

#script location
cd /glade/u/home/kwheelan/scripts/Cellwise

SCRIPT=saveModels_validation.py
SCRIPTNAME=saveModels_validation
MODELFOLDER=/glade/scratch/kwheelan/models/cellwise_RF/run7 #where to save the models
mkdir -p $MODELFOLDER
SOURCEDATA=/glade/scratch/kwheelan/datasets/PNW_1979-2010.nc #the data with which to fit the models

for yr in $(seq 1 200 3001)
do
    python $SCRIPT $MODELFOLDER $SOURCEDATA $yr ${yr+200} &
done
wait 

#redoing any missing models
python fillholes.py $SCRIPTNAME $MODELFOLDER $SOURCEDATA

