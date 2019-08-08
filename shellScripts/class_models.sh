#!/bin/bash -l
#SBATCH --job-name=classifier
#SBATCH --account=p48500028
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=modelGeneration.output.%j                                              
#SBATCH --mem=200G

#A batch script to run a cellwise random forest for predicting probability

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

#The location of the python script that predicts cellwise RF classifier
cd /glade/u/home/kwheelan/scripts/Cellwise

SCRIPT=class_cw_savemodels.py #the python script to call
SCRIPTNAME=class_cw_savemodels #same scripts
FOLDER=/glade/scratch/kwheelan/models/cellwise_classifier/run2 #The folder in which to save all the models for each cell.
mkdir -p $FOLDER
SOURCEDATA=/glade/scratch/kwheelan/datasets/PNW_1979-2010.nc #The training data location

time(
for i in $(seq 1 200 3001)
do
    python $SCRIPT $FOLDER $SOURCEDATA $i ${i+200} &
done
wait
#This fills in any missing models. This should probably be paralellized...
python fillholes.py $SCRIPTNAME $FOLDER $SOURCEDATA
)
