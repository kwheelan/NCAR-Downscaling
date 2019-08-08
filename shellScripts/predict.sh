#!/bin/bash -l
#SBATCH --job-name=p_class
#SBATCH --account=p48500028
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=future_class.%j

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
module load python
ncar_pylib

#makes predictions using random forests.
#This should be run twice: once using the classifier model and once using the intensity random foests

cd /glade/u/home/kwheelan/scripts/Cellwise

SCRIPT=predict_cellwiseRF.py #script to run
MODELFOLDER=/glade/scratch/kwheelan/models/cellwise_RF/run6 #where the models are saved
DATALOCATION=/glade/scratch/kwheelan/datasets/cellwiseRFpreds/future-climate-run/future-intensity #where to save the final predictions

mkdir -p $DATALOCATION
PARTIALDATASPOT=/glade/scratch/kwheelan/datasets/partialData/future-intensity #where to save chunks of cellwise predictions

STARTYR=2070 #first year
ENDYR=2070 #last year
INC=1 #increment; ie. 1 means every year and 2 means every other year

for VALIDATIONYEAR in $(seq $STARTYR $INC $ENDYR)
do
    INPUTDATA=/glade/scratch/kwheelan/datasets/future_climate/PNW_${VALIDATIONYEAR}_prepped.nc ##where the prepped data is to predict with
    mkdir -p $PARTIALDATASPOT/yr$VALIDATIONYEAR

    for cell in $(seq 1 100 3101)
    do
	python $SCRIPT $MODELFOLDER $VALIDATIONYEAR $PARTIALDATASPOT $cell $((cell+100)) & #predict in chunks of 100 cells
    done
    wait
    python knit.py $DATALOCATION $VALIDATIONYEAR $INPUTDATA $PARTIALDATASPOT #put predictions together in a single file
    echo $VALIDATIONYEAR
done

echo done.
