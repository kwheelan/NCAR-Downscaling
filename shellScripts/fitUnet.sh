#!/bin/bash -l                                                                  
#SBATCH --job-name=OpenMP_job                                                   
#SBATCH --account=p48500028                                                     
#SBATCH --ntasks=1                                                             
#SBATCH --cpus-per-task=8                                                       
#SBATCH --time=00:10:00                                                         
#SBATCH --output=Unet.%j                                     
#SBATCH --constraint=gpu                                                                                
#SBATCH --mem=300G

#A script to train the Unet and run it on teh training data

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

#Script folder location for the script that creates the Unet
cd /glade/u/home/kwheelan/scripts/NeuralNets 

NAME=/glade/scratch/kwheelan/models/nn/unet20.h5 #Name of the model filepath
LOCATION=/glade/scratch/kwheelan/datasets/predictions/unet/unet_train_preds20.nc #Where to save the predictions of the training data

time(
python Unet0.py $NAME $LOCATION
)
