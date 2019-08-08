#!/bin/bash -l                                                                                                                                                                       
#SBATCH --job-name=p_class                                                                                                                                                           
#SBATCH --account=p48500028                                                                                                                                                          
#SBATCH --ntasks=20                                                                                                                                                                  
#SBATCH --cpus-per-task=8                                                                                                                                                            
#SBATCH --time=02:00:00                                                                                                                                                              
#SBATCH --output=p_class.out.%j                                                                                                                                                      

#A script to 'prep' the data

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load python
ncar_pylib

cd /glade/u/home/kwheelan/scripts/Cellwise

start=2070
end=2099

for yr in $(seq $start 1 $end)
do
    python prep.py $yr &
done

