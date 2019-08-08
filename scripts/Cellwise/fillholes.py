"""
A file to fill any areas that did not get a pickled model.
I'm not sure what's causing these holes (maybe a conflict while trying to write to the folder when the bash script runs in parallel?

This is a quick fix. Ideally I would figure out what's causing the original conflict. This script also isn't parallelized, and that 
would slow it down if there are a lot of models that the original run is failing to create.

K. Wheelan 7.17.19
"""

import pickle
import os
import sys

script=sys.argv[1] #script to run to create the models where they're missing
modelFolder=sys.argv[2] #where to save the models
sourceData=sys.argv[3] #training data location

n_cells = 40*80 #change if area changes

for cell in range(n_cells):
    #see if a model exists for each cell and if not, re-run the model
    filename = modelFolder+"/"+str(cell)+'_RF.sav'
    try:
        pickle.load(open(filename, 'rb'))
    except:
        os.system('python '+script+'.py '+modelFolder + " " + sourceData + " " + str(cell) + " "+str(cell+1))
