Files in this directory:

  prep.sh - 'preps' data for analysis including unit correction and and getting all the neighboring cells as inputs. 

  predict.sh - predicts cellwise random forest precipitation using saved models. (This should be run once for the classifier and once for the intensity random forests.)

  class_models.sh - This fits and saves random forest models that predict the probability of precipiation in each cell.

  generateModels.sh - This fits and saves random forest models for each individual cell to predict precipiation intensity.

  predictOLS.sh - This script predicts precipitation intensity using the linear model.

  fitUnet.sh - Trains the Unet using some input data and saves a dataset with the predictions for the training data.

to add:

  fitOLS.sh - editing and will add
