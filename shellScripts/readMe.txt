Files in this directory:

prep.sh - 'preps' data for analysis including unit correction and and getting all the neighboring cells as inputs. 

predict.sh - predicts cellwise random forest precipitation using saved models. (this should be run once for the classifier and once for the intensity random forests.)

class_models.sh - This fits and saves random forest models that predict the probability of precipiation in each cell.

generateModels.sh - This fits and saves frandom forest models for each individual cell to predict precipiation intensity.

predictOLS.sh - this script predicts precipitation intensity  

fitUnet.sh - 

to add:

fitOLS.sh
