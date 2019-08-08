Files to do cellwise linear predictions:

Logistic_cellwise.py - fits a logistic model to each cell in the grid and saves the coefficients as a csv file for later predictions.

OLS_cellwise.py - fits a linear model for each cell and saves the coefficients.

predict_logistic.py - predicts the probability of precipitation for some test data using the saved coefficents. 

predict_OLS = predicts the precipitation intensity for each cell in the test data using saved logistic coefficients. 
