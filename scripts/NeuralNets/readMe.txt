Unet0.py - A file that builds a U-Net to predict precipitation. You can toggle a few of the comments to include elevation as an input or not. Writes predictions using training data into a netCDF file at the end.

Unet1.py - A second script to build a U-Net. This one includes some dense layers at the end (although this does not seem to improve the prediction quality at all (in fact, it looks worse). This file also writes predictions for the training data as a netCDF.

Unet_predict.py - A script to run new data (validation data) through the UNet to get predictions. It writes these predictions to a file.
