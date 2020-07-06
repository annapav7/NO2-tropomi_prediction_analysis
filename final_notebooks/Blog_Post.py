#!/usr/bin/env python
# coding: utf-8

# <img style="float: left;" src="33.JPG" width="150" height="150" />
# 
# 
# #  NO2 Prediction by using Machine Learning Regression Analyses in Google EE

# # Contents:

# ## Nitrogen Dioxide (NO2) air pollution.
# The World Health Organization estimates that air pollution kills 4.2 million people every year.  
# The main effect of breathing in raised levels of NO2 is the increased likelihood of respiratory problems. NO2 inflames the lining of the lungs, and it can reduce immunity to lung infections.
# 
# Even there are connections between respiratory deceases / also exposure to viruses and more deadly cases and level of NO2 pollution in our atmosphere.
# 
# ##### Sources of NO2:
# The rapid population growth, 
# The fast urbanization: 
#     - Industrial facilities
#     - Fossil fuels (coal, oil and gas)
#     - The increase of transportation – 80 %.
# 
# 
# The affect air pollution (NO2): 
# population health, and global warming.
# 
# <img style="float: left;" src="4.jpg" width="1250" height="1250" />
# 
# Fig_1. Pollution in Industrial cities

# ### Workflow:
# 

# <img style="float: center;" src="WorkFlow.jpg" width="1500" height="1500" />
# Fig_2. WorkFlow of this project

# ### Study Area / Input Data:
# Study area for research Project: Los Angeles,  CA.
# Data: Image collection of Landsat 8 for 2014 – 2019 years, Sentinel 5-P (TROPOMI) 2018 -2019.
# <img style="float: center;" src="data.JPG" width="1500" height="1500" />
# Fig_3. Los Angeles,  CA / Landsat 8  / Sentinel 5-P (TROPOMI)

# ### ML Regression Analysis uses in this Project
# 
# The machine learning toolbox includes several linear and non-linear supervised learners, predicting either numeric outputs (regressors) or nominal outputs (classifiers).
# 
# ##### Classification Workflow
# 1. Build
# 2. Train
# 3. Apply
# 4. Assessment
# 
# ##### Classification Workflow
# var training = image.sample(region, scale)
# var classifier = ee.Classifier.randomForest().train(training)
# var result = image.classify(classifier)
# var predictor = classifier.setOutputMode(Regression)
# var confusionMatrix = classifier.confusionMatrix()
# var accuracy = confusionMatrix.accuracy()
# 
# ##### Classifiers
# Classification and regression trees.
# 
# Linear Regression: Random Forest - Random Decision Forest, SVM - Support Vector Machine
# 
# ##### Classifier Output Mode
# classifier.setOutputMode(mode):
# 
# Classification - Discrete input/output classes
# 
# Regression - Continuous valued output
# 
# Probability - binary classifiers only
# 
# // _______________________________
# Support Regression: Random Forest, SVM
# // _______________________________
# Support Probability: Cart, NaiveBayes, IKPamir*, Pegasos, SVM, Perceptron

# ##  Methods in this project
# ### Random Sampling
# Training DataSEt created by using Random Points. Random Points were collected by areas selected in 9 different levels of NO2 by TROPOMI satellite imagery 2019.
# 
# var training = image.sample(region, numPixels)
# 
# var training = image.sample(ee.FeatureCollection.RandomPoints(numPixels))
# 
# <img style="float: center;" src="tropomi.jpg" width="1500" height="1500" />
# Fig_4. TROPOMI imagery 2019
# 
# 
# <img style="float: center;" src="Random_Points.JPG" width="1500" height="1500" />
# 
# 
# <img style="float: center;" src="tropomi_points.jpg" width="1500" height="1500" />
# Fig_5. Random points collection from TROPOMI 2019

# ### Supervised Classification
# 
# In project was used Random Forest method.
# 
# // ___ https://code.earthengine.google.com/?accept_repo=EE101-B&scriptPath=users%2Fpavlenkoanna2011%2Fex1%3AClassifications%2FRandom_Forest
# <img style="float: center;" src="Prediction.jpg" width="1500" height="1500" />
# Fig_6. Random Forest Classification 2019
# 
# We can assess the accuracy of the trained classifier using a confusionMatrix.
# 
# // Get a confusion matrix representing resubstitution accuracy.
# 
# print('RF error matrix: ', classifier.confusionMatrix());
# <img style="float: center;" src="Error_matrix.JPG" width="450" height="450" />
# 
# 
# print('RF accuracy: ', classifier.confusionMatrix().accuracy());
# <img style="float: center;" src="valid_matrix.JPG" width="450" height="450" />
# 
# 

# ## Two ways of Predict continuous values: Across Space & Over Time:
# ### Regression: Predict continuous values output Across Space
# 
# Contexts of Linear Regression in GEE:
# <img style="float: center;" src="across_space.jpg" width="1500" height="1500" />
# Fig_7.  Regression Across Space 2018
# 
# // PREDICTION FOR 2018 YEAR:
# 
# var predict_2018 = landsat8_2018.select(bands).classify(predictor_all_data);
# 
# 
# // NDVI_2018
# 
# var ndvi_2018 = landsat8_2018.normalizedDifference(['B5', 'B4']);
# 
# 
# // ___https://code.earthengine.google.com/?accept_repo=EE101-B&scriptPath=users%2Fpavlenkoanna2011%2Fex1%3ANDVI_Landsat%2FNDVI_Landsat_2018
# 
# <img style="float: center;" src="ndvi.JPG" width="1500" height="1500" />
# Fig_8. NDVI and Regression 2018
# 
# #### Linear Regression 2018
# // https://code.earthengine.google.com/?accept_repo=EE101-B&scriptPath=users%2Fpavlenkoanna2011%2Fex1%3AClassifications%2FRandom_Forest
# 
# <img style="float: center;" src="ee-chart_2018.png" width="1500" height="1500" />
# Fig_9.  Regression 2018
# 
# <img style="float: center;" src="coff_2018_1.jpg" width="250" height="250" />
# Fig_10. NDVI and Regression 2018
# 

# ### Model Accuracy:
# 
# For 2018 there are TROPOMI data exist, also Predicted value NO2 calculated from Predictor 2019. It is possible to evaluate quality of Predictor.
# <img style="float: center;" src="difference_raster.jpg" width="1500" height="1500" />
# Fig_11.  Difference Raster 2018 - Difference between actual value NO2 and Predicted value NO2

# #### Predictions 2018 -2015 years:
# // 2018 - https://code.earthengine.google.co.in/?scriptPath=users%2Fpavlenkoanna2011%2Fex1%3APredictions%2FPrediction_2018
# 
# // 2017 - https://code.earthengine.google.co.in/?scriptPath=users%2Fpavlenkoanna2011%2Fex1%3APredictions%2FPrediction_2017
# 
# // 2016 - https://code.earthengine.google.co.in/?scriptPath=users%2Fpavlenkoanna2011%2Fex1%3APredictions%2FPrediction_2016
# 
# // 2015 - https://code.earthengine.google.co.in/?scriptPath=users%2Fpavlenkoanna2011%2Fex1%3APredictions%2FPrediction_2015
# 
# <img style="float: center;" src="predictions.jpg" width="1500" height="1500" />
# Fig_12.  NO2 Predictions for 2018 -2015 years

# ### Predict continuous values output Across Time (Random Forest REGRESSION):
# //  NO2 PREDICTIONS for 2014 - 2018 years:
# // Landsat 2018:
# 
# var predict_2018 = landsat8_2018.select(bands).classify(predictor_all_data);
# 
# var predict_2017 = landsat8_2017.select(bands).classify(predictor_all_data);
# 
# var predict_2016 = landsat8_2016.select(bands).classify(predictor_all_data);
# 
# var predict_2015 = landsat8_2015.select(bands).classify(predictor_all_data);
# 
# var predict_2014 = landsat8_2014.select(bands).classify(predictor_all_data);
# 
# <img style="float: center;" src="over_time.jpg" width="1500" height="1500" />
# Fig_13.  Context of Linear Regression in GEE - over Time

# 
# <img style="float: center;" src="chart_over_time.jpg" width="1500" height="1500" />
# Fig_14. Predicted NO2 level Over Time (2018 - 2014)
# 

# <img style="float: center;" src="power_plants.jpg" width="1500" height="1500" />
# Fig_15. Power Plants in GEE
# 

# <img style="float: center;" src="point_prediction.jpg" width="1500" height="1500" />
# Fig_16. Power Plants Predictions in GEE

# # Result
# The goal to create data of NO2 for past years (2018, 2017, 2016, 2015) by using data for 2019 (TROPOMI and Landsat 8) was reached.

# # Summary / Conclusions:
# To improve accuracy of Regression / Prediction Model combination from multiple Machine Learning Algorithms or Multiple Predictions several times from the same Algorithm to make more accurate predictions - Ensemble Model.
# 

# # References:
# 1. Bert Brunekreef, Stephen T Holgate "Air pollution and health". THE LANCET • Vol 360 • October 19, 2002 • www.thelancet.com
# 
# 2. M.L. Brusseau, A.D. Matthias, A.C. Comrie and S.A. Musil "Atmospheric Pollution". Environmental and Pollution Science. https://doi.org/10.1016/B978-0-12-814719-1.00017-3 Copyright © 2019 Elsevier Inc. All rights reserved. 
# 
# 3. M.L. Brusseau "Physical Processes Affecting Contaminant Transport and Fate". Environmental and Pollution Science. https://doi.org/10.1016/B978-0-12-814719-1.00007-0 Copyright © 2019 Elsevier Inc. All rights reserved. 103.
# 
# 4. Gustavo Camps-Valls "Machine Learning in Remote Sensing Data Processing" Conference Paper · September 2009 DOI: 10.1109/MLSP.2009.5306233.
# 
# 5. LEO BREIMAN "Random Forests". Machine Learning, 45, 5–32, 2001 c 2001 Kluwer Academic Publishers. Manufactured in The Netherlands.
# 
# 6. Robert H. Shumway • David S. Stoffer "Third edition Time Series Analysis and Its Applications". 2011
# 
# 7. "Sentinel-5 precursor/TROPOMI Level 2 Product User Manual Nitrogendioxide". 2017

# In[ ]:




