# Propensity Score
Propensity.py : This contains two functions. Train and Predict. 
Train will train model on new data and save the model using pickle.
Predict will predict the outcome using saved model.

smote.py: New data points are created using SMOTE to balance the categories.

gSearch: Grid search of LGBM model for hyperparameters tuning.

Pipelined.ipynb: Model analysis and comparisons to pick the best model among RandomForest, Adaboost, LGBM.
LGBM out performed which was used to create model on given data.
