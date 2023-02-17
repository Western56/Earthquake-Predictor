"""
Shohei Maeda
tests.py

Source code used to run tests for the earthquake data set and run the regression
tests needed to determine if there is some correlation among the data.


"""

#Importing all necessary librarys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error, median_absolute_error
'exec(%matplotlib inline)'

pd.set_option('display.max_columns', 20)
eq_data = pd.read_csv('Japan earthquakes 2001 - 2018.csv')

#Check the data

print(eq_data.head())
print(eq_data.info())

#Set up the training sets for each case, analyzing 4 columns to use

depth_x= eq_data[['depth']]
lat_x= eq_data[['latitude']]
long_x= eq_data[['longitude']]
mag_y= eq_data[['mag']]

#Setting up the training sets using the built-in train_test_split method
x_train, x_test, y_train, y_test = train_test_split(
    long_x,mag_y, test_size = 0.2, shuffle = True)


#Creating an object of each of the Regression types
lin_reg = linear_model.LinearRegression()
lasso_reg = linear_model.Lasso()
enet_reg = linear_model.ElasticNet()

#Training the data sets
lin_reg.fit(x_train, y_train)

lasso_reg.fit(x_train, y_train)

enet_reg.fit(x_train, y_train)

#Creating the predictions for each of the tests
lin_y = lin_reg.predict(x_test)
las_test_y = lasso_reg.predict(x_test)
e_test_y= enet_reg.predict(x_test)


#Ideal value is 0
print("MAE ",mean_absolute_error(y_test, e_test_y))
#Ideal value is 0
print("MSE ", mean_squared_error(y_test, e_test_y))
#Ideal value is 1
print("R2 ",r2_score(y_test, e_test_y))
#Ideal is 1
print("Variance ",explained_variance_score(y_test, e_test_y))

#Ideal is 0
print("ME ", max_error(y_test, e_test_y))

#Ideal is 0
print("MedianAE " ,median_absolute_error(y_test, e_test_y))




