#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\21st\RANDOM FOREST\Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Split the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""


'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''


#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=129, criterion="absolute_error", min_samples_split=4)
regressor.fit(x, y)


#Predicting a new result
y_pred = regressor.predict([[6.5]])


#Visualising the Random Forest Regression results (higher resolution)one
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
