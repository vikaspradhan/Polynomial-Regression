#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#splitting the dataset into training set and test set

"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
"""
#feature scaling

#fittig linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizing the linear regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear regrssion)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#visualizing the Polynomial Regression results(degree=2)
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#visualizing the Polynomial Regression results(degree=3)
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#visualizing the Polynomial Regression results(degree=4)
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#visualizing the Polynomial Regression results(degree=4 and resolution points=100)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with linear regression
lin_reg.predict([[6.5]])

#predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))





