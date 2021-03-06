#Regression Template

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
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""


#fittig Regression model to the dataset

#Predicting a new result
y_pred=regressor.predict([[6.5]])

#visualizing the Regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Regrssion model)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#visualizing the Regression results for higher resolution and smoother curve
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Regression regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()




