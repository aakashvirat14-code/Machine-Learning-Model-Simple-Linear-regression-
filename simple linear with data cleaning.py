#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data set
data=pd.read_csv("C:\imagecon\Dataset\linear_regression_100rows_with_nulls.csv")
data

#data inspection
data.head()
data.tail()
data.shape
data.describe()
data.info()

#data cleaning
#checking null values 
data.isnull().sum()

#replace null values
#age
data["Age"]=data["Age"].fillna(data["Age"].mean()).astype(int)

#Experience
data["Experience"]=data["Experience"].fillna(data["Experience"].median()).astype(int)
#salary
data["Salary"]=data["Salary"].fillna(data["Salary"].median()).astype(int)
#project
data.drop("Projects",axis=1,inplace=True)
#DATA Split
x=data.iloc[:,[0]].values
y=data.iloc[:,-1].values

plt.scatter(x,y)


#taking simple Linear regression
#splitting data
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.30,random_state=0)
# import algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predict result
y_pred=regressor.predict(x_test)
y_pred
#predecting score for both train and test
score=regressor.score(x_train,y_train)
score
score=regressor.score(x_test,y_test)
score
#visualize for train data
plt.figure(dpi=500)
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.xlabel("age")
plt.ylabel("salary")
plt.title("age vs salary")
plt.show()
#visulazise for test data
plt.figure(dpi=500)
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_test,y_pred,color="green")
plt.xlabel("age")
plt.ylabel("salary")
plt.title("age vs salary")
plt.show()

new_data = [[30],[27],[34]]

prediction = regressor.predict(new_data)

print("Predicted Salary:", prediction[0])