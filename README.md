# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the needed packages
2.Read the txt file using read_csv
3.Use numpy to find theta,x,y values
4.To visualize the data use plt.plot
```

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vidya Neela.M
RegisterNumber: 212221230120 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('student_scores - student_scores.csv')
df.head()
df.tail()

#checking for null values in dataset
df.isnull().sum()

#To calculate Gradient decent and Linear Descent
x=df.Hours
x.head()

y=df.Scores
y.head()

n=len(x)
m=0
c=0
L=0.001
loss=[]
for i in range(10000):
    ypred = m*x + c
    MSE = (1/n) * sum((ypred - y)*2)
    dm = (2/n) * sum(x*(ypred-y))
    dc = (2/n) * sum(ypred-y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
print(m,c)

#plotting Linear Regression graph
y_pred=m*x+c
plt.scatter(x,y,color="violet")
plt.plot(x,y_pred,color="purple")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study hours vs Scores")
plt.show()

#plotting Gradient Descent graph
plt.plot(loss, color="skyblue")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
```

## Output:

![o1](https://user-images.githubusercontent.com/94169318/204546435-d4df05b3-69a2-4dbe-bc88-9e261ff759cd.png)
![o2](https://user-images.githubusercontent.com/94169318/204546485-b0d870f0-139b-42bd-8ed2-00633599767a.png)

![o3a](https://user-images.githubusercontent.com/94169318/204546521-ee2f84cf-85c0-4e32-9a09-ec521480b157.png)

![o4](https://user-images.githubusercontent.com/94169318/204546553-ca9e2152-ef13-4838-9d2e-7e3f652b0eea.png)
![o5](https://user-images.githubusercontent.com/94169318/204546586-9984ef83-c9f3-4ffe-91f5-67570c89018c.png)

![o6](https://user-images.githubusercontent.com/94169318/204546621-41b54ad6-74a8-4bb4-9db1-8ce93d91b702.png)

![o7](https://user-images.githubusercontent.com/94169318/204546662-8245e7f8-0956-4030-aed6-6b6cc0d1c31c.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
