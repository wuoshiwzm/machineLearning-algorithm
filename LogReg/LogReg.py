import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import os

path = 'data' + os.sep + 'LogiReg_data.txt'

pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
pdData.head()

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] != 1]

fig, ax = plt.subplots(figsize=(10, 5))  # get area for plots
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')


# targe: make classification machine
# set threshold, then do classify by the threshold(Admitted or Not)  (set 0.5 this time)

# module need to be done:
# sigmoid function: map input to a  probability value
# model:return the forecast value
# cost: calculate cost according to the parameters
# gradient: calculate the direction of gradient of every parameters(theta0 theta1 theta2)
# descent:update parameter by gradient decent (迭代计算参数值)
# accuracy: the accuracy of calculation

# sigmoid function : 1/(1+exp(-z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# model:
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

# 数据预处理
pdData.insert(0, 'Ones', 1) # in a try / except structure so as not to return an error if the block si executed several times


# set X (training data) and y (target variable)
orig_data = pdData.as_matrix() # convert the Pandas representation of the data to an array useful for further computations
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1]
y = orig_data[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
#X = np.matrix(X.values)
#y = np.matrix(data.iloc[:,3:4].values) #np.array(y.values)
theta = np.zeros([1, 3])

pdData.insert(0, 'Ones', 1) # in a try / except structure so as not to return an error if the block si executed several times

#损失函数

def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))