# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
import math

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np



def ridge(data):
    #1.读取数据
    X,y=read_data()
    #2.根据算法求解weight
    #Xw=y
    a=0.5
    add11=X[:,4]*X[:,5]
    add12=X[:,2]*X[:,3]
    add13=X[:,0]*X[:,1]
    X=np.column_stack((X,add11,add12,add13))
    X=np.concatenate((np.ones((404, 1)), X), axis=1)
    weight=np.dot(np.linalg.inv(np.dot(X.T,X)+a*np.eye(X.shape[1])),np.dot(X.T,y))
    add21=data[:,4]*data[:,5]
    add22=data[:,2]*data[:,3]
    add23=data[:,0]*data[:,1]
    data=np.column_stack((data,add21,add22,add23))
    data=np.concatenate((np.ones((10,1)),data),axis=1)
    return data@weight






def lasso(data):
    X,y=read_data()

    add13 = X[:, 0] * X[:, 3]/10
    add14 = X[:,1] * X[:,2]/10
    add15 = X[:,1] * X[:,3]/10
    add16 = X[:,1] * X[:,4]/10
    add17 = X[:,2] * X[:,3]/10
    add18 = X[:,2] * X[:,4]/10
    add19 = X[:,3] * X[:,5]/10
    X=np.column_stack((X,add13,add14,add15,add16,add17,add18,add19))

    a=0.5
    w=np.zeros(X.shape[1])


    learning_rate=1e-8
    num_train = X.shape[0]


    add23 = data[:,0] * data[:,3]/10
    add24 = data[:, 1] * data[:, 2]/10
    add25 = data[:, 1] * data[:, 3]/10
    add26 = data[:, 1] * data[:, 4]/10
    add27 = data[:, 2] * data[:, 3]/10
    add28 = data[:, 2] * data[:, 4]/10
    add29 = data[:, 3] * data[:, 5]/10

    data=np.column_stack((data,add23,add24,add25,add26,add27,add28,add29))


    epochs=500000

    for i in range(1,epochs):
        grad = (np.dot(X.T, np.dot(X, w) - y) / num_train) + a * np.sign(w)
        w = w - learning_rate * grad
    y_pred = np.dot(data, w)
    return y_pred




def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y





