# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

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
    a=0.03
    weight=np.dot(np.linalg.inv(np.dot(X.T,X)+a*np.eye(X.shape[1])),np.dot(X.T,y))
    return weight@data

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def lasso(data):
    X,y=read_data()
    a=0.03
    w=np.zeros((X.shape[1]),1)
    b=0
    learning_rate=0.5

    loss_list=[]
    epochs=5
    params = {
        'w': w,
        'b': b
    }
    for i in range(1,epochs):
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / X.shape[0] + np.sum(a * abs(w))
        dw = np.dot(X.T, (y_hat - y)) / X.shape[0] + a * np.vectorize(sign)(w)
        db = np.sum((y_hat - y)) / X.shape[0]
        w+=-learning_rate*dw
        b+=-learning_rate*db
        loss_list.append(loss)
        if i%5==0:
            params={
                'w':w,
                'b':b
            }
    w=params['w']
    b=params['b']
    y_pred = np.dot(X,w)+b
    return y_pred



def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y



