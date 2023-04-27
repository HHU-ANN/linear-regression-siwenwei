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
    a=0.035
    weight=np.dot(np.linalg.inv(np.dot(X.T,X)+a*np.eye(X.shape[1])),np.dot(X.T,y))

    return weight@data






def lasso(data):
    X,y=read_data()
    #print(X.shape[1])
    a=0.6
    w=np.zeros(X.shape[1])
    #print(w)
    b=0
    learning_rate=0.0000003
    num_train = X.shape[0]
    num_feature = X.shape[1]

    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(a * abs(w))
    dw = np.dot(X.T, (y_hat - y)) / num_train + a * np.sign(w)
    db = np.sum((y_hat - y)) / num_train
    loss_list=[]
    epochs=10
    params = {
        'w': w,
        'b': b
    }
    for i in range(1,epochs):
        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(a * abs(w))
        dw = np.dot(X.T, (y_hat - y)) / num_train + a * np.sign(w)
        db = np.sum((y_hat - y)) / num_train
        w+=-learning_rate*dw
        b+=-learning_rate*db
        loss_list.append(loss)
        params={
                'w':w,
                'b':b
            }

    w=params['w']
    b=params['b']
    y_pred = np.dot(data, w)+b
    #print(y_pred.shape)
    return y_pred




def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y




