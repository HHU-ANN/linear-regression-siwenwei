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
    a=math.exp(-4.5)
    #0.018189888888888888882611
    weight=np.dot(np.linalg.inv(np.dot(X.T,X)+a*np.eye(X.shape[1])),np.dot(X.T,y))
    return data@weight






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

features = np.array([
    [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
    [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
    [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
    [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
    [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
    [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
    [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
    [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
    [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
    [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
    ])
print(ridge(features))
labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])
print(labels)
print(lasso(features))
print(ridge(features)-labels)
print(lasso(features)-labels)




