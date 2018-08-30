# coding:utf-8

"""
@author:zb
@date:2018.8 
"""


import math
import numpy as np 

def basic_sigmoid(x):
    return 1./(1+math.exp(-x))

def sigmoid(x):
    return 1./(1+np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))

def im2vec(img):
    # convert rgb image to a column vector
    return img.reshape(img.shape[0]*img.shape[1]*img.shape[2],1)

def norm_rows(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True)

def softmax(x):
    num = np.exp(x)
    dem = np.sum(num,axis=1,keepdims=True)
    return num/dem

def loss_l1(y_pred,y_true):
    return np.sum(np.abs(y_pred-y_true))

def loss_l2(y_pred,y_true):
    # return np.sum(np.square(y_pred-y_true))
    return np.dot(y_pred-y_true,y_pred-y_true)


if __name__ == "__main__":
    # print(basic_sigmoid(5))
    # x = np.array([1,2,5])
    # print(np.exp(x))
    # print(sigmoid(x))
    # print(sigmoid_gradient(x))

    # x = np.array([[0,3,4],[2,6,4]])
    # print(norm_rows(x))
    # print(softmax(x))

    # vectorization

    # import time 
    # x1 = [9,2,5,0,0,10000]
    # x2 = [9,2,2,9,0,10000]
    # tic = time.process_time()
    # dot = 0
    # for i in range(len(x1)):
    #     dot += x1[i]*x2[i]
    # toc = time.process_time()
    # print('elapse time:'+str(1000*(toc-tic)))
    # print('the outer product:\n',np.outer(x1,x2))
    # print('the inner product:\n',np.dot(x1,x2))
    # print('element-wise product:\n',np.multiply(x1,x2))

    # loss
    yhat = np.array([0.9,0.2])
    y = np.array([1,0])
    print('l1 loss:'+str(loss_l1(yhat,y)))
    print('l2 loss:'+str(loss_l2(yhat,y)))