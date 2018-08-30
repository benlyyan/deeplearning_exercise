# -*- coding:utf-8 -*-
# @author:zb
# date:2018.8

import numpy as np  
import matplotlib.pyplot as plt 
from utils.reg_utils import * 
import sklearn
import sklearn.datasets 
import scipy.io 
# from testCases import * 

def model(x,y,learning_rate=0.3,num_iter=3000,print_cost=True,lambd=0,keep_prob=1):
    grads = {}
    costs = []
    m = x.shape[1]
    layers_dims = [x.shape[0],20,3,1]
    params = initialize_parameters(layers_dims)
    for i in range(0,num_iter):
        if keep_prob==1:
            a3,cache = forward_propagation(x,params)
        elif keep_prob<1:
            a3,cache = forward_propagation_with_dropout(x,params,keep_prob)
        if lambd == 0:
            cost = compute_cost(a3,y)
        else:
            cost = compute_cost_with_reg(a3,y,params,lambd)

        if lambd ==0  and keep_prob ==1:
            grads = backward_propagation(x,y,cache)
        elif lambd!=0:
            grads = backward_propagation_with_reg(x,y,cache,lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(x,y,cache,keep_prob)
        params = update_parameters(params,grads,learning_rate)

        if print_cost and i%1000==0:
            print('cost after iter {}:{}'.format(i,cost))
            costs.append(cost)
    plt.plot(costs)
    plt.show()
    return params 

def compute_cost_with_reg(a3,y,params,lambd):
    m = y.shape[1]
    w1 = params['W1']
    w2 = params['W2']
    w3 = params['W3']
    cross_entropy = compute_cost(a3,y)
    l2 = lambd*(np.sum(np.square(w1))+np.sum(np.square(w2))+np.sum(np.square(w3)))/(2*m)
    cost = cross_entropy + l2 
    return cost 

def backward_propagation_with_reg(x,y,cache,lambd):
    m = x.shape[1]
    (z1,a1,w1,b1,z2,a2,w2,b2,z3,a3,w3,b3) = cache
    dz3 = a3-y 
    dw3 = 1./m*np.dot(dz3,a2.T)+lambd*w3/m
    db3 = 1./m*np.sum(dz3,axis=1,keepdims=True)
    da2 = np.dot(w3.T,dz3)
    dz2 = np.multiply(da2,np.int64(a2>0))
    dw2 = 1./m*np.dot(dz2,a1.T)+lambd*w2/m 
    db2 = 1./m*np.sum(dz2,axis=1,keepdims=True)
    da1 = np.dot(w2.T,dz2)
    dz1 = np.multiply(da1,np.int64(a1>0))
    dw1 = 1./m*np.dot(dz1,x.T)+lambd*w1/m
    db1 = 1./m*np.sum(dz1,axis=1,keepdims=True)
    grads = {'dz3':dz3,'dW3':dw3,'db3':db3,'da2':da2,'dz2':dz2,\
        'dW2':dw2,'db2':db2,'da1':da1,'dz1':dz1,'dW1':dw1,'db1':db1}
    return grads 

def forward_propagation_with_dropout(x,params,keep_prob=0.5):
    np.random.seed(1)
    w1 = params["W1"]
    b1 = params["b1"]
    w2 = params["W2"]
    b2 = params["b2"]
    w3 = params["W3"]
    b3 = params["b3"]
    z1 = np.dot(w1,x) + b1 
    a1 = relu(z1)
    d1 = np.random.rand(a1.shape[0],a1.shape[1])
    d1 = (d1<keep_prob)
    a1 = a1*d1 
    a1 = a1/keep_prob 
    z2 = np.dot(w2,a1) + b2 
    a2 = relu(z2)
    d2 = np.random.rand(a2.shape[0],a2.shape[1])
    d2 = (d2<keep_prob)
    a2 = a2*d2 
    a2 = a2/keep_prob 
    z3 = np.dot(w3,a2)+b3 
    a3 = sigmoid(z3)
    cache = (z1,d1,a1,w1,b1,z2,d2,a2,w2,b2,z3,a3,w3,b3)
    return a3,cache 

def backward_propagation_with_dropout(x,y,cache,keep_prob):
    m = x.shape[1]
    (z1,d1,a1,w1,b1,z2,d2,a2,w2,b2,z3,a3,w3,b3) = cache 
    dz3 = a3-y  
    dw3 = 1./m*np.dot(dz3,a2.T)
    db3 = 1./m*np.sum(dz3,axis=1,keepdims=True)
    da2 = np.dot(w3.T,dz3)
    da2 = da2*d2 
    da2 = da2/keep_prob
    dz2 = np.multiply(da2,np.int64(a2>0))
    dw2 = 1./m*np.dot(dz2,a1.T)
    db2 = 1./m*np.sum(dz2,axis=1,keepdims=True)
    da1 = np.dot(w2.T,dz2)
    da1 = da1*d1
    da1 = da1/keep_prob
    dz1 = np.multiply(da1,np.int64(a1>0))
    dw1 = 1./m*np.dot(dz1,x.T)
    db1 = 1./m*np.sum(dz1,axis=1,keepdims=True)
    grads = {'dz3':dz3,'dW3':dw3,'db3':db3,'da2':da2,'dz2':dz2,\
        'dW2':dw2,'db2':db2,'da1':da1,'dz1':dz1,'dW1':dw1,'db1':db1}
    return grads



if __name__=="__main__":
    x,y,test_x,test_y = load_2D_dataset()
    # print(x.shape)
    params = model(x,y,keep_prob=0.86,learning_rate=0.3)
    pred_train = predict(x,y,params)
    pred_test = predict(test_x,test_y,params)
    plot_decision_boundary(lambda x:predict_dec(params,x.T),x,y)

