# -*- coding:utf-8 -*-
# @author:zb
# date:2018.8

import numpy as np  
import matplotlib.pyplot as plt 
import sklearn
import sklearn.datasets
from utils.init_utils import update_parameters,predict,load_dataset,plot_decision_boundary,predict_dec,\
    forward_propagation,backward_propagation,compute_loss,update_parameters

def model(x,y,learning_rate=0.01,num_iter=1000,
        print_cost=True,init='he'):
    grads = {}
    costs = []
    m = x.shape[1]
    layers_dims = [x.shape[0],10,5,1]
    if init=='zeros':
        params = init_params_zeros(layers_dims)
    elif init=='random':
        params = init_params_random(layers_dims)
    elif init=='he':
        params = init_params_he(layers_dims)
    for i in range(0,num_iter):
        a3,cache = forward_propagation(x,params)
        cost = compute_loss(a3,y)
        grads = backward_propagation(x,y,cache)
        params = update_parameters(params,grads,learning_rate)
        if print_cost and i%100==0:
            print('cost after iter {}:{}'.format(i,cost))
            costs.append(cost)
    plt.plot(costs)
    plt.show()
    return params 

def init_params_zeros(layers_dims):
    params = {} 
    l = len(layers_dims)
    for i in range(1,l):
        params['W'+str(i)] = np.zeros((layers_dims[i],layers_dims[i-1]))
        params['b'+str(i)] = np.zeros((layers_dims[i],1))
    return params 

def init_params_random(layers_dims):
    params = {} 
    l = len(layers_dims)
    for i in range(1,l):
        params['W'+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])
        params['b'+str(i)] = np.zeros((layers_dims[i],1))
    return params 

def init_params_he(layers_dims):
    params = {} 
    l = len(layers_dims)
    for i in range(1,l):
        params['W'+str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])*np.sqrt(2./layers_dims[i-1])
        params['b'+str(i)] = np.zeros((layers_dims[i],1))
    return params 

if __name__=="__main__":
    x,y,x_test,y_test = load_dataset()
    # params = model(x,y,init='zeros')
    # params = model(x,y,init='random')
    params = model(x,y,init='he')
    pred_train = predict(x,y,params)
    pred_test = predict(x_test,y_test,params)

