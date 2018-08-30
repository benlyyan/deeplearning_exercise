# -*- coding:utf-8 -*-
# @author:zb
# date:2018.8

import sklearn
import sklearn.datasets 
from utils.opt_utils import * 
import matplotlib.pyplot as plt 
import math

def update_params_with_gd(params,grads,learning_rate):
    l = len(params)//2
    for i in range(l):
        params['W'+str(i+1)] -= learning_rate*grads['dW'+str(i+1)]
        params['b'+str(i+1)] -= learning_rate*grads['db'+str(i+1)]
    return params 

def update_params_with_moment(params,grads,v,learning_rate,beta):
    l = len(params)//2 
    for i in range(l):
        v['dW'+str(i+1)] = beta*v['dW'+str(i+1)]+(1-beta)*grads['dW'+str(i+1)]
        v['db'+str(i+1)] = beta*v['db'+str(i+1)]+(1-beta)*grads['db'+str(i+1)]
        params['W'+str(i+1)] -= learning_rate*v['dW'+str(i+1)]
        params['b'+str(i+1)] -= learning_rate*v['db'+str(i+1)]
    return params,v


def init_v(params):
    l = len(params)//2
    v = {}
    for i in range(l):
        v['dW'+str(i+1)] = np.zeros(params['W'+str(i+1)].shape)
        v['db'+str(i+1)] = np.zeros(params['b'+str(i+1)].shape)
    return v

def init_s(params):
    l = len(params)//2
    s ={}
    for i in range(l):
        s['dW'+str(i+1)] = np.zeros(params['W'+str(i+1)].shape)
        s['db'+str(i+1)] = np.zeros(params['b'+str(i+1)].shape)
    return s 

def update_params_with_adam(params,grads,v,s,learning_rate,beta1,beta2,t,eps=1e-6):
    l = len(params)//2
    v_c = {}
    s_c = {}
    for i in range(l):
        v['dW'+str(i+1)] = beta1*v['dW'+str(i+1)]+(1-beta1)*grads['dW'+str(i+1)]
        v_c['dW'+str(i+1)] =v['dW'+str(i+1)]/(1-math.pow(beta1,t))
        s['dW'+str(i+1)] = beta2*s['dW'+str(i+1)]+(1-beta2)*grads['dW'+str(i+1)]**2
        s_c['dW'+str(i+1)] = s['dW'+str(i+1)]/(1-math.pow(beta2,t))
        params['W'+str(i+1)] -= learning_rate*v_c['dW'+str(i+1)]/(np.sqrt(s_c['dW'+str(i+1)])+eps)
        v['db'+str(i+1)] = beta1*v['db'+str(i+1)]+(1-beta1)*grads['db'+str(i+1)]
        v_c['db'+str(i+1)] =v['db'+str(i+1)]/(1-math.pow(beta1,t))
        s['db'+str(i+1)] = beta2*s['db'+str(i+1)]+(1-beta2)*grads['db'+str(i+1)]**2
        s_c['db'+str(i+1)] = s['db'+str(i+1)]/(1-math.pow(beta2,t))
        params['b'+str(i+1)] -= learning_rate*v_c['db'+str(i+1)]/(np.sqrt(s_c['db'+str(i+1)])+eps)
    return params,v,s


def random_batch(x,y,batch_size=64,seed=0):
    np.random.seed(seed)
    m = x.shape[1]
    num_comp = int(m/batch_size)
    batches = []
    for i in range(0,num_comp):
        batch_x = x[:,i*batch_size:(i+1)*batch_size]
        batch_y = y[:,i*batch_size:(i+1)*batch_size]
        batches.append((batch_x,batch_y))
    if batch_size*num_comp<m:
        batch_x = x[:,num_comp*batch_size:]
        batch_y = y[:,num_comp*batch_size:]
        batches.append((batch_x,batch_y))
    return batches 

def test(x,y,layers_dims,num_iter=1000,learning_rate=0.001,beta=0.9,method='gd',print_cost=True,beta1=0.9,beta2=0.99):
    m = x.shape[1]
    costs = []
    params = initialize_parameters(layers_dims)
    v = init_v(params)
    s = init_s(params)
    t = 0 
    for i in range(0,num_iter):
        a,cache = forward_propagation(x,params)
        cost = compute_cost(a,y)
        grads = backward_propagation(x,y,cache)
        if method == 'gd':
            params = update_params_with_gd(params,grads,learning_rate)
        elif method == 'moment':
            params,v = update_params_with_moment(params,grads,v,learning_rate,beta)
        elif method =='adam':
            t += 1 
            params,v,s = update_params_with_adam(params,grads,v,s,learning_rate,beta1,beta2,t)
        if print_cost and i%100==0:
            print('cost after iter {}:{}'.format(i,cost))
            costs.append(cost)
    plt.plot(costs)
    plt.show()
    return params 

if __name__=="__main__":
    x,y = load_dataset(is_plot=False)
    layers_dims = [x.shape[0],5,2,1]
    # params = test(x,y,layers_dims,num_iter=2000,learning_rate=0.01,method='gd')
    # pred_x = predict(x,y,params)
    # plot_decision_boundary(lambda x:predict(x.T,y,params),x,y)
    # params = test(x,y,layers_dims,num_iter=2000,learning_rate=0.01,method='moment')
    # params = test(x,y,layers_dims,num_iter=2000,learning_rate=0.01,method='adam')
    # pred_x = predict(x,y,params)
    # plot_decision_boundary(lambda x:predict(x.T,y,params),x,y)
    batches = random_batch(x,y)

