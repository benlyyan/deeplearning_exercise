# coding:utf-8
"""
@author:zb
@date:2018.7
"""

import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
import scipy as sp 
from PIL import Image 
from utils.lr_utils import load_dataset

def sigmoid(x):
    return 1./(1+np.exp(-x))

def init_w(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape==(dim,1))
    return w,b 

def propagate(w,b,x,y):
    m = x.shape[1]
    a = sigmoid(np.dot(w.T,x) + b) 
    cost = -1./m*(y.dot(np.log(a.T)) + (1-y).dot(np.log((1-a).T)))
    dw = 1./m*x.dot((a-y).T) 
    db = 1./m*np.sum(a-y)
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    grads = {'dw':dw,'db':db}
    cost = np.squeeze(cost)
    return grads,cost


def opt(w,b,x,y,num_iter,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iter):
        grads,cost = propagate(w,b,x,y)
        w -= grads['dw']*learning_rate 
        b -= grads['db']*learning_rate 
        if i%100==0:
            costs.append(cost)
        if print_cost and i%100==0:
            print('iteration num:'+str(i)+'\ncost:'+str(cost))
    params = {'w':w,'b':b}
    return params,grads,costs

def opt_early_stop(w,b,x,y,xtest,ytest,num_iter,learning_rate,print_cost=False):
    costs = []
    costs_test = []
    train_errors = []
    test_errors = []
    for i in range(num_iter):
        grads,cost = propagate(w,b,x,y)
        w -= grads['dw']*learning_rate
        b -= grads['db']*learning_rate
        _,cost_test = propagate(w,b,xtest,ytest)
        if i%100==0:
            costs.append(cost)
            costs_test.append(cost_test)
            ypred = predict(w,b,x)
            ypred_test = predict(w,b,xtest)
            train_error = np.mean(np.abs(y-ypred))*100
            test_error = np.mean(np.abs(ytest-ypred_test))*100
            train_errors.append(train_error)
            test_errors.append(test_error)
        if print_cost and i%100==0:
            print('iteration num:'+str(i)+'\ntrain error:'+str(cost))
    bestparams = {'w':w,'b':b}
    return bestparams,grads,costs,costs_test,train_errors,test_errors    



def predict(w,b,x):
    m = x.shape[1]
    ypred = np.zeros((1,m))
    a = sigmoid(np.dot(w.T,x)+b)
    ypred[a>0.5] = 1
    assert(ypred.shape==(1,m))
    return ypred 


def test():
    w,b,x,y = np.array([[1.],[2.]]),2.,np.array([[1.,2.,-1],[3.,4.,-3.2]]),np.array([[1,0,1]])
    grads,cost = propagate(w,b,x,y)
    print(grads)
    print(cost)

    params,grads,costs = opt(w,b,x,y,100,0.009)
    print(params)
    print(grads)

    ypred = predict(params['w'],params['b'],x)
    print(ypred)

def model(x,y,xtest,ytest,num_iter,learning_rate=0.5,print_cost=False):
    w,b = init_w(x.shape[0])
    params,grads,costs = opt(w,b,x,y,num_iter,learning_rate,print_cost)
    w = params['w']
    b = params['b']
    ypred_test = predict(w,b,xtest)
    ypred_train = predict(w,b,x)
    print('train acc:{}%'.format(100-np.mean(np.abs(y-ypred_train))*100))
    print('test acc:{}%'.format(100-np.mean(np.abs(ytest-ypred_test))*100))
    d = {'costs':costs,'ypred_test':ypred_test,
    'ypred_train':ypred_train,'w':w,'b':b}
    return d 

def model_early_stop(x,y,xtest,ytest,num_iter,learning_rate=0.5,print_cost=False):
    w,b = init_w(x.shape[0])
    params,grads,costs,costs_test,train_errors,test_errors = opt_early_stop(w,b,x,y,xtest,ytest,num_iter,learning_rate,print_cost)
    w = params['w']
    b = params['b']
    ypred_test = predict(w,b,xtest)
    ypred_train = predict(w,b,x)
    print('train acc:{}%'.format(100-np.mean(np.abs(y-ypred_train))*100))
    print('test acc:{}%'.format(100-np.mean(np.abs(ytest-ypred_test))*100))
    d = {'costs':costs,'costs_test':costs_test,'ypred_test':ypred_test,
    'ypred_train':ypred_train,'w':w,'b':b,'train_errors':train_errors,'test_errors':test_errors}
    return d 

    


if __name__=="__main__":
    train_x_orig,train_y,test_x_orig,test_y,classes = load_dataset()
    print('the train x shape:\n',train_x_orig.shape)
    print('the train y shape:\n',train_y.shape)
    print('the class shape:\n',classes.shape)

    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

    print('the train x flatten shape:\n',train_x_flatten.shape)
    print('the test x flatten shape:\n',test_x_flatten.shape)

    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    # d = model(train_x,train_y,test_x,test_y,num_iter=2000,learning_rate=0.005,print_cost=True)
    d = model_early_stop(train_x,train_y,test_x,test_y,num_iter=2000,learning_rate=0.005,print_cost=True)
    costs = np.squeeze(d['costs'])
    fig,axes = plt.subplots(1,2)
    axes[0].plot(costs)
    axes[0].plot(np.squeeze(d['costs_test']))
    axes[1].plot(d['train_errors'])
    axes[1].plot(d['test_errors'])
    axes[0].set_ylabel('cost')
    axes[0].set_xlabel('num iter')
    axes[0].set_title('evolve curve')
    axes[1].set_xlabel('num iter')
    axes[1].set_ylabel('acc')
    axes[1].set_title('evolve curve')
    plt.show()


    

