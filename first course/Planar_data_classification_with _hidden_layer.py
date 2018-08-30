# coding:utf-8
"""
@author:zb
@date:2018.8
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
from utils.planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

def sample_linear_logistic(X,Y):
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X.T,Y.T)
    plot_decision_boundary(lambda x:clf.predict(x),X,Y)
    plt.title('lr')
    plt.show()
    ypred = clf.predict(X.T)
    print('the acc is:\n',np.mean(ypred==Y.T))

def layer_size(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x,n_h,n_y

def init_w(n_x,n_h,n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h,n_x)*1./np.sqrt(n_x)
    b1 = np.zeros((n_h,1)) 
    w2 = np.random.randn(n_y,n_h)*1./np.sqrt(n_h)
    b2 = np.zeros((n_y,1))
    assert (w1.shape==(n_h,n_x))
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    return params 

def forward_propagate(x,params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    z1 = w1.dot(x) + b1 
    a1 = np.tanh(z1)
    z2 = w2.dot(a1) + b2 
    a2 = sigmoid(z2)
    assert(a2.shape==(1,x.shape[1]))
    cache = {'z1':z1,'a1':a1,'z2':z2,'a2':a2}
    return a2,cache 

def compute_cost(a2,y,params):
    m = y.shape[1]
    cost = y.dot(a2.T) + (1-y).dot(1-a2.T)
    cost = -cost/m
    cost = np.squeeze(cost)
    return cost 

def backward_propagate(params,cache,x,y):
    m = y.shape[1]
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    a1 = cache['a1']
    a2 = cache['a2']
    dz2 = a2-y
    dw2 = dz2.dot(a1.T)/m
    db2 = np.sum(dz2,axis=1,keepdims=True)/m 
    dz1 = np.dot(w2.T,dz2)*(1-np.power(a1,2))
    dw1 = np.dot(dz1,x.T)/m 
    db1 = np.sum(dz1,axis=1,keepdims=True)/m 
    grads = {'dw1':dw1,'db1':db1,'dw2':dw2,'db2':db2}
    return grads 

def update_params(params,grads,learning_rate=1.2):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    dw1 = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']
    w1 -= learning_rate*dw1 
    b1 -= learning_rate*db1 
    w2 -= learning_rate*dw2 
    b2 -= learning_rate*db2 
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    return params 

def nn_model(x,y,n_h,num_iter=10000,learning_rate=0.001,print_cost=False):
    np.random.seed(3)
    n_x = layer_size(x,y)[0]
    n_y = layer_size(x,y)[2]
    params = init_w(n_x,n_h,n_y)
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    for i in range(0,num_iter):
        a2,cache = forward_propagate(x,params)
        cost = compute_cost(a2,y,params)
        grads = backward_propagate(params,cache,x,y)
        params = update_params(params,grads,learning_rate)
        if print_cost and i%1000==0:
            print('cost after iter %i:%f' %(i,cost))
    return params 

def predict(params,x):
    a2,cache = forward_propagate(x,params)
    pred = np.round(a2)
    return pred 

if __name__=="__main__":
    np.random.seed(1)
    x,y = load_planar_dataset()
    # plt.figure(figsize=(5,8))
    # plt.scatter(x[0,:],x[1,:],c=y,s=20,cmap=plt.cm.Spectral)
    # plt.show()
    # sample_linear_logistic(x,y)
    print('the shape x is:'+str(x.shape))
    print('the shape y is:'+str(y.shape))
    m = x.shape[1]
    # params = nn_model(x,y,n_h=4,num_iter=10000,print_cost=True,learning_rate=1.2)
    # plot_decision_boundary(lambda x:predict(params,x.T),x,y)
    # plt.show()
    # pred = predict(params,x)
    # print('Accuracy: %d' % float((np.dot(y,pred.T) +np.dot(1-y,1-pred.T))/float(y.size)*100) + '%')

    plt.figure(figsize=(20,20))
    hidden_layer_sizes = [1,2,3,4,5,20,50]
    for i,n_h in enumerate(hidden_layer_sizes):
        plt.subplot(4,2,i+1)
        params = nn_model(x,y,n_h,num_iter=5000,learning_rate=0.1)
        plot_decision_boundary(lambda x:predict(params,x.T),x,y)
        pred = predict(params,x)
        acc = np.mean(pred==y)
        print('acc for {} hidden units:{}%'.format(n_h,acc))
    plt.show()





