# coding:utf-8
# author:zb
# date:2018.8


import numpy as np 
import matplotlib.pyplot as plt 
from utils.dnn_utils_v2 import relu_backward,sigmoid_backward
import h5py
import scipy 
from PIL import Image 
from utils.lr_utils import load_dataset 

def init_w(n_x,n_h,n_y):
    w1 = np.random.randn(n_h,n_x)*1./np.sqrt(n_x)
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h)*1./np.sqrt(n_h)
    b2 = np.zeros((n_y,1))
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    b2 = np.zeros((n_y,1))
    return params 

def init_w_deep(layer_dims):
    l = len(layer_dims)
    params = {}
    for i in range(1,l):
        params['w'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*1./np.sqrt(layer_dims[i-1])
        params['b'+str(i)] = np.zeros((layer_dims[i],1))

    return params

def linear_forward(w,a,b):
    z = np.dot(w,a) + b 
    cache = (a,w,b)
    return z,cache 

def sigmoid(x):
    a = 1./(1+np.exp(-x)) 
    return a,x 

def relu(x):
    a = np.maximum(0,x)
    return a,x 

def linear_activation_forward(a_prev,w,b,activation):
    if activation=='sigmoid':
        z,linear_cache = linear_forward(w,a_prev,b)
        a,activation_cache = sigmoid(z)
    elif activation=='relu':
        z,linear_cache = linear_forward(w,a_prev,b)
        a,activation_cache = relu(z)
    cache =(linear_cache,activation_cache)
    return a,cache 

def l_model_forward(x,params):
    l = len(params)//2 
    a = x 
    caches = []
    for i in range(1,l):
        a_prev = a 
        a,cache = linear_activation_forward(a_prev,params['w'+str(i)],params['b'+str(i)],'relu')
        caches.append(cache)
    al,cache = linear_activation_forward(a,params['w'+str(l)],params['b'+str(l)],'sigmoid')
    caches.append(cache) 
    return al,caches 

def compute_cost(al,y):
    m = al.shape[1]
    cost = -(np.dot(y,np.log(al.T))+np.dot(1-y,np.log(1-al.T)))/m
    cost = np.squeeze(cost)
    # print(cost)
    return cost 

def linear_backward(dz,cache):
    a_prev,w,b = cache
    m = a_prev.shape[1]
    dw = np.dot(dz,a_prev.T)/m 
    db = np.sum(dz,axis=1,keepdims=True)/m
    da_prev = np.dot(w.T,dz)
    assert(da_prev.shape==a_prev.shape)
    assert(dw.shape==w.shape)
    assert(db.shape==b.shape)
    return da_prev,dw,db

def linear_activation_backward(da,cache,activation):
    linear_cache,activation_cache = cache
    if activation == 'sigmoid':
        dz = sigmoid_backward(da,activation_cache)
        da_prev,dw,db = linear_backward(dz,linear_cache)
    elif activation == 'relu':
        dz = relu_backward(da,activation_cache)
        da_prev,dw,db = linear_backward(dz,linear_cache)
    return da_prev,dw,db 

def l_model_backward(al,y,caches):
    grads = {}
    l = len(caches)
    m = al.shape[1]
    y = y.reshape(al.shape)
    dal = -(np.divide(y,al) - np.divide(1-y,1-al))
    current_cache = caches[l-1]
    grads['da'+str(l)],grads['dw'+str(l)],grads['db'+str(l)] = linear_activation_backward(dal,current_cache,'sigmoid')
    for i in reversed(range(l-1)):
        current_cache = caches[i]
        da_prev,dw,db = linear_activation_backward(grads['da'+str(i+2)],current_cache,'relu')
        grads['da'+str(i+1)] = da_prev
        grads['dw'+str(i+1)] = dw 
        grads['db'+str(i+1)] = db 
    return grads 

def update_params(params,grads,learning_rate):
    l = len(params)//2
    for i in range(l):
        params['w'+str(i+1)] = params['w'+str(i+1)] - learning_rate*grads['dw'+str(i+1)]
        params['b'+str(i+1)] = params['b'+str(i+1)] - learning_rate*grads['db'+str(i+1)]
    return params 

def two_layer_model(x,y,layer_dims,learning_rate=0.001,num_iter=3000,print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = x.shape[1]
    n_x,n_h,n_y = layer_dims 
    params = init_w(n_x,n_h,n_y)
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    for i in range(0,num_iter):
        a1,cache1 = linear_activation_forward(x,w1,b1,'relu')
        a2,cache2 = linear_activation_forward(a1,w2,b2,'sigmoid')
        cost = compute_cost(a2,y)
        da2 = -(np.divide(y,a2)-np.divide(1-y,1-a2))
        da1,dw2,db2 = linear_activation_backward(da2,cache2,'sigmoid')
        da0,dw1,db1 = linear_activation_backward(da1,cache1,'relu')
        grads['dw1'] = dw1
        grads['dw2'] = dw2 
        grads['db1'] = db1 
        grads['db2'] = db2 
        params = update_params(params,grads,learning_rate)
        w1 = params['w1']
        w2 = params['w2']
        b1 = params['b1']
        b2 = params['b2']
        if print_cost and i%100==0:
            print('cost after iter:{}:{}'.format(i,np.squeeze(cost)))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.show()
    return params 

def l_layer_model(x,y,layer_dims,learning_rate=0.005,num_iter=3000,print_cost=False):
    np.random.seed(1)
    costs = []
    params = init_w_deep(layer_dims)
    for i in range(0,num_iter):
        al,caches = l_model_forward(x,params)
        cost = compute_cost(al,y)
        grads = l_model_backward(al,y,caches)
        params = update_params(params,grads,learning_rate)
        if print_cost and i%100 == 0:
            print('cost after iter:{}:{}'.format(i,cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.show()
    return params 


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = l_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))   
    return p

if __name__ == "__main__":
    np.random.seed(1)
    train_x_orig,train_y,test_x_orig,test_y,classes = load_dataset()
    train_x = train_x_orig.reshape(train_x_orig.shape[0],-1).T/255.
    test_x = test_x_orig.reshape(test_x_orig.shape[0],-1).T/255. 
    # index = 10
    # plt.imshow(train_x_orig[index])
    # plt.show()
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]

    n_h = 7
    n_y =1
    layer_dims = (num_px*num_px*3,5,7,2,1)
    # layer_dims = (num_px*num_px*3,n_h,n_y)
    # params = two_layer_model(train_x,train_y,layer_dims=layer_dims,learning_rate=0.05,num_iter=2500,print_cost=True)
    # pred_train = predict(train_x,train_y,params)
    # pred_test = predict(test_x,test_y,params)
    params = l_layer_model(train_x,train_y,layer_dims,learning_rate=0.001,num_iter=500,print_cost=True)
    pred_train = predict(train_x,train_y,params)
    pred_test = predict(test_x,test_y,params)




