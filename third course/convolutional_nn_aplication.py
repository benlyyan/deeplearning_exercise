# -*- coding:utf-8 -*-

import math 
import numpy as np  
import h5py  
import matplotlib.pyplot as plt 
import scipy 
from scipy import ndimage 
import tensorflow as tf 
from tensorflow.python.framework import ops 
from utils.cnn_utils import * 

def create_placeholders(n_h,n_w,n_c,n_y):
    X = tf.placeholder(tf.float32,shape=[None,n_h,n_w,n_c])
    Y = tf.placeholder(tf.float32,shape=[None,n_y])
    return X,Y 
    
def init_w():
    tf.set_random_seed(1)
    w1 = tf.get_variable('W1',[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2 = tf.get_variable('W2',[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    params = {'W1':w1,'W2':w2}
    return params 

def forward_propagation(x,params):
    w1 = params['W1']
    w2 = params['W2']
    z1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
    a1 = tf.nn.relu(z1)
    p1 = tf.nn.max_pool(a1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
    z2 = tf.nn.conv2d(p1,w2,strides=[1,1,1,1],padding='SAME')
    a2 = tf.nn.relu(z2)
    p2 = tf.nn.max_pool(a2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    p2 = tf.contrib.layers.flatten(p2)
    z3 = tf.contrib.layers.fully_connected(p2,6,activation_fn=None)
    return z3 

def compute_cost(z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3,labels=Y))
    return cost 

def model(x,y,xtest,ytest,learning_rate=0.009,num_epochs=100,minibatch_size=64,print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    (m,h,w,c) = x.shape
    n_y = y.shape[1]
    costs = []
    seed = 3
    X,Y = create_placeholders(h,w,c,n_y)
    params = init_w()
    z3 = forward_propagation(X,params)
    cost = compute_cost(z3,Y)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_batches = int(m/minibatch_size)
            seed = seed + 1 
            minibatches = random_mini_batches(x,y,minibatch_size,seed)
            for minibatch in minibatches:
                (batchx,batchy) = minibatch
                _,temp_cost = sess.run([opt,cost],feed_dict={X:batchx,Y:batchy})
                minibatch_cost += temp_cost/num_batches
            if print_cost == True and epoch%10==0:
                print('cost after epoch %i:%f' %(epoch,minibatch_cost))
                costs.append(minibatch_cost)
        plt.plot(np.squeeze(costs))
        plt.show()
        pred = tf.argmax(z3,1)
        correct_pred = tf.equal(pred,tf.argmax(Y,1))
        acc = tf.reduce_mean(tf.cast(correct_pred,'float'))
        print(acc)
        trainacc = acc.eval({X:x,Y:y})
        testacc = acc.eval({X:xtest,Y:ytest})
        print(trainacc,testacc)
        sess.run(params)
        return trainacc,testacc,params




if __name__=="__main__":
    x,y,xtest,ytest,classes = load_dataset()
    x =x/255.
    xtest = xtest/255.
    y = convert_to_one_hot(y,6).T 
    ytest = convert_to_one_hot(ytest,6).T 
    # print('x shape:'+str(x.shape))
    # print('y shape:'+str(y.shape))
    X,Y = create_placeholders(64,64,3,6)
    print(X,Y)
    _,_,params = model(x,y,xtest,ytest,learning_rate=0.01,num_epochs=200)
    print(type(params),params.keys())
