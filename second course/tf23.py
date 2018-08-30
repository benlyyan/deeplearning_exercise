import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from utils.tf_utils import load_dataset, random_mini_batches,convert_to_one_hot, predict
import scipy as sp 

def to_hot(y,c):
    m = len(y)
    x = np.zeros((m,c))
    for i in range(len(y)):
        x[i,y[i]]=1
    return x    

def ones(shape):
    res = tf.ones(shape,tf.float32)
    sess = tf.Session()
    res = sess.run(res)
    sess.close()
    return res 

def create_placeholders(n_x,n_y):
    x = tf.placeholder(tf.float32,[n_x,None],name='x')
    y = tf.placeholder(tf.float32,[n_y,None],name='y')
    return x,y  

def init_w():
    w1 = tf.get_variable('W1',[25,64*64*3],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1',[25,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    w2 = tf.get_variable('W2',[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2',[12,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    w3 = tf.get_variable('W3',[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3',[6,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    params = {'W1':w1,'W2':w2,'W3':w3,'b1':b1,'b2':b2,'b3':b3}
    return params 

def forward_propagate(x,params):
    w1 = params['W1']
    b1 = params['b1']
    w2 = params['W2']
    w3 = params['W3']
    b2 = params['b2']
    b3 = params['b3']
    z1 = tf.add(tf.matmul(w1,x),b1)
    a1 = tf.nn.relu(z1)
    z2 = tf.add(tf.matmul(w2,a1),b2)
    a2 = tf.nn.relu(z2)
    z3 = tf.add(tf.matmul(w3,a2),b3)
    return z3 

def compute_cost(z3,y):
    logits = tf.transpose(z3)
    labels = tf.transpose(y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return loss 

def model(x,y,x_test,y_test,learning_rate=0.0001,num_epochs=1500,batch_size=32,print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3 
    (n_x,m) = x.shape
    n_y = y.shape[0]
    costs = []
    X,Y = create_placeholders(n_x,n_y)
    params = init_w()
    z3 = forward_propagate(X,params)
    loss = compute_cost(z3,Y)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0. 
            num_minibatches = int(m/batch_size)
            seed = seed + 1
            batches = random_mini_batches(x,y,batch_size,seed)
            for minibatch in batches:
                (batch_x,batch_y) = minibatch 
                _,batch_cost = sess.run([opt,loss],
                    feed_dict={X:batch_x,Y:batch_y})
                epoch_cost += batch_cost/num_minibatches
            if print_cost==True and epoch%100==0:
                print('cost after epoch %i:%f' % (epoch,epoch_cost))
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.show()
        params = sess.run(params)
        pred = tf.equal(tf.argmax(z3),tf.argmax(Y))
        acc = tf.reduce_mean(tf.cast(pred,'float'))
        print('train accuracy:',acc.eval({X:x,Y:y}))
        print('test accuracy:',acc.eval({X:x_test,Y:y_test}))
        return params 

if __name__=="__main__":
    np.random.seed(1)
    x,y,x_test,y_test,classes = load_dataset()
    print(x.shape)
    print(y.shape)
    x = x.reshape(x.shape[0],-1).T 
    x_test = x_test.reshape(x_test.shape[0],-1).T
    x = x/255.
    x_test = x_test/255. 
    y_ = convert_to_one_hot(y,6)
    y_test_ = convert_to_one_hot(y_test,6)
    # print('train x shape:'+str(x.shape))
    # print('test x shape:'+str(x_test.shape))
    # print('train y shape:'+str(y_.shape))
    # print('test y shape:'+str(y_test_.shape))
    tf.reset_default_graph()
    params = model(x,y_,x_test,y_test_,num_epochs=700,learning_rate=0.001,print_cost=True,batch_size=32)

    img = sp.ndimage.imread('datasets/1.png',flatten=False)
    img = sp.misc.imresize(np.array(img),size=(64,64))
    img = img.reshape((1,-1)).T/255. 
    pred = predict(img,params)
    print(pred)




